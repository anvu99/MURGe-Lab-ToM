"""
DebateArena — orchestrator for multi-agent debates.

Constructs agents from class lists, runs multi-round debates on MMLU-style
questions, determines a final answer via majority vote, and updates agent
memories after each debate.
"""

import logging
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Type

from vllm import LLM, SamplingParams

from agents.base import BaseAgent
from configs.configs import AgentResponse, Conversation, RoundEntry
from memory.base import BaseMemory
from utils import strip_hallucinated_turns, strip_think_blocks

# Imported lazily to avoid a circular dependency at module load time.
# The arena checks isinstance(agent, ThinkThenSpeakDebater) only at runtime.
try:
    from agents.tts_debater import ThinkThenSpeakDebater as _TTS
except ImportError:
    _TTS = None  # type: ignore

logger = logging.getLogger(__name__)


class DebateArena:
    """
    Orchestrates multi-agent debates.

    Construction:
        - Takes lists of agent classes, memory classes, and model names
          (all same length).
        - Builds a shared LLM cache so the same model is never loaded twice.

    Execution:
        - ``run(question_data)`` runs a full debate and returns a result dict
          compatible with ``RoundAnalyzer``.
    """

    def __init__(
        self,
        agent_classes: List[Type[BaseAgent]],
        memory_classes: List[Optional[Type[BaseMemory]]],
        model_names: List[str],
        num_rounds: int = 3,
        agent_names: Optional[List[str]] = None,
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        sampling_params: Optional[List[Optional[Any]]] = None,
        observer: Optional[Any] = None,
    ):
        """
        Args:
            agent_classes: List of agent classes to instantiate
                           (e.g., [DebaterAgent, DebaterAgent]).
            memory_classes: Parallel list of memory classes (or None)
                            for each agent (e.g., [ToMMemory, None]).
            model_names: Parallel list of model identifiers, one per agent
                         (e.g., ["Qwen/Qwen3-32B", "Qwen/Qwen3-32B"]).
                         Same model name → same cached LLM instance.
            num_rounds: Number of debate rounds.
            agent_names: Optional custom names. Auto-generated if None.
            model_kwargs: Optional dict mapping model_name to kwargs for vllm.LLM().
            sampling_params: Optional list of vLLM SamplingParams, one per agent,
                             to override the default temperature/max_tokens.
            observer: Optional ObserverAgent instance to monitor and steer debates.
        """
        n = len(agent_classes)
        if len(memory_classes) != n or len(model_names) != n:
            raise ValueError(
                f"agent_classes ({n}), memory_classes ({len(memory_classes)}), "
                f"and model_names ({len(model_names)}) must all have the "
                f"same length."
            )
        if sampling_params is not None and len(sampling_params) != n:
            raise ValueError(
                f"sampling_params ({len(sampling_params)}) must match "
                f"agent_classes ({n})."
            )
        if n == 0:
            raise ValueError("Must provide at least one agent class.")

        self.num_rounds = num_rounds
        self.n_agents = n
        self._sampling_params = sampling_params or [None] * n
        self.observer = observer

        # Generate names if not provided
        if agent_names is None:
            agent_names = [f"Agent_{i}" for i in range(self.n_agents)]
        elif len(agent_names) != self.n_agents:
            raise ValueError(
                f"agent_names ({len(agent_names)}) must match "
                f"agent_classes ({self.n_agents})."
            )

        # ---- Load LLMs (deduplicated) ----
        self._llm_cache: Dict[str, LLM] = {}
        for model_name in model_names:
            if model_name not in self._llm_cache:
                kwargs = {}
                if model_kwargs and model_name in model_kwargs:
                    kwargs = dict(model_kwargs[model_name])  # copy to avoid mutation

                # 'visible_devices' pins the model to a specific GPU via
                # CUDA_VISIBLE_DEVICES. Strip it before passing to vLLM since
                # it is not a valid vLLM LLM() parameter.
                visible_devices = kwargs.pop("visible_devices", None)
                if visible_devices is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
                    logger.info(
                        "Loading LLM '%s' on GPU(s) %s with kwargs: %s ...",
                        model_name, visible_devices, kwargs,
                    )
                else:
                    logger.info("Loading LLM '%s' with kwargs: %s ...", model_name, kwargs)

                self._llm_cache[model_name] = LLM(model=model_name, **kwargs)


        # ---- Construct agents ----
        self.agents: List[BaseAgent] = []
        for i, (agent_cls, mem_cls, model_name, s_params) in enumerate(
            zip(agent_classes, memory_classes, model_names, self._sampling_params)
        ):
            llm = self._llm_cache[model_name]

            # Build memory (if any)
            memory = None
            if mem_cls is not None:
                memory = mem_cls(llm=llm, owner_name=agent_names[i])

            # Build agent — uses each class's DEFAULT_SYSTEM_PROMPT
            agent = agent_cls(
                agent_id=f"agent_{i}",
                name=agent_names[i],
                model_name=model_name,
                llm=llm,
                memory=memory,
                sampling_params=s_params,
            )
            self.agents.append(agent)

        logger.info(
            "DebateArena created: %d agents, %d rounds, models=%s",
            self.n_agents,
            self.num_rounds,
            list(self._llm_cache.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a full multi-round debate on one question.

        Args:
            question_data: A dict from the MMLU loader with keys:
                           'question', 'options', 'answer', 'category',
                           'question_id'.

        Returns:
            A result dict compatible with RoundAnalyzer:
            {
                "question_id": ...,
                "correct_answer": ...,
                "category": ...,
                "rounds": [{agent_id: answer, ...}, ...],
                "final_answer": ... or "unresolved",
            }
        """
        question_prompt = self._format_question(question_data)
        conversation: Conversation = []

        logger.info(
            "Starting debate on question '%s' (%s)",
            question_data.get("question_id", "?"),
            question_data.get("category", "?"),
        )

        # ---- Run debate rounds ----
        all_observer_flags = []
        for round_idx in range(self.num_rounds):
            logger.info("--- Round %d ---", round_idx)
            round_entry = self._run_round(
                question_prompt, conversation
            )
            # Collect flags if this round had any
            if hasattr(round_entry, "observer_flags"):
                all_observer_flags.extend(round_entry.observer_flags)
            conversation.append(round_entry)

        # ---- Determine final answer ----
        final_answer = self._determine_final_answer(conversation)

        # ---- Update memories ----
        self._update_memories(conversation, result=final_answer, question_data=question_data)

        # ---- Build result dict ----
        rounds_data: List[Dict[str, str]] = []
        for round_entry in conversation:
            round_answers = {}
            for agent_id, response in round_entry.agent_responses.items():
                round_answers[agent_id] = response.answer
            rounds_data.append(round_answers)

        result = {
            "question_id": question_data.get("question_id", ""),
            "correct_answer": question_data.get("answer", ""),
            "category": question_data.get("category", ""),
            "rounds": rounds_data,
            "final_answer": final_answer,
            "observer_flags": all_observer_flags,
            "total_sycophancy_flags": sum(1 for f in all_observer_flags if f.get("flag_type") == "sycophancy"),
            "total_repetition_flags": sum(1 for f in all_observer_flags if f.get("flag_type") == "repetition"),
            "total_inconsistency_flags": sum(1 for f in all_observer_flags if f.get("flag_type") == "inconsistency"),
        }

        logger.info(
            "Debate finished. Final answer: %s (correct: %s)",
            final_answer,
            result["correct_answer"],
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_question(self, question_data: Dict[str, Any]) -> str:
        """
        Format an MMLU-style question into a prompt string.

        Args:
            question_data: Dict with 'question' and 'options' keys.

        Returns:
            A formatted question string with lettered options.
        """
        question = question_data["question"]
        options = question_data["options"]

        option_lines = "\n".join(
            f"  {letter}. {text}" for letter, text in sorted(options.items())
        )
        return f"Question: {question}\n\nOptions:\n{option_lines}"

    def _run_round(
        self,
        question_prompt: str,
        conversation: Conversation,
    ) -> RoundEntry:
        """
        Execute one debate round with LLM-grouped batching.

        Agents sharing the same LLM instance are batched together in a single
        call. Agents using different LLMs (heterogeneous setup) are handled in
        separate sequential calls — one per model.

        This means:
          Same-model setup (2 agents, 1 LLM):
            2 LLM calls per round (1 reasoning batch + 1 answer batch)
          Hetero setup (2 agents, 2 LLMs):
            4 LLM calls per round (1 reasoning + 1 answer per model)

        Args:
            question_prompt: The formatted question string.
            conversation: The conversation history so far (previous rounds only).

        Returns:
            A RoundEntry with all agents' responses.
        """
        # ---- Phase 1: Build all reasoning prompts ----
        states = [
            agent.prepare_round(question_prompt, conversation)
            for agent in self.agents
        ]

        # ---- Phase 2: Group agents by LLM instance ----
        # Each group is a list of (global_index, agent, state) tuples.
        # Agents in the same group share the same LLM and can be batched.
        llm_groups: dict = defaultdict(list)
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            llm_groups[id(agent.llm)].append((i, agent, state))

        # ---- Phase 3: Batch reasoning generation per LLM group ----
        reasonings: list = [""] * len(self.agents)
        for group in llm_groups.values():
            llm = group[0][1].llm
            sampling_params = group[0][1].sampling_params
            is_gemma = "gemma" in group[0][1].model_name.lower()
            if is_gemma:
                reasoning_messages = [
                    [
                        {"role": "user", "content": f"{state['system']}\n\n{state['reasoning_prompt']}"},
                    ]
                    for _, _, state in group
                ]
            else:
                reasoning_messages = [
                    [
                        {"role": "system", "content": state["system"]},
                        {"role": "user",   "content": state["reasoning_prompt"]},
                    ]
                    for _, _, state in group
                ]
            try:
                outputs = llm.chat(
                    messages=reasoning_messages,
                    sampling_params=sampling_params,
                )
                for (i, _, _), out in zip(group, outputs):
                    reasonings[i] = out.outputs[0].text.strip() if out.outputs else ""
            except Exception as e:
                logger.error("Batch reasoning call failed: %s", e)

        # ---- Phase 4: Assemble round entry (finish_round per agent) ----
        round_entry = RoundEntry()
        for agent, state, reasoning in zip(
            self.agents, states, reasonings
        ):
            clean_reasoning = strip_think_blocks(reasoning)
            clean_reasoning = strip_hallucinated_turns(clean_reasoning)
            response = agent.finish_round(state, clean_reasoning)
            round_entry.agent_responses[agent.agent_id] = response
            logger.info(
                "  %s (%s):\n--- REASONING & ANSWER ---\n%s\n--- EXTRACTED ANSWER ---: %s",
                agent.name,
                agent.agent_id,
                reasoning,
                response.answer,
            )

        # ---- Phase 5: Stage-2 speak call (ThinkThenSpeakDebater only) ----
        # Batch speak prompts per LLM group, exactly like Phase 2/3 above.
        tts_agents = [
            (i, agent, states[i])
            for i, agent in enumerate(self.agents)
            if _TTS is not None and isinstance(agent, _TTS)
        ]
        if tts_agents:
            # Group TTS agents by LLM instance for batching
            speak_groups: dict = defaultdict(list)
            for i, agent, state in tts_agents:
                speak_groups[id(agent.llm)].append((i, agent, state))

            for group in speak_groups.values():
                llm = group[0][1].llm
                speak_messages = [
                    agent.build_speak_messages(state)
                    for _, agent, state in group
                ]
                try:
                    speak_outputs = llm.chat(
                        messages=speak_messages,
                        sampling_params=group[0][1]._speak_params,
                    )
                    for (i, agent, state), out in zip(group, speak_outputs):
                        public_msg = out.outputs[0].text.strip() if out.outputs else ""
                        agent_id = agent.agent_id
                        old_response = round_entry.agent_responses[agent_id]
                        new_response = agent.attach_public_message(old_response, public_msg)
                        round_entry.agent_responses[agent_id] = new_response
                        logger.info(
                            "  %s (%s):\n--- PUBLIC MESSAGE ---\n%s",
                            agent.name,
                            agent_id,
                            public_msg,
                        )

                        # ---- Phase 5b: Consistency check ----
                        # Only flag genuine mismatches where extraction succeeded
                        # (not fallback injections where answers were force-aligned).
                        private_answer = old_response.answer
                        public_answer = new_response.answer
                        public_extracted = getattr(new_response, "_public_extracted", True)
                        if public_extracted and private_answer != public_answer and private_answer != "?":
                            logger.info(
                                "  INCONSISTENCY: %s (%s) thought %s privately but said %s publicly — correcting.",
                                agent.name, agent_id, private_answer, public_answer,
                            )
                            correction_msgs = agent.build_inconsistency_correction_messages(
                                private_reasoning=old_response.reasoning,
                                private_answer=private_answer,
                                public_answer=public_answer,
                            )
                            try:
                                corr_out = llm.chat(
                                    messages=[correction_msgs],
                                    sampling_params=agent._speak_params,
                                )
                                if corr_out and corr_out[0].outputs:
                                    corrected_pub = corr_out[0].outputs[0].text.strip()
                                    corrected_response = agent.attach_public_message(old_response, corrected_pub)
                                    round_entry.agent_responses[agent_id] = corrected_response
                                    logger.info(
                                        "  %s (%s) corrected public message: %s → %s",
                                        agent.name, agent_id, public_answer, corrected_response.answer,
                                    )
                                    # Record the inconsistency flag
                                    if not hasattr(round_entry, "observer_flags"):
                                        round_entry.observer_flags = []
                                    round_entry.observer_flags.append({
                                        "agent_id": agent_id,
                                        "flag_type": "inconsistency",
                                        "original_answer": public_answer,
                                        "corrected_answer": corrected_response.answer,
                                        "notice": (
                                            f"Agent privately concluded {private_answer} "
                                            f"but publicly stated {public_answer}."
                                        ),
                                    })
                            except Exception as e:
                                logger.error("Inconsistency correction failed for %s: %s", agent_id, e)

                except Exception as e:
                    logger.error("Batch speak call failed: %s", e)

        # ---- Phase 6: Observer Analysis & Correction ----
        if self.observer is not None and conversation:
            notices = self.observer.analyze_round(conversation, round_entry.agent_responses)
            if notices:
                round_entry.observer_flags = []
                for agent_id, notice in notices.items():
                    # Parse flag type
                    flag_type = "unknown"
                    if notice.startswith("[sycophancy]"):
                        flag_type = "sycophancy"
                        notice = notice[len("[sycophancy]"):].strip()
                    elif notice.startswith("[repetition]"):
                        flag_type = "repetition"
                        notice = notice[len("[repetition]"):].strip()

                    agent = next(a for a in self.agents if a.agent_id == agent_id)
                    original_resp = round_entry.agent_responses[agent_id]
                    
                    logger.info("  %s (%s) FLAGGED for %s", agent.name, agent_id, flag_type.upper())
                    logger.info("  --- ORIGINAL (FLAGGED) ---\n%s\n--- END ORIGINAL ---", original_resp.reasoning)
                    
                    flag_record = {
                        "agent_id": agent_id,
                        "flag_type": flag_type,
                        "original_answer": original_resp.answer,
                        "notice": notice,
                        "corrected_answer": original_resp.answer, # fallback
                    }
                    round_entry.observer_flags.append(flag_record)

                    state_idx = self.agents.index(agent)
                    state = states[state_idx]

                    if hasattr(agent, "prepare_corrected_round"):
                        corrected_state = agent.prepare_corrected_round(state, notice)
                        
                        is_gemma = "gemma" in agent.model_name.lower()
                        if is_gemma:
                            msg = [{"role": "user", "content": f"{corrected_state['system']}\n\n{corrected_state['reasoning_prompt']}"}]
                        else:
                            msg = [
                                {"role": "system", "content": corrected_state["system"]},
                                {"role": "user", "content": corrected_state["reasoning_prompt"]}
                            ]
                        
                        try:
                            out = agent.llm.chat(messages=[msg], sampling_params=agent.sampling_params)
                            if out and out[0].outputs:
                                corrected_reasoning = out[0].outputs[0].text.strip()
                                corrected_reasoning = strip_think_blocks(corrected_reasoning)
                                corrected_reasoning = strip_hallucinated_turns(corrected_reasoning)
                                
                                new_resp = agent.finish_round(corrected_state, corrected_reasoning)
                                
                                if _TTS is not None and isinstance(agent, _TTS):
                                    speak_msg = agent.build_speak_messages(corrected_state)
                                    speak_out = agent.llm.chat(messages=[speak_msg], sampling_params=agent._speak_params)
                                    if speak_out and speak_out[0].outputs:
                                        public_msg = speak_out[0].outputs[0].text.strip()
                                        new_resp = agent.attach_public_message(new_resp, public_msg)

                                logger.info("  --- CORRECTED ---\n%s\n--- END CORRECTED ---", new_resp.reasoning)
                                round_entry.agent_responses[agent_id] = new_resp
                                flag_record["corrected_answer"] = new_resp.answer
                        except Exception as e:
                            logger.error("Error during agent correction for %s: %s", agent_id, e)

        return round_entry

    def _determine_final_answer(self, conversation: Conversation) -> str:
        """
        Determine the system's final answer via strict majority vote on
        the last round's answers.

        For 2 agents, this means both must agree — any disagreement returns
        "unresolved". This is intentional: unresolved cases are tracked by
        the NAR/DCR metrics to measure debate quality.

        Args:
            conversation: The full conversation.

        Returns:
            The majority answer letter, or "unresolved" if no strict majority.
        """
        if not conversation:
            return "unresolved"

        last_round = conversation[-1]
        answers = [
            resp.answer
            for resp in last_round.agent_responses.values()
            if resp.answer != "?"
        ]

        if not answers:
            return "unresolved"

        counts = Counter(answers)
        most_common_answer, most_common_count = counts.most_common(1)[0]

        # Strict majority: must have more votes than half the total agents
        if most_common_count > self.n_agents / 2:
            return most_common_answer

        return "unresolved"

    def _update_memories(
        self, conversation: Conversation, result: Any, question_data: Dict[str, Any]
    ) -> None:
        """
        Update memories for all agents that have them.

        Args:
            conversation: The full conversation.
            result: The debate result (final answer).
            question_data: The MMLU question data (for domain stats).
        """
        for agent in self.agents:
            if hasattr(agent, "update_memory"):
                agent.update_memory(conversation, result=result, question_data=question_data)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        agent_info = [
            f"{a.name}({a.model_name})" for a in self.agents
        ]
        return (
            f"DebateArena(agents={agent_info}, "
            f"rounds={self.num_rounds})"
        )
