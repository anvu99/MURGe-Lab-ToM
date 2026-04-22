"""
ThinkThenSpeakDebater — a two-stage debate agent.

Stage 1 (Private):  Full-depth chain-of-thought reasoning with no word limit.
                    This reasoning is stored in AgentResponse.reasoning but is
                    NOT shared with other agents.

Stage 2 (Public):   A concise ≤300-word debate message distilled from the
                    private reasoning, explicitly addressing peers' arguments.
                    This is stored in AgentResponse.public_message and is the
                    ONLY text peers see in subsequent rounds.

Prompt growth is bounded at O(agents × rounds × 300 words) regardless of how
deeply the agent reasons privately.
"""

import logging
from typing import Any, List, Optional

from vllm import LLM, SamplingParams

from agents.base import BaseAgent
from configs.configs import AgentResponse, Conversation
from memory.base import BaseMemory
from utils import strip_hallucinated_turns, strip_think_blocks

logger = logging.getLogger(__name__)

# Hard token cap for the public message generation to enforce the 300-word
# limit at the generation level, not just via prompt instruction.
_SPEAK_MAX_TOKENS = 480  # ~300 words + answer tag headroom


class ThinkThenSpeakDebater(BaseAgent):
    """
    A debate agent that reasons privately before speaking concisely.

    In each round the agent:
      1. Reasons privately (full chain-of-thought, private reasoning prompt).
      2. Distills that reasoning into a ≤300-word public message addressing
         its peers, which is what peers see in subsequent rounds.

    The DebateArena orchestrates both stages via:
      - ``prepare_round()``  →  builds the private reasoning prompt (Stage 1).
      - ``finish_round()``   →  returns state with reasoning + speak prompt.
      - ``build_speak_messages()`` →  called by the arena AFTER finish_round to
         batch the Stage 2 (speak) LLM calls per model group.
      - ``attach_public_message()`` → attaches the generated public message to
         the AgentResponse.
    """

    # ---- System prompts ----

    SOLO_SYSTEM_PROMPT = (
        "You are a knowledgeable assistant answering a multiple-choice question. "
        "You MUST show your working step-by-step in detail before giving your answer. "
        "Do NOT just output the answer letter immediately. "
        "After thoroughly writing out your reasoning trace, provide your final answer as {{X}} "
        "where X is a single capital letter (e.g., A, B, C, ..., J). "
        "NEVER use a number — always use the letter label shown in the options."
    )

    THINK_SYSTEM_PROMPT = (
        "You are a critical thinker in a multi-agent debate. "
        "You MUST write out your private reasoning step-by-step in detail. "
        "Do NOT just output the final answer letter immediately. "
        "Carefully evaluate your peers' arguments: do they introduce new evidence, "
        "stronger reasoning, or a perspective you had not considered? "
        "Update your position when you find genuine reason to — but hold your ground "
        "when peers are simply asserting without new substance. "
        "After thoroughly writing your reasoning trace, end with your final answer as {{X}} "
        "where X is a single capital letter (A–J). "
        "NEVER output a number — always use the letter label shown in the options."
    )

    SPEAK_SYSTEM_PROMPT = (
        "You are a rigorous debate participant. "
        "Back every claim with a specific reason or piece of evidence — do not simply assert your answer. "
        "When you disagree with a peer, name the specific flaw in their argument or the evidence they are missing. "
        "If a peer has challenged your position, address their objection directly — "
        "whether you are updating your view or standing by it, explain your reasoning clearly. "
        "Do not gloss over disagreements with vague consensus — resolve them with substance. "
        "Be concise and persuasive."
    )

    def __init__(
        self,
        agent_id: str,
        name: str,
        model_name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        memory: Optional[BaseMemory] = None,
        sampling_params: Optional[SamplingParams] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_role="tts_debater",
            model_name=model_name,
            system_prompt=system_prompt or self.THINK_SYSTEM_PROMPT,
            llm=llm,
            sampling_params=sampling_params,
        )
        self.memory = memory
        self.name = name

        # Sampling params for Stage 2 (speak). Use a tighter max_tokens to
        # enforce the 300-word budget at the generation level.
        self._speak_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            max_tokens=_SPEAK_MAX_TOKENS,
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def build_public_history(self, history: Conversation) -> str:
        """
        Build a peer context from the *public messages* of all previous rounds.

        Args:
            history: Full conversation history (all previous rounds).

        Returns:
            A formatted string of all public messages across the debate.
        """
        if not history:
            return ""

        lines: List[str] = []
        for i, past_round in enumerate(history):
            lines.append(f"--- Round {i} ---")
            for agent_id, response in past_round.agent_responses.items():
                if response.name == self.name:
                    speaker = "You"
                    # Use own public message if available, else fall back to answer only
                    msg = response.public_message if response.public_message else f"Answer: {response.answer}"
                else:
                    speaker = f"Peer ({response.name})"
                    msg = response.public_message if response.public_message else f"Answer: {response.answer}"
                lines.append(f"{speaker}: {msg}")

        return "\n\n".join(lines)

    def get_memory(self) -> str:
        """Retrieve memory context, or empty string if no memory is attached."""
        if self.memory is None:
            return ""
        return self.memory.retrieve_memory()

    # ------------------------------------------------------------------
    # Round protocol  (used by DebateArena)
    # ------------------------------------------------------------------

    def prepare_round(
        self, question_prompt: str, history: Conversation, **kwargs
    ) -> dict:
        """
        Build the Stage 1 (private reasoning) prompt.

        Round 0: solo mode — question only, no peer context.
        Round N>0: question + own last public message + peers' last public messages.

        Args:
            question_prompt: The formatted question string.
            history: Full conversation from previous rounds.
            **kwargs: Additional parameters. Support `current_round_responses` to
                      inject messages from peers who have already spoken in the
                      current round of an asynchronous debate.

        Returns:
            A state dict with keys:
              - ``reasoning_prompt``: full prompt for Stage 1 LLM call.
              - ``system``: system prompt to use for Stage 1.
              - ``is_solo``: True if this is Round 0.
              - ``next_round``: integer index of this round.
        """
        current_round_responses = kwargs.get("current_round_responses", {})
        
        # If history is empty, it is Round 0. Force independent reasoning.
        is_solo = not history
        if is_solo:
            current_round_responses = {}
            
        next_round = len(history)

        # Extract peer names and last round messages from the conversation history
        peer_names: List[str] = []
        recent_peer_messages: List[str] = []
        if history:
            for _, response in history[-1].agent_responses.items():
                if response.name != self.name:
                    peer_names.append(response.name)
                    msg = response.public_message if response.public_message else f"Answer: {response.answer}"
                    recent_peer_messages.append(f"Peer ({response.name}): {msg}")

        # Inject current round peer messages if this is an async debate
        if current_round_responses:
            for _, response in current_round_responses.items():
                if response.name != self.name:
                    if response.name not in peer_names:
                        peer_names.append(response.name)
                    msg = response.public_message if getattr(response, "public_message", "") else f"Answer: {response.answer}"
                    recent_peer_messages.append(f"Peer ({response.name}) [CURRENT ROUND UPDATE]: {msg}")

        if is_solo:
            system = self.SOLO_SYSTEM_PROMPT
            instruction = (
                "You MUST show your working and write out your reasoning step-by-step. "
                "Do NOT jump straight to the answer. "
                "After writing your detailed reasoning, provide your final answer as {{X}} "
                "where X is the capital LETTER of your chosen option (e.g., A, B, C, ..., J). "
                "Do NOT write a number — use the letter label shown in the options."
            )
            reasoning_prompt = f"You are {self.name}.\n\n{question_prompt}\n\n{instruction}"
            logger.debug("[TTS Round 0 SOLO] %s", self.agent_id)
        else:
            public_history = self.build_public_history(history)
            
            # Incorporate current round into public history visually so agent understands context
            if current_round_responses:
                curr_lines = [f"--- Round {next_round} (In Progress) ---"]
                for _, r in current_round_responses.items():
                    if r.name != self.name:
                        msg = r.public_message if getattr(r, "public_message", "") else f"Answer: {r.answer}"
                        curr_lines.append(f"Peer ({r.name}): {msg}")
                if len(curr_lines) > 1:  # If there are actually peers who spoke
                    if public_history:
                        public_history += "\n\n" + "\n".join(curr_lines)
                    else:
                        public_history = "\n".join(curr_lines)

            mem = self.get_memory()

            # Extract own private reasoning from the immediate previous round
            own_past_reasoning = ""
            if history:
                for _, response in history[-1].agent_responses.items():
                    if response.name == self.name:
                        own_past_reasoning = response.reasoning
                        break

            parts = [question_prompt]
            if own_past_reasoning:
                parts.append(f"[CONTEXT: Your previous reasoning]\n{own_past_reasoning}\n[END CONTEXT]")
            if mem:
                parts.append(mem)
            parts.append(public_history)

            mem_instruction = self.memory.get_instruction() if self.memory else ""
            base_instruction = (
                "Based on the debate messages above, you MUST write out your private reasoning "
                "step-by-step to decide whether to update your position. Consider each peer's arguments carefully. "
                "Do NOT jump straight to the answer. "
                "Do NOT reproduce section headers like '[Private Thoughts]', '[CONTEXT]', or '[Your Private Thoughts from Round N]' in your output — "
                "write your reasoning as plain flowing text. "
                "After writing your detailed reasoning trace, end with your final answer as {{X}} "
                "where X is the capital LETTER of your chosen option (e.g., A, B, C, ..., J). "
                "Do NOT write a number — use the letter label shown in the options."
            )
            instruction = f"{mem_instruction}\n{base_instruction}" if mem_instruction else base_instruction
            reasoning_prompt = f"You are {self.name}.\n\n" + "\n\n".join(parts) + f"\n\n{instruction}"
            system = self.system_prompt  # THINK_SYSTEM_PROMPT
            logger.debug("[TTS Round %d THINK] %s", next_round, self.agent_id)

        return {
            "reasoning_prompt": reasoning_prompt,
            "system": system,
            "is_solo": is_solo,
            "next_round": next_round,
            "peer_names": peer_names,
            "recent_peer_messages": "\n\n".join(recent_peer_messages),
        }

    def prepare_corrected_round(self, state: dict, observer_notice: str) -> dict:
        """
        Re-build the reasoning prompt with an Observer notice appended.
        Returns a new state dict for re-generation.
        """
        corrected_prompt = (
            state["reasoning_prompt"]
            + f"\n\n[Observer Notice] {observer_notice}\n\n"
            + "Given this feedback, re-examine your reasoning independently. "
            + "If you are changing your position, you MUST explain specifically "
            + "what flaw you found in your previous reasoning. If you are "
            + "maintaining your position, address the specific peer arguments "
            + "mentioned above."
        )
        return {**state, "reasoning_prompt": corrected_prompt}

    def finish_round(self, state: dict, private_reasoning: str) -> AgentResponse:
        """
        Build AgentResponse from Stage 1 output and prepare Stage 2 prompt.

        The AgentResponse is returned with `public_message=""` — the arena will
        call ``build_speak_messages()`` to batch Stage 2, then
        ``attach_public_message()`` to fill in the field.

        Args:
            state: The dict returned by ``prepare_round()``.
            private_reasoning: The Stage 1 reasoning text from the LLM.

        Returns:
            AgentResponse with reasoning and answer filled; public_message="".
        """
        answer = self.extract_answer(private_reasoning)

        # Attach speak prompt to the response object via a side-channel key
        # on the state dict so the arena can retrieve it without changing the
        # AgentResponse dataclass contract.
        state["_private_reasoning"] = private_reasoning
        state["_speak_prompt"] = self._build_speak_prompt(state, private_reasoning)

        return AgentResponse(
            name=self.name,
            reasoning=private_reasoning,
            answer=answer,
            public_message="",  # filled in by arena after Stage 2 batch call
        )

    def build_speak_messages(self, state: dict) -> list:
        """
        Build the chat messages list for the Stage 2 (speak) LLM call.

        Called by the arena to assemble the batch for all TTS agents sharing
        the same LLM instance.

        Args:
            state: The state dict from ``prepare_round()`` (already has
                   ``_speak_prompt`` populated by ``finish_round()``).

        Returns:
            A messages list suitable for ``llm.chat()``.
        """
        speak_prompt = state.get("_speak_prompt", "")
        if "gemma" in self.model_name.lower():
            return [
                {"role": "user", "content": f"{self.SPEAK_SYSTEM_PROMPT}\n\n{speak_prompt}"},
            ]
        return [
            {"role": "system", "content": self.SPEAK_SYSTEM_PROMPT},
            {"role": "user", "content": speak_prompt},
        ]

    def attach_public_message(
        self, response: AgentResponse, public_message: str
    ) -> AgentResponse:
        """
        Return a new AgentResponse with the public_message filled in.

        Args:
            response: The AgentResponse from ``finish_round()``.
            public_message: The Stage 2 output text.

        Returns:
            Updated AgentResponse. The arena uses the hidden ``_public_extracted``
            attribute to distinguish genuine answer mismatches from fallback injections.
        """
        # Ensure the answer tag is present in the public message
        answer = self.extract_answer(public_message)
        public_extracted = answer != "?"
        if not public_extracted:
            # Fall back to the answer extracted from private reasoning
            answer = response.answer
            public_message = public_message.rstrip() + f"\n\n{{{{answer}}}}"
            public_message = public_message.replace("{answer}", answer)

        resp = AgentResponse(
            name=response.name,
            reasoning=response.reasoning,
            answer=answer if answer != "?" else response.answer,
            public_message=public_message,
        )
        # Side-channel flag used by the arena for inconsistency detection
        resp._public_extracted = public_extracted  # type: ignore[attr-defined]
        return resp

    def build_inconsistency_correction_messages(
        self,
        private_reasoning: str,
        private_answer: str,
        public_answer: str,
        original_speak_prompt: str = "",
    ) -> list:
        """
        Build a Stage 2 correction prompt when the agent's public answer
        differs from its private reasoning conclusion.

        Uses the original speak prompt (which already contains peer messages,
        private reasoning, and formatting instructions) as the base, then
        appends a short correction notice so the model has full context.

        Args:
            private_reasoning: The full Stage 1 reasoning text.
            private_answer: The answer letter extracted from Stage 1.
            public_answer: The mismatched answer letter from the draft public message.
            original_speak_prompt: The Stage 2 prompt built by ``_build_speak_prompt``.
                                   If provided, it is used as the base so the model
                                   retains peer context during correction.

        Returns:
            A messages list suitable for ``llm.chat()``.
        """
        private_tag = "{{" + private_answer + "}}"
        public_tag   = "{{" + public_answer  + "}}"

        correction_notice = (
            f"\n\nINCONSISTENCY DETECTED: Your private reasoning concluded {private_tag}, "
            f"but your public message stated {public_tag}. "
            f"This is internally inconsistent — you must be honest and align your public "
            f"message with your actual private conclusion. "
            f"Rewrite your response below (≤300 words), addressing your peers as before, "
            f"but end with {private_tag}, which is what your reasoning actually supports."
        )

        # Use the original speak prompt as base so the model retains full peer
        # context (peer messages, debate history, formatting rules). Fall back
        # to private reasoning only if original_speak_prompt was not supplied.
        base = (
            original_speak_prompt
            if original_speak_prompt
            else f"Your private reasoning:\n{private_reasoning}"
        )
        correction_prompt = base + correction_notice

        if "gemma" in self.model_name.lower():
            return [
                {"role": "user", "content": f"{self.SPEAK_SYSTEM_PROMPT}\n\n{correction_prompt}"},
            ]
        return [
            {"role": "system", "content": self.SPEAK_SYSTEM_PROMPT},
            {"role": "user", "content": correction_prompt},
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_speak_prompt(self, state: dict, private_reasoning: str) -> str:
        """
        Build the Stage 2 prompt that distills private reasoning into a
        ≤300-word public message.

        Args:
            state: The state dict from ``prepare_round()``.
            private_reasoning: The Stage 1 output.

        Returns:
            A formatted string for the Stage 2 LLM call.
        """
        if state.get("is_solo"):
            # Round 0: no peers to address, just summarise the answer concisely
            return (
                "You just reasoned about a multiple-choice question. "
                "Write a concise summary (≤300 words) of your reasoning and conclusion. "
                "End with your final answer as {{X}} where X is the capital LETTER of your chosen option "
                "(e.g., A, B, C, ..., J). Do NOT write a number.\n\n"
                f"Your reasoning:\n{private_reasoning}"
            )

        peer_names = state.get("peer_names", [])
        # Strictly exclude own name from the peer list (safety guarantee)
        peer_names = [n for n in peer_names if n != self.name]
        peer_list = ", ".join(peer_names) if peer_names else "your peers"
        recent_msgs = state.get("recent_peer_messages", "")

        # Peer context placed FIRST so it doesn't bleed into generation
        peer_context_block = ""
        if recent_msgs:
            peer_context_block = f"Peer messages to address:\n{recent_msgs}\n\n"

        return (
            # 1. Peer messages FIRST — context before reasoning
            f"{peer_context_block}"
            # 2. Private reasoning SECOND
            f"Your private reasoning:\n{private_reasoning}\n\n"
            # 3. Instruction LAST — final thing read before generation starts
            f"You are {self.name}. "
            "Now write ONLY your own response below. "
            "Do NOT reproduce or repeat any peer messages above.\n\n"
            f"Your debate peers are: {peer_list}. "
            "You MUST ONLY address these peers — do NOT invent or hallucinate other agent names.\n\n"
            "Your response MUST:\n"
            f"1. Address each peer by their actual name ({peer_list}) — state whether you agree or disagree and why.\n"
            "2. Present your strongest supporting argument clearly.\n"
            "3. End with your final answer as {{X}} where X is the capital LETTER of your chosen option "
            "(e.g., A, B, C, ..., J). Do NOT write a number.\n\n"
            "Do not exceed 300 words."
        )

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def update_memory(self, history: Conversation, result: Any = None, **kwargs) -> None:
        """Update the agent's memory after a debate concludes."""
        if self.memory is not None:
            self.memory.update_memory(history, result=result, **kwargs)

    # ------------------------------------------------------------------
    # Convenience wrapper (single-agent use / testing)
    # ------------------------------------------------------------------

    def observe_and_response(
        self, question_prompt: str, history: Conversation
    ) -> AgentResponse:
        """Single-agent convenience: full two-stage call in sequence."""
        state = self.prepare_round(question_prompt, history)

        # Stage 1
        from utils import strip_hallucinated_turns, strip_think_blocks
        raw_reasoning = self.call_llm(
            state["reasoning_prompt"],
            system_prompt_override=state["system"],
        )
        private_reasoning = strip_think_blocks(raw_reasoning)
        private_reasoning = strip_hallucinated_turns(private_reasoning)

        response = self.finish_round(state, private_reasoning)

        # Stage 2
        speak_prompt = state.get("_speak_prompt", "")
        raw_public = self.call_llm(
            speak_prompt,
            params=self._speak_params,
            system_prompt_override=self.SPEAK_SYSTEM_PROMPT,
        )
        return self.attach_public_message(response, raw_public.strip())
