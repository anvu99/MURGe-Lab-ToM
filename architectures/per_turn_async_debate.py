"""
PerTurnAsyncDebateArena — per-turn conversation logging for async debates.

Unlike AsyncDebateArena which groups both agents' responses into one RoundEntry
per debate round (misrepresenting the sequential async nature), this arena
logs ONE RoundEntry per individual agent turn.

For 2 agents × 5 debate rounds → 10 RoundEntries in the conversation log,
accurately reflecting the sequential speaking order.

This makes the conversation history transparent for the CSADebater's
Step 1 audit: history[-1] is always the peer's last message, history[-2]
is always the owner's last message.

The result dict converts per-turn to grouped format for metrics compatibility
with the existing Evaluator / RoundAnalyzer pipeline.
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict

from architectures.debate import DebateArena
from configs.configs import Conversation, RoundEntry
from utils import strip_hallucinated_turns, strip_think_blocks

logger = logging.getLogger(__name__)


class PerTurnAsyncDebateArena(DebateArena):
    """
    Async debate arena with per-turn conversation logging.

    Each agent speaks sequentially within each debate round, and each
    individual speaking turn is stored as a separate RoundEntry.

    Key properties:
    - No current_round_responses injection needed — every prior turn is
      already in the conversation history before the next agent speaks.
    - history[-1] when agent A speaks is always agent B's most recent turn,
      making the CSA send audit unambiguous.
    - update_memory receives the per-turn conversation directly.
    - result["rounds"] is converted back to grouped (one dict per debate round)
      for compatibility with the existing Evaluator / RoundAnalyzer.
    """

    def run(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a full debate with per-turn conversation logging.

        Each agent speaks in order within each debate round. The conversation
        log grows by one entry per agent turn (n_agents × num_rounds entries).

        Args:
            question_data: MMLU question dict.

        Returns:
            Result dict with grouped rounds for metrics compatibility.
        """
        question_prompt = self._format_question(question_data)
        conversation: Conversation = []  # per-turn: one RoundEntry per agent turn
        all_observer_flags = []

        # Convergence: track how many consecutive turns both agents have agreed.
        # Check starts at len=3 (Turn 2).
        _CONVERGENCE_CHECK_FROM = 3
        _CONVERGENCE_STREAK_REQUIRED = 2
        convergence_streak = 0
        converged_early = False

        for debate_round in range(self.num_rounds):
            logger.info("--- Debate Round %d ---", debate_round)
            for agent in self.agents:
                logger.info("  [TURN] %s is taking their turn...", agent.name)
                turn_entry = self._run_agent_turn(agent, question_prompt, conversation)
                if hasattr(turn_entry, "observer_flags"):
                    all_observer_flags.extend(turn_entry.observer_flags)
                conversation.append(turn_entry)

                # --- Early convergence check ---
                if len(conversation) >= _CONVERGENCE_CHECK_FROM:
                    agreed = self._check_convergence(conversation)
                    if agreed is not None:
                        convergence_streak += 1
                        logger.info(
                            "Convergence check (turn %d): agents agree on '%s'. Streak = %d / %d",
                            len(conversation) - 1,
                            agreed,
                            convergence_streak,
                            _CONVERGENCE_STREAK_REQUIRED
                        )
                    else:
                        convergence_streak = 0  # Reset on disagreement

                    if convergence_streak >= _CONVERGENCE_STREAK_REQUIRED:
                        logger.info(
                            "CONVERGENCE ACHIEVED at turn %d: streak of 3 reached for '%s'. Stopping early.",
                            len(conversation) - 1,
                            agreed,
                        )
                        converged_early = True
                        break

            if converged_early:
                break

        # Determine final answer from the last turn of each agent
        final_answer = self._determine_final_answer_per_turn(conversation)

        # update_memory receives the full per-turn conversation
        self._update_memories(conversation, result=final_answer, question_data=question_data)

        # Convert per-turn log to grouped rounds for Evaluator / RoundAnalyzer
        grouped_rounds = self._group_turns_to_rounds(conversation)

        result = {
            "question_id": question_data.get("question_id", ""),
            "correct_answer": question_data.get("answer", ""),
            "category": question_data.get("category", ""),
            "rounds": grouped_rounds,
            "final_answer": final_answer,
            "observer_flags": all_observer_flags,
            "converged_early": converged_early,
            "turns_taken": len(conversation),
            "total_inconsistency_flags": sum(
                1 for f in all_observer_flags if f.get("flag_type") == "inconsistency"
            ),
        }

        logger.info(
            "Debate finished. Final answer: %s (correct: %s) [turns=%d, converged=%s]",
            final_answer,
            result["correct_answer"],
            len(conversation),
            converged_early,
        )
        return result

    # ------------------------------------------------------------------
    # Per-turn execution
    # ------------------------------------------------------------------

    def _run_agent_turn(
        self,
        agent: Any,
        question_prompt: str,
        conversation: Conversation,
    ) -> RoundEntry:
        """
        Execute a single agent's speaking turn.

        The conversation already contains all previous turns (no
        current_round_responses injection needed), so prepare_round
        receives the full history up to this moment.

        Args:
            agent: The agent whose turn it is.
            question_prompt: Formatted question string.
            conversation: All turns so far (may be empty for the first turn).

        Returns:
            A RoundEntry containing only this agent's response.
        """
        turn_entry = RoundEntry()
        turn_entry.agent_responses = {}
        turn_entry.turn_order = [agent.agent_id]

        # ---- Stage 1: Build reasoning prompt & generate ----
        # No current_round_responses — the conversation already reflects reality.
        state = agent.prepare_round(question_prompt, conversation)

        is_gemma = "gemma" in agent.model_name.lower()
        if is_gemma:
            reasoning_messages = [
                [{"role": "user", "content": f"{state['system']}\n\n{state['reasoning_prompt']}"}]
            ]
        else:
            reasoning_messages = [
                [
                    {"role": "system", "content": state["system"]},
                    {"role": "user", "content": state["reasoning_prompt"]},
                ]
            ]

        try:
            out = agent.llm.chat(
                messages=reasoning_messages,
                sampling_params=agent.sampling_params,
            )
            reasoning = out[0].outputs[0].text.strip() if out and out[0].outputs else ""
        except Exception as e:
            logger.error("Reasoning call failed for %s: %s", agent.agent_id, e)
            reasoning = ""

        clean_reasoning = strip_think_blocks(reasoning)
        clean_reasoning = strip_hallucinated_turns(clean_reasoning)
        old_response = agent.finish_round(state, clean_reasoning)
        turn_entry.agent_responses[agent.agent_id] = old_response

        logger.info(
            "  %s:\n--- REASONING ---\n%s\n--- EXTRACTED ANSWER ---: %s",
            agent.name,
            reasoning,
            old_response.answer,
        )

        # ---- Stage 2: Speak (ThinkThenSpeakDebater and subclasses) ----
        # On the agent's initial solo turn (first time they reason), we deliberately
        # skip Stage 2. No public message is produced. Both agents form private opinions
        # independently, and the public debate starts on their SECOND turn.
        # This prevents a spurious Step 1 SEND AUDIT on Turn 2
        # (the peer was also silent in Turn 1, so engagement-auditing makes no sense).
        if hasattr(agent, "build_speak_messages") and not state.get("is_solo"):
            speak_msgs = agent.build_speak_messages(state)
            speak_params = getattr(agent, "_speak_params", agent.sampling_params)

            try:
                speak_out = agent.llm.chat(
                    messages=[speak_msgs],
                    sampling_params=speak_params,
                )
                public_msg = (
                    speak_out[0].outputs[0].text.strip()
                    if speak_out and speak_out[0].outputs
                    else ""
                )
                new_response = agent.attach_public_message(old_response, public_msg)
                turn_entry.agent_responses[agent.agent_id] = new_response

                logger.info(
                    "  %s:\n--- PUBLIC MESSAGE ---\n%s", agent.name, public_msg
                )

                # ---- Consistency check ----
                private_answer = old_response.answer
                public_answer = new_response.answer
                public_extracted = getattr(new_response, "_public_extracted", True)

                if (
                    public_extracted
                    and private_answer != public_answer
                    and private_answer != "?"
                ):
                    logger.info(
                        "  INCONSISTENCY: %s thought %s privately but said %s publicly — correcting.",
                        agent.name, private_answer, public_answer,
                    )
                    correction_msgs = agent.build_inconsistency_correction_messages(
                        private_reasoning=old_response.reasoning,
                        private_answer=private_answer,
                        public_answer=public_answer,
                        original_speak_prompt=state.get("_speak_prompt", ""),
                    )
                    try:
                        corr_out = agent.llm.chat(
                            messages=[correction_msgs],
                            sampling_params=speak_params,
                        )
                        if corr_out and corr_out[0].outputs:
                            corrected_pub = corr_out[0].outputs[0].text.strip()
                            corrected_response = agent.attach_public_message(
                                old_response, corrected_pub
                            )
                            turn_entry.agent_responses[agent.agent_id] = corrected_response
                            if not hasattr(turn_entry, "observer_flags"):
                                turn_entry.observer_flags = []
                            turn_entry.observer_flags.append({
                                "agent_id": agent.agent_id,
                                "flag_type": "inconsistency",
                                "original_answer": public_answer,
                                "corrected_answer": corrected_response.answer,
                                "notice": (
                                    f"Privately concluded {private_answer} "
                                    f"but publicly stated {public_answer}."
                                ),
                            })
                    except Exception as e:
                        logger.error(
                            "Inconsistency correction failed for %s: %s", agent.agent_id, e
                        )

            except Exception as e:
                logger.error("Stage 2 speak call failed for %s: %s", agent.agent_id, e)

        return turn_entry

    # ------------------------------------------------------------------
    # Final answer & metrics helpers
    # ------------------------------------------------------------------

    def _check_convergence(self, conversation: Conversation):
        """
        Check if all agents currently agree on the same answer.

        Scans backward to find the most recent answer for each known agent.
        Returns the agreed answer letter if all n_agents agree on a non-"?" answer,
        otherwise returns None.

        Args:
            conversation: Current per-turn conversation.

        Returns:
            The agreed answer letter (str), or None if no convergence.
        """
        last_answers: Dict[str, str] = {}
        for turn_entry in reversed(conversation):
            for agent_id, response in turn_entry.agent_responses.items():
                if agent_id not in last_answers and response.answer != "?":
                    last_answers[agent_id] = response.answer
            if len(last_answers) == self.n_agents:
                break  # found the latest answer for every agent

        if len(last_answers) < self.n_agents:
            return None  # not all agents have answered yet

        unique_answers = set(last_answers.values())
        if len(unique_answers) == 1:
            return unique_answers.pop()  # unanimous agreement
        return None

    def _determine_final_answer_per_turn(self, conversation: Conversation) -> str:
        """
        Determine final answer from per-turn conversation.

        Finds the last answer produced by each agent and applies
        strict majority vote.

        Args:
            conversation: Per-turn conversation (one agent per entry).

        Returns:
            Majority answer letter, or "unresolved".
        """
        last_answers: Dict[str, str] = {}
        for turn_entry in conversation:
            for agent_id, response in turn_entry.agent_responses.items():
                if response.answer != "?":
                    last_answers[agent_id] = response.answer

        answers = list(last_answers.values())
        if not answers:
            return "unresolved"

        counts = Counter(answers)
        most_common_answer, most_common_count = counts.most_common(1)[0]
        if most_common_count > self.n_agents / 2:
            return most_common_answer
        return "unresolved"

    def _group_turns_to_rounds(self, conversation: Conversation):
        """
        Convert per-turn conversation to grouped rounds for Evaluator.

        Groups consecutive turns into debate rounds. With 2 agents and
        alternating order (Qwen, Llama, Qwen, Llama, ...):
          - Turn 0 + Turn 1 → Round 0
          - Turn 2 + Turn 3 → Round 1
          etc.

        Args:
            conversation: Per-turn conversation.

        Returns:
            List of dicts {agent_id: answer_letter} — one per debate round.
        """
        n = self.n_agents
        grouped = []
        for round_start in range(0, len(conversation), n):
            round_data: Dict[str, str] = {}
            for offset in range(n):
                idx = round_start + offset
                if idx < len(conversation):
                    for agent_id, response in conversation[idx].agent_responses.items():
                        round_data[agent_id] = response.answer
            grouped.append(round_data)
        return grouped

    # Override the parent's _determine_final_answer so it won't be called
    # accidentally (the overridden run() uses _determine_final_answer_per_turn).
    def _determine_final_answer(self, conversation: Conversation) -> str:
        return self._determine_final_answer_per_turn(conversation)
