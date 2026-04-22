"""
Debater agent for multi-agent debate.
Handles reasoning and answer generation during debate rounds.
"""

import logging
from typing import Optional, List

from vllm import LLM, SamplingParams

from agents.base import BaseAgent
from configs.configs import AgentResponse, Conversation, RoundEntry
from memory.base import BaseMemory
from utils import strip_hallucinated_turns, strip_think_blocks

logger = logging.getLogger(__name__)


class DebaterAgent(BaseAgent):
    """
    A debate participant that reasons about questions and produces answers.

    In each round, the debater:
    1. Builds a prompt from the question + memory + other agents' prior responses
    2. Generates reasoning (chain-of-thought)
    3. Generates a final answer letter from that reasoning
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a participant in a multi-agent debate working collaboratively to find the correct answer. "
        "In each round, you will see your peers' latest reasoning and answers. "
        "You must engage with their arguments directly — do not reason in isolation. "
        "Specifically: "
        "(1) Acknowledge your peers' positions. "
        "(2) If you agree with a peer, explain what in their reasoning supports the correct answer. "
        "(3) If you disagree with a peer, explain specifically why their argument is flawed and defend your own position with evidence. "
        "(4) If you are changing your mind, clearly state what convinced you. "
        "Never ignore your peers' reasoning. "
        "Always end with your final answer formatted as {{X}} where X is the answer letter."
    )

    SOLO_SYSTEM_PROMPT = (
        "You are a knowledgeable assistant answering a multiple-choice question. "
        "Reason step by step through the question, then provide your final answer as {{X}} "
        "where X is the answer letter. "
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
            agent_role="debater",
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            llm=llm,
            sampling_params=sampling_params,
        )
        self.memory = memory
        self.name = name

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_history(self, history: Conversation) -> str:
        """
        Flatten the full conversation history into a readable string,
        clearly distinguishing the agent's own responses from its peers'.

        Args:
            history: A Conversation (List[RoundEntry]), where each
                     RoundEntry contains a dict of AgentResponse objects.

        Returns:
            A formatted string where each line is labeled as either
            "You" (own response) or "Peer" (another agent's response).
        """
        lines: List[str] = []
        for round_idx, round_entry in enumerate(history):
            for agent_id, response in round_entry.agent_responses.items():
                if response.name == self.name:
                    speaker = f"You ({response.name})"
                else:
                    speaker = f"Peer ({response.name})"
                lines.append(
                    f"[Round {round_idx}] {speaker}: "
                    f"{response.reasoning}. Answer: {response.answer}"
                )
        return "\n".join(lines)

    def get_memory(self) -> str:
        """Retrieve memory context, or empty string if no memory is attached."""
        if self.memory is None:
            return ""
        return self.memory.retrieve_memory()

    def prepare_round(
        self, question_prompt: str, history: Conversation
    ) -> dict:
        """
        Build all prompt components for this round WITHOUT calling the LLM.
        Returns a state dict to be consumed by build_answer_prompt() and finish_round().

        Round 0 (history is empty): solo mode — question only, clean system prompt.
        Round 1+ (history non-empty): debate mode — full context with memory + history.

        Args:
            question_prompt: The formatted question string.
            history: The full conversation from previous rounds.

        Returns:
            A dict containing:
              - "reasoning_prompt": the full prompt to feed for reasoning generation.
              - "system": the system prompt string to use for this round.
              - "prompt": the base prompt (without completion cue).
              - "instruction": the in-context instruction injected before the cue.
              - "next_round": integer index of this round.
        """
        is_solo = not history
        next_round = len(history)

        if is_solo:
            prompt = question_prompt
            system = self.SOLO_SYSTEM_PROMPT
            instruction = (
                "Reason step by step through the question, "
                "then provide your final answer as {{X}}."
            )
            logger.debug("[Round 0 SOLO] %s", self.agent_id)
        else:
            history_str = self.build_history(history)
            mem = self.get_memory()
            parts = [question_prompt]
            if mem:
                parts.append(mem)
            if history_str:
                parts.append(history_str)
            prompt = "\n\n".join(parts)
            system = self.system_prompt  # DEFAULT_SYSTEM_PROMPT
            base_instruction = (
                "Based on the conversation above, respond to your peers' latest arguments. "
                "Explicitly state whether you agree or disagree with each peer's reasoning and why, "
                "then provide your final answer as {{X}}."
            )
            mem_instruction = self.memory.get_instruction() if self.memory else ""
            instruction = f"{mem_instruction}\n{base_instruction}" if mem_instruction else base_instruction
            logger.debug("[Round %d DEBATE] %s", next_round, self.agent_id)

        reasoning_prompt = (
            prompt + f"\n\n{instruction}"
        )

        return {
            "prompt": prompt,
            "reasoning_prompt": reasoning_prompt,
            "system": system,
            "instruction": instruction,
            "next_round": next_round,
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

    def finish_round(self, state: dict, reasoning: str) -> AgentResponse:
        """
        Assemble an AgentResponse from the single LLM output.

        Args:
            state: The dict returned by prepare_round().
            reasoning: The reasoning text which also contains the final answer.

        Returns:
            An AgentResponse with name, reasoning, and extracted answer.
        """
        answer = self.extract_answer(reasoning)
        return AgentResponse(name=self.name, reasoning=reasoning, answer=answer)

    # ------------------------------------------------------------------
    # Convenience wrapper (single-agent use / testing)
    # ------------------------------------------------------------------

    def observe_and_response(
        self, question_prompt: str, history: Conversation
    ) -> AgentResponse:
        """Single-agent convenience method. Calls prepare_round → LLM → finish_round."""
        state = self.prepare_round(question_prompt, history)
        reasoning = self.call_llm(
            state["reasoning_prompt"], system_prompt_override=state["system"]
        )
        clean_reasoning = strip_think_blocks(reasoning)
        clean_reasoning = strip_hallucinated_turns(clean_reasoning)
        return self.finish_round(state, clean_reasoning)

    def update_memory(self, history: Conversation, result) -> None:
        """Update the agent's memory after a debate concludes."""
        if self.memory is not None:
            self.memory.update_memory(history, result=result)