"""
ThinkThenSpeakDebater — a two-stage debate agent.

Stage 1 (Private):  Full-depth chain-of-thought reasoning with no word limit.
                    This reasoning is stored in AgentResponse.reasoning but is
                    NOT shared with other agents.

Stage 2 (Public):   A concise ≤200-word debate message distilled from the
                    private reasoning, explicitly addressing peers' arguments.
                    This is stored in AgentResponse.public_message and is the
                    ONLY text peers see in subsequent rounds.

Prompt growth is bounded at O(agents × rounds × 200 words) regardless of how
deeply the agent reasons privately.
"""

import logging
from typing import List, Optional

from vllm import LLM, SamplingParams

from agents.base import BaseAgent
from configs.configs import AgentResponse, Conversation
from memory.base import BaseMemory
from utils import strip_hallucinated_turns, strip_think_blocks

logger = logging.getLogger(__name__)

# Hard token cap for the public message generation to enforce the 200-word
# limit at the generation level, not just via prompt instruction.
_SPEAK_MAX_TOKENS = 320  # ~200 words + answer tag headroom


class ThinkThenSpeakDebater(BaseAgent):
    """
    A debate agent that reasons privately before speaking concisely.

    In each round the agent:
      1. Reasons privately (full chain-of-thought, private reasoning prompt).
      2. Distills that reasoning into a ≤200-word public message addressing
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
        "Reason step by step through the question, then provide your final answer as {{X}} "
        "where X is the answer letter."
    )

    THINK_SYSTEM_PROMPT = (
        "You are a critical thinker in a multi-agent debate. "
        "Reason privately and thoroughly about the question. "
        "Consider your peers' arguments carefully — do they reveal flaws in your reasoning? "
        "End with your final answer as {{X}} where X is the answer letter."
    )

    SPEAK_SYSTEM_PROMPT = (
        "You are a debate participant. Write a concise, persuasive message for your peers."
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
        # enforce the 200-word budget at the generation level.
        self._speak_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            max_tokens=_SPEAK_MAX_TOKENS,
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def build_public_history(self, history: Conversation) -> str:
        """
        Build a compact peer context from the *public messages* only.

        Only the most recent round's public messages are included — the agent
        already received older rounds in its previous reasoning prompt, so there
        is no information loss, only prompt savings.

        Args:
            history: Full conversation history (all previous rounds).

        Returns:
            A formatted string of last-round public messages.
        """
        if not history:
            return ""

        last_round = history[-1]
        lines: List[str] = []
        for agent_id, response in last_round.agent_responses.items():
            if response.name == self.name:
                speaker = f"You ({response.name})"
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
        self, question_prompt: str, history: Conversation
    ) -> dict:
        """
        Build the Stage 1 (private reasoning) prompt.

        Round 0: solo mode — question only, no peer context.
        Round N>0: question + own last public message + peers' last public messages.

        Args:
            question_prompt: The formatted question string.
            history: Full conversation from previous rounds.

        Returns:
            A state dict with keys:
              - ``reasoning_prompt``: full prompt for Stage 1 LLM call.
              - ``system``: system prompt to use for Stage 1.
              - ``is_solo``: True if this is Round 0.
              - ``next_round``: integer index of this round.
        """
        is_solo = not history
        next_round = len(history)

        if is_solo:
            system = self.SOLO_SYSTEM_PROMPT
            instruction = (
                "Reason step by step through the question, "
                "then provide your final answer as {{X}}."
            )
            reasoning_prompt = f"{question_prompt}\n\n{instruction}"
            logger.debug("[TTS Round 0 SOLO] %s", self.agent_id)
        else:
            public_history = self.build_public_history(history)
            mem = self.get_memory()

            parts = [question_prompt]
            if mem:
                parts.append(mem)
            parts.append(public_history)

            mem_instruction = self.memory.get_instruction() if self.memory else ""
            base_instruction = (
                "Based on the debate messages above, reason privately about whether to "
                "update your position. Consider each peer's arguments carefully. "
                "End with your final answer as {{X}}."
            )
            instruction = f"{mem_instruction}\n{base_instruction}" if mem_instruction else base_instruction
            reasoning_prompt = "\n\n".join(parts) + f"\n\n{instruction}"
            system = self.system_prompt  # THINK_SYSTEM_PROMPT
            logger.debug("[TTS Round %d THINK] %s", next_round, self.agent_id)

        return {
            "reasoning_prompt": reasoning_prompt,
            "system": system,
            "is_solo": is_solo,
            "next_round": next_round,
        }

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
            Updated AgentResponse.
        """
        # Ensure the answer tag is present in the public message
        answer = self.extract_answer(public_message)
        if answer == "?":
            # Fall back to the answer extracted from private reasoning
            answer = response.answer
            public_message = public_message.rstrip() + f"\n\n{{{{answer}}}}"
            public_message = public_message.replace("{answer}", answer)

        return AgentResponse(
            name=response.name,
            reasoning=response.reasoning,
            answer=answer if answer != "?" else response.answer,
            public_message=public_message,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_speak_prompt(self, state: dict, private_reasoning: str) -> str:
        """
        Build the Stage 2 prompt that distills private reasoning into a
        ≤200-word public message.

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
                "Write a concise summary (≤200 words) of your reasoning and conclusion. "
                "End with your final answer as {{X}}.\n\n"
                f"Your reasoning:\n{private_reasoning}"
            )

        return (
            "You just reasoned privately about this debate question. "
            "Now write a concise message (≤200 words) for your debate peers.\n\n"
            "Your message MUST:\n"
            "1. Address each peer by name — state whether you agree or disagree and why.\n"
            "2. Present your strongest supporting argument clearly.\n"
            "3. End with your final answer as {{X}} where X is the answer letter.\n\n"
            "Do not exceed 200 words.\n\n"
            f"Your private reasoning:\n{private_reasoning}"
        )

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def update_memory(self, history: Conversation, result) -> None:
        """Update the agent's memory after a debate concludes."""
        if self.memory is not None:
            self.memory.update_memory(history, result=result)

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
