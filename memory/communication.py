"""
CommunicationStrategyMemory — tracks what argument styles and formats
cause a specific peer agent to engage vs. ignore arguments.

Updated after each debate by synthesising the engagement observations the
CSA debater already produced during the debate (Step 1 SEND AUDIT and Step 2
COMMUNICATION ANALYSIS sections extracted from private reasoning). The update
LLM receives these pre-computed observations and only needs to distil them
into actionable forward-looking directives — it does NOT re-analyse the raw
transcript.

Injected into Stage 1 Step 2 (Communication Analysis) as prior guidance.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams

from configs.configs import Conversation
from memory.base import BaseMemory
from utils import is_deepseek_model, strip_think_blocks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Update prompt
# ---------------------------------------------------------------------------

_UPDATE_PROMPT = """\
You are refining a communication strategy for how {owner_name} should package \
arguments to maximize engagement from {peer_name}.

During the debate just completed, {owner_name} already performed engagement \
audit observations and communication adaptation analysis at each turn \
(Step 1 — SEND AUDIT + COMM ANALYSIS). These are extracted directly \
from {owner_name}'s private reasoning below:

{evidence}

Previous directives for reaching {peer_name}:
{existing_strategy}

Your output MUST use exactly this two-section format:

[ANALYSIS]
<Your free-form reasoning: identify structural patterns across multiple turns. \
Cite specific turns as evidence.>

[DIRECTIVES]
DIRECTIVE_1: <1-2 sentences, second person, structural and transferable>
DIRECTIVE_2: <1-2 sentences, second person, structural and transferable>
... (up to DIRECTIVE_3 maximum)

Rules for [DIRECTIVES]:
- Always start at DIRECTIVE_1 — never continue numbering from prior directives.
- Generate exactly 2–3 directives.
- Each directive must describe a STRUCTURAL or BEHAVIORAL pattern: how to frame, \
sequence, or challenge arguments to elicit a response from {peer_name}.
- Write in second person: "When arguing with {peer_name}...", "{peer_name} tends to engage with..."
- Only state patterns supported by evidence from multiple turns in your [ANALYSIS].
- 1–2 sentences per directive.
- Do NOT reference specific answer letters, question content, or any \
domain knowledge (no subject-matter terms like "enthalpy", "legal precedents", \
"evolutionary biology", "Gaussian distribution", etc.).

CRITICAL — Domain-agnostic filter:
Before writing each directive, ask: “Would this directive still be useful if the \
next question were about a completely different subject?” If the answer is NO, \
rewrite it more abstractly or discard it.

BAD (topic-specific, never output these):
  ✗ "Provide legal precedents to support your points."
  ✗ "Include step-by-step enthalpy calculations."

GOOD (structural, transferable):
  ✓ "{peer_name} tends to engage more when you open with a direct challenge \
rather than a statement — lead each argument with a targeted question."
  ✓ "When {peer_name} only partially addresses a point, explicitly name the \
un-addressed part and ask them to respond to it directly."
"""


def _extract_directives_from_output(text: str) -> List[str]:
    """
    Extract DIRECTIVE_N entries from the [DIRECTIVES] section of the LLM output.
    """
    # Isolate the [DIRECTIVES] section
    section_match = re.search(
        r"\[DIRECTIVES\](.*?)(?=\[|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    section = section_match.group(1) if section_match else text

    # Extract each DIRECTIVE_N: ... entry
    matches = re.findall(
        r"DIRECTIVE_\d+:\s*(.+?)(?=DIRECTIVE_\d+:|$)",
        section,
        re.DOTALL | re.IGNORECASE,
    )
    return [m.strip() for m in matches if m.strip()]


class CommunicationStrategyMemory(BaseMemory):
    """
    Per-peer communication strategy memory.

    Learns what argument styles and formats maximize a specific peer's
    engagement with the owner's reasoning. Updated after each debate.
    Injected into Stage 1 Step 2 as a prior for communication analysis.
    """

    def __init__(
        self,
        llm: LLM,
        owner_name: str,
        peer_name: str,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """
        Args:
            llm: vLLM instance for generating strategy updates.
            owner_name: Name of the agent that owns this memory (the sender).
            peer_name: Name of the specific peer this strategy targets.
            sampling_params: Optional override for strategy generation.
        """
        super().__init__(name="CommunicationStrategyMemory")
        self.llm = llm
        self.owner_name = owner_name
        self.peer_name = peer_name
        self._directive_list: List[str] = []  # accumulated directives as a list
        
        if sampling_params is not None:
            self.sampling_params = sampling_params
        else:
            is_deepseek = is_deepseek_model(self.llm)
            self.sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=1024 if is_deepseek else 512,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> str:
        """Backward-compatible string view of strategy (for logging/snapshots)."""
        if not self._directive_list:
            return ""
        return "\n".join(
            f"DIRECTIVE_{i + 1}: {d}" for i, d in enumerate(self._directive_list)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_step_observations(reasoning: str) -> str:
        """
        Extract the Step 1 (SEND AUDIT + COMM ANALYSIS) section from an agent's
        private reasoning string.
        """
        # Match STEP 1 content from its header to the next STEP header or end
        pattern = re.compile(
            r"(\[STEP 1[^\]]*\].*?)(?=\[STEP [2-5][^\]]*\]|$)",
            re.DOTALL,
        )
        matches = pattern.findall(reasoning)
        return "\n\n".join(m.strip() for m in matches if m.strip())

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def update_memory(
        self,
        conversation: Conversation,
        result: Any = None,
        question_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Update communication strategy after a debate.
        """
        if not conversation:
            return

        # Extract Step 1 observations from each of the owner's turns.
        observation_blocks: List[str] = []
        for turn_idx, turn_entry in enumerate(conversation):
            for _, response in turn_entry.agent_responses.items():
                if response.name != self.owner_name:
                    continue
                reasoning = response.reasoning or ""
                obs = self._extract_step_observations(reasoning)
                if obs:
                    observation_blocks.append(f"[Turn {turn_idx} observations]\n{obs}")

        if not observation_blocks:
            logger.info(
                "CommunicationStrategyMemory: no Step 1 observations found for '%s'. "
                "Skipping update.",
                self.owner_name,
            )
            return

        evidence_str = "\n\n".join(observation_blocks)

        prompt = _UPDATE_PROMPT.format(
            owner_name=self.owner_name,
            peer_name=self.peer_name,
            existing_strategy=(
                "\n".join(
                    f"DIRECTIVE_{i + 1}: {d}"
                    for i, d in enumerate(self._directive_list)
                )
                if self._directive_list
                else "(none yet \u2014 first debate)"
            ),
            evidence=evidence_str,
        )

        is_gemma = False
        try:
            if hasattr(self.llm, "llm_engine") and "gemma" in str(
                self.llm.llm_engine.model_config.model
            ).lower():
                is_gemma = True
        except Exception:
            pass

        system_content = (
            f"You are generating a communication strategy profile for how {self.owner_name} "
            f"can make its arguments more engaging to {self.peer_name}. "
            "Focus on structural and stylistic patterns — not on the correctness of arguments. "
            "Every directive must be specific and evidence-based."
        )

        if is_gemma:
            messages = [{"role": "user", "content": f"{system_content}\n\n{prompt}"}]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

        try:
            outputs = self.llm.chat(
                messages=[messages],
                sampling_params=self.sampling_params,
            )
            if outputs and outputs[0].outputs:
                raw_output = outputs[0].outputs[0].text.strip()
                raw_output = strip_think_blocks(raw_output)
                extracted = _extract_directives_from_output(raw_output)
                if extracted:
                    self._directive_list = extracted
                    logger.info(
                        "CommunicationStrategyMemory updated for '%s' → '%s': "
                        "%d directives extracted.",
                        self.owner_name,
                        self.peer_name,
                        len(extracted),
                    )
                else:
                    logger.warning(
                        "Directive extraction returned empty for '%s' → '%s'; "
                        "keeping old strategy.",
                        self.owner_name,
                        self.peer_name,
                    )
            else:
                logger.warning(
                    "Empty comm strategy update for '%s' → '%s', keeping old strategy.",
                    self.owner_name,
                    self.peer_name,
                )
        except Exception as e:
            logger.error(
                "CommunicationStrategyMemory update failed for '%s' → '%s': %s",
                self.owner_name, self.peer_name, e,
            )

    def retrieve_memory(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Return the current strategy string for prompt injection.
        """
        if not self._directive_list:
            return ""
        return "\n".join(
            f"{i + 1}. {d}" for i, d in enumerate(self._directive_list)
        )

    def get_instruction(self) -> str:
        return (
            f"Use the communication strategy above when deciding how to package "
            f"your arguments for {self.peer_name} this round."
        )

    def clear(self) -> None:
        """Reset the stored strategy."""
        self._directive_list = []
        logger.info(
            "CommunicationStrategyMemory cleared for '%s' → '%s'.",
            self.owner_name,
            self.peer_name,
        )

    def __repr__(self) -> str:
        return (
            f"CommunicationStrategyMemory("
            f"owner='{self.owner_name}', peer='{self.peer_name}', "
            f"directives={len(self._directive_list)})"
        )
