"""
ReasoningMemory — tracks how an agent consumes peer arguments.

Stores directives about three failure modes:
  1. Dismissal    — ignoring valid peer arguments without addressing them
  2. Sycophancy   — changing answers before completing thorough reasoning
  3. Under-engagement — partial reading of peer arguments

Updated after each debate via LLM analysis of public message pairs.
Injected into Stage 1 (Step 3) as behavioral guardrails.
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
You are analyzing an AI agent's debate performance to generate actionable directives \
for how the agent should better consume and integrate peer arguments in future debates.

Agent being analyzed: {agent_name}
Peer they debated: {peer_name}
Correct answer: {correct_answer}
Category: {category}

Previous directives for this agent:
{existing_directives}

Full debate transcript (peer public messages, agent public responses, agent private reasoning):
{evidence}

Your task: Identify patterns in how this agent consumed peer arguments.
Focus on these three failure modes:

1. DISMISSAL: Did the agent ignore valid peer arguments without addressing them?
   (An argument is "valid" if it pointed toward the correct answer — cross-reference the
   Correct answer field provided above. Do NOT flag dismissal of peer arguments that were
   leading away from the correct answer.)

2. SYCOPHANCY: Did the agent ever change their answer without articulating WHY the \
peer's logic was correct?
   (Signal: answer changes between rounds but the agent's reasoning does not explicitly \
explain what was wrong with their previous position.)

3. UNDER-ENGAGEMENT: Did the agent only partially engage with peer arguments, \
addressing surface points while missing the core challenge?

Your output MUST use exactly this two-section format:

[ANALYSIS]
<Your free-form reasoning: cite specific turns as evidence. Verify each pattern is genuine.>

[DIRECTIVES]
DIRECTIVE_1: <1-2 sentences, second person, specific and actionable>
DIRECTIVE_2: <1-2 sentences, second person, specific and actionable>
... (up to DIRECTIVE_5 maximum)

Rules for [DIRECTIVES]:
- Always start at DIRECTIVE_1 — never continue numbering from prior directives.
- Generate 2–5 directives total.
- Each must be specific, not generic \
  (e.g., not "engage more" but "when peer introduces a worked example, \
verify their calculation step-by-step before dismissing it").
- Write in second person: "You tend to...", "Before dismissing...", "When peer..."
- Only write directives supported by evidence in your [ANALYSIS].
- Do not reference specific answer letters or question details.
"""


def _extract_directives_from_output(text: str) -> List[str]:
    """
    Extract DIRECTIVE_N entries from the [DIRECTIVES] section of the LLM output.
    Falls back to storing the full output as a single entry if parsing fails.
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
    directives = [m.strip() for m in matches if m.strip()]

    # Fallback: store raw text as one entry if nothing matched
    if not directives and text.strip():
        directives = [text.strip()]

    return directives


class ReasoningMemory(BaseMemory):
    """
    Tracks how an agent consumes peer arguments across debates.

    Stores a single string of directives that grows more specific over time.
    Injected into Stage 1 Step 3 (Answer Reasoning) as behavioral guardrails.
    """

    def __init__(
        self,
        llm: LLM,
        owner_name: str,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """
        Args:
            llm: vLLM instance for generating directive updates.
            owner_name: Name of the agent that owns this memory.
            sampling_params: Optional override for directive generation.
        """
        super().__init__(name="ReasoningMemory")
        self.llm = llm
        self.owner_name = owner_name
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
    def directives(self) -> str:
        """Backward-compatible string view of directives (for logging/snapshots)."""
        if not self._directive_list:
            return ""
        return "\n".join(
            f"DIRECTIVE_{i + 1}: {d}" for i, d in enumerate(self._directive_list)
        )

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
        Update reasoning directives after a debate.

        Analyzes which peer arguments this agent failed to engage with,
        whether any answer changes were sycophantic, and refines directives.

        Args:
            conversation: Full debate conversation (List[RoundEntry]).
            result: Final answer of the debate.
            question_data: MMLU question data with 'answer' and 'category'.
        """
        if not conversation:
            return

        correct_answer = question_data.get("answer", "?") if question_data else "?"
        category = question_data.get("category", "general") if question_data else "general"

        # Find the peer name from any turn we didn't speak
        peer_name = "peer"
        for turn_entry in conversation:
            for _, response in turn_entry.agent_responses.items():
                if response.name != self.owner_name:
                    peer_name = response.name
                    break
            if peer_name != "peer":
                break

        # Build per-turn evidence.
        # Per-turn conversation: each RoundEntry has exactly one agent.
        # We tag each turn as an owner-turn or peer-turn and pair them
        # for the sycophancy analysis (look for answer changes in owner turns).
        evidence_lines: List[str] = []
        prev_owner_answer = None
        turn_idx = 0
        for turn_entry in conversation:
            for _, response in turn_entry.agent_responses.items():
                if response.name == self.owner_name:
                    owner_pub = response.public_message or f"(Answer: {response.answer})"
                    owner_reasoning = response.reasoning or "(no reasoning captured)"
                    answer_changed = (
                        prev_owner_answer is not None
                        and response.answer != "?"
                        and response.answer != prev_owner_answer
                    )
                    line = (
                        f"[Turn {turn_idx}] {self.owner_name} said: {owner_pub}\n"
                        f"  private reasoning: {owner_reasoning}"
                    )
                    if answer_changed:
                        line += f"\n  *** ANSWER CHANGED: {prev_owner_answer} → {response.answer} ***"
                    prev_owner_answer = response.answer
                else:
                    peer_pub = response.public_message or f"(Answer: {response.answer})"
                    line = f"[Turn {turn_idx}] {peer_name} said: {peer_pub}"
                evidence_lines.append(line)
            turn_idx += 1

        evidence_str = "\n\n".join(evidence_lines)

        # Build update prompt
        prompt = _UPDATE_PROMPT.format(
            agent_name=self.owner_name,
            peer_name=peer_name,
            correct_answer=correct_answer,
            category=category,
            existing_directives=(
                "\n".join(
                    f"DIRECTIVE_{i + 1}: {d}"
                    for i, d in enumerate(self._directive_list)
                )
                if self._directive_list
                else "(none yet — first debate)"
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
            "You are generating actionable self-improvement directives for an AI debate agent. "
            "Every directive must be a specific, concrete instruction — not a vague observation. "
            "Focus only on how the agent receives and evaluates peer arguments. "
            "The evidence includes each turn's private reasoning tagged as 'private reasoning: ...'. "
            "Use this field to detect sycophancy: if an answer changed but the private reasoning "
            "does not explicitly articulate why the previous answer was wrong, that is a sycophantic change."
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
                        "ReasoningMemory updated for '%s': %d directives extracted.",
                        self.owner_name,
                        len(extracted),
                    )
                else:
                    logger.warning(
                        "Directive extraction returned empty for '%s'; keeping old directives.",
                        self.owner_name,
                    )
            else:
                logger.warning(
                    "Empty reasoning memory update for '%s', keeping old directives.",
                    self.owner_name,
                )
        except Exception as e:
            logger.error("ReasoningMemory update failed for '%s': %s", self.owner_name, e)

    def retrieve_memory(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Return the current directives string for prompt injection.

        Returns:
            Formatted string ready for Stage 1 Step 3 injection,
            or empty string if no directives have been accumulated yet.
        """
        if not self._directive_list:
            return ""
        return "\n".join(
            f"{i + 1}. {d}" for i, d in enumerate(self._directive_list)
        )

    def get_instruction(self) -> str:
        return (
            "Apply every directive above before evaluating your peer's arguments. "
            "Pay special attention to the sycophancy guard: "
            "Only change your answer if you can articulate a specific, evidence-based reason why your previous reasoning was flawed."
        )

    def clear(self) -> None:
        """Reset all stored directives."""
        self._directive_list = []
        logger.info("ReasoningMemory cleared for owner '%s'.", self.owner_name)

    def __repr__(self) -> str:
        return (
            f"ReasoningMemory(owner='{self.owner_name}', "
            f"directives={len(self._directive_list)})"
        )
