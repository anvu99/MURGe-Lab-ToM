"""
Base agent for the multi-agent debate system.
Shared functionality used by both debaters and judges.
"""

import logging
import re
from typing import Optional

from vllm import LLM, SamplingParams

from configs.configs import Conversation, AgentResponse
from utils import is_deepseek_model

logger = logging.getLogger(__name__)


class BaseAgent():
    """
    Base class with shared functionality for all agent types.

    Handles LLM interaction, memory integration, answer extraction,
    and conversation history. Debater and Judge subclasses add
    role-specific prompt building and generation logic.
    """

    def __init__(
        self,
        agent_id: str,
        agent_role: str,
        model_name: str,
        system_prompt: str,
        llm: LLM,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """
        Args:
            agent_id: Unique identifier (e.g., "agent_0").
            agent_role: Role in the debate ("debater" or "judge").
            model_name: LLM model name (e.g., "Qwen/Qwen3-32B").
            system_prompt: Base system prompt for this agent.
            llm: A shared vLLM LLM instance (loaded externally).
            sampling_params: Optional vLLM sampling parameters.
                             Defaults to temperature=0.7, max_tokens=1024.
        """
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm = llm

        if sampling_params is not None:
            self.sampling_params = sampling_params
        else:
            is_deepseek = is_deepseek_model(self.llm)
            self.sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=1024 if is_deepseek else 512,
            )
    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def call_llm(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
        system_prompt_override: Optional[str] = None,
    ) -> str:
        """
        Send a prompt to the LLM using the chat API so that the correct
        chat template (Qwen, Llama, etc.) is applied automatically.

        Args:
            prompt: The user-turn prompt string.
            params: Optional override for sampling parameters.
            system_prompt_override: If provided, replaces self.system_prompt
                                    for this call only (e.g., solo vs. debate mode).

        Returns:
            The generated text from the LLM.
        """
        system = system_prompt_override if system_prompt_override is not None else self.system_prompt
        if "gemma" in self.model_name.lower():
            messages = [
                {"role": "user", "content": f"{system}\n\n{prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        logger.debug("LLM chat call by %s: %s", self.agent_id, prompt[:200])
        try:
            outputs = self.llm.chat(
                messages=[messages],
                sampling_params=params or self.sampling_params,
            )
            if outputs and outputs[0].outputs:
                return outputs[0].outputs[0].text.strip()
            logger.warning("Empty LLM output for %s", self.agent_id)
            return ""
        except Exception as e:
            logger.error("LLM call failed for %s: %s", self.agent_id, e)
            return ""

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def extract_answer(self, raw_response: str) -> str:
        """
        Parse the answer letter (A-J) from the LLM's raw output.

        Strategy (in order of priority):

        1. Last-token pass — check the very end of the response for any
           recognised answer format after stripping trailing whitespace.
           Formats tried (in decreasing specificity):
             - {{X}}      template-escaped double braces
             - {X}        single-brace (produced when not inside an f-string)
             - \\boxed{X}
             - bare X     lone capital letter not attached to a word
           This step is strict (tail-only) so it cannot be fooled by letters
           embedded inside prose like "P = I / (RT)" in a bullet point.

        2. High-priority full-text scan — {{X}}, {X}, \\boxed{X}, and the
           explicit arrow directives (My private conclusion / My final answer).
           We pick the LAST occurrence.

        3. Low-priority fallback scan — "answer is X", numbered lists.
           Only reached when the model failed to follow the format instructions.

        Args:
            raw_response: The raw text output from the LLM.

        Returns:
            A single uppercase answer letter (A-J), or "?" if not found.
        """
        if not raw_response:
            return "?"

        # ------------------------------------------------------------------
        # Step 1: Last-token pass (tail of the text, all recognised formats).
        # Strip trailing whitespace only — don't eat closing braces.
        # ------------------------------------------------------------------
        tail = raw_response.rstrip(" \t\n\r")

        # 1a. {{X}} — template-escaped double braces
        m = re.search(r"\{\{([A-J])\}\}$", tail)
        if m:
            return m.group(1)

        # 1b. {X} — single-brace variant
        m = re.search(r"\{([A-J])\}$", tail)
        if m:
            return m.group(1)

        # 1c. \boxed{X}
        m = re.search(r"\\boxed\{([A-J])\}$", tail)
        if m:
            return m.group(1)

        # 1d. Bare capital letter at the very end, not attached to a word.
        #     Strip common sentence-ending punctuation (but NOT braces) first.
        tail_stripped = tail.rstrip(".,!?;:'\"* \t\n\r")
        m = re.search(r"(?:^|[^A-Za-z0-9])([A-J])$", tail_stripped)
        if m:
            return m.group(1)

        # 1e. Digit at the very end — convert to letter (0->A … 9->J)
        m = re.search(r"(?:^|[^A-Za-z0-9])([0-9])$", tail_stripped)
        if m:
            return chr(65 + int(m.group(1)))

        # ------------------------------------------------------------------
        # Step 2: High-priority full-text scan.
        # Only explicit structured answer formats — pick the LAST occurrence.
        # ------------------------------------------------------------------
        high_priority = []

        for m in re.finditer(r"\{\{([A-J])\}\}", raw_response):
            high_priority.append((m.end(), m.group(1)))

        for m in re.finditer(r"\{([A-J])\}", raw_response):
            high_priority.append((m.end(), m.group(1)))

        for m in re.finditer(r"\\boxed\{([A-J])\}", raw_response):
            high_priority.append((m.end(), m.group(1)))

        # Explicit skeleton directive arrows
        for m in re.finditer(
            r"\u2192\s*My\s+(?:private\s+conclusion|final\s+answer)\s*:\s*(?:\{\{?)?([A-J])(?:\}?\})?",
            raw_response,
        ):
            high_priority.append((m.end(), m.group(1)))

        if high_priority:
            high_priority.sort(key=lambda x: x[0])
            return high_priority[-1][1]

        # ------------------------------------------------------------------
        # Step 3: Low-priority fallback scan.
        # Weak heuristics — only reached when the model ignored all formats.
        # ------------------------------------------------------------------
        low_priority = []

        for m in re.finditer(r"(?:[Ff]inal|[Tt]he)\s+answer\s+is\s*\(?([A-J])\)?", raw_response):
            low_priority.append((m.end(), m.group(1)))

        for m in re.finditer(r"(?:^|\n)([A-J])(?:[.)\s]|$)(?!\w)", raw_response):
            low_priority.append((m.end(), m.group(1)))

        for m in re.finditer(r"(?:answer|option|choice)[:\s]+([0-9])\b", raw_response, re.IGNORECASE):
            low_priority.append((m.end(), chr(65 + int(m.group(1)))))

        for m in re.finditer(r"(?:^|\n)([0-9])(?:[.)\s]|$)", raw_response):
            low_priority.append((m.end(), chr(65 + int(m.group(1)))))

        if low_priority:
            low_priority.sort(key=lambda x: x[0])
            return low_priority[-1][1]

        return "?"

        # We only want to strip common sentence-ending punctuation/whitespace
        # We avoid string.punctuation so we don't strip bracketed answers like {{A}}
        strip_chars = " \t\n\r" + ".,!?;:'\"*"
        clean_end = raw_response.rstrip(strip_chars)
        
        # 1. First prioritize {{X}} if it's explicitly at the end
        match_end = re.search(r"\{\{([A-Z])\}\}$", clean_end)
        if match_end:
            return match_end.group(1)
            
        # 2. Check if there's a standalone answer X at the end
        match_end = re.search(r"(?:^|[^a-zA-Z0-9])([A-Z])$", clean_end)
        if match_end:
            return match_end.group(1)
            
        # 3. Check for \boxed{X} at the end
        match_end = re.search(r"\\boxed\{([A-J])\}$", clean_end)
        if match_end:
            return match_end.group(1)
            
        # 4. Check for standalone digit at the end
        match_end = re.search(r"(?:^|[^a-zA-Z0-9])([0-9])$", clean_end)
        if match_end:
            return chr(65 + int(match_end.group(1)))

        # ---------------------------------------------------------------------
        # Fallbacks: if the answer wasn't correctly positioned at the very end
        # we scan the whole text and pick the *last* explicit answer pattern.
        # We use a priority system so that explicit formats like {{X}} cannot
        # be overridden by weak fallbacks like a numbered list "4." later on.
        # ---------------------------------------------------------------------
        high_priority_matches = []
        low_priority_matches = []
        
        # High priority: Explicitly requested formats
        for m in re.finditer(r"\{\{([A-Z])\}\}", raw_response):
            high_priority_matches.append((m.end(), m.group(1)))
            
        for m in re.finditer(r"\\boxed\{([A-J])\}", raw_response):
            high_priority_matches.append((m.end(), m.group(1)))

        # High priority: Our explicit skeleton prompt directive
        for m in re.finditer(r"\u2192 My (?:private conclusion|final answer):\s*(?:\{\{)?([A-Z])(?:\}\})?", raw_response):
            high_priority_matches.append((m.end(), m.group(1)))
            
        # Low priority: Weak fallbacks (often hallucinated or part of regular text)
        for m in re.finditer(r"(?:[Ff]inal|[Tt]he) answer is\s*\(?([A-Z])\)?", raw_response):
            low_priority_matches.append((m.end(), m.group(1)))
            
        for m in re.finditer(r"(?:^|\n)([A-Z])(?:[.)\s]|$)(?!\w)", raw_response):
            low_priority_matches.append((m.end(), m.group(1)))
            
        for m in re.finditer(r"(?:answer|option|choice)[:\s]+([0-9])\b", raw_response, re.IGNORECASE):
            low_priority_matches.append((m.end(), chr(65 + int(m.group(1)))))
            
        for m in re.finditer(r"(?:^|\n)([0-9])(?:[.)\s]|$)", raw_response):
            low_priority_matches.append((m.end(), chr(65 + int(m.group(1)))))
            
        if high_priority_matches:
            high_priority_matches.sort(key=lambda x: x[0])
            return high_priority_matches[-1][1]
            
        if low_priority_matches:
            low_priority_matches.sort(key=lambda x: x[0])
            return low_priority_matches[-1][1]

        return "?"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id='{self.agent_id}', "
            f"role='{self.agent_role}', "
            f"model='{self.model_name}')"
        )
