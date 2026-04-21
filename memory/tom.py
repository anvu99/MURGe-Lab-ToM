"""
Theory of Mind (ToM) memory architecture.

Maintains a per-agent mental model: a dictionary mapping agent names to
high-level belief strings describing their reasoning style, tendencies,
and behavior during debates. Uses an LLM to generate and refine beliefs
after each conversation.
"""

import logging
import re
from typing import Any, Dict, Optional

from vllm import LLM, SamplingParams

from configs.configs import Conversation
from memory.base import BaseMemory
from utils import is_deepseek_model, strip_think_blocks

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompt template for belief updates
# ------------------------------------------------------------------

_UPDATE_PROMPT = """\
You are building a Theory of Mind model to support better collaboration in a multi-agent debate system.
Your goal is to produce a profile of another agent that helps a debating partner decide \
how much to trust that agent's answers and when to defer vs. push back.

Agent being profiled: {agent_name}
Previous belief about this agent: {existing_belief}

New evidence from this debate (reasoning and answers across rounds):
{evidence}

Debate outcome: {result}

Update the profile of {agent_name} by addressing all five aspects below.
Integrate the previous belief with the new evidence — do not simply repeat or discard prior observations.

1. REASONING STYLE
   How does this agent approach problems? Is its reasoning systematic or intuitive? \
Verbose or concise? Does it show a clear chain of logic or jump to conclusions?

2. DOMAIN STRENGTHS
   What types of questions or subject areas does this agent answer confidently and correctly? \
Look for consistent patterns across topics.

3. DOMAIN WEAKNESSES
   Where does this agent tend to struggle, show uncertainty, or produce unreliable answers? \
Identify recurring gaps rather than isolated mistakes.

4. TRUSTABILITY
   How reliable are this agent's answers overall? Does it maintain its position \
when challenged, or does it change its mind under pressure? \
Does its expressed confidence match its actual correctness?

5. COLLABORATION SIGNAL
   Given the profile above, provide concrete guidance for a debating partner: \
in which domains or situations should the partner defer to this agent's answer, \
and in which should the partner maintain their own position or challenge this agent? \
This must be domain-specific — avoid blanket trust or blanket distrust.

Writing rules:
- Write in third person, referring to the agent by name ({agent_name}).
- Stay high-level: describe patterns and tendencies, not individual question details.
- Do not quote the evidence directly or reference specific answer letters.
- 1-2 sentences per aspect (5-8 sentences total).
- Every sentence must serve a debating partner trying to decide how to weight this agent's input.\
"""

_SELF_REFLECTION_PROMPT = """\
You are an AI agent analyzing your own reasoning and performance in a multi-agent debate system.
Your goal is to produce a self-reflection profile that helps you improve your future reasoning, \
recognize your own biases, and decide when to trust your own answers vs. when to defer to others.

Agent evaluating: {agent_name} (You)
Previous reflection about yourself: {existing_belief}

New evidence of your reasoning and answers from this debate:
{evidence}

Debate outcome: {result}

Update your self-reflection profile by addressing the aspects below.
Integrate the previous reflection with the new evidence \u2014 do not simply repeat or discard prior observations.

1. REASONING STYLE
   How do you typically approach problems? Do you show a clear chain of logic or jump to conclusions?

2. DOMAIN STRENGTHS
   What types of questions or subject areas do you answer confidently and correctly?

3. DOMAIN WEAKNESSES
   Where do you tend to struggle, show uncertainty, or produce unreliable answers?

4. TRUSTABILITY & OVERCONFIDENCE
   Are you generally reliable? Do you tend to maintain your position when challenged, \
   or do you easily fold under pressure? Do you express unwarranted confidence when you are wrong?

5. COLLABORATION GUIDANCE
   Given your profile, provide yourself concrete guidance for future debates: \
   in which domains should you hold your ground, and when should you be more willing to listen to peers?

Writing rules:
- Write in the second person ("You tend to...", "Your reasoning...").
- Stay high-level: describe patterns and tendencies, not individual question details.
- Do not quote the evidence directly or reference specific answer letters.
- 1-2 sentences per aspect.
"""


class ToMMemory(BaseMemory):
    """
    Theory of Mind memory.

    Stores a dictionary of belief strings, one per agent encountered.
    After each debate, uses the LLM to generate or refine beliefs based
    on the conversation log. The LLM is injected at construction time
    (Option B) so the base interface stays clean.
    """

    def __init__(
        self,
        llm: LLM,
        owner_name: str,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """
        Args:
            llm: A shared vLLM instance for generating belief updates.
            owner_name: Name of the agent that owns this memory.
                        Also used as the key for self-reflection beliefs.
            sampling_params: Optional sampling parameters for belief
                             generation. Defaults to low temperature
                             for consistent summaries.
        """
        super().__init__(name="ToMMemory")
        self.llm = llm
        self.owner_name = owner_name
        self.beliefs: Dict[str, str] = {}

        if sampling_params is not None:
            self.sampling_params = sampling_params
        else:
            # DeepSeek R1-distill models emit <think>...</think> reasoning
            # blocks that consume ~300-400 tokens. Give them a larger budget
            # so the actual profile isn't truncated after stripping.
            is_deepseek = is_deepseek_model(self.llm)
            self.sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=1024 if is_deepseek else 512,
            )



    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def update_memory(
        self,
        conversation: Conversation,
        result: Any = None,
        **kwargs,
    ) -> None:
        """
        Update beliefs about every agent observed in the conversation,
        including the owner itself (self-reflection).

        For each agent, extracts their reasoning and answers across all
        rounds, then calls the LLM to produce an updated belief string.

        Args:
            conversation: The full debate conversation (List[RoundEntry]).
            result: The outcome of the debate (e.g., correct answer, score).
        """
        # Collect per-agent evidence from the conversation
        agent_evidence: Dict[str, list] = {}
        for round_idx, round_entry in enumerate(conversation):
            for agent_id, response in round_entry.agent_responses.items():
                agent_name = response.name
                if agent_name not in agent_evidence:
                    agent_evidence[agent_name] = []
                agent_evidence[agent_name].append(
                    f"[Round {round_idx}] Reasoning: {response.reasoning} "
                    f"| Answer: {response.answer}"
                )

        # Generate / refine a belief for each observed agent
        for agent_name, evidence_lines in agent_evidence.items():
            existing_belief = self.beliefs.get(agent_name, "No prior belief.")
            evidence_str = "\n".join(evidence_lines)
            result_str = str(result) if result is not None else "Unknown"

            if agent_name == self.owner_name:
                prompt_template = _SELF_REFLECTION_PROMPT
                system_content = (
                    "You are an AI agent analyzing your own prior reasoning to improve your future performance. "
                    "Your task is to produce a concise, high-level self-reflection profile that captures your "
                    "reasoning patterns, strengths, weaknesses, and biases. Focus on observable patterns."
                )
            else:
                prompt_template = _UPDATE_PROMPT
                system_content = (
                    "You are an expert observer building Theory of Mind models of AI agents. "
                    "Your task is to produce concise, high-level profiles that capture each agent's "
                    "reasoning patterns, strengths, weaknesses, and trustability. "
                    "Focus on observable patterns, not specific details."
                )

            prompt = prompt_template.format(
                agent_name=agent_name,
                existing_belief=existing_belief,
                evidence=evidence_str,
                result=result_str,
            )

            is_gemma = False
            try:
                if hasattr(self.llm, "llm_engine") and "gemma" in str(self.llm.llm_engine.model_config.model).lower():
                    is_gemma = True
            except Exception:
                pass

            if is_gemma:
                messages = [
                    {"role": "user", "content": f"{system_content}\n\n{prompt}"},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ]

            logger.debug(
                "Updating ToM belief for '%s'. Evidence rounds: %d",
                agent_name,
                len(evidence_lines),
            )

            try:
                outputs = self.llm.chat(
                    messages=[messages],
                    sampling_params=self.sampling_params,
                )
                if outputs and outputs[0].outputs:
                    updated_belief = outputs[0].outputs[0].text.strip()
                    # Strip DeepSeek's <think>...</think> reasoning blocks
                    updated_belief = strip_think_blocks(updated_belief)
                else:
                    logger.warning("Empty belief update for '%s', keeping old belief.", agent_name)
                    continue
            except Exception as e:
                logger.error("ToM belief update failed for '%s': %s", agent_name, e)
                continue

            self.beliefs[agent_name] = updated_belief

            logger.info(
                "ToM belief updated for '%s': %s",
                agent_name,
                updated_belief[:100],
            )

    def retrieve_memory(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Retrieve stored beliefs, clearly separating self-reflection from peer profiles.

        Args:
            query: A specific agent name to look up. If None, returns all beliefs
                   split into self-reflection and peer profile sections.

        Returns:
            A formatted string ready for prompt injection.
        """
        if query is not None:
            return self.beliefs.get(query, "")

        if not self.beliefs:
            return ""

        lines = []

        # --- Self-reflection (own belief) ---
        if self.owner_name in self.beliefs:
            lines.append("[Self-reflection] Your own observed tendencies:")
            lines.append(f"  {self.beliefs[self.owner_name]}")

        # --- Peer profiles ---
        peer_beliefs = {
            name: belief
            for name, belief in self.beliefs.items()
            if name != self.owner_name
        }
        if peer_beliefs:
            if lines:
                lines.append("")  # blank line between sections
            lines.append("[Peer profiles] Your beliefs about other agents:")
            for agent_name, belief in peer_beliefs.items():
                lines.append(f"  - {agent_name}: {belief}")

        return "\n".join(lines)

    def get_instruction(self) -> str:
        return (
            "You MUST use your [Self-reflection] and the [Peer profiles] provided above "
            "to evaluate the reliability of your own reasoning versus your peers. "
            "Use these profiles to decide when to trust them and when to hold your ground. "
        )

    def clear(self) -> None:
        """Reset all stored beliefs."""
        self.beliefs.clear()
        logger.info("ToM memory cleared for owner '%s'.", self.owner_name)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ToMMemory(owner='{self.owner_name}', "
            f"agents_modeled={list(self.beliefs.keys())})"
        )
