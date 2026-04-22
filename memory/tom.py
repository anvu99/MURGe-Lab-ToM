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
You are building a Theory of Mind profile for an AI agent to guide future debate interactions.
Your goal is to produce actionable guidance: every sentence should directly help the owner \
decide how to interact with this agent, how much to trust them, and what strategies work.

Agent being profiled: {agent_name}
Previous belief about this agent: {existing_belief}

Quantitative performance data for {agent_name}:
{agent_stats}

Your own quantitative performance data (for comparison):
{owner_stats}

New evidence from this debate (Public Debate Transcript and your own Private Thinking):
{evidence}

Debate outcome: {result}

Update the profile of {agent_name} by addressing the two aspects below.
Integrate the previous belief with the new evidence and stats.

1. DOMAIN COMPETENCE
   Using the quantitative stats and debate evidence, which domains can this agent be trusted on \
   and where should their answers be scrutinized? Compare against your own stats to note \
   where they are stronger or weaker than you.

2. INTERACTION STRATEGY
   Provide concrete, actionable instructions for future interactions with this agent. \
   Focus on patterns most supported by the evidence — for example: how to critically \
   evaluate their arguments, how much weight to give their evidence in specific contexts, \
   or what to watch out for in their reasoning style. \
   Only include advice you have genuine evidence for.

Writing rules:
- Write in third person, referring to the agent by name ({agent_name}).
- Stay high-level: describe patterns and tendencies, not individual question details.
- Do not quote the evidence directly or reference specific answer letters.
- Write 2-4 sentences per aspect (4-8 sentences total).
- Every sentence must be a direct instruction or actionable insight — no pure description.\
"""

_SELF_REFLECTION_PROMPT = """\
You are an AI agent analyzing your own reasoning to produce actionable self-improvement guidance.
Your goal is to produce directives that directly improve your future debate performance: \
when to trust yourself, when to defer, and what to do differently.

Agent evaluating: {agent_name} (You)
Previous reflection about yourself: {existing_belief}

Your own quantitative performance data:
{agent_stats}

New evidence of your reasoning and answers from this debate:
{evidence}

Debate outcome: {result}

Update your self-reflection profile by addressing the two aspects below.
Integrate the previous reflection with the new evidence and stats.

1. DOMAIN COMPETENCE
   Using the quantitative stats, which domains should you trust your own reasoning in \
   and where are you making systematic errors? Be specific about error patterns.

2. SELF-IMPROVEMENT STRATEGY
   Provide yourself concrete directives for future debates. \
   Focus on patterns most supported by the evidence — for example: when to demand \
   stronger evidence before changing your mind, or specific reasoning fixes. \
   Only include advice you have genuine evidence for.

Writing rules:
- Write in the second person ("You tend to...", "Your reasoning...").
- Stay high-level: describe patterns and tendencies.
- Do not quote the evidence directly or reference specific answer letters.
- Write 2-4 sentences per aspect (4-8 sentences total).
- Every sentence must be a direct instruction or actionable insight — no pure description.
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
        # Structure: {agent_name: {domain: {"correct": N, "total": N, "persuaded_peers": N, "was_persuaded": N}}}
        self.stats: Dict[str, Dict[str, Dict[str, int]]] = {}

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
        question_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Update beliefs about every agent observed in the conversation,
        including the owner itself (self-reflection).

        For each agent, extracts their reasoning and answers across all
        rounds, then calls the LLM to produce an updated belief string.
        Also tracks deterministic domain stats.

        Args:
            conversation: The full debate conversation (List[RoundEntry]).
            result: The outcome of the debate (e.g., correct answer, score).
            question_data: The MMLU question data containing category and answer.
        """
        if not conversation:
            return

        # ------------------------------------------------------------------
        # Update Deterministic Domain Stats
        # ------------------------------------------------------------------
        category = question_data.get("category", "general") if question_data else "general"
        correct_answer = question_data.get("answer", "") if question_data else ""
        
        initial_answers = {}
        final_answers = {}
        
        if len(conversation) > 0:
            for _, response in conversation[0].agent_responses.items():
                initial_answers[response.name] = response.answer
            for _, response in conversation[-1].agent_responses.items():
                final_answers[response.name] = response.answer
                
        if category and correct_answer:
            for agent_name, final_ans in final_answers.items():
                if agent_name not in self.stats:
                    self.stats[agent_name] = {}
                if category not in self.stats[agent_name]:
                    self.stats[agent_name][category] = {
                        "correct": 0, "total": 0, 
                        "persuaded_peers": 0, "was_persuaded": 0
                    }
                
                # Accuracy
                if final_ans != "?":
                    self.stats[agent_name][category]["total"] += 1
                    if final_ans == correct_answer:
                        self.stats[agent_name][category]["correct"] += 1
                        
                # Persuadability
                initial_ans = initial_answers.get(agent_name, "?")
                if initial_ans != "?" and final_ans != "?" and initial_ans != final_ans:
                    self.stats[agent_name][category]["was_persuaded"] += 1
                    
                # Persuasiveness
                for other_agent, other_final in final_answers.items():
                    if other_agent == agent_name:
                        continue
                    other_initial = initial_answers.get(other_agent, "?")
                    if other_initial != "?" and other_final != "?" and other_initial != other_final:
                        if other_final == initial_ans:
                            self.stats[agent_name][category]["persuaded_peers"] += 1

        # ------------------------------------------------------------------
        # Build Epistemic Boundary Evidence
        # ------------------------------------------------------------------
        agent_names = list({resp.name for round_entry in conversation for resp in round_entry.agent_responses.values()})
        evidence_built: Dict[str, str] = {}
        
        for target_agent in agent_names:
            evidence_lines = []
            for round_idx, round_entry in enumerate(conversation):
                lines = [f"[Round {round_idx}]"]
                
                owner_response = None
                public_transcript = []
                
                for _, response in round_entry.agent_responses.items():
                    if response.name == self.owner_name:
                        owner_response = response
                    
                    pub_msg = response.public_message if response.public_message else "(No public message provided)"
                    if not response.public_message and hasattr(response, 'reasoning') and response.reasoning:
                        pub_msg = response.reasoning[-200:] # fallback
                        
                    pub_line = f"    {response.name}: {pub_msg} (answered {response.answer})"
                    public_transcript.append(pub_line)

                if owner_response:
                    lines.append(f"  Your Private Thinking: {owner_response.reasoning}")
                    if target_agent == self.owner_name:
                        lines.append(f"  Your Public Message: {owner_response.public_message}")
                        lines.append(f"  Your Answer: {owner_response.answer}")
                        
                lines.append("  Public Debate Transcript:")
                lines.extend(public_transcript)
                evidence_lines.append("\n".join(lines))
                
            evidence_built[target_agent] = "\n\n".join(evidence_lines)

        # Generate / refine a belief for each observed agent
        for agent_name, evidence_str in evidence_built.items():
            existing_belief = self.beliefs.get(agent_name, "No prior belief.")
            result_str = str(result) if result is not None else "Unknown"
            
            # Format stats helper
            def format_stats(a_name: str) -> str:
                a_stats = self.stats.get(a_name, {})
                s_lines = []
                for dom, metrics in a_stats.items():
                    if metrics["total"] >= 3:
                        pct = int((metrics["correct"] / metrics["total"]) * 100)
                        s_lines.append(
                            f"  {dom}: {metrics['correct']}/{metrics['total']} correct ({pct}%) | "
                            f"persuaded peers {metrics['persuaded_peers']} times | "
                            f"was persuaded {metrics['was_persuaded']} times"
                        )
                if not s_lines:
                    return "  (Not enough data points collected yet)"
                return "\n".join(s_lines)

            stats_str = format_stats(agent_name)
            owner_stats_str = format_stats(self.owner_name)

            if agent_name == self.owner_name:
                prompt_template = _SELF_REFLECTION_PROMPT
                system_content = (
                    "You are an AI agent producing actionable self-improvement directives "
                    "based on your own past reasoning and performance data. "
                    "Every sentence must be a concrete instruction on how to reason better or evaluate evidence more critically. "
                    "Only state what the evidence directly supports — do not invent patterns."
                )
                prompt = prompt_template.format(
                    agent_name=agent_name,
                    existing_belief=existing_belief,
                    agent_stats=stats_str,
                    evidence=evidence_str,
                    result=result_str,
                )
            else:
                prompt_template = _UPDATE_PROMPT
                system_content = (
                    "You are building an actionable Theory of Mind profile of another AI agent. "
                    "Every sentence must directly instruct the reader on how to critically evaluate "
                    "this agent's arguments and weight their evidence — not merely describe the agent. "
                    "Only state what the evidence directly supports — do not invent patterns."
                )
                prompt = prompt_template.format(
                    agent_name=agent_name,
                    existing_belief=existing_belief,
                    agent_stats=stats_str,
                    owner_stats=owner_stats_str,
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
                "Updating ToM belief for '%s'.",
                agent_name,
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
        Retrieve stored beliefs, clearly separating self-reflection from peer profiles,
        and injecting quantitative stats for domains with >= 3 observations.

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
            "You MUST actively apply the [Self-reflection] and [Peer profiles] above in your private reasoning:\n"
            "- Use your self-reflection to identify where your reasoning is historically unreliable and compensate for it.\n"
            "- Use peer profiles to critically evaluate the quality of their arguments — not to follow their lead, but to know how much scrutiny to apply to their evidence."
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
