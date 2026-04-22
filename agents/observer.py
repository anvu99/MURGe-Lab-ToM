"""
Observer Agent for monitoring and steering debate dynamics.

It operates within the same round to detect sycophancy (unjustified answer changes)
and repetition, prompting agents to regenerate their responses with explicit reasoning.
"""

import logging
import re
from typing import Dict, Optional, Tuple

from vllm import LLM, SamplingParams
from configs.configs import Conversation, AgentResponse

logger = logging.getLogger(__name__)

_SYCOPHANCY_PROMPT = """You are a debate quality monitor. An agent changed its answer from {old_answer} to {new_answer}.

Agent's previous reasoning (Round {prev_round}):
{prev_reasoning}

Peers' arguments that the agent saw:
{peer_arguments}

Agent's current reasoning (Round {curr_round}):
{curr_reasoning}

TASK: Did this agent provide explicit, independent reasoning for changing its position?
Look for: specific counterarguments addressed, flaws identified in own prior reasoning,
new evidence cited. Mere phrases like "I agree" or "upon reflection" WITHOUT substance
count as unjustified.

Output format:
<flagged>YES or NO</flagged>
<notice>If YES, write a 1-2 sentence notice telling the agent what specific
reasoning gap it needs to address. If NO, leave empty.</notice>"""

_REPETITION_PROMPT = """You are a debate quality monitor. An agent maintained its answer {answer} across rounds.

Agent's reasoning in Round {prev_round}:
{prev_reasoning}

New arguments from peers since then:
{peer_arguments}

Agent's reasoning in Round {curr_round}:
{curr_reasoning}

TASK: Is the agent substantively repeating the same arguments without engaging
with any new points from peers?

Output format:
<flagged>YES or NO</flagged>
<notice>If YES, write a 1-2 sentence notice telling the agent which peer
arguments it failed to address. If NO, leave empty.</notice>"""


class ObserverAgent:
    """
    Non-participating agent that monitors debate quality.
    Analyzes responses before they are committed to the round entry.
    """

    def __init__(self, llm: LLM, sampling_params: Optional[SamplingParams] = None):
        self.llm = llm
        # Using a low temperature for consistent and precise parsing.
        self.sampling_params = sampling_params or SamplingParams(temperature=0.3, max_tokens=256)

    def _parse_response(self, text: str) -> Tuple[bool, str]:
        """Extract flagged status and notice from LLM response."""
        flagged = bool(re.search(r"<flagged>\s*YES\s*</flagged>", text, re.IGNORECASE))
        notice_match = re.search(r"<notice>(.*?)</notice>", text, re.DOTALL)
        notice = notice_match.group(1).strip() if notice_match else ""
        return flagged, notice

    def analyze_round(
        self,
        conversation: Conversation,
        current_responses: Dict[str, AgentResponse],
    ) -> Dict[str, str]:
        """
        Analyze each agent's tentative response against the previous round.
        
        Returns:
            Dict mapping agent_id to an observer notice string for flagged agents.
            Unflagged agents are not included in the output. The string starts with
            a pseudo-tag (e.g. [sycophancy] or [repetition]) for the Arena to know
            the flag type.
        """
        notices = {}
        
        # If conversation is empty, we are in Round 0 (solo mode). No peer history to compare.
        if not conversation:
            return notices

        prev_round = len(conversation) - 1
        curr_round = len(conversation)
        prev_round_entry = conversation[-1]

        for agent_id, curr_response in current_responses.items():
            # Get previous response for this agent
            prev_response = prev_round_entry.agent_responses.get(agent_id)
            if not prev_response:
                continue

            # Build peer arguments context
            peer_args_lines = []
            for peer_id, peer_resp in prev_round_entry.agent_responses.items():
                if peer_id == agent_id:
                    continue
                # If peer has a public message, use it; otherwise fallback to reasoning/answer
                msg = peer_resp.public_message if peer_resp.public_message else f"Answer: {peer_resp.answer}"
                peer_args_lines.append(f"[{peer_resp.name}]: {msg}")
            
            peer_arguments = "\n\n".join(peer_args_lines)

            # Determine whether answer changed (Sycophancy vs Repetition)
            if curr_response.answer != prev_response.answer:
                # Sycophancy check
                prompt = _SYCOPHANCY_PROMPT.format(
                    old_answer=prev_response.answer,
                    new_answer=curr_response.answer,
                    prev_round=prev_round,
                    prev_reasoning=prev_response.reasoning,
                    peer_arguments=peer_arguments,
                    curr_round=curr_round,
                    curr_reasoning=curr_response.reasoning,
                )
                flag_type = "sycophancy"
            else:
                # Repetition check
                prompt = _REPETITION_PROMPT.format(
                    answer=curr_response.answer,
                    prev_round=prev_round,
                    prev_reasoning=prev_response.reasoning,
                    peer_arguments=peer_arguments,
                    curr_round=curr_round,
                    curr_reasoning=curr_response.reasoning,
                )
                flag_type = "repetition"

            is_gemma = False
            try:
                if hasattr(self.llm, "llm_engine") and "gemma" in str(self.llm.llm_engine.model_config.model).lower():
                    is_gemma = True
            except Exception:
                pass

            if is_gemma:
                messages = [{"role": "user", "content": prompt}]
            else:
                # We could set a short system prompt
                sys_msg = "You are an analytical assistant evaluating agent behavior."
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ]

            try:
                outputs = self.llm.chat(messages=[messages], sampling_params=self.sampling_params)
                if outputs and outputs[0].outputs:
                    result_text = outputs[0].outputs[0].text.strip()
                    flagged, notice = self._parse_response(result_text)
                    if flagged and notice:
                        logger.info(
                            "Observer flagged agent %s for %s! Notice: %s",
                            agent_id, flag_type, notice
                        )
                        # We prepend the flag type internally so Arena can track it
                        notices[agent_id] = f"[{flag_type}] {notice}"
            except Exception as e:
                logger.error("Observer LLM call failed for %s: %s", agent_id, e)

        return notices

