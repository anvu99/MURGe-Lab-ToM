"""
AsyncDebateArena — orchestrator for sequential multi-agent debates.

Inherits from DebateArena but runs agents asynchronously (sequentially) 
instead of batching them. This allows an agent to read the public messages 
of peers who have already spoken in the current round, eliminating 
conversation lag.
"""

import logging
from collections import defaultdict
from typing import Any, Dict

from architectures.debate import DebateArena
from configs.configs import Conversation, RoundEntry
from utils import strip_hallucinated_turns, strip_think_blocks

logger = logging.getLogger(__name__)


class AsyncDebateArena(DebateArena):
    """
    Executes debate rounds sequentially.
    
    Instead of generating all agents' round N responses in a single block 
    (where agents are oblivious to peers' round N messages), agents take turns.
    Agent B's prompt will include Agent A's round N message.
    """

    def _run_round(
        self,
        question_prompt: str,
        conversation: Conversation,
    ) -> RoundEntry:
        """
        Execute one debate round sequentially across all agents.
        """
        round_entry = RoundEntry()
        round_entry.agent_responses = {}
        turn_order: list = []  # tracks intra-round speaking sequence for ToM memory

        # Execute agents in the exact order they were provided in agent_names
        ordered_agents = self.agents

        for agent in ordered_agents:
            agent_id = agent.agent_id
            logger.info("  [TURN] %s (%s) is taking their turn...", agent.name, agent_id)

            # ---- Phase 1: Prepare State ----
            # Pass the currently recorded responses so the agent sees peers who have spoken.
            state = agent.prepare_round(
                question_prompt, 
                conversation,
                current_round_responses=round_entry.agent_responses
            )

            # ---- Phase 2: Reasoning Generation ----
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
                    sampling_params=agent.sampling_params
                )
                reasoning = out[0].outputs[0].text.strip() if out and out[0].outputs else ""
            except Exception as e:
                logger.error("Reasoning call failed for %s: %s", agent_id, e)
                reasoning = ""

            clean_reasoning = strip_think_blocks(reasoning)
            clean_reasoning = strip_hallucinated_turns(clean_reasoning)
            
            # This creates the preliminary AgentResponse (without public_message)
            old_response = agent.finish_round(state, clean_reasoning)

            logger.info(
                "  %s (%s):\n--- REASONING & ANSWER ---\n%s\n--- EXTRACTED ANSWER ---: %s",
                agent.name,
                agent_id,
                reasoning,
                old_response.answer,
            )

            # Assign to round_entry provisionally just in case
            round_entry.agent_responses[agent_id] = old_response
            turn_order.append(agent_id)

            # ---- Phase 3: Stage-2 Speak (ThinkThenSpeakDebater only) ----
            if hasattr(agent, "build_speak_messages"):
                speak_msgs = agent.build_speak_messages(state)
                speak_params = getattr(agent, "_speak_params", agent.sampling_params)
                
                try:
                    speak_out = agent.llm.chat(
                        messages=[speak_msgs],
                        sampling_params=speak_params,
                    )
                    public_msg = speak_out[0].outputs[0].text.strip() if speak_out and speak_out[0].outputs else ""
                    
                    new_response = agent.attach_public_message(old_response, public_msg)
                    round_entry.agent_responses[agent_id] = new_response
                    
                    logger.info(
                        "  %s (%s):\n--- PUBLIC MESSAGE ---\n%s",
                        agent.name,
                        agent_id,
                        public_msg,
                    )

                    # --- Consistency check ---
                    private_answer = old_response.answer
                    public_answer = new_response.answer
                    public_extracted = getattr(new_response, "_public_extracted", True)
                    
                    if public_extracted and private_answer != public_answer and private_answer != "?":
                        logger.info(
                            "  INCONSISTENCY: %s (%s) thought %s privately but said %s publicly — correcting.",
                            agent.name, agent_id, private_answer, public_answer,
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
                                corrected_response = agent.attach_public_message(old_response, corrected_pub)
                                round_entry.agent_responses[agent_id] = corrected_response
                                logger.info(
                                    "  %s (%s) corrected public message: %s → %s",
                                    agent.name, agent_id, public_answer, corrected_response.answer,
                                )
                                # Record the inconsistency flag
                                if not hasattr(round_entry, "observer_flags"):
                                    round_entry.observer_flags = []
                                round_entry.observer_flags.append({
                                    "agent_id": agent_id,
                                    "flag_type": "inconsistency",
                                    "original_answer": public_answer,
                                    "corrected_answer": corrected_response.answer,
                                    "notice": (
                                        f"Agent privately concluded {private_answer} "
                                        f"but publicly stated {public_answer}."
                                    ),
                                })
                        except Exception as e:
                            logger.error("Inconsistency correction failed for %s: %s", agent_id, e)

                except Exception as e:
                    logger.error("Stage 2 speak call failed for %s: %s", agent_id, e)

        # Record the intra-round speaking order for ToM memory annotation
        round_entry.turn_order = turn_order

        # ---- Phase 4: Observer Analysis & Correction ----
        if self.observer is not None and conversation:
            notices = self.observer.analyze_round(conversation, round_entry.agent_responses)
            if notices:
                if not hasattr(round_entry, "observer_flags"):
                    round_entry.observer_flags = []
                    
                for agent_id, notice in notices.items():
                    # Parse flag type
                    flag_type = "unknown"
                    if notice.startswith("[sycophancy]"):
                        flag_type = "sycophancy"
                        notice = notice[len("[sycophancy]"):].strip()
                    elif notice.startswith("[repetition]"):
                        flag_type = "repetition"
                        notice = notice[len("[repetition]"):].strip()

                    agent = next(a for a in self.agents if a.agent_id == agent_id)
                    original_resp = round_entry.agent_responses[agent_id]
                    
                    logger.info("  %s (%s) FLAGGED for %s", agent.name, agent_id, flag_type.upper())
                    
                    flag_record = {
                        "agent_id": agent_id,
                        "flag_type": flag_type,
                        "original_answer": original_resp.answer,
                        "notice": notice,
                        "corrected_answer": original_resp.answer, # fallback
                    }
                    round_entry.observer_flags.append(flag_record)

                    # We must run the agent sequentially again to perform correction
                    # We need the agent's state, but state is not centrally stored in AsyncDebateArena
                    # So we reconstruct the state using prepare_round again, ensuring it's accurate
                    state = agent.prepare_round(
                        question_prompt, 
                        conversation,
                        current_round_responses=round_entry.agent_responses
                    )

                    if hasattr(agent, "prepare_corrected_round"):
                        corrected_state = agent.prepare_corrected_round(state, notice)
                        
                        is_gemma = "gemma" in agent.model_name.lower()
                        if is_gemma:
                            msg = [{"role": "user", "content": f"{corrected_state['system']}\n\n{corrected_state['reasoning_prompt']}"}]
                        else:
                            msg = [
                                {"role": "system", "content": corrected_state["system"]},
                                {"role": "user", "content": corrected_state["reasoning_prompt"]}
                            ]
                        
                        try:
                            out = agent.llm.chat(messages=[msg], sampling_params=agent.sampling_params)
                            if out and out[0].outputs:
                                corrected_reasoning = out[0].outputs[0].text.strip()
                                corrected_reasoning = strip_think_blocks(corrected_reasoning)
                                corrected_reasoning = strip_hallucinated_turns(corrected_reasoning)
                                
                                new_resp = agent.finish_round(corrected_state, corrected_reasoning)
                                
                                if hasattr(agent, "build_speak_messages"):
                                    speak_msg = agent.build_speak_messages(corrected_state)
                                    speak_params = getattr(agent, "_speak_params", agent.sampling_params)
                                    speak_out = agent.llm.chat(messages=[speak_msg], sampling_params=speak_params)
                                    if speak_out and speak_out[0].outputs:
                                        public_msg = speak_out[0].outputs[0].text.strip()
                                        new_resp = agent.attach_public_message(new_resp, public_msg)

                                logger.info("  --- CORRECTED ---\n%s\n--- END CORRECTED ---", new_resp.reasoning)
                                round_entry.agent_responses[agent_id] = new_resp
                                flag_record["corrected_answer"] = new_resp.answer
                        except Exception as e:
                            logger.error("Error during agent correction for %s: %s", agent_id, e)

        return round_entry
