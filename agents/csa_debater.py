"""
CSADebater — Communication Strategy Aware debate agent.

Subclass of ThinkThenSpeakDebater that extends Stage 1 (Think) with a
4-step structured reasoning protocol and Stage 2 (Speak) with a
locate-and-use instruction format.

Stage 1 (4 steps):
  Step 1 — SEND AUDIT: which of my last arguments did the peer engage with?
  Step 2 — COMMUNICATION ANALYSIS: why? what should I adapt this round?
  Step 3 — ANSWER REASONING: evaluate peer args; apply ReasoningMemory guardrails.
  Step 4 — ARGUMENT BULLETS: key points to cover, styled per Step 2.

Stage 2:
  Locate [STEP 2] and [STEP 4] sections in private reasoning and build
  a public message that applies the adaptation to the bullet points.

Backward compatible with AsyncDebateArena — implements the same interface
as ThinkThenSpeakDebater (prepare_round, finish_round, build_speak_messages,
build_inconsistency_correction_messages, update_memory).
"""

import logging
import re
from typing import Any, List, Optional

from vllm import LLM, SamplingParams

from agents.tts_debater import ThinkThenSpeakDebater
from configs.configs import AgentResponse, Conversation
from memory.reasoning import ReasoningMemory
from memory.communication import CommunicationStrategyMemory

logger = logging.getLogger(__name__)

# Stage 1 token budget: 5 steps need more headroom than a single reasoning blob.
_CSA_THINK_MAX_TOKENS = 5500


def _extract_step2_adaptation(private_reasoning: str) -> str:
    """
    Deterministically extract the 'Adaptation this round: ...' line from the
    [STEP 2 — COMMUNICATION ANALYSIS] block in the agent's private reasoning.

    Doing this in Python (rather than asking the model to self-locate it)
    guarantees the adaptation is always surfaced, even when the model buries
    it deep in a long reasoning trace.

    Returns the extracted adaptation text, or '' if the pattern is not found.
    """
    match = re.search(
        r"Adaptation this round:\s*(.+?)(?:\n|$)",
        private_reasoning,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _extract_step_n_block(private_reasoning: str, step_n: int) -> str:
    """
    Extract the content of STEP N from private reasoning.
    Matches from [STEP N ...] header to the next [STEP ...] header or end of string.
    Returns the block content (without the header line), or '' if not found.
    """
    pattern = rf"\[STEP {step_n}[^\]]*\](.*?)(?=\[STEP \d|\Z)"
    match = re.search(pattern, private_reasoning, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_private_conclusion(private_reasoning: str) -> str:
    """
    Extract the answer letter from the '→ My private conclusion: {X}' line.
    Returns the single capital letter (e.g., 'A') or '' if not found.
    """
    match = re.search(
        r"→\s*My private conclusion:\s*\{?([A-J])\}?",
        private_reasoning,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


class CSADebater(ThinkThenSpeakDebater):
    """
    Communication Strategy Aware debater.

    Extends ThinkThenSpeakDebater with a 4-step Stage 1 protocol and two
    memory types:
      - reasoning_memory: how to better consume/integrate peer arguments
      - comm_memory: how to package arguments for maximum peer engagement
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        model_name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        reasoning_memory: Optional[ReasoningMemory] = None,
        comm_memory: Optional[CommunicationStrategyMemory] = None,
        sampling_params: Optional[SamplingParams] = None,
        # Accept but ignore the parent `memory` param for arena compat
        memory: Any = None,
    ):
        """
        Args:
            agent_id: Unique agent identifier (e.g., "agent_0").
            name: Human-readable name (e.g., "Agent_Qwen").
            model_name: vLLM model identifier.
            llm: Shared vLLM instance.
            system_prompt: Override for Stage 1 system prompt.
            reasoning_memory: ReasoningMemory for peer-argument consumption directives.
            comm_memory: CommunicationStrategyMemory for per-peer engagement strategy.
            sampling_params: Stage 1 sampling params. max_tokens overridden to
                             _CSA_THINK_MAX_TOKENS if not already >= that value.
            memory: Accepted for arena compatibility but not used.
        """
        # Ensure Stage 1 has enough token budget for 4 steps
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.0, max_tokens=_CSA_THINK_MAX_TOKENS
            )
        elif sampling_params.max_tokens < _CSA_THINK_MAX_TOKENS:
            sampling_params = SamplingParams(
                temperature=sampling_params.temperature,
                max_tokens=_CSA_THINK_MAX_TOKENS,
            )

        super().__init__(
            agent_id=agent_id,
            name=name,
            model_name=model_name,
            llm=llm,
            system_prompt=system_prompt,
            memory=None,  # parent memory not used
            sampling_params=sampling_params,
        )
        self.reasoning_memory = reasoning_memory
        self.comm_memory = comm_memory

    # ------------------------------------------------------------------
    # Stage 1 — 4-step structured reasoning prompt
    # ------------------------------------------------------------------

    def prepare_round(
        self, question_prompt: str, history: Conversation, **kwargs
    ) -> dict:
        """
        Build the Stage 1 prompt.

        Round 0 (history empty): delegates to parent (solo mode).
        Round N > 0: builds the 4-step CSA prompt.

        With PerTurnAsyncDebateArena, history[-1] is always the peer's
        most recent turn and history[-2] is always the owner's most recent
        turn, making the Step 1 audit unambiguous.
        """
        has_spoken = any(
            self.name == response.name
            for turn_entry in history
            for _, response in turn_entry.agent_responses.items()
        )
        is_solo = not has_spoken
        if is_solo:
            return super().prepare_round(question_prompt, history, **kwargs)

        # ---- Classify this turn's role in the 4-phase debate flow ----
        own_prior_responses = [
            r
            for turn_entry in history
            for _, r in turn_entry.agent_responses.items()
            if r.name == self.name
        ]
        peer_has_public = any(
            r.public_message
            for turn_entry in history
            for _, r in turn_entry.agent_responses.items()
            if r.name != self.name
        )
        own_has_public = any(r.public_message for r in own_prior_responses)

        # Turn 2 (speak-only): has prior solo turn, no public messages from self or peer.
        # Delegate to parent TTS so Turn 0 reasoning is injected as context anchor.
        is_speak_only = bool(own_prior_responses and not own_has_public and not peer_has_public)

        # Turn 3 (first-public-with-peer): has prior solo turn, hasn't spoken publicly yet,
        # but peer HAS posted a public message (Llama's Turn 2 statement is visible).
        is_first_public_with_peer = bool(own_prior_responses and not own_has_public and peer_has_public)

        if is_speak_only:
            # Reorder history: put own solo turn last so TTS's own_past_reasoning lookup
            # finds it at history[-1] and injects the full Turn 0 reasoning as context.
            own_turns = [
                t for t in history
                if any(r.name == self.name for r in t.agent_responses.values())
            ]
            other_turns = [
                t for t in history
                if not any(r.name == self.name for r in t.agent_responses.values())
            ]
            state = super().prepare_round(question_prompt, other_turns + own_turns, **kwargs)
            # Patch peer names: TTS won't find them in reordered history[-1] (= own solo turn)
            state["peer_names"] = list({
                r.name
                for turn_entry in history
                for _, r in turn_entry.agent_responses.items()
                if r.name != self.name
            })
            state["recent_peer_messages"] = ""  # no peer public messages yet
            state["is_speak_only"] = True        # informational flag for logging
            return state

        # In per-turn mode, history[-1] is always the peer's last turn
        # (guaranteed to exist because we are not solo).
        peer_names: List[str] = []
        for _, response in history[-1].agent_responses.items():
            if response.name != self.name:
                peer_names.append(response.name)
        # Fallback: scan earlier turns if peer hasn't spoken yet
        if not peer_names:
            for turn_entry in reversed(history):
                for _, response in turn_entry.agent_responses.items():
                    if response.name != self.name and response.name not in peer_names:
                        peer_names.append(response.name)
                if peer_names:
                    break

        peer_name = peer_names[0] if len(peer_names) == 1 else (
            ", ".join(peer_names) if peer_names else "your peer"
        )

        # ---- Recent peer message (for Stage 2) ----
        # history[-1] is the peer's last turn in per-turn mode.
        recent_peer_messages: List[str] = []
        for _, response in history[-1].agent_responses.items():
            if response.name != self.name:
                msg = response.public_message or f"Answer: {response.answer}"
                recent_peer_messages.append(f"Peer ({response.name}): {msg}")

        # ---- Public history (sequential turn log) ----
        public_history = self.build_public_history(history)

        # ---- Owner's last public message (needed for Step 1 audit) ----
        # Scan backward for most recent turn where I spoke.
        own_last_public = ""
        for turn_entry in reversed(history):
            for _, response in turn_entry.agent_responses.items():
                if response.name == self.name and response.public_message:
                    own_last_public = response.public_message
            if own_last_public:
                break

        # ---- Memory retrieval ----
        reasoning_mem_str = (
            self.reasoning_memory.retrieve_memory()
            if self.reasoning_memory is not None
            else ""
        )
        comm_mem_str = (
            self.comm_memory.retrieve_memory()
            if self.comm_memory is not None
            else ""
        )

        # ---- Build 4-step prompt ----
        parts = [f"You are {self.name}.", "", question_prompt]

        if reasoning_mem_str:
            parts.append(
                "[Reasoning Memory — How you consume peer arguments]\n"
                + reasoning_mem_str
            )

        if comm_mem_str:
            parts.append(
                f"[Communication Strategy for {peer_name}]\n" + comm_mem_str
            )

        parts.append(f"[Debate History]\n{public_history}")

        # Remember: {{X}} in a regular string literal appears as {{X}} to the model,
        # which is the format the answer extractor looks for ({B}, {C}, etc.)
        is_first_public_turn = not own_last_public

        # ---- Conditional memory directive for Step 3 ----
        # Forces the agent to quote the single most-relevant directive before reasoning,
        # turning memory injection from passive context into an active generation anchor.
        reasoning_mem_directive = (
            "Memory check \u2014 required before reasoning:\n"
            "Quote or closely paraphrase the ONE directive from your Reasoning Memory "
            "most relevant to how you should evaluate the peer\u2019s argument this turn. "
            "Write it on the line below, then apply it throughout your reasoning.\n"
            "\u2192 Directive I am applying: <write it here>\n\n"
            if reasoning_mem_str
            else ""
        )

        if is_first_public_with_peer:
            # ---- Turn 3: First public turn while peer has already spoken ----
            # Inject own solo reasoning as a concrete anchor in Step 3 so the agent
            # treats its Turn 1 conclusion as the default and only deviates on evidence.
            solo_response = own_prior_responses[-1]
            solo_answer = solo_response.answer
            solo_reasoning = solo_response.reasoning
            skeleton = (
                "---\n"
                "Your output MUST follow this exact format. "
                "Fill in every section completely before moving to the next.\n\n"
                f"DEBATE CONTEXT: This is a 2-agent debate. You are {self.name}. "
                f"The ONLY other agent is {peer_name}. "
                "Do NOT reference or evaluate arguments from any other agents.\n\n"
                "[STEP 1 \u2014 SEND AUDIT + COMM ANALYSIS: N/A]\n"
                "This is your first public response. You have not yet made any public arguments. "
                "There is no prior exchange of yours to audit.\n"
                + (
                    f"Refer to [Communication Strategy for {peer_name}] injected at the top of this prompt.\n"
                    if comm_mem_str
                    else "No prior communication strategy yet.\n"
                )
                + "Opening strategy: <ONE specific choice about how to package your first public argument "
                f"(e.g., lead with a direct question, use a concrete example, challenge {peer_name}'s answer head-on)>\n\n"
                "[STEP 2 \u2014 PEER CLAIM ANALYSIS]\n"
                f"Read {peer_name}'s opening public message carefully.\n\n"
                f"List each distinct claim {peer_name} made. For each, evaluate:\n"
                "  Claim: <exact claim>\n"
                "  Logical basis: <what reasoning or evidence did they provide?>\n"
                "  My Evaluation: <Do you agree or disagree with their logic? Provide a step-by-step evaluation of their evidence. Do they rely on facts or just confident assertions?>\n"
                "  Impact on my position: <Based on your evaluation, does this effectively challenge your current stance?>\n\n"
                "CRITICAL OBJECTIVITY RULE: If the peer provides a specific factual counter-example "
                "(e.g., a specific person, event, or data point), you MUST evaluate it objectively. "
                "Do NOT dismiss concrete evidence as a 'fallacy' simply because it contradicts your view.\n\n"
                "After listing all claims, write a one-sentence summary:\n"
                "\u2192 Peer argument summary: <overall strength and what, if anything, "
                "genuinely challenges your view>\n\n"
                "[STEP 3 \u2014 ANSWER REASONING]\n"
                + reasoning_mem_directive
                + "Sycophancy guard: Only change your answer if you can articulate a specific, evidence-based reason why your previous reasoning was flawed.\n"
                f"[Your initial solo reasoning]\n{solo_reasoning}\n[End of solo reasoning]\n\n"
                f"Your solo reasoning concluded: {solo_answer}. Treat this as your default anchor.\n"
                f"Now evaluate {peer_name}'s opening argument above carefully. "
                "Only deviate from your solo conclusion if you identify a specific, "
                "evidence-based flaw in your own prior reasoning above \u2014 not because the peer sounds confident.\n"
                "Show your full step-by-step reasoning below. "
                "After completing all reasoning steps, write your conclusion on the final line:\n"
                "\u2192 My private conclusion: {{X}}\n\n"
                "[STEP 4 \u2014 RECEIVE AUDIT]\n"
                f"Review {peer_name}'s opening message again.\n"
                f"List every DIRECT QUESTION or IMPLICIT CHALLENGE {peer_name} posed "
                "that requires you to respond. This includes explicit questions and implicit challenges to your stance.\n"
                "For each, write a direct one-sentence answer grounded in your Step 3 reasoning:\n"
                "  \"<question or challenge>\": My answer: <direct response>\n\n"
                "If no direct questions or challenges exist, write: 'No inbound questions.'\n"
                "IMPORTANT: Every item listed here is MANDATORY content for your public message.\n\n"
                "[STEP 5 \u2014 ARGUMENT BULLETS]\n"
                "List 2\u20134 key points to make in your first public message.\n"
                "Rules:\n"
                "  - Apply the opening strategy from Step 1 to each bullet\n"
                "  - At least one bullet must directly answer each item from your Step 4 Receive Audit\n"
                "  - Format each bullet as: \u2022 <point> (do NOT use {{}} letter format in bullets)\n\n"
                "After your bullets, you MUST write: \u2192 My final answer: {{X}} "
                "where X is the SAME letter you wrote in '\u2192 My private conclusion' above.\n"
            )
        elif is_first_public_turn:
            # Agent hasn't made any public speech yet (Turn 0 and Turn 1 were silent).
            # Both agents reasoned independently. Skip Step 1 (no engagement to audit).
            # Step 2 becomes "choose an opening strategy" rather than "adapt based on history".
            skeleton = (
                "---\n"
                "Your output MUST follow this exact format. "
                "Fill in every section completely before moving to the next.\n\n"
                f"DEBATE CONTEXT: This is a 2-agent debate. You are {self.name}. "
                f"The ONLY other agent is {peer_name}. "
                "Do NOT reference or evaluate arguments from any other agents.\n\n"
                "[STEP 1 — SEND AUDIT + COMM ANALYSIS: N/A]\n"
                "This is your first public response. Both you and your peer reasoned independently "
                "in your initial turns (neither of you had seen the other's arguments). "
                "There is no prior public exchange to audit.\n"
                + (
                    f"Refer to [Communication Strategy for {peer_name}] injected at the top of this prompt.\n"
                    if comm_mem_str
                    else "No prior communication strategy yet.\n"
                )
                + "Opening strategy: <ONE specific choice about how to package your first public argument "
                f"(e.g., lead with a direct question, use a concrete example, challenge {peer_name}'s answer head-on)>\n\n"
                "[STEP 2 \u2014 PEER CLAIM ANALYSIS: N/A]\n"
                f"{peer_name} has not yet made a public argument. Skip to Step 3.\n\n"
                "[STEP 3 — ANSWER REASONING]\n"
                + reasoning_mem_directive
                + "Sycophancy guard: Only change your answer if you can articulate a specific, evidence-based reason why your previous reasoning was flawed.\n"
                f"{peer_name} has not yet made a public argument. "
                "Make your opening case based on your own private reasoning only. "
                "Show your full step-by-step reasoning below. "
                "After completing all reasoning steps, write your conclusion on the final line:\n"
                "\u2192 My private conclusion: {{X}}\n\n"
                "[STEP 4 \u2014 RECEIVE AUDIT: N/A]\n"
                f"{peer_name} has not yet made a public argument. Write: 'No inbound questions.'\n\n"
                "[STEP 5 — ARGUMENT BULLETS]\n"
                "List 2\u20134 key points to make in your first public message. "
                "Apply the opening strategy from Step 1 to each bullet.\n"
                "Format each bullet as: \u2022 <point> (do NOT use {{}} letter format in bullets)\n"
                "After your bullets, you MUST write: \u2192 My final answer: {{X}} "
                "where X is the SAME letter you wrote in '\u2192 My private conclusion' above.\n"
            )
        else:
            # Agent has spoken before — run the full 5-step protocol.

            # Carry forward the previous Step 3 private conclusion as a belief anchor.
            own_last_reasoning = ""
            for r in reversed(own_prior_responses):
                if getattr(r, "reasoning", None):
                    own_last_reasoning = r.reasoning
                    break
            prior_step3_conclusion = (
                _extract_private_conclusion(own_last_reasoning)
                if own_last_reasoning
                else ""
            )

            # ---- Step 1: SEND AUDIT + COMM ANALYSIS ----
            step1 = (
                "---\n"
                "Your output MUST follow this exact format. "
                "Fill in every section completely before moving to the next.\n\n"
                f"DEBATE CONTEXT: This is a 2-agent debate. You are {self.name}. "
                f"The ONLY other agent is {peer_name}. "
                "Do NOT reference or evaluate arguments from any other agents.\n\n"
                "[STEP 1 \u2014 SEND AUDIT + COMM ANALYSIS]\n"
                "Part A \u2014 Send Audit:\n"
                "Review your last public message shown in the Debate History above.\n"
                "Your last public message was:\n"
                f"{own_last_public}\n\n"
                f"List EVERY substantive argument, claim, or challenge you made "
                f"that was directed at {peer_name} or required {peer_name} to respond.\n"
                "This includes:\n"
                "  - Any claim about the correctness or weakness of their position\n"
                "  - Any counter-argument you raised against a point they made\n"
                "  - Any question (explicit or implicit) you posed about their reasoning\n"
                "  - Any evidence or example you introduced that they should address\n\n"
                f"For each argument, state whether {peer_name}:\n"
                "  - ENGAGED: addressed it directly with reasoning\n"
                "  - PARTIAL: briefly acknowledged it without real engagement\n"
                "  - IGNORED: did not respond at all\n"
                "Format: \"<your argument>\": ENGAGED / PARTIAL / IGNORED \u2014 <one sentence why>\n\n"
                "IMPORTANT: Only write 'No engagement-seeking arguments.' if your prior message "
                "contained zero claims or challenges directed at the peer. "
                "If your message challenged the peer's reasoning in any way, you MUST list those.\n\n"
                "Part B \u2014 Comm Analysis:\n"
            )
            if comm_mem_str:
                step1 += (
                    f"Communication strategy check \u2014 required before adapting:\n"
                    f"Quote or closely paraphrase the ONE directive from your "
                    f"Communication Strategy Memory for {peer_name} most relevant "
                    f"to this turn. Write it on the line below.\n"
                    f"\u2192 Strategy I am applying: <write it here>\n\n"
                    f"Based on your Send Audit above and that strategy directive:\n"
                )
            else:
                step1 += f"Based on your Send Audit above:\n"
            step1 += (
                f"IF your Send Audit found IGNORED or PARTIAL arguments:\n"
                f"  Ignored/partial reason: <why did {peer_name} skip those arguments?>\n"
                "  Adaptation this round: <ONE specific change to demand better engagement>\n"
                f"IF your Send Audit found no engagement-seeking arguments OR all were ENGAGED:\n"
                "  What worked well: <what communication pattern succeeded>\n"
                "  Adaptation this round: <ONE specific change to build on what worked>\n\n"
            )

            # ---- Step 2: PEER CLAIM ANALYSIS ----
            step2 = (
                "[STEP 2 \u2014 PEER CLAIM ANALYSIS]\n"
                f"Read {peer_name}'s most recent public message carefully.\n\n"
                f"List each distinct claim {peer_name} made. For each, evaluate:\n"
                "  Claim: <exact claim>\n"
                "  Logical basis: <what reasoning or evidence did they provide?>\n"
                "  My Evaluation: <Do you agree or disagree with their logic? Provide a step-by-step evaluation of their evidence. Do they rely on facts or just confident assertions?>\n"
                "  Impact on my position: <Based on your evaluation, does this effectively challenge your current stance?>\n\n"
                "CRITICAL OBJECTIVITY RULE: If the peer provides a specific factual counter-example "
                "(e.g., a specific person, event, or data point), you MUST evaluate it objectively. "
                "Do NOT dismiss concrete evidence as a 'fallacy' simply because it contradicts your view.\n\n"
                "After listing all claims, write a one-sentence summary:\n"
                "\u2192 Peer argument summary: <overall strength and what, if anything, "
                "genuinely challenges your view>\n\n"
            )

            # ---- Step 3: ANSWER REASONING (with prior-conclusion anchor) ----
            step3_anchor = (
                f"Prior private conclusion (from your last reasoning turn): {prior_step3_conclusion}\n"
                "Treat this as your default. Only deviate if Step 2 revealed a STRONG "
                "claim that exposes a specific flaw in your prior reasoning.\n\n"
                if prior_step3_conclusion
                else ""
            )
            step3 = (
                "[STEP 3 \u2014 ANSWER REASONING]\n"
                + reasoning_mem_directive
                + step3_anchor
                + (
                    "Sycophancy guard: Only change your answer if you can identify a "
                    "SPECIFIC flaw in your prior reasoning \u2014 not because the peer "
                    "sounds confident.\n"
                    f"Using your Step 2 analysis, evaluate {peer_name}'s arguments. "
                    "What do you actually believe and why? "
                    "Show your full step-by-step reasoning below. "
                    "After completing all reasoning steps, write your conclusion on the final line:\n"
                    "\u2192 My private conclusion: {X}\n\n"
                )
            )

            # ---- Step 4: RECEIVE AUDIT ----
            step4 = (
                "[STEP 4 \u2014 RECEIVE AUDIT]\n"
                f"Review {peer_name}'s most recent message again.\n"
                f"List every DIRECT QUESTION or IMPLICIT CHALLENGE {peer_name} posed "
                "that requires you to respond. This includes:\n"
                "  - Explicit questions (ending in '?')\n"
                "  - Implicit challenges (e.g., 'How can your position hold if ...')\n"
                "  - Claims about your prior reasoning that you must accept or refute\n\n"
                "For each, write a direct one-sentence answer grounded in your Step 3 reasoning:\n"
                "  \"<question or challenge>\": My answer: <direct response>\n\n"
                "If no direct questions or challenges exist, write: 'No inbound questions.'\n"
                "IMPORTANT: Every item listed here is MANDATORY content for your public message.\n\n"
            )

            # ---- Step 5: ARGUMENT BULLETS ----
            step5 = (
                "[STEP 5 \u2014 ARGUMENT BULLETS]\n"
                "List 2\u20134 key points to make in your public message.\n"
                "Rules:\n"
                "  - Apply the adaptation from Step 1 Part B to each bullet\n"
                "  - At least one bullet must directly answer each item from your Step 4 Receive Audit\n"
                "  - Format each bullet as: \u2022 <point> (do NOT use {} letter format in bullets)\n\n"
                "After your bullets, you MUST write: \u2192 My final answer: {X} "
                "where X is the SAME letter you wrote in '\u2192 My private conclusion' above.\n"
            )

            skeleton = step1 + step2 + step3 + step4 + step5



        parts.append(skeleton)
        reasoning_prompt = "\n\n".join(parts)

        return {
            "reasoning_prompt": reasoning_prompt,
            "system": self.system_prompt,
            "is_solo": False,
            "next_round": len([
                t for t in history if any(r.name == self.name for r in t.agent_responses.values())
            ]),
            "peer_names": peer_names,
            "recent_peer_messages": "\n\n".join(recent_peer_messages),
        }

    # ------------------------------------------------------------------
    # Stage 2 — locate-and-use speak prompt
    # ------------------------------------------------------------------

    def _build_speak_prompt(self, state: dict, private_reasoning: str) -> str:
        """
        Build Stage 2 prompt that uses Step 2 and Step 4 from Stage 1.

        Round 0 delegates to parent. Round N>0 injects the Step 2 adaptation
        as a hard-anchored prefix so the model cannot ignore it.

        Key design: the Step 2 'Adaptation this round' is extracted in Python
        (deterministic) and placed immediately before the generation instructions,
        so recency bias + explicit labeling forces the model to apply it rather
        than defaulting to recycling the previous message.
        """
        if state.get("is_solo"):
            return super()._build_speak_prompt(state, private_reasoning)

        peer_names = state.get("peer_names", [])
        peer_names = [n for n in peer_names if n != self.name]
        peer_list = ", ".join(peer_names) if peer_names else "your peers"
        recent_msgs = state.get("recent_peer_messages", "")

        peer_context_block = ""
        if recent_msgs:
            peer_context_block = f"Peer messages to address:\n{recent_msgs}\n\n"

        # --- Extract adaptation from Step 1 Part B (deterministic Python extraction) ---
        # Note: In early rounds (is_first_public_turn), "Opening strategy:" acts as the adaptation.
        adaptation = _extract_step2_adaptation(private_reasoning)
        if not adaptation:
            # Fallback for early rounds: try to extract the Opening strategy from Step 1: N/A
            import re
            m = re.search(r"Opening strategy:\s*(.+)", private_reasoning, re.IGNORECASE)
            if m:
                adaptation = m.group(1).strip()

        if adaptation:
            adaptation_block = (
                f"REQUIRED ADAPTATION THIS ROUND: {adaptation}\n"
                "Your public message MUST apply this adaptation. "
                "Your FIRST sentence must directly enact it "
                "(e.g., a targeted question naming a specific un-addressed point, "
                "a direct challenge, or a concrete example as the opening move). "
                "Do NOT save the adaptation for the end of your message.\n\n"
            )
            adaptation_instruction = (
                "1. REQUIRED ADAPTATION THIS ROUND is shown above — "
                "your first sentence must directly enact it.\n"
            )
        else:
            adaptation_block = ""
            adaptation_instruction = (
                "1. Find 'Adaptation this round' in your Step 1 Part B "
                "and apply it as your opening move.\n"
            )

        # --- Inject only Steps 3, 4, 5 — the synthesis steps that drive the public message ---
        step3_block = _extract_step_n_block(private_reasoning, 3)
        step4_block = _extract_step_n_block(private_reasoning, 4)
        step5_block = _extract_step_n_block(private_reasoning, 5)

        reasoning_parts = []
        if step3_block:
            reasoning_parts.append(
                f"[Your Step 3 \u2014 Answer Reasoning]\n{step3_block}"
            )
        if step4_block:
            reasoning_parts.append(
                f"[Your Step 4 \u2014 Mandatory Answers (Receive Audit)]\n{step4_block}"
            )
        if step5_block:
            reasoning_parts.append(
                f"[Your Step 5 \u2014 Argument Bullets]\n{step5_block}"
            )
        # Fallback to full reasoning if all extractions fail
        reasoning_summary = (
            "\n\n".join(reasoning_parts) if reasoning_parts else private_reasoning
        )

        if not recent_msgs:
            instruction_steps = (
                "2. Base your response purely on your Step 5 Argument Bullets.\n"
                "3. Do NOT address any peer arguments as none exist yet.\n"
                "4. Write ONLY your debate response. Do NOT reproduce your step analysis.\n"
                "5. Do NOT address any peers by name since you are opening the debate.\n"
                "6. Do NOT exceed 300 words.\n"
                "7. End with your final answer as {{X}} where X is the capital letter "
                "(A\u2013J). Your answer must match your Step 3 private conclusion.\n"
            )
        else:
            instruction_steps = (
                "2. Your public message MUST directly answer every item listed in "
                "[Your Step 4 \u2014 Mandatory Answers] above. "
                "Address each question or challenge explicitly before making your own arguments.\n"
                "3. Build your remaining arguments from "
                "[Your Step 5 \u2014 Argument Bullets].\n"
                "4. Write ONLY your debate response. "
                "Do NOT reproduce your step analysis.\n"
                f"5. This is a 2-agent debate between you and {peer_list}. "
                f"Address {peer_list} by name. "
                "Do NOT mention, address, or reference any agent not listed here.\n"
                "6. Do NOT exceed 300 words.\n"
                "7. End with your final answer as {{X}} where X is the capital letter "
                "(A\u2013J). Your answer must match your Step 3 private conclusion.\n"
            )

        instructions = (
            f"You are {self.name}. Write ONLY your own debate response below.\n\n"
            "Instructions:\n"
            + adaptation_instruction
            + instruction_steps
        )

        return (
            f"{peer_context_block}"
            f"Your private reasoning (key sections only):\n{reasoning_summary}\n\n"
            f"{adaptation_block}"
            f"{instructions}"
        )

    # ------------------------------------------------------------------
    # Memory update
    # ------------------------------------------------------------------

    def update_memory(
        self, history: Conversation, result: Any = None, **kwargs
    ) -> None:
        """Update both memory types after a debate concludes."""
        if self.reasoning_memory is not None:
            self.reasoning_memory.update_memory(history, result=result, **kwargs)
        if self.comm_memory is not None:
            self.comm_memory.update_memory(history, result=result, **kwargs)

    # ------------------------------------------------------------------
    # Convenience repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CSADebater(name='{self.name}', model='{self.model_name}', "
            f"reasoning_memory={self.reasoning_memory is not None}, "
            f"comm_memory={self.comm_memory is not None})"
        )
