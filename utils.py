import re

# Matches fake conversation turn headers that a model may hallucinate inside
# its own response, e.g.:
#   [Round 2] You (Agent_Llama):
#   [Round 1] Peer (Agent_Qwen):
_HALLUCINATED_TURN_RE = re.compile(
    r"^\s*\[Round\s+\d+\]\s+(?:You|Peer)\s*\([^)]+\)\s*:",
    re.MULTILINE,
)

# Matches the {{X}} final-answer format emitted by debater agents.
_ANSWER_RE = re.compile(r"\{\{([A-Z])\}\}")


def strip_hallucinated_turns(text: str) -> str:
    """Truncate a model response at the first hallucinated conversation turn.

    Some smaller models (e.g. Llama-3.1-8B-Instruct) will, after producing
    their genuine reply, continue generating a fake multi-turn transcript that
    mimics the ``[Round N] You/Peer (Name):`` history format used in the
    debate prompt. Everything from the first such fake header onward is noise
    and should be discarded before the response is stored in history.

    **Self-labeling vs. hallucination:**
    Llama sometimes begins its own reply with a ``[Round N] You (Name):``
    header as a harmless self-label. That match occurs at position 0 and must
    NOT be stripped; only matches that appear AFTER real content (i.e. the
    model has already written something meaningful) are genuine hallucinations.

    **Answer rescue:**
    If stripping a hallucinated turn accidentally removes the ``{{X}}`` final
    answer (e.g. the model wrote its answer inside the hallucinated block), the
    last ``{{X}}`` found anywhere in the original text is appended back so that
    answer extraction never fails due to truncation.

    Examples of patterns that are removed::

        ...genuine reply... {{G}}
        [Round 3] Peer (Agent_Qwen): ...   ← stripped (appears after content)
        [Round 3] You (Agent_Llama): ...   ← stripped (appears after content)

    Examples that are kept::

        [Round 2] You (Agent_Llama): genuine reply... {{G}}
        ↑ at position 0 — benign self-label, NOT stripped

    Args:
        text: Raw (or think-stripped) output from the model.

    Returns:
        The text up to (but not including) the first hallucinated turn header
        that appears after real content, stripped of trailing whitespace, with
        the final answer rescued if it was inadvertently removed.
        If no hallucinated header is found the original text is returned
        unchanged.
    """
    if not text:
        return text

    original = text

    # How many characters of leading whitespace precede the real content?
    content_start = len(text) - len(text.lstrip())

    for match in _HALLUCINATED_TURN_RE.finditer(text):
        # Skip matches at the very start of the text — that's a self-label.
        if match.start() <= content_start:
            continue
        # First match that appears *after* real content → truncate here.
        text = text[: match.start()].rstrip()
        break

    # Safety net: if truncation removed the {{X}} answer, rescue the last
    # occurrence from the original full text and append it.
    if not _ANSWER_RE.search(text):
        last_answer = None
        for ans_match in _ANSWER_RE.finditer(original):
            last_answer = ans_match.group(0)   # e.g. "{{G}}"
        if last_answer:
            text = text + f"\n\n{last_answer}"

    return text



def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output.

    DeepSeek R1-distill models emit chain-of-thought inside these tags.
    The reasoning is useful for generation quality but should not be
    stored in beliefs or injected into future prompts.

    Handles two cases:
      1. Explicit:  <think>...</think>  (full tags)
      2. Implicit:  reasoning...</think> (opening tag omitted by model)
    """
    if not text:
        return text
    # Case 1: explicit <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Case 2: implicit — model starts reasoning without <think>,
    # so everything before the first </think> is the think block
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text

def is_deepseek_model(llm) -> bool:
    """Check if the underlying LLM is a DeepSeek model."""
    try:
        if hasattr(llm, "llm_engine"):
            model_str = str(llm.llm_engine.model_config.model).lower()
            return "deepseek" in model_str
    except Exception:
        pass
    return False
