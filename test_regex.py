import re
import string

def extract_answer(raw_response: str) -> str:
    # We only want to strip common sentence-ending punctuation, NOT brackets/braces
    # so we don't accidentally strip the \} from \{\{A\}\}
    strip_chars = " \t\n\r" + ".,!?;:'\""
    clean_end = raw_response.rstrip(strip_chars)
    
    match_end = re.search(r"\{\{([A-Z])\}\}$", clean_end)
    if match_end:
        return f"Found {{X}} at end: {match_end.group(1)}"
        
    match_end = re.search(r"(?:^|[^a-zA-Z0-9])([A-Z])$", clean_end)
    if match_end:
        return f"Found standalone X at end: {match_end.group(1)}"
        
    return "?"

print(extract_answer("So the answer is {{A}}."))
print(extract_answer("My final answer is B."))
print(extract_answer("Therefore, B!"))
print(extract_answer("Wait, {{C}} was wrong. Finally, it's A."))
print(extract_answer("The DNA."))
