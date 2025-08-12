import re

def _enforce_succinctness(text: str, max_sent=3, max_words=20):
    # Split on sentence enders while keeping order
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    clean = []
    for s in parts:
        if not s:
            continue
        words = s.split()
        if len(words) > max_words:
            words = words[:max_words]
            s = " ".join(words).rstrip(",;:") + "."
        clean.append(s)
        if len(clean) == max_sent:
            break
    # If nothing looked like sentences, fall back to word chunking
    if not clean:
        words = text.split()[:max_words]
        clean = [" ".join(words) + "."]
    return " ".join(clean)