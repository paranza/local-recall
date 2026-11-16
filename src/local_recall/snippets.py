"""Query-focused snippets: show the part of a document that matches the query.

A preview made from the first lines of an email is often a greeting. This
picks the window of words that covers the most distinct query terms (ties go
to the window with more matches in total, then to the earlier one).
"""

from __future__ import annotations

from .tokenizer import tokenize


def best_snippet(text: str, query: str, width: int = 45) -> str:
    """The `width`-word window of `text` that best covers the query terms."""
    words = text.split()
    if len(words) <= width:
        return " ".join(words)
    terms = set(tokenize(query))
    # One term set per word, using the same normalization as the index.
    keys = [set(tokenize(w)) & terms for w in words]
    if not any(keys):
        return " ".join(words[:width]) + " ..."
    best_start, best_score = 0, (-1, -1)
    for start in range(0, len(words) - width + 1):
        window = keys[start : start + width]
        distinct = len(set().union(*window))
        total = sum(1 for k in window if k)
        if (distinct, total) > best_score:
            best_start, best_score = start, (distinct, total)
    # Center the window on its matches, so context shows on both sides.
    hits = [i for i in range(best_start, best_start + width) if keys[i]]
    center = (hits[0] + hits[-1]) // 2
    start = max(0, min(center - width // 2, len(words) - width))
    snippet = " ".join(words[start : start + width])
    return ("... " if start > 0 else "") + snippet + (" ..." if start + width < len(words) else "")
