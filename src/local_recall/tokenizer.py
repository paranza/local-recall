"""Tokenizer for the lexical index.

Steps: Unicode-normalize, strip accents, lowercase, split on anything that is
not a letter or digit, drop stopwords, then apply an optional light stemmer.

The stemmer is Harman's "S-stemmer" (1991). It only folds plural endings, so
it is easy to reason about and cannot merge unrelated words the way an
aggressive stemmer can. Its effect is measured in the ablation, not assumed.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Short, standard English stopword list (the Lucene default set).
STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with""".split()
)


def s_stem(token: str) -> str:
    """Harman's S-stemmer: fold plural endings only.

    Rules, first match wins:
      1. "...ies" -> "...y"   unless the word ends in "eies" or "aies"
      2. "...es"  -> "...e"   unless it ends in "aes", "ees" or "oes"
      3. "...s"   -> "..."    unless it ends in "us" or "ss"
    Tokens of three characters or fewer are left alone.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


def normalize(text: str) -> str:
    """Lowercase and strip accents ("Café" -> "cafe")."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def tokenize(text: str, *, stopwords: bool = True, stem: bool = True) -> list[str]:
    """Turn raw text into index terms."""
    tokens = _TOKEN_RE.findall(normalize(text))
    if stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [s_stem(t) for t in tokens]
    return tokens
