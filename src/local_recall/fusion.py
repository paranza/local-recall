"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009).

fused(d) = sum over rankings r that contain d of 1 / (k + rank_r(d))

Only ranks are used, never raw scores, so BM25 scores and cosine similarities
can be combined without putting them on a common scale.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def rrf(rankings: Sequence[Sequence[T]], k: int = 60, top: int | None = None) -> list[tuple[T, float]]:
    """Fuse ranked lists (best first). Ties are broken by first appearance."""
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])  # stable sort keeps insertion order on ties
    return fused[:top] if top is not None else fused
