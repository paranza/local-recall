"""Cross-encoder re-ranking.

A bi-encoder embeds the query and the document separately, which is fast but
never lets the two texts attend to each other. A cross-encoder reads the pair
together and scores it directly. That is more accurate and far slower, so it
only runs on a short candidate list from the first-stage retriever.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, *, device: str = "cpu", max_length: int = 512) -> None:
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def scores(self, query: str, docs: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        pairs = [(query, d) for d in docs]
        return np.asarray(self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=np.float32)

    def rerank(self, query: str, candidates: Sequence[int], texts: Sequence[str]) -> list[tuple[int, float]]:
        """Re-order candidate ids by cross-encoder score. `texts[i]` belongs to `candidates[i]`."""
        s = self.scores(query, texts)
        order = np.argsort(-s, kind="stable")
        return [(candidates[i], float(s[i])) for i in order]
