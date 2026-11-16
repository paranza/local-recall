"""One search engine object that runs every retrieval mode over a document list.

Modes:
  bm25    lexical only
  dense   embedding similarity only (exact search)
  hybrid  reciprocal rank fusion of the two
  rerank  hybrid, then a cross-encoder over the top candidates

Every mode accepts `candidates`, an array of allowed document indices. This is
how structured filters (mailbox owner, sender, date range) are applied: the
filter runs first and ranking only sees what passed. Filtering first is exact.
Filtering after an approximate top-k search can return too few results.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .bm25 import BM25Index, top_k
from .dense import Encoder
from .fusion import rrf
from .rerank import Reranker

MODES = ("bm25", "dense", "hybrid", "rerank")


class SearchEngine:
    def __init__(
        self,
        texts: Sequence[str],
        *,
        encoder: Encoder,
        embeddings: np.ndarray | None = None,
        reranker: Reranker | None = None,
        depth: int = 100,
        rerank_depth: int = 50,
    ) -> None:
        self.texts = texts
        self.encoder = encoder
        self.reranker = reranker
        self.depth = depth
        self.rerank_depth = rerank_depth
        self.bm25 = BM25Index().build(texts)
        self.emb = embeddings if embeddings is not None else encoder.encode_docs(texts, progress=True)

    def _bm25(self, query: str, cands: np.ndarray | None, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.scores(query)
        if cands is None:
            return top_k(scores, k, drop_zero=True)
        return [(int(cands[i]), s) for i, s in top_k(scores[cands], k, drop_zero=True)]

    def _dense(self, q_emb: np.ndarray, cands: np.ndarray | None, k: int) -> list[tuple[int, float]]:
        if cands is None:
            return top_k(self.emb @ q_emb, k)
        return [(int(cands[i]), s) for i, s in top_k(self.emb[cands] @ q_emb, k)]

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        candidates: np.ndarray | None = None,
        q_emb: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Top-k (doc index, score), best first. Scores are only comparable within one mode."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if candidates is not None and len(candidates) == 0:
            return []
        if mode == "bm25":
            return self._bm25(query, candidates, k)
        if q_emb is None:
            q_emb = self.encoder.encode_queries([query])[0]
        if mode == "dense":
            return self._dense(q_emb, candidates, k)
        fused = rrf(
            [
                [i for i, _ in self._bm25(query, candidates, self.depth)],
                [i for i, _ in self._dense(q_emb, candidates, self.depth)],
            ],
            top=max(k, self.rerank_depth),
        )
        if mode == "hybrid":
            return fused[:k]
        if self.reranker is None:
            raise ValueError("mode 'rerank' needs a reranker")
        ids = [i for i, _ in fused[: self.rerank_depth]]
        return self.reranker.rerank(query, ids, [self.texts[i] for i in ids])[:k]
