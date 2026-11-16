"""BM25 over an inverted index, written from scratch.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

Two idf variants:
  "lucene": log(1 + (N - n + 0.5) / (n + 0.5)), always positive (default)
  "okapi":  log((N - n + 0.5) / (n + 0.5)), the original, can go negative
The "okapi" variant exists so the implementation can be checked term for term
against the rank_bm25 reference library in the tests.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from .tokenizer import tokenize


class BM25Index:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        idf: str = "lucene",
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if idf not in ("lucene", "okapi"):
            raise ValueError(f"unknown idf variant: {idf}")
        self.k1 = k1
        self.b = b
        self.idf_variant = idf
        self.tokenizer = tokenizer
        self.n_docs = 0
        self.avgdl = 0.0
        self.doc_len = np.zeros(0, dtype=np.float32)
        # term -> (doc indices, term frequencies), both sorted by doc index
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}

    def build(self, docs: Sequence[str]) -> "BM25Index":
        """Index a list of documents. Document i gets internal id i."""
        self.n_docs = len(docs)
        doc_len = np.zeros(self.n_docs, dtype=np.float32)
        tmp: dict[str, tuple[list[int], list[int]]] = {}
        for i, text in enumerate(docs):
            counts = Counter(self.tokenizer(text))
            doc_len[i] = sum(counts.values())
            for term, tf in counts.items():
                ids, tfs = tmp.setdefault(term, ([], []))
                ids.append(i)
                tfs.append(tf)
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0
        self.postings = {
            term: (np.asarray(ids, dtype=np.int32), np.asarray(tfs, dtype=np.float32))
            for term, (ids, tfs) in tmp.items()
        }
        self.idf = {term: self._idf(len(ids)) for term, (ids, _) in self.postings.items()}
        return self

    def _idf(self, df: int) -> float:
        ratio = (self.n_docs - df + 0.5) / (df + 0.5)
        return math.log(1.0 + ratio) if self.idf_variant == "lucene" else math.log(ratio)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for one query."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0:
            return out
        # A term repeated in the query counts once per occurrence, as in Okapi.
        for term, qtf in Counter(self.tokenizer(query)).items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            ids, tf = posting
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[ids] / self.avgdl)
            out[ids] += qtf * self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm)
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (doc index, score), best first. Documents scoring 0 are dropped."""
        return top_k(self.scores(query), k, drop_zero=True)


def top_k(scores: np.ndarray, k: int, *, drop_zero: bool = False) -> list[tuple[int, float]]:
    """Indices of the k largest scores, best first, ties broken by lower index."""
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    order = part[np.lexsort((part, -scores[part]))]
    hits = [(int(i), float(scores[i])) for i in order]
    if drop_zero:
        hits = [h for h in hits if h[1] > 0.0]
    return hits
