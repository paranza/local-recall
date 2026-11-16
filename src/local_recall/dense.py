"""Dense retrieval: a sentence-embedding encoder plus exact and approximate indexes.

Embeddings are L2-normalized, so the inner product is the cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge v1.5 models are trained with this instruction on the query side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        backend: str = "torch",
        onnx_file: str | None = None,
        query_prefix: str | None = None,
        max_seq_length: int = 512,
    ) -> None:
        kwargs: dict = {}
        if backend == "onnx":
            kwargs["model_kwargs"] = {"file_name": onnx_file or "onnx/model.onnx"}
        self.model = SentenceTransformer(model_name, device=device, backend=backend, **kwargs)
        self.model.max_seq_length = max_seq_length
        if query_prefix is None:
            query_prefix = BGE_QUERY_PREFIX if "bge-" in model_name and "en" in model_name else ""
        self.query_prefix = query_prefix
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: Sequence[str], batch_size: int, progress: bool) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode_docs(self, texts: Sequence[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        return self._encode(texts, batch_size, progress)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts], batch_size, progress)


class ExactIndex:
    """Brute-force search: one matrix product against every document."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.emb = embeddings

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores), each of shape (n_queries, k), best first."""
        queries = np.atleast_2d(queries)
        sims = queries @ self.emb.T
        k = min(k, self.emb.shape[0])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        return np.take_along_axis(part, order, axis=1), np.take_along_axis(part_scores, order, axis=1)


class HNSWIndex:
    """Approximate search with a Hierarchical Navigable Small World graph (FAISS).

    m:               links per node. More links: better recall, more memory.
    ef_construction: candidate list size while building. Higher: better graph, slower build.
    ef_search:       candidate list size while searching. The recall/latency dial.
    """

    def __init__(self, embeddings: np.ndarray, m: int = 32, ef_construction: int = 200) -> None:
        self.index = faiss.IndexHNSWFlat(embeddings.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(embeddings)

    def search(self, queries: np.ndarray, k: int, ef_search: int = 64) -> tuple[np.ndarray, np.ndarray]:
        self.index.hnsw.efSearch = ef_search
        scores, ids = self.index.search(np.atleast_2d(queries), k)
        return ids, scores
