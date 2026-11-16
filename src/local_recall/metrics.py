"""Ranking metrics, computed per query so they can be bootstrapped.

Conventions follow trec_eval (and so BEIR):
  - nDCG uses linear gain (the relevance grade itself) and a log2(rank + 1) discount.
  - The ideal ranking is built from all judged documents for the query.
  - A document counts as relevant when its grade is > 0.
  - Queries with no relevant document are skipped.

Types used throughout:
  qrels: {query_id: {doc_id: grade}}
  run:   {query_id: [doc_id, ...]}   ranked best first
"""

from __future__ import annotations

import math

import numpy as np

Qrels = dict[str, dict[str, int]]
Run = dict[str, list[str]]


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(
        rels.get(doc, 0) / math.log2(rank + 2) for rank, doc in enumerate(ranked[:k])
    )
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    for rank, doc in enumerate(ranked[:k]):
        if rels.get(doc, 0) > 0:
            return 1.0 / (rank + 1)
    return 0.0


_METRICS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k}


def judged_queries(qrels: Qrels) -> list[str]:
    """Query ids with at least one relevant document, in sorted order."""
    return sorted(q for q, rels in qrels.items() if any(g > 0 for g in rels.values()))


def per_query(run: Run, qrels: Qrels, metric: str, k: int) -> np.ndarray:
    """Metric value for each judged query, in `judged_queries` order.

    A judged query missing from the run scores 0: a system is not allowed to
    look better by skipping hard queries.
    """
    fn = _METRICS[metric]
    return np.array(
        [fn(run.get(q, []), qrels[q], k) for q in judged_queries(qrels)], dtype=np.float64
    )


def evaluate(run: Run, qrels: Qrels, spec: tuple[tuple[str, int], ...]) -> dict[str, float]:
    """Mean of each (metric, k) over judged queries, e.g. {"ndcg@10": 0.71}."""
    return {f"{m}@{k}": float(per_query(run, qrels, m, k).mean()) for m, k in spec}
