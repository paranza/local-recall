"""Personal-search benchmark: find the source email for a question about one mailbox.

Every question is searched inside its owner's mailbox only. By default emails
dated after the question's date are hidden, since a real assistant cannot see
the future. One extra run drops that date filter to measure what it is worth.

Writes results/enron.json, results/runs/enron/*.json and results/per_query/enron.npz.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import metrics as M
from .benchmark import CACHE, RESULTS, _percentiles
from .dense import DEFAULT_MODEL, Encoder
from .engine import SearchEngine
from .enron import load_mail
from .rerank import DEFAULT_RERANKER, Reranker
from .stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_randomization_test

SPEC = (("recall", 1), ("recall", 5), ("recall", 10), ("mrr", 10), ("ndcg", 10))
MODES = {"bm25": "bm25", "dense": "dense", "hybrid": "hybrid", "hybrid_rerank": "rerank"}


def load_engine(device: str) -> tuple:
    """Mail, questions and a ready SearchEngine. Embeddings are cached on disk."""
    mail, questions = load_mail()
    texts = mail.texts
    encoder = Encoder(DEFAULT_MODEL, device=device)
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "enron_bge-small_fp32.npy"
    if path.exists():
        emb = np.load(path)
    else:
        emb = encoder.encode_docs(texts, batch_size=128, progress=True)
        np.save(path, emb)
    engine = SearchEngine(texts, encoder=encoder, embeddings=emb, reranker=Reranker(DEFAULT_RERANKER, device=device))
    return mail, questions, engine


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mail, questions, engine = load_engine(device)
    qids = [str(i) for i in questions["id"]]
    qrels = {q: {g: 1 for g in gold} for q, gold in zip(qids, questions["gold"])}
    q_emb = engine.encoder.encode_queries(questions["question"].tolist(), batch_size=128)

    id_to_doc = {m: i for i, m in enumerate(mail.ids)}
    cutoffs = [np.datetime64(d, "D").astype("datetime64[s]") for d in questions["query_date"]]
    gold_after_cutoff = float(np.mean([
        any(mail.dates[id_to_doc[g]] >= c + np.timedelta64(1, "D") for g in gold)
        for gold, c in zip(questions["gold"], cutoffs)
    ]))

    runs: dict[str, dict[str, list[tuple[str, float]]]] = {name: {} for name in (*MODES, "hybrid_no_date_filter")}
    pool_sizes = []
    timing: dict[str, list[float]] = {name: [] for name in MODES}
    for n, (qid, text, owner, cutoff) in enumerate(zip(qids, questions["question"], questions["owner"], cutoffs)):
        cands = mail.candidates(owner, date_to=cutoff)
        pool_sizes.append(len(cands))
        for name, mode in MODES.items():
            t0 = time.perf_counter()
            hits = engine.search(text, k=10, mode=mode, candidates=cands, q_emb=q_emb[n])
            timing[name].append((time.perf_counter() - t0) * 1000)
            runs[name][qid] = [(mail.ids[i], s) for i, s in hits]
        hits = engine.search(text, k=10, mode="hybrid", candidates=mail.candidates(owner), q_emb=q_emb[n])
        runs["hybrid_no_date_filter"][qid] = [(mail.ids[i], s) for i, s in hits]
        if (n + 1) % 400 == 0:
            print(f"  {n + 1}/{len(qids)} questions", flush=True)

    systems, per_q = {}, {}
    for name, run in runs.items():
        ranked = {q: [d for d, _ in hits] for q, hits in run.items()}
        systems[name] = {}
        for metric, k in SPEC:
            values = M.per_query(ranked, qrels, metric, k)
            lo, hi = bootstrap_ci(values)
            systems[name][f"{metric}@{k}"] = {"mean": float(values.mean()), "ci95": [lo, hi]}
        per_q[name] = M.per_query(ranked, qrels, "mrr", 10)

    comparisons = []
    for a, b in (("dense", "bm25"), ("hybrid", "bm25"), ("hybrid", "dense"), ("hybrid_rerank", "hybrid"),
                 ("hybrid_rerank", "bm25"), ("hybrid", "hybrid_no_date_filter")):
        lo, hi = paired_bootstrap_ci(per_q[a], per_q[b])
        comparisons.append({
            "a": a, "b": b, "metric": "mrr@10", "diff": float((per_q[a] - per_q[b]).mean()),
            "ci95": [lo, hi], "p": paired_randomization_test(per_q[a], per_q[b]),
        })
    for comp, p_adj in zip(comparisons, holm([c["p"] for c in comparisons])):
        comp["p_holm"] = p_adj

    result = {
        "dataset": "enron", "n_docs": len(mail.ids), "n_queries": len(qids),
        "n_mailboxes": len(mail.owner_index),
        "mailbox_sizes": {o: int(len(v)) for o, v in sorted(mail.owner_index.items())},
        "candidate_pool": {"mean": float(np.mean(pool_sizes)), "min": int(np.min(pool_sizes)), "max": int(np.max(pool_sizes))},
        "share_of_questions_with_gold_after_query_date": gold_after_cutoff,
        "systems": systems, "comparisons": comparisons,
        # Wall time per query for the whole search call, query embedding excluded.
        # The cross-encoder runs on the GPU here when one is present; see `device`.
        "search_ms": {name: _percentiles(ms) for name, ms in timing.items()},
        "device": device,
        "embedding_model": DEFAULT_MODEL, "reranker": DEFAULT_RERANKER,
    }
    (RESULTS / "runs" / "enron").mkdir(parents=True, exist_ok=True)
    (RESULTS / "per_query").mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        (RESULTS / "runs" / "enron" / f"{name}.json").write_text(json.dumps(run), encoding="utf-8")
    (RESULTS / "runs" / "enron" / "qrels.json").write_text(json.dumps(qrels), encoding="utf-8")
    np.savez(RESULTS / "per_query" / "enron.npz", qids=np.array(qids), **per_q)
    (RESULTS / "enron.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, s in systems.items():
        print(f"  {name:24s} R@1 {s['recall@1']['mean']:.4f}  R@10 {s['recall@10']['mean']:.4f}  MRR@10 {s['mrr@10']['mean']:.4f}")


if __name__ == "__main__":
    main()
