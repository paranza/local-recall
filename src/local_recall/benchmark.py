"""Run every retrieval stage on the BEIR test sets and write the results to disk.

Outputs, all under results/:
  <dataset>.json            metrics with confidence intervals, comparisons, latency, sizes
  runs/<dataset>/<sys>.json ranked lists with scores, so anyone can re-score them
  per_query/<dataset>.npz   per-query nDCG@10 for each system

Quality numbers do not depend on the device. Bulk encoding uses the GPU when one
is present, only to save time. Every latency number is measured on the CPU, one
query at a time, because that is the on-device setting this project is about.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from . import metrics as M
from .bm25 import BM25Index, top_k
from .data import Dataset, load_beir
from .dense import BGE_QUERY_PREFIX, DEFAULT_MODEL, Encoder, ExactIndex, HNSWIndex
from .fusion import rrf
from .rerank import DEFAULT_RERANKER, Reranker
from .stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_randomization_test
from .tokenizer import tokenize

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"

SPEC = (("ndcg", 10), ("recall", 10), ("recall", 100), ("mrr", 10))
DEPTH = 100  # candidates kept per first-stage system
RERANK_DEPTH = 50  # candidates passed to the cross-encoder
EF_SWEEP = (8, 16, 32, 64, 128, 256)
LATENCY_QUERIES = 100
RERANK_LATENCY_QUERIES = 20
ONNX_DATASETS = ("scifact", "nfcorpus")  # small enough to encode on the CPU three times


def _percentiles(ms: list[float]) -> dict[str, float]:
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)), "n": len(ms)}


def _time_each(fn, items) -> list[float]:
    out = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        out.append((time.perf_counter() - t0) * 1000)
    return out


def _cached_embeddings(tag: str, encoder: Encoder, texts: list[str], batch_size: int) -> tuple[np.ndarray, float | None]:
    """Document embeddings, cached on disk. Returns (embeddings, encode seconds or None if cached)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{tag}.npy"
    if path.exists():
        return np.load(path), None
    t0 = time.perf_counter()
    emb = encoder.encode_docs(texts, batch_size=batch_size, progress=True)
    seconds = time.perf_counter() - t0
    np.save(path, emb)
    return emb, seconds


def _run_from_ids(ds: Dataset, qids: list[str], ids: np.ndarray, scores: np.ndarray) -> dict[str, list[tuple[str, float]]]:
    return {
        q: [(ds.doc_ids[int(i)], float(s)) for i, s in zip(row_i, row_s) if i >= 0]
        for q, row_i, row_s in zip(qids, ids, scores)
    }


def _summarize(run: dict[str, list[tuple[str, float]]], ds: Dataset) -> tuple[dict, np.ndarray]:
    ranked = {q: [d for d, _ in hits] for q, hits in run.items()}
    summary = {}
    for metric, k in SPEC:
        values = M.per_query(ranked, ds.qrels, metric, k)
        lo, hi = bootstrap_ci(values)
        summary[f"{metric}@{k}"] = {"mean": float(values.mean()), "ci95": [lo, hi]}
    return summary, M.per_query(ranked, ds.qrels, "ndcg", 10)


def export_onnx(model_name: str = DEFAULT_MODEL) -> dict[str, Path]:
    """Save the ONNX fp32 model and a dynamically quantized int8 copy under models/."""
    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

    out = MODELS / model_name.split("/")[-1]
    fp32 = out / "onnx" / "model.onnx"
    int8 = out / "onnx" / "model_quint8_avx2.onnx"
    if not fp32.exists():
        SentenceTransformer(model_name, backend="onnx", device="cpu").save_pretrained(str(out))
    if not int8.exists():
        model = SentenceTransformer(str(out), backend="onnx", device="cpu", model_kwargs={"file_name": "onnx/model.onnx"})
        export_dynamic_quantized_onnx_model(model, "avx2", str(out))
    return {"dir": out, "fp32": fp32, "int8": int8}


def run_dataset(name: str, *, device: str, with_onnx: bool) -> dict:
    print(f"\n=== {name} ===", flush=True)
    ds = load_beir(name)
    qids = M.judged_queries(ds.qrels)
    qtexts = [ds.queries[q] for q in qids]
    rng = np.random.default_rng(0)
    sample = [qtexts[i] for i in rng.choice(len(qtexts), size=min(LATENCY_QUERIES, len(qtexts)), replace=False)]

    runs: dict[str, dict[str, list[tuple[str, float]]]] = {}
    latency: dict[str, dict] = {}
    sizes: dict[str, float] = {}
    build: dict[str, float] = {}

    # ---- 1. BM25, with and without the plural stemmer ----
    for label, stem in (("bm25_nostem", False), ("bm25", True)):
        t0 = time.perf_counter()
        index = BM25Index(tokenizer=lambda t, s=stem: tokenize(t, stem=s)).build(ds.doc_texts)
        build[label] = time.perf_counter() - t0
        runs[label] = {
            q: [(ds.doc_ids[i], s) for i, s in index.search(text, DEPTH)] for q, text in zip(qids, qtexts)
        }
        if stem:
            bm25 = index
            latency["bm25"] = _percentiles(_time_each(lambda t: index.search(t, DEPTH), sample))
            sizes["bm25_postings_mb"] = sum(i.nbytes + t.nbytes for i, t in index.postings.values()) / 1e6
    print("bm25 done", flush=True)

    # ---- 2. Dense retrieval, exact and HNSW ----
    bulk = Encoder(DEFAULT_MODEL, device=device)
    doc_emb, enc_seconds = _cached_embeddings(f"{name}_bge-small_fp32", bulk, ds.doc_texts, batch_size=128)
    if enc_seconds is not None:
        build[f"encode_docs_{device}"] = enc_seconds
    q_emb = bulk.encode_queries(qtexts, batch_size=128)
    del bulk

    exact = ExactIndex(doc_emb)
    ids, scores = exact.search(q_emb, DEPTH)
    runs["dense"] = _run_from_ids(ds, qids, ids, scores)
    exact_top10 = ids[:, :10]
    sizes["embeddings_mb"] = doc_emb.nbytes / 1e6

    t0 = time.perf_counter()
    hnsw = HNSWIndex(doc_emb, m=32, ef_construction=200)
    build["hnsw"] = time.perf_counter() - t0
    sizes["hnsw_index_mb"] = faiss.serialize_index(hnsw.index).nbytes / 1e6

    sample_idx = rng.choice(len(qtexts), size=min(LATENCY_QUERIES, len(qtexts)), replace=False)
    ann = []
    exact_ms = _time_each(lambda i: exact.search(q_emb[i], 10), sample_idx)
    for ef in EF_SWEEP:
        ids_a, scores_a = hnsw.search(q_emb, DEPTH, ef_search=max(ef, DEPTH))
        ids_10, _ = hnsw.search(q_emb, 10, ef_search=ef)
        overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids_10, exact_top10)])
        ranked = {q: [ds.doc_ids[int(i)] for i in row if i >= 0] for q, row in zip(qids, ids_10)}
        ann.append(
            {
                "ef_search": ef,
                "recall_vs_exact@10": float(overlap),
                "ndcg@10": float(M.per_query(ranked, ds.qrels, "ndcg", 10).mean()),
                **_percentiles(_time_each(lambda i, e=ef: hnsw.search(q_emb[i], 10, ef_search=e), sample_idx)),
            }
        )
        if ef == 64:
            runs["dense_hnsw"] = _run_from_ids(ds, qids, ids_a, scores_a)
    latency["dense_search_exact"] = _percentiles(exact_ms)
    print("dense done", flush=True)

    # ---- 3. Hybrid: reciprocal rank fusion of BM25 and dense ----
    runs["hybrid"] = {
        q: rrf([[d for d, _ in runs["bm25"][q]], [d for d, _ in runs["dense"][q]]], top=DEPTH) for q in qids
    }

    # ---- 4. Cross-encoder re-ranking of the hybrid top candidates ----
    doc_pos = {d: i for i, d in enumerate(ds.doc_ids)}
    reranker = Reranker(DEFAULT_RERANKER, device=device)
    rerank_run = {}
    for q, text in zip(qids, qtexts):
        cands = [d for d, _ in runs["hybrid"][q][:RERANK_DEPTH]]
        rerank_run[q] = reranker.rerank(text, cands, [ds.doc_texts[doc_pos[d]] for d in cands])
    runs["hybrid_rerank"] = rerank_run
    del reranker
    print("rerank done", flush=True)

    # ---- latency on the CPU: query encoding and re-ranking ----
    cpu_encoder = Encoder(DEFAULT_MODEL, device="cpu")
    cpu_encoder.encode_queries(sample[:3])  # warm-up
    latency["encode_query_torch_fp32"] = _percentiles(_time_each(lambda t: cpu_encoder.encode_queries([t]), sample))
    del cpu_encoder
    cpu_reranker = Reranker(DEFAULT_RERANKER, device="cpu")
    pairs = []
    for i in sample_idx[:RERANK_LATENCY_QUERIES]:
        cands = [d for d, _ in runs["hybrid"][qids[i]][:RERANK_DEPTH]]
        pairs.append((qtexts[i], [ds.doc_texts[doc_pos[d]] for d in cands]))
    cpu_reranker.scores(*pairs[0])  # warm-up
    latency[f"rerank_top{RERANK_DEPTH}"] = _percentiles(_time_each(lambda p: cpu_reranker.scores(*p), pairs))
    del cpu_reranker

    # ---- 6. ONNX fp32 and int8 on the CPU ----
    onnx_info: dict = {}
    if with_onnx and name in ONNX_DATASETS:
        paths = export_onnx(DEFAULT_MODEL)
        for label, file_name in (("onnx_fp32", "onnx/model.onnx"), ("onnx_int8", "onnx/model_quint8_avx2.onnx")):
            enc = Encoder(
                str(paths["dir"]), device="cpu", backend="onnx", onnx_file=file_name, query_prefix=BGE_QUERY_PREFIX
            )
            emb, seconds = _cached_embeddings(f"{name}_bge-small_{label}", enc, ds.doc_texts, batch_size=32)
            if seconds is not None:
                build[f"encode_docs_{label}_cpu"] = seconds
            q = enc.encode_queries(qtexts, batch_size=32)
            ids_o, scores_o = ExactIndex(emb).search(q, DEPTH)
            runs[f"dense_{label}"] = _run_from_ids(ds, qids, ids_o, scores_o)
            enc.encode_queries(sample[:3])
            latency[f"encode_query_{label}"] = _percentiles(_time_each(lambda t: enc.encode_queries([t]), sample))
            onnx_info[label] = {
                "model_file_mb": (paths["dir"] / file_name).stat().st_size / 1e6,
                "mean_cosine_to_torch_fp32": float(np.mean(np.sum(emb * doc_emb, axis=1))),
            }
            del enc
        print("onnx done", flush=True)

    # ---- metrics, confidence intervals, paired comparisons ----
    systems, per_q = {}, {}
    for label, run in runs.items():
        systems[label], per_q[label] = _summarize(run, ds)

    pairs_to_test = [
        ("bm25", "bm25_nostem"), ("dense", "bm25"), ("dense_hnsw", "dense"), ("hybrid", "bm25"),
        ("hybrid", "dense"), ("hybrid_rerank", "hybrid"), ("hybrid_rerank", "dense"),
    ]
    if "dense_onnx_int8" in runs:
        pairs_to_test += [("dense_onnx_fp32", "dense"), ("dense_onnx_int8", "dense_onnx_fp32")]
    comparisons = []
    for a, b in pairs_to_test:
        lo, hi = paired_bootstrap_ci(per_q[a], per_q[b])
        comparisons.append(
            {
                "a": a, "b": b, "metric": "ndcg@10",
                "diff": float((per_q[a] - per_q[b]).mean()), "ci95": [lo, hi],
                "p": paired_randomization_test(per_q[a], per_q[b]),
            }
        )
    for comp, p_adj in zip(comparisons, holm([c["p"] for c in comparisons])):
        comp["p_holm"] = p_adj

    result = {
        "dataset": name, "n_docs": len(ds.doc_ids), "n_queries": len(qids),
        "systems": systems, "comparisons": comparisons, "ann_sweep": ann,
        "latency_cpu": latency, "sizes": sizes, "build_seconds": build, "onnx": onnx_info,
    }

    (RESULTS / "runs" / name).mkdir(parents=True, exist_ok=True)
    (RESULTS / "per_query").mkdir(parents=True, exist_ok=True)
    for label, run in runs.items():
        (RESULTS / "runs" / name / f"{label}.json").write_text(json.dumps(run), encoding="utf-8")
    np.savez(RESULTS / "per_query" / f"{name}.npz", qids=np.array(qids), **per_q)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def environment() -> dict:
    import sentence_transformers

    cpu = platform.processor()
    try:
        import subprocess

        cpu = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True
        ).strip() or cpu
    except Exception:
        pass
    return {
        "cpu": cpu, "os": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
        "faiss": faiss.__version__, "torch_threads": torch.get_num_threads(),
        "gpu_for_bulk_encoding": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_model": DEFAULT_MODEL, "reranker": DEFAULT_RERANKER,
        "depth": DEPTH, "rerank_depth": RERANK_DEPTH,
    }


def main(datasets: tuple[str, ...] = ("scifact", "nfcorpus", "fiqa"), with_onnx: bool = True) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    for name in datasets:
        res = run_dataset(name, device=device, with_onnx=with_onnx)
        for label, s in res["systems"].items():
            print(f"  {label:18s} nDCG@10 {s['ndcg@10']['mean']:.4f}  R@100 {s['recall@100']['mean']:.4f}", flush=True)
