"""Error-analysis counts quoted in the README, computed from the saved runs and transcripts.

Writes results/analysis.json. Needs no model and no dataset download.
"""

from __future__ import annotations

import json
from collections import Counter
from statistics import median

from .benchmark import RESULTS

MISS = 99  # rank used when the gold email is not in the top 10


def _first_gold_rank(hits: list, gold: dict) -> int:
    for rank, (doc, _) in enumerate(hits, start=1):
        if doc in gold:
            return rank
    return MISS


def mailbox() -> dict:
    root = RESULTS / "runs" / "enron"
    qrels = json.loads((root / "qrels.json").read_text(encoding="utf-8"))
    names = ("bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_no_date_filter")
    rank = {
        n: {q: _first_gold_rank(hits, qrels[q]) for q, hits in json.loads((root / f"{n}.json").read_text(encoding="utf-8")).items()}
        for n in names
    }
    qs = list(qrels)
    return {
        "bm25_rank1_dense_below5": sum(rank["bm25"][q] == 1 and rank["dense"][q] > 5 for q in qs),
        "dense_rank1_bm25_below5": sum(rank["dense"][q] == 1 and rank["bm25"][q] > 5 for q in qs),
        "rerank_promoted_to_rank1": sum(rank["hybrid"][q] > 1 and rank["hybrid_rerank"][q] == 1 for q in qs),
        "rerank_demoted_from_rank1": sum(rank["hybrid"][q] == 1 and rank["hybrid_rerank"][q] > 1 for q in qs),
        "rerank_missed_top10": sum(rank["hybrid_rerank"][q] == MISS for q in qs),
        "rerank_missed_top10_and_absent_from_both_first_stages": sum(
            rank["hybrid_rerank"][q] == MISS and rank["bm25"][q] == MISS and rank["dense"][q] == MISS for q in qs
        ),
        "date_filter_moved_to_rank1": sum(rank["hybrid"][q] == 1 and rank["hybrid_no_date_filter"][q] > 1 for q in qs),
    }


def agent(path=None) -> dict:
    path = path or RESULTS / "agent_transcripts.jsonl"
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    ans = {s: [r for r in recs if r["setting"] == "answerable" and r["system"] == s] for s in ("rag", "agent")}
    a, r = ans["agent"], ans["rag"]
    steps = [t for rec in a for t in rec["trace"]]
    searches = [t for t in steps if t.get("tool") == "search_emails"]
    return {
        "episodes": len(a),
        "pushbacks": dict(Counter(t["pushed_back"] for t in steps if t.get("pushed_back"))),
        "tool_calls": dict(Counter(t["tool"] for t in steps if t.get("tool"))),
        "searches_with_a_filter": sum(any(k in t["args"] for k in ("sender", "date_from", "date_to")) for t in searches),
        "searches_that_fell_back_to_unfiltered": sum(t["result"].startswith("No emails matched those filters") for t in searches),
        "source_hit_both": sum(bool(x["source_hit"] and y["source_hit"]) for x, y in zip(a, r)),
        "source_hit_agent_only": sum(bool(x["source_hit"] and not y["source_hit"]) for x, y in zip(a, r)),
        "source_hit_single_step_only": sum(bool(y["source_hit"] and not x["source_hit"]) for x, y in zip(a, r)),
        "median_answer_words": {
            s: median(len(rec["answer"].split()) for rec in rows if not rec["abstained"]) for s, rows in ans.items()
        },
    }


def main() -> dict:
    out = {}
    if (RESULTS / "runs" / "enron" / "qrels.json").exists():
        out["mailbox"] = mailbox()
    if (RESULTS / "agent_transcripts.jsonl").exists():
        out["agent"] = agent()
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
