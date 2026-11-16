import math

import numpy as np
import pytest

from local_recall.bm25 import BM25Index, top_k
from local_recall.fusion import rrf
from local_recall.metrics import evaluate, mrr_at_k, ndcg_at_k, per_query, recall_at_k
from local_recall.stats import bootstrap_ci, holm, paired_bootstrap_ci, paired_randomization_test
from local_recall.tokenizer import normalize, s_stem, tokenize

# ---------- tokenizer ----------


def test_normalize_strips_accents_and_case():
    assert normalize("Café DÉJÀ") == "cafe deja"


def test_tokenize_splits_drops_stopwords_and_stems():
    assert tokenize("The cats and the dog's 2 queries!") == ["cat", "dog", "s", "2", "query"]


def test_tokenize_flags():
    assert tokenize("The cats", stopwords=False, stem=False) == ["the", "cats"]


@pytest.mark.parametrize(
    "word,stem",
    [("queries", "query"), ("houses", "house"), ("cats", "cat"), ("glass", "glass"),
     ("virus", "virus"), ("shoes", "shoe"), ("bus", "bus"), ("is", "is")],
)
def test_s_stemmer_rules(word, stem):
    assert s_stem(word) == stem


# ---------- BM25 ----------

DOCS = [
    "the quick brown fox jumps over the lazy dog",
    "a quick movement of the enemy will jeopardize six gunboats",
    "the five boxing wizards jump quickly",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz judge my vow",
    "the dog barks at the brown fox and the fox runs",
]


def _ws(text):
    return text.split()


def test_bm25_hand_computed_score():
    idx = BM25Index(k1=1.2, b=0.75, idf="lucene", tokenizer=_ws).build(["a b b", "a c", "c c c c"])
    # term "b": df=1, N=3 -> idf = ln(1 + 2.5/1.5); doc 0: tf=2, |d|=3, avgdl=3
    idf = math.log(1 + 2.5 / 1.5)
    expected = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 3))
    scores = idx.scores("b")
    assert scores[0] == pytest.approx(expected, rel=1e-6)
    assert scores[1] == 0 and scores[2] == 0


def test_bm25_matches_rank_bm25_reference():
    from rank_bm25 import BM25Okapi

    ref = BM25Okapi([_ws(d) for d in DOCS], k1=1.5, b=0.75)
    mine = BM25Index(k1=1.5, b=0.75, idf="okapi", tokenizer=_ws).build(DOCS)
    # Queries use terms found in fewer than half the documents: rank_bm25 replaces
    # negative idf values with a floor, which the plain Okapi formula does not do.
    for query in ["fox dog", "quick zebras", "five box jump", "fox fox brown", "missing term"]:
        np.testing.assert_allclose(mine.scores(query), ref.get_scores(_ws(query)), rtol=1e-5, atol=1e-6)


def test_bm25_search_orders_and_drops_zero():
    idx = BM25Index(tokenizer=_ws).build(DOCS)
    hits = idx.search("fox", k=5)
    assert [h[0] for h in hits] == [6, 0]  # doc 6 mentions fox twice
    assert hits[0][1] > hits[1][1]
    assert idx.search("nonexistent", k=5) == []


def test_top_k_tie_break_is_lower_index():
    assert [i for i, _ in top_k(np.array([1.0, 3.0, 3.0, 2.0]), 3)] == [1, 2, 3]


# ---------- metrics ----------

RELS = {"d1": 2, "d2": 1, "d3": 0, "d4": 1}


def test_ndcg_hand_computed():
    ranked = ["d3", "d1", "d9", "d2"]
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, RELS, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["d1", "d2", "d4"], RELS, 10) == pytest.approx(1.0)


def test_recall_and_mrr():
    ranked = ["d3", "d1", "d9", "d2"]
    assert recall_at_k(ranked, RELS, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, RELS, 10) == pytest.approx(2 / 3)
    assert mrr_at_k(ranked, RELS, 10) == pytest.approx(0.5)
    assert mrr_at_k(ranked, RELS, 1) == 0.0


def test_missing_query_scores_zero_and_unjudged_query_is_skipped():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 0}}
    run = {"q1": ["d1"]}
    assert per_query(run, qrels, "ndcg", 10).tolist() == [1.0, 0.0]
    assert evaluate(run, qrels, (("recall", 10),)) == {"recall@10": 0.5}


def test_metrics_match_ranx_reference():
    from ranx import Qrels, Run
    from ranx import evaluate as ranx_evaluate

    rng = np.random.default_rng(0)
    docs = [f"d{i}" for i in range(60)]
    qrels, run_scores, run = {}, {}, {}
    for q in range(40):
        judged = rng.choice(docs, size=8, replace=False)
        qrels[f"q{q}"] = {d: int(g) for d, g in zip(judged, rng.integers(0, 3, size=8))}
        if not any(qrels[f"q{q}"].values()):
            qrels[f"q{q}"][judged[0]] = 1
        ranked = rng.permutation(docs)[:25].tolist()
        run[f"q{q}"] = ranked
        run_scores[f"q{q}"] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
    ref = ranx_evaluate(Qrels(qrels), Run(run_scores), ["ndcg@10", "recall@10", "mrr@10", "recall@20"])
    mine = evaluate(run, qrels, (("ndcg", 10), ("recall", 10), ("mrr", 10), ("recall", 20)))
    for key, value in mine.items():
        assert value == pytest.approx(float(ref[key]), abs=1e-9), key


# ---------- fusion ----------


def test_rrf_scores_and_order():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["c"] == pytest.approx(1 / 63)
    assert [d for d, _ in fused] == ["a", "b", "c", "d"]  # ties keep first-seen order
    assert len(rrf([["a", "b"], ["c"]], top=2)) == 2


# ---------- statistics ----------


def test_bootstrap_ci_covers_mean_and_narrows_with_n():
    rng = np.random.default_rng(1)
    small, large = rng.normal(0.5, 0.2, 50), rng.normal(0.5, 0.2, 2000)
    lo_s, hi_s = bootstrap_ci(small, n_boot=2000)
    lo_l, hi_l = bootstrap_ci(large, n_boot=2000)
    assert lo_s < small.mean() < hi_s
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_randomization_test_detects_real_difference_only():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, 300)
    noise = rng.normal(0, 0.05, 300)
    assert paired_randomization_test(base + 0.05, base + noise * 0, n_perm=2000) < 0.01
    assert paired_randomization_test(base + noise, base, n_perm=2000) > 0.05
    lo, hi = paired_bootstrap_ci(base + 0.05, base, n_boot=2000)
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.05)


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
