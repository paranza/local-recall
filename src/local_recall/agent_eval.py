"""Score the agent and the single-step baseline on mailbox questions.

Two settings, same questions, same local model:
  answerable        the mailbox is intact. Score: did it cite a gold source email, and how
                    close is the answer text to the reference answer.
  evidence removed  the gold emails are hidden. The right behaviour is NOT_FOUND. Score:
                    how often the system answers anyway.

All scores are automatic and repeatable. Nothing here uses a language model as a judge.
  source_hit     1 if any cited email is a gold source
  answer_f1      token-overlap F1 between answer and reference (the SQuAD metric)
  abstained      1 if the answer is NOT_FOUND

Writes results/agent.json and results/agent_transcripts.jsonl.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

import numpy as np
import torch

from .agent import DEFAULT_LLM, NOT_FOUND, LocalLLM, MailboxEnv, rag_answer, run_agent
from .benchmark import RESULTS
from .enron_benchmark import load_engine
from .stats import bootstrap_ci, paired_bootstrap_ci, paired_randomization_test

N_ANSWERABLE = 150
N_REMOVED = 75
SEED = 0

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def _norm_tokens(text: str) -> list[str]:
    return _ARTICLES.sub(" ", text.lower().translate(_PUNCT)).split()


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1."""
    pred, ref = _norm_tokens(prediction), _norm_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    return NOT_FOUND.lower() in answer.lower().replace(" ", "_")


def _mean_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = bootstrap_ci(arr)
    return {"mean": float(arr.mean()), "ci95": [lo, hi]}


def main(model_name: str = DEFAULT_LLM, load_in_4bit: bool = False, tag: str | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mail, questions, engine = load_engine(device)
    id_to_doc = {m: i for i, m in enumerate(mail.ids)}
    llm = LocalLLM(model_name, load_in_4bit=load_in_4bit)

    rng = np.random.default_rng(SEED)
    picked = rng.choice(len(questions), size=N_ANSWERABLE, replace=False)
    rows = questions.iloc[picked].reset_index(drop=True)

    records = []
    scores: dict[str, dict[str, list[float]]] = {}

    def record(setting: str, system: str, row, episode, gold: set[int]) -> None:
        abstained = is_abstention(episode.answer)
        entry = {
            "source_hit": float(bool(gold.intersection(episode.sources))),
            "answer_f1": 0.0 if abstained else token_f1(episode.answer, row["answer"]),
            "abstained": float(abstained),
            "steps": float(episode.steps),
            "seconds": episode.seconds,
            "forced": float(episode.forced),
        }
        bucket = scores.setdefault(f"{setting}/{system}", {k: [] for k in entry})
        for k, v in entry.items():
            bucket[k].append(v)
        records.append({
            "setting": setting, "system": system, "question_id": int(row["id"]), "owner": row["owner"],
            "question": row["question"], "reference": row["answer"], "answer": episode.answer,
            "cited": [mail.ids[i] for i in episode.sources], "gold": list(row["gold"]),
            **entry, "trace": episode.trace,
        })

    for n, row in rows.iterrows():
        gold = {id_to_doc[g] for g in row["gold"]}
        settings = [("answerable", frozenset())]
        if n < N_REMOVED:
            settings.append(("evidence_removed", frozenset(gold)))
        for setting, exclude in settings:
            for system, fn in (("rag", rag_answer), ("agent", run_agent)):
                env = MailboxEnv(mail, engine, row["owner"], row["query_date"], exclude=exclude)
                record(setting, system, row, fn(llm, env, row["question"]), gold)
        if (n + 1) % 10 == 0:
            a, r = scores["answerable/agent"], scores["answerable/rag"]
            print(
                f"  {n + 1}/{len(rows)}  agent hit {np.mean(a['source_hit']):.2f} f1 {np.mean(a['answer_f1']):.2f} | "
                f"rag hit {np.mean(r['source_hit']):.2f} f1 {np.mean(r['answer_f1']):.2f}", flush=True,
            )

    summary = {key: {m: _mean_ci(v) for m, v in metrics.items()} for key, metrics in scores.items()}
    comparisons = []
    for setting, metric in (("answerable", "source_hit"), ("answerable", "answer_f1"), ("evidence_removed", "abstained")):
        a = np.asarray(scores[f"{setting}/agent"][metric])
        b = np.asarray(scores[f"{setting}/rag"][metric])
        lo, hi = paired_bootstrap_ci(a, b)
        comparisons.append({
            "setting": setting, "metric": metric, "a": "agent", "b": "rag",
            "diff": float((a - b).mean()), "ci95": [lo, hi], "p": paired_randomization_test(a, b),
        })

    result = {
        "llm": model_name, "load_in_4bit": load_in_4bit, "device": device,
        "retrieval_mode": "rerank", "k": 5, "max_steps": 6, "seed": SEED,
        "n_answerable": N_ANSWERABLE, "n_evidence_removed": N_REMOVED,
        "summary": summary, "comparisons": comparisons,
    }
    suffix = f"_{tag}" if tag else ""
    (RESULTS / f"agent{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(RESULTS / f"agent_transcripts{suffix}.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for key, metrics in summary.items():
        print(f"  {key:26s} " + "  ".join(f"{m} {v['mean']:.3f}" for m, v in metrics.items()))


if __name__ == "__main__":
    main()
