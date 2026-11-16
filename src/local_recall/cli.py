"""Command line entry point: `local-recall <command>`."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="local-recall", description="Local personal search, measured end to end.")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="run every retrieval stage on the BEIR test sets")
    bench.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    bench.add_argument("--no-onnx", action="store_true", help="skip the ONNX fp32 / int8 comparison")

    sub.add_parser("mailbox-benchmark", help="personal-search benchmark on the Enron mailboxes")

    agent = sub.add_parser("agent-eval", help="score the agent and the single-step baseline")
    agent.add_argument("--model", default=None, help="Hugging Face id of the chat model")
    agent.add_argument("--4bit", dest="four_bit", action="store_true", help="load the model in 4-bit")
    agent.add_argument("--tag", default=None, help="suffix for the output files")

    sub.add_parser("report", help="rebuild results/RESULTS.md from the result files")
    sub.add_parser("all", help="benchmark, mailbox-benchmark, agent-eval and report, in that order")

    search = sub.add_parser("search", help="search one Enron mailbox from the terminal")
    search.add_argument("query")
    search.add_argument("--owner", default="greg.whalley@enron.com")
    search.add_argument("--mode", default="rerank", choices=["bm25", "dense", "hybrid", "rerank"])
    search.add_argument("-k", type=int, default=5)

    args = parser.parse_args()
    if args.command in ("benchmark", "all"):
        from .benchmark import main as run

        run(tuple(getattr(args, "datasets", ("scifact", "nfcorpus", "fiqa"))), with_onnx=not getattr(args, "no_onnx", False))
    if args.command in ("mailbox-benchmark", "all"):
        from .enron_benchmark import main as run

        run()
    if args.command in ("agent-eval", "all"):
        from .agent import DEFAULT_LLM
        from .agent_eval import main as run

        run(getattr(args, "model", None) or DEFAULT_LLM, getattr(args, "four_bit", False), getattr(args, "tag", None))
    if args.command in ("report", "all"):
        from .report import main as run

        run()
    if args.command == "search":
        import numpy as np
        import torch

        from .enron_benchmark import load_engine
        from .query_filters import parse_query

        mail, _, engine = load_engine("cuda" if torch.cuda.is_available() else "cpu")
        parsed = parse_query(args.query)
        to64 = lambda d: np.datetime64(d.isoformat(), "s") if d else None  # noqa: E731
        cands = mail.candidates(args.owner, sender=parsed.sender, date_from=to64(parsed.date_from), date_to=to64(parsed.date_to))
        print(f"filters: sender={parsed.sender} from={parsed.date_from} to={parsed.date_to} | {len(cands):,} candidate emails")
        for rank, (i, score) in enumerate(engine.search(parsed.text or args.query, k=args.k, mode=args.mode, candidates=cands), 1):
            print(f"{rank}. [{score:.3f}] {str(mail.dates[i])[:10]} {mail.senders[i]} | {mail.subjects[i][:80]}")


if __name__ == "__main__":
    main()
