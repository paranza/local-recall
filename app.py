"""Web demo: the four retrieval modes side by side on one mailbox, plus the agent.

Run:  python app.py
The agent needs a local chat model. It loads on first use. Set LOCAL_RECALL_AGENT=0 to hide it.
"""

from __future__ import annotations

import os
from datetime import date

import gradio as gr
import numpy as np
import torch

from local_recall.agent import DEFAULT_LLM, MailboxEnv, run_agent
from local_recall.enron_benchmark import load_engine
from local_recall.query_filters import parse_query

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGENT_ON = os.environ.get("LOCAL_RECALL_AGENT", "1") != "0"
MODES = (("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid (RRF)"), ("rerank", "Hybrid + cross-encoder"))

mail, questions, engine = load_engine(DEVICE)
OWNERS = sorted(mail.owner_index)
_llm = None


def _to64(d: date | None) -> np.datetime64 | None:
    return np.datetime64(d.isoformat(), "s") if d else None


def search(query: str, owner: str, today: str):
    if not query.strip():
        return ["Type a query."] + [""] * len(MODES)
    try:
        today_d = date.fromisoformat(today.strip())
    except ValueError:
        return ["'Today' must be a date like 2001-06-30."] + [""] * len(MODES)
    parsed = parse_query(query, today=today_d)
    upper = min(parsed.date_to or today_d, today_d)
    cands = mail.candidates(owner, sender=parsed.sender, date_from=_to64(parsed.date_from), date_to=_to64(upper))
    info = (
        f"**Parsed filters:** sender = `{parsed.sender}`, from = `{parsed.date_from}`, to = `{upper}`  \n"
        f"**Text sent to the rankers:** `{parsed.text or query}`  \n"
        f"**Emails that pass the filters:** {len(cands):,} of {len(mail.owner_index[owner]):,} in this mailbox"
    )
    columns = []
    for mode, _ in MODES:
        hits = engine.search(parsed.text or query, k=5, mode=mode, candidates=cands)
        lines = [
            f"**{rank}. {mail.subjects[i][:70] or '(no subject)'}**  \n"
            f"{str(mail.dates[i])[:10]} · {mail.senders[i]} · score {score:.3f}  \n"
            f"<small>{' '.join(mail.bodies[i].split())[:180]}</small>"
            for rank, (i, score) in enumerate(hits, 1)
        ]
        columns.append("\n\n".join(lines) or "_no results_")
    return [info, *columns]


def ask(question: str, owner: str, today: str):
    global _llm
    if not question.strip():
        return "Type a question.", ""
    if _llm is None:
        from local_recall.agent import LocalLLM

        _llm = LocalLLM(DEFAULT_LLM)
    env = MailboxEnv(mail, engine, owner, today.strip())
    episode = run_agent(_llm, env, question)
    cited = "\n".join(
        f"- **{env.handle(i)}** {str(mail.dates[i])[:10]} · {mail.senders[i]} · {mail.subjects[i][:80]}" for i in episode.sources
    ) or "_no sources cited_"
    steps = "\n".join(
        f"{t['step']}. `{t.get('tool', 'text')}` {t.get('args', '')}" for t in episode.trace
    )
    answer = f"### {episode.answer}\n\n**Sources**\n{cited}\n\n{episode.steps} steps, {episode.seconds:.1f} s on {DEVICE}"
    return answer, steps


def sample_question(owner: str):
    rows = questions[questions["owner"] == owner]
    row = rows.sample(1).iloc[0]
    return row["question"], row["query_date"]


with gr.Blocks(title="local-recall") as demo:
    gr.Markdown(
        "# local-recall\nSearch one person's mailbox, fully on this machine. Public Enron emails. "
        "Try a filter in plain words: `emails from tim belden in May 2001 about the forecast`."
    )
    with gr.Row():
        owner = gr.Dropdown(OWNERS, value=OWNERS[0], label="Mailbox")
        today = gr.Textbox("2001-12-31", label="Today (later emails are hidden)")
        sample = gr.Button("Load a benchmark question")
    query = gr.Textbox(label="Query or question", lines=1)
    with gr.Row():
        go = gr.Button("Search", variant="primary")
        ask_btn = gr.Button("Ask the agent", visible=AGENT_ON)
    info = gr.Markdown()
    with gr.Row():
        outputs = []
        for _, label in MODES:
            with gr.Column():
                gr.Markdown(f"### {label}")
                outputs.append(gr.Markdown())
    answer = gr.Markdown()
    trace = gr.Markdown()

    go.click(search, [query, owner, today], [info, *outputs])
    query.submit(search, [query, owner, today], [info, *outputs])
    ask_btn.click(ask, [query, owner, today], [answer, trace])
    sample.click(sample_question, [owner], [query, today])

if __name__ == "__main__":
    demo.launch()
