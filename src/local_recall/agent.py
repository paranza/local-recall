"""A tool-calling agent that answers questions about one person's mailbox.

The language model runs locally, so no email text leaves the machine. It gets
three tools:
  search_emails(query, sender?, date_from?, date_to?)  ranked snippets
  read_email(email_id)                                  the full text of one hit
  final_answer(answer, sources)                         ends the episode

The loop is plain: the model either calls a tool or stops. Every answer must
name the emails it rests on, and when the mailbox does not hold the answer the
model is told to answer NOT_FOUND instead of guessing.

`rag_answer` is the single-step baseline: retrieve once with the question,
hand the top emails to the model, ask for the answer. Comparing the two shows
what the multi-step loop adds, or does not add.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .engine import SearchEngine
from .enron import Mail

DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
NOT_FOUND = "NOT_FOUND"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search the mailbox. Returns the best matching emails with id, date, sender, subject and a short snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or a short description of what to find."},
                    "sender": {"type": "string", "description": "Optional. Part of the sender's address, e.g. 'greg.whalley'."},
                    "date_from": {"type": "string", "description": "Optional. Earliest date, YYYY-MM-DD."},
                    "date_to": {"type": "string", "description": "Optional. Latest date, YYYY-MM-DD."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Read the full text of one email returned by search_emails.",
            "parameters": {
                "type": "object",
                "properties": {"email_id": {"type": "string", "description": "An id such as E3."}},
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": f"Give the final answer. Use the answer '{NOT_FOUND}' if the mailbox does not contain it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "A short, direct answer."},
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "Ids of the emails that support the answer, e.g. ['E3']."},
                },
                "required": ["answer", "sources"],
            },
        },
    },
]

SYSTEM = (
    "You answer questions about the mailbox of {owner}. Today is {today}.\n"
    "Rules:\n"
    "1. Use search_emails to find evidence. Read an email with read_email before you rely on it.\n"
    "2. If a search does not help, try different keywords or filters.\n"
    "3. Finish with final_answer. Keep the answer short and list the ids of the emails that support it.\n"
    f"4. If the mailbox does not contain the answer, call final_answer with the answer '{NOT_FOUND}' and no sources. Never guess."
)

RAG_SYSTEM = (
    "You answer a question about the mailbox of {owner} using only the emails shown. Today is {today}.\n"
    "Call final_answer with a short answer and the ids of the emails that support it.\n"
    f"If the emails do not contain the answer, call final_answer with the answer '{NOT_FOUND}' and no sources. Never guess."
)

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)


class LocalLLM:
    """A chat model loaded with transformers. Greedy decoding, so runs are repeatable."""

    def __init__(self, model_name: str = DEFAULT_LLM, device: str | None = None, load_in_4bit: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)}
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
        self.model.eval()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_new_tokens: int = 256) -> str:
        inputs = self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract the first {"name": ..., "arguments": {...}} tool call from model output."""
    m = _TOOL_CALL.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    name, args = call.get("name"), call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return name, args


def _parse_date(value) -> np.datetime64 | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return None
    try:
        return np.datetime64(value.strip(), "D").astype("datetime64[s]")
    except ValueError:
        return None


@dataclass
class MailboxEnv:
    """The tools, bound to one mailbox and one "today". Maps short ids (E1, E2) to documents."""

    mail: Mail
    engine: SearchEngine
    owner: str
    today: str  # YYYY-MM-DD; emails after this date are invisible
    mode: str = "rerank"
    k: int = 5
    exclude: frozenset[int] = frozenset()  # doc indices hidden from this episode (used to test abstention)
    handles: dict[str, int] = field(default_factory=dict)
    _by_doc: dict[int, str] = field(default_factory=dict)

    def handle(self, doc: int) -> str:
        if doc not in self._by_doc:
            h = f"E{len(self._by_doc) + 1}"
            self._by_doc[doc] = h
            self.handles[h] = doc
        return self._by_doc[doc]

    def search(self, query: str, sender: str | None = None, date_from=None, date_to=None, k: int | None = None) -> list[int]:
        today = _parse_date(self.today)
        upper = _parse_date(date_to)
        upper = today if upper is None or upper > today else upper
        cands = self.mail.candidates(self.owner, sender=sender or None, date_from=_parse_date(date_from), date_to=upper)
        if self.exclude:
            cands = cands[~np.isin(cands, list(self.exclude))]
        return [i for i, _ in self.engine.search(query, k=k or self.k, mode=self.mode, candidates=cands)]

    def search_emails(self, query: str = "", sender: str | None = None, date_from=None, date_to=None, **_) -> str:
        if not isinstance(query, str) or not query.strip():
            return "Error: 'query' must be a non-empty string."
        hits = self.search(query, sender, date_from, date_to)
        if not hits:
            return "No emails matched. Try other keywords or remove filters."
        lines = []
        for i in hits:
            snippet = " ".join(self.mail.bodies[i].split())[:220]
            lines.append(
                f"[{self.handle(i)}] {str(self.mail.dates[i])[:10]} | from {self.mail.senders[i]} | "
                f"subject: {self.mail.subjects[i][:90]} | {snippet}"
            )
        return "\n".join(lines)

    def read_email(self, email_id: str = "", **_) -> str:
        doc = self.handles.get(str(email_id).strip().strip("[]").upper())
        if doc is None:
            return f"Error: unknown email id '{email_id}'. Use an id returned by search_emails."
        return self.mail.render(doc, max_chars=2500)


@dataclass
class Episode:
    answer: str
    sources: list[int]  # doc indices
    steps: int
    seconds: float
    forced: bool  # True when the step limit ended the episode
    trace: list[dict]


def _finish(env: MailboxEnv, args: dict) -> tuple[str, list[int]]:
    answer = str(args.get("answer", "")).strip() or NOT_FOUND
    raw = args.get("sources", [])
    raw = raw if isinstance(raw, list) else [raw]
    sources = []
    for s in raw:
        doc = env.handles.get(str(s).strip().strip("[]").upper())
        if doc is not None and doc not in sources:
            sources.append(doc)
    return answer, sources


def run_agent(llm, env: MailboxEnv, question: str, max_steps: int = 6) -> Episode:
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM.format(owner=env.owner, today=env.today)},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    for step in range(1, max_steps + 1):
        last = step == max_steps
        if last:
            messages.append({"role": "user", "content": "You are out of steps. Call final_answer now."})
        reply = llm.chat(messages, tools=[TOOLS[2]] if last else TOOLS)
        messages.append({"role": "assistant", "content": reply})
        call = parse_tool_call(reply)
        if call is None:
            if last:
                trace.append({"step": step, "raw": reply})
                return Episode(reply or NOT_FOUND, [], step, time.perf_counter() - t0, True, trace)
            trace.append({"step": step, "raw": reply, "error": "no tool call"})
            messages.append({"role": "user", "content": "Reply with exactly one tool call."})
            continue
        name, args = call
        if name == "final_answer":
            answer, sources = _finish(env, args)
            trace.append({"step": step, "tool": name, "args": args})
            return Episode(answer, sources, step, time.perf_counter() - t0, last, trace)
        if name == "search_emails":
            result = env.search_emails(**args)
        elif name == "read_email":
            result = env.read_email(**args)
        else:
            result = f"Error: unknown tool '{name}'."
        trace.append({"step": step, "tool": name, "args": args, "result": result[:600]})
        messages.append({"role": "tool", "content": result})
    return Episode(NOT_FOUND, [], max_steps, time.perf_counter() - t0, True, trace)


def rag_answer(llm, env: MailboxEnv, question: str) -> Episode:
    """Single-step baseline: one retrieval with the question, one model call."""
    t0 = time.perf_counter()
    hits = env.search(question)
    shown = "\n\n".join(f"[{env.handle(i)}]\n{env.mail.render(i, max_chars=1200)}" for i in hits) or "(no emails found)"
    messages = [
        {"role": "system", "content": RAG_SYSTEM.format(owner=env.owner, today=env.today)},
        {"role": "user", "content": f"Emails:\n\n{shown}\n\nQuestion: {question}"},
    ]
    reply = llm.chat(messages, tools=[TOOLS[2]])
    call = parse_tool_call(reply)
    trace = [{"step": 1, "retrieved": [env.handle(i) for i in hits], "raw": reply}]
    if call is None or call[0] != "final_answer":
        return Episode(reply or NOT_FOUND, [], 1, time.perf_counter() - t0, True, trace)
    answer, sources = _finish(env, call[1])
    return Episode(answer, sources, 1, time.perf_counter() - t0, False, trace)
