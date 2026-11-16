"""A tool-calling agent that answers questions about one person's mailbox.

The language model runs locally, so no email text leaves the machine. It gets
two tools:
  search_emails(query, sender?, date_from?, date_to?)  ranked hits with previews
  read_email(email_id)                                  the full text of one hit

The loop follows how chat models are trained to use tools: the model calls
tools for as long as it needs, and its first reply that is not a tool call is
the final answer. That answer must end with a "Sources:" line naming the
emails it rests on, or say NOT_FOUND.

The environment, not the prompt, enforces the grounding rules, because a small
model does not reliably follow instructions:
  - An email can be cited only after read_email showed its full text.
  - An answer given before any email was opened is sent back.
  - Giving up after a single search is sent back once.
  - A filter that matches nothing falls back to an unfiltered search and says
    so. Models invent filters (a sender "avi.marscorp"), and a wrong hard
    filter would otherwise hide the right email completely.
  - An identical repeated call is refused.

`rag_answer` is the single-step baseline: retrieve once with the question,
show the top emails in full, ask for the answer in the same format.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .engine import SearchEngine
from .enron import Mail
from .snippets import best_snippet

DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
NOT_FOUND = "NOT_FOUND"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search the mailbox. Returns the best matching emails with id, sent date, sender, subject and a short preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or a short description of what to find."},
                    "sender": {"type": "string", "description": "Optional. Part of the sender's email address, e.g. 'greg.whalley'. Only use it when you are sure who sent the email."},
                    "date_from": {"type": "string", "description": "Optional. Earliest sent date, YYYY-MM-DD."},
                    "date_to": {"type": "string", "description": "Optional. Latest sent date, YYYY-MM-DD."},
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
]

ANSWER_FORMAT = (
    "Final answer format, exactly two lines:\n"
    "Answer: <short, direct answer>\n"
    "Sources: <ids of the emails that support it, e.g. E2, E5>\n"
    f"If the emails do not contain the answer, write 'Answer: {NOT_FOUND}' and 'Sources: none'. Never guess."
)

SYSTEM = (
    "You answer questions about the mailbox of {owner}. Today is {today}.\n"
    "How to work:\n"
    "1. Call search_emails to find candidates. Previews are short, and the 'sent' date is only when the email was sent.\n"
    "2. Call read_email to open the most promising email. You may only cite emails you have opened.\n"
    "3. If it does not answer the question, open another one or search again with different keywords.\n"
    "4. When you know the answer, reply without calling a tool.\n\n" + ANSWER_FORMAT
)

RAG_SYSTEM = (
    "You answer a question about the mailbox of {owner} using only the emails shown. Today is {today}.\n\n" + ANSWER_FORMAT
)

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.DOTALL)
_HANDLE = re.compile(r"\bE\d+\b", re.IGNORECASE)
_SOURCES_LINE = re.compile(r"^\W*sources?\W*:(.*)$", re.IGNORECASE | re.MULTILINE)
_ANSWER_PREFIX = re.compile(r"^\W*(?:final\s+)?answer\W*:[\s*]*", re.IGNORECASE)


class LocalLLM:
    """A chat model loaded with transformers. Greedy decoding, so runs are repeatable."""

    def __init__(self, model_name: str = DEFAULT_LLM, device: str | None = None, load_in_4bit: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, quantization_config=quant)
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
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


def _first_json_object(text: str) -> dict | None:
    """The first {...} in `text` that parses as a JSON object."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract the first {"name": ..., "arguments": {...}} tool call from model output.

    Small models sometimes drop the <tool_call> tags and print the bare JSON,
    so a JSON object with a "name" key is accepted with or without the tags.
    """
    m = _TOOL_CALL.search(text)
    call = None
    if m:
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            call = None
    if call is None:
        call = _first_json_object(text)
    if not isinstance(call, dict) or "name" not in call:
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


def parse_final(text: str) -> tuple[str, list[str]]:
    """Split a final reply into (answer text, cited ids such as ["E2", "E5"])."""
    m = _SOURCES_LINE.search(text)
    if m:
        handles = [h.upper() for h in _HANDLE.findall(m.group(1))]
        body = text[: m.start()]
    else:
        handles = [h.upper() for h in _HANDLE.findall(text)]
        body = text
    answer = _ANSWER_PREFIX.sub("", body.strip()).strip()
    return answer or NOT_FOUND, list(dict.fromkeys(handles))


def is_abstention(answer: str) -> bool:
    return bool(re.search(r"not[_ ]found", answer, re.IGNORECASE))


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
    seen_in_full: set[int] = field(default_factory=set)  # docs whose full text the model was shown
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
        note = ""
        if not hits and (sender or date_from or date_to):
            hits = self.search(query)
            if hits:
                note = "No emails matched those filters. Results without the filters:\n"
        if not hits:
            return "No emails matched. Try other keywords."
        lines = [
            f"[{self.handle(i)}] sent {str(self.mail.dates[i])[:10]} | from {self.mail.senders[i]} | "
            f"subject: {self.mail.subjects[i][:90]} | preview: {best_snippet(self.mail.bodies[i], query)}"
            for i in hits
        ]
        return note + "\n".join(lines)

    def read_email(self, email_id: str = "", **_) -> str:
        doc = self.handles.get(str(email_id).strip().strip("[]").upper())
        if doc is None:
            return f"Error: unknown email id '{email_id}'. Use an id returned by search_emails."
        self.seen_in_full.add(doc)
        return f"[{self.handle(doc)}]\n" + self.mail.render(doc, max_chars=2500)


@dataclass
class Episode:
    answer: str
    sources: list[int]  # doc indices, only emails the model was shown in full
    steps: int  # model calls
    seconds: float
    forced: bool  # True when the step limit ended the episode
    trace: list[dict]


PUSH_READ_FIRST = "You have not opened any email yet. Call read_email on the most promising result before you answer."
PUSH_SEARCH_FIRST = "You have not searched the mailbox yet. Call search_emails first."
PUSH_UNREAD ="You cited {ids} without opening it. A preview is not enough. Call read_email on it first."
PUSH_SEARCH_AGAIN = "Do not give up after one search. Search once more with different, shorter keywords and no filters."
PUSH_SOURCES = "Add the 'Sources:' line with the ids of the opened emails that support your answer."
REPEAT = "You already made this exact call. Open one of the results with read_email, or change the query."


def run_agent(llm, env: MailboxEnv, question: str, max_steps: int = 6) -> Episode:
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM.format(owner=env.owner, today=env.today)},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    seen_calls: set[str] = set()
    pushed: set[str] = set()
    searches = 0

    def done(answer: str, handles: list[str], step: int, forced: bool) -> Episode:
        docs = [env.handles[h] for h in handles if h in env.handles]
        sources = [d for d in docs if d in env.seen_in_full]
        return Episode(answer, sources, step, time.perf_counter() - t0, forced, trace)

    for step in range(1, max_steps + 1):
        last = step == max_steps
        if last:
            messages.append({"role": "user", "content": "You are out of steps. Give your final answer now, in the two-line format."})
        reply = llm.chat(messages, tools=None if last else TOOLS)
        messages.append({"role": "assistant", "content": reply})
        call = None if last else parse_tool_call(reply)

        if call is None:
            answer, handles = parse_final(reply)
            trace.append({"step": step, "final": reply})
            if last:
                return done(answer, handles, step, True)
            cited = [env.handles[h] for h in handles if h in env.handles]
            unread = [h for h in handles if h in env.handles and env.handles[h] not in env.seen_in_full]
            push = None
            if is_abstention(answer):
                if searches < 2 and "search_again" not in pushed:
                    push = ("search_again", PUSH_SEARCH_AGAIN)
            elif not env.seen_in_full:
                push = ("read_first", PUSH_READ_FIRST if env.handles else PUSH_SEARCH_FIRST)
            elif unread and not any(d in env.seen_in_full for d in cited):
                push = ("unread", PUSH_UNREAD.format(ids=", ".join(unread)))
            elif not cited and "sources" not in pushed:
                push = ("sources", PUSH_SOURCES)
            if push is None:
                return done(answer, handles, step, False)
            pushed.add(push[0])
            trace[-1]["pushed_back"] = push[0]
            messages.append({"role": "user", "content": push[1]})
            continue

        name, args = call
        key = json.dumps([name, args], sort_keys=True)
        if key in seen_calls:
            result = REPEAT
        elif name == "search_emails":
            searches += 1
            result = env.search_emails(**args)
        elif name == "read_email":
            result = env.read_email(**args)
        else:
            result = f"Error: unknown tool '{name}'. The tools are search_emails and read_email."
        seen_calls.add(key)
        trace.append({"step": step, "tool": name, "args": args, "result": result[:600]})
        messages.append({"role": "tool", "content": result})
    return Episode(NOT_FOUND, [], max_steps, time.perf_counter() - t0, True, trace)


def rag_answer(llm, env: MailboxEnv, question: str) -> Episode:
    """Single-step baseline: one retrieval with the question, one model call."""
    t0 = time.perf_counter()
    hits = env.search(question)
    env.seen_in_full.update(hits)  # the baseline is shown every retrieved email in full
    shown = "\n\n".join(f"[{env.handle(i)}]\n{env.mail.render(i, max_chars=1200)}" for i in hits) or "(no emails found)"
    messages = [
        {"role": "system", "content": RAG_SYSTEM.format(owner=env.owner, today=env.today)},
        {"role": "user", "content": f"Emails:\n\n{shown}\n\nQuestion: {question}"},
    ]
    reply = llm.chat(messages)
    answer, handles = parse_final(reply)
    sources = [env.handles[h] for h in handles if h in env.handles]
    trace = [{"step": 1, "retrieved": [env.handle(i) for i in hits], "final": reply}]
    return Episode(answer, sources, 1, time.perf_counter() - t0, False, trace)
