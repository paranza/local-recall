"""Agent loop tests with a scripted fake model, so they need no GPU and no downloads."""

import json

import numpy as np

from local_recall.agent import (
    NOT_FOUND, PUSH_READ_FIRST, PUSH_SEARCH_AGAIN, REPEAT, MailboxEnv, is_abstention,
    parse_final, parse_tool_call, rag_answer, run_agent,
)
from local_recall.agent_eval import token_f1
from local_recall.enron import Mail
from local_recall.snippets import best_snippet


class KeywordEngine:
    """Stand-in for SearchEngine: ranks candidates by how many query words they contain."""

    def __init__(self, texts):
        self.texts = texts

    def search(self, query, k=10, mode="rerank", candidates=None, q_emb=None):
        words = query.lower().split()
        cands = range(len(self.texts)) if candidates is None else candidates
        scored = [(int(i), float(sum(w in self.texts[i].lower() for w in words))) for i in cands]
        return sorted((s for s in scored if s[1] > 0), key=lambda s: -s[1])[:k]


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, max_new_tokens=256):
        self.calls.append((messages[-1], [t["function"]["name"] for t in tools or []]))
        return self.replies.pop(0)


def call(name, **args):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


def final(answer, sources="none"):
    return f"Answer: {answer}\nSources: {sources}"


def make_env(**kwargs):
    mail = Mail(
        ids=["m0", "m1", "m2"],
        dates=np.array(["2000-06-01", "2000-06-20", "2000-07-15"], dtype="datetime64[s]"),
        senders=["ann@x.com", "bob@x.com", "ann@x.com"],
        recipients=[["me@x.com"]] * 3,
        subjects=["budget draft", "video conference", "budget final"],
        bodies=["The budget draft is 10k.", "Conference with Avi on Friday June 30 at 11:30.", "The final budget is 12k."],
        owner_index={"me@x.com": np.array([0, 1, 2])},
    )
    return MailboxEnv(mail, KeywordEngine(mail.texts), "me@x.com", kwargs.pop("today", "2000-06-30"), **kwargs)


def test_parse_tool_call_variants():
    assert parse_tool_call(call("read_email", email_id="E1")) == ("read_email", {"email_id": "E1"})
    assert parse_tool_call('<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>') == ("x", {"a": 1})
    assert parse_tool_call('Sure: {"name": "read_email", "arguments": {"email_id": "E2"}}') == ("read_email", {"email_id": "E2"})
    assert parse_tool_call("just text") is None
    assert parse_tool_call("<tool_call>{not json}</tool_call>") is None


def test_parse_final_variants():
    assert parse_final("Answer: June 30\nSources: E1, e3") == ("June 30", ["E1", "E3"])
    assert parse_final("**Answer:** June 30\n**Sources:** [E2]") == ("June 30", ["E2"])
    assert parse_final("It is June 30, see E4 and E4.") == ("It is June 30, see E4 and E4.", ["E4"])
    answer, sources = parse_final(f"Answer: {NOT_FOUND}\nSources: none")
    assert is_abstention(answer) and sources == []
    assert is_abstention("Not found.") and not is_abstention("June 30")


def test_search_hides_future_emails_and_applies_filters():
    env = make_env()
    assert [env.mail.ids[i] for i in env.search("budget")] == ["m0"]  # m2 is dated after "today"
    assert env.search("budget", sender="bob") == []
    assert env.search("budget", date_from="2000-06-10") == []
    later = make_env(today="2000-12-31")
    assert sorted(later.mail.ids[i] for i in later.search("budget")) == ["m0", "m2"]
    assert [later.mail.ids[i] for i in later.search("budget", date_to="2099-01-01", date_from="2000-07-01")] == ["m2"]


def test_filter_that_matches_nothing_falls_back_and_says_so():
    env = make_env()
    out = env.search_emails(query="budget", sender="avi.marscorp")
    assert out.startswith("No emails matched those filters.") and "[E1]" in out
    assert env.search_emails(query="zebra") == "No emails matched. Try other keywords."


def test_agent_searches_reads_and_cites():
    env = make_env()
    llm = ScriptedLLM([
        call("search_emails", query="video conference Avi"),
        call("read_email", email_id="E1"),
        final("Friday June 30 at 11:30", "E1, E9"),
    ])
    ep = run_agent(llm, env, "When is the conference with Avi?")
    assert ep.answer == "Friday June 30 at 11:30"
    assert [env.mail.ids[i] for i in ep.sources] == ["m1"]  # the unknown id E9 is dropped
    assert ep.steps == 3 and not ep.forced
    assert "Conference with Avi" in ep.trace[1]["result"]


def test_answer_before_opening_any_email_is_sent_back():
    env = make_env()
    llm = ScriptedLLM([
        call("search_emails", query="budget"),
        final("10k", "E1"),  # only saw the preview
        call("read_email", email_id="E1"),
        final("10k", "E1"),
    ])
    ep = run_agent(llm, env, "What is the budget?")
    assert llm.calls[2][0]["content"] == PUSH_READ_FIRST
    assert [env.mail.ids[i] for i in ep.sources] == ["m0"] and ep.steps == 4


def test_giving_up_after_one_search_is_sent_back_once():
    env = make_env()
    llm = ScriptedLLM([call("search_emails", query="zebra"), final(NOT_FOUND), final(NOT_FOUND)])
    ep = run_agent(llm, env, "Where is the zebra?")
    assert llm.calls[2][0]["content"] == PUSH_SEARCH_AGAIN
    assert is_abstention(ep.answer) and ep.sources == [] and ep.steps == 3


def test_identical_repeated_call_is_refused_and_unknown_id_is_reported():
    env = make_env()
    llm = ScriptedLLM([
        call("search_emails", query="budget"), call("search_emails", query="budget"),
        call("read_email", email_id="E7"), final(NOT_FOUND), final(NOT_FOUND),
    ])
    run_agent(llm, env, "What is the budget?")
    assert llm.calls[2][0]["content"] == REPEAT
    assert "unknown email id" in llm.calls[3][0]["content"]


def test_step_limit_forces_an_answer_and_unread_citations_do_not_count():
    env = make_env()
    llm = ScriptedLLM([call("search_emails", query="budget"), call("search_emails", query="budget draft"), final("10k", "E1")])
    ep = run_agent(llm, env, "What is the budget?", max_steps=3)
    assert ep.forced and ep.steps == 3 and ep.answer == "10k"
    assert ep.sources == []  # E1 was never opened
    assert llm.calls[-1][1] == []  # no tools are offered on the last step


def test_hidden_evidence_is_never_shown():
    env = make_env(exclude=frozenset({0}))
    llm = ScriptedLLM([final(NOT_FOUND)])
    ep = rag_answer(llm, env, "budget draft")
    assert ep.trace[0]["retrieved"] == [] and is_abstention(ep.answer)
    assert "(no emails found)" in llm.calls[0][0]["content"]


def test_rag_baseline_cites_shown_emails():
    env = make_env()
    ep = rag_answer(ScriptedLLM([final("10k", "E1")]), env, "budget draft")
    assert [env.mail.ids[i] for i in ep.sources] == ["m0"] and ep.answer == "10k"


def test_token_f1():
    assert token_f1("Friday, June 30 at 11:30 AM", "friday june 30 at 1130 am") == 1.0
    assert token_f1("the budget", "a schedule") == 0.0
    assert 0 < token_f1("June 30", "Friday June 30") < 1


def test_snippet_shows_the_matching_part():
    text = " ".join(["filler"] * 100 + ["the", "meeting", "is", "on", "Friday"] + ["filler"] * 100)
    snippet = best_snippet(text, "when is the meeting", width=20)
    assert "meeting is on Friday" in snippet and snippet.startswith("...") and snippet.endswith("...")
    assert best_snippet("short text", "anything") == "short text"
    assert best_snippet(" ".join(["w"] * 60), "absent", width=10) == " ".join(["w"] * 10) + " ..."
