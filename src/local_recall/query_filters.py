"""Query understanding: pull structured filters out of a free-text query.

"emails from greg whalley in June 2000 about the QQQ short"
  -> sender="greg.whalley", date_from=2000-06-01, date_to=2000-06-30,
     text="emails about the QQQ short"

This is a small rule-based parser on purpose. The rules are easy to read, to
test and to explain, and they never invent a filter that is not in the query.
A wrong hard filter hides the right answer completely, so the parser only acts
on patterns it is sure about and leaves everything else in the text for the
ranker.

Supported:
  sender   "from <email>" anywhere; "emails from <name>", "sent by <name>", or a query
           that starts with "from <name>"
  dates    "in June 2000", "in 2001", "on 2000-12-13", "before / after / since <date>",
           "between <date> and <date>", "yesterday", "today", "last week", "last month"
A <date> is an ISO date (2000-12-13), a month and year (June 2000) or a year.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_RE = rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|(?:{_MONTH_RE})\.?\s+\d{{4}}|\d{{4}})"

_NAME_STOP = {"in", "on", "before", "after", "since", "between", "about", "regarding", "re", "with", "to",
              "last", "yesterday", "today", "that", "which", "and", "or", "the", "a", "an", "me", "my"}


@dataclass
class ParsedQuery:
    text: str
    sender: str | None = None
    date_from: date | None = None
    date_to: date | None = None


def _span(token: str) -> tuple[date, date] | None:
    """First and last day covered by a date expression."""
    token = token.strip().lower().replace(".", "")
    if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", token):
        try:
            d = date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
        return d, d
    if m := re.fullmatch(rf"({_MONTH_RE})\s+(\d{{4}})", token):
        year, month = int(m[2]), _MONTHS[m[1]]
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    if m := re.fullmatch(r"(19|20)\d{2}", token):
        year = int(token)
        return date(year, 1, 1), date(year, 12, 31)
    return None


def parse_query(query: str, today: date | None = None) -> ParsedQuery:
    today = today or date.today()
    text = query
    out = ParsedQuery(text=query)

    def cut(match: re.Match) -> None:
        nonlocal text
        text = text[: match.start()] + " " + text[match.end():]

    flags = re.IGNORECASE

    # --- sender ---
    if m := re.search(r"\b(?:from|sent by)\s+([\w.+-]+@[\w.-]+)", text, flags):
        out.sender = m[1].lower()
        cut(m)
    elif m := re.search(
        # A bare "from <words>" is ambiguous ("Avi from Mars Corp."), so a name only counts as a
        # sender when the query starts with "from", or says "emails from ..." or "sent by ...".
        r"(?:^\s*|\b(?:e-?mails?|mails?|messages?|sent|received)\s+)(from\s+|by\s+)((?:[A-Za-z][\w'-]*\s*){1,3})",
        text,
        flags,
    ):
        raw = m[2].split()
        words = []
        for w in raw:
            if w.lower() in _NAME_STOP or w.lower() in _MONTHS:
                break
            words.append(w.lower())
        if words:
            out.sender = ".".join(words)  # matches addresses like greg.whalley@enron.com
            consumed = re.match(r"\s*".join(re.escape(w) for w in raw[: len(words)]), m[2])
            start = m.start() if m[1].lower().startswith("by") else m.start(1)
            text = text[:start] + " " + text[m.start(2) + consumed.end():]

    # --- date ranges ---
    if m := re.search(rf"\bbetween\s+({_DATE_RE})\s+and\s+({_DATE_RE})", text, flags):
        a, b = _span(m[1]), _span(m[2])
        if a and b:
            out.date_from, out.date_to = a[0], b[1]
            cut(m)
    if out.date_from is None and out.date_to is None:
        for pattern, kind in (
            (rf"\b(?:before|until|prior to)\s+({_DATE_RE})", "before"),
            (rf"\b(?:after|since)\s+({_DATE_RE})", "after"),
            (rf"\b(?:in|on|during)\s+({_DATE_RE})", "in"),
        ):
            if (m := re.search(pattern, text, flags)) and (span := _span(m[1])):
                if kind == "before":
                    out.date_to = span[0] - timedelta(days=1)
                elif kind == "after":
                    out.date_from = span[0] if m[0].lower().startswith("since") else span[1] + timedelta(days=1)
                else:
                    out.date_from, out.date_to = span
                cut(m)
                break

    # --- relative dates ---
    if out.date_from is None and out.date_to is None:
        if m := re.search(r"\byesterday\b", text, flags):
            out.date_from = out.date_to = today - timedelta(days=1)
            cut(m)
        elif m := re.search(r"\btoday\b", text, flags):
            out.date_from = out.date_to = today
            cut(m)
        elif m := re.search(r"\blast week\b", text, flags):
            monday = today - timedelta(days=today.weekday() + 7)
            out.date_from, out.date_to = monday, monday + timedelta(days=6)
            cut(m)
        elif m := re.search(r"\blast month\b", text, flags):
            end = today.replace(day=1) - timedelta(days=1)
            out.date_from, out.date_to = end.replace(day=1), end
            cut(m)

    out.text = re.sub(r"\s+", " ", text).strip(" ,.;")
    return out
