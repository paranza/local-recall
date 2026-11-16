"""Personal-search benchmark built from public Enron data.

Two public datasets are joined:
  corbt/enron-emails                   the email corpus (about 517k messages)
  corbt/enron_emails_sample_questions  questions a mailbox owner might ask, each with
                                       an answer, the source message ids, the owner's
                                       address and the date the question is asked

Each question is scoped to one person's mailbox, which makes this a personal
search task: find the right email among the 7k to 23k messages that person
sent or received. The questions were written by a language model from the
source emails (see the dataset card), so they are synthetic, and each one
comes with a realism score.

Cleaning done here:
  - keep only emails sent or received by one of the mailbox owners in the test split
  - merge exact duplicates (same sender, subject and body). The Enron corpus stores
    the same message in several folders under different ids, so without this step a
    system is punished for returning a copy of the right email.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .data import DATA_DIR

ENRON_DIR = DATA_DIR / "enron"
CORPUS_REPO = "corbt/enron-emails"
QUESTIONS_REPO = "corbt/enron_emails_sample_questions"
MAX_BODY_CHARS = 6000

_WS = re.compile(r"\s+")


def _clean_list(values) -> list[str]:
    return [v for v in values if v]


def build(force: bool = False) -> Path:
    """Download, filter and de-duplicate. Writes emails.parquet and questions.parquet."""
    emails_path = ENRON_DIR / "emails.parquet"
    questions_path = ENRON_DIR / "questions.parquet"
    if emails_path.exists() and questions_path.exists() and not force:
        return ENRON_DIR
    ENRON_DIR.mkdir(parents=True, exist_ok=True)

    questions = pd.read_parquet(
        hf_hub_download(QUESTIONS_REPO, "data/test-00000-of-00001.parquet", repo_type="dataset")
    )
    owners = set(questions["inbox_address"])

    frames = []
    for shard in range(3):
        df = pd.read_parquet(
            hf_hub_download(CORPUS_REPO, f"data/train-{shard:05d}-of-00003.parquet", repo_type="dataset"),
            columns=["message_id", "subject", "from", "to", "cc", "bcc", "date", "body"],
        )
        people = [
            {f, *t, *c, *b} & owners
            for f, t, c, b in zip(df["from"], df["to"], df["cc"], df["bcc"])
        ]
        keep = np.array([bool(p) for p in people])
        df = df[keep].copy()
        df["owners"] = [sorted(p) for p, k in zip(people, keep) if k]
        frames.append(df)
    emails = pd.concat(frames, ignore_index=True)

    # Merge exact duplicates. The first copy becomes the canonical one.
    key = (
        emails["from"].fillna("")
        + "\x1f" + emails["subject"].fillna("")
        + "\x1f" + emails["body"].fillna("").map(lambda b: _WS.sub(" ", b).strip())
    )
    emails["canonical"] = emails.groupby(key, sort=False)["message_id"].transform("first")
    to_canonical = dict(zip(emails["message_id"], emails["canonical"]))
    owners_by_canonical = emails.groupby("canonical")["owners"].agg(lambda rows: sorted({o for r in rows for o in r}))
    first_date = emails.groupby("canonical")["date"].min()

    unique = emails[emails["message_id"] == emails["canonical"]].copy()
    unique["owners"] = unique["message_id"].map(owners_by_canonical)
    unique["date"] = unique["message_id"].map(first_date)
    out = pd.DataFrame(
        {
            "id": unique["message_id"],
            "date": unique["date"],
            "sender": unique["from"].fillna(""),
            "to": [_clean_list([*t, *c, *b]) for t, c, b in zip(unique["to"], unique["cc"], unique["bcc"])],
            "subject": unique["subject"].fillna(""),
            "body": unique["body"].fillna("").str.slice(0, MAX_BODY_CHARS),
            "owners": unique["owners"],
        }
    ).sort_values("date", kind="stable").reset_index(drop=True)
    out.to_parquet(emails_path)

    questions["gold"] = [
        sorted({to_canonical[m] for m in ids if m in to_canonical}) for ids in questions["message_ids"]
    ]
    questions = questions[questions["gold"].map(len) > 0]
    questions = questions.rename(columns={"inbox_address": "owner"})[
        ["id", "question", "answer", "gold", "owner", "query_date", "how_realistic"]
    ].reset_index(drop=True)
    questions.to_parquet(questions_path)
    return ENRON_DIR


@dataclass
class Mail:
    """The email corpus in column form, plus the lookups search needs."""

    ids: list[str]
    dates: np.ndarray  # datetime64[s], sorted ascending
    senders: list[str]
    recipients: list[list[str]]
    subjects: list[str]
    bodies: list[str]
    owner_index: dict[str, np.ndarray]  # owner address -> sorted doc indices in that mailbox

    @property
    def texts(self) -> list[str]:
        """What gets indexed: subject, then body."""
        return [f"{s}\n{b}".strip() for s, b in zip(self.subjects, self.bodies)]

    def candidates(
        self,
        owner: str | None = None,
        *,
        sender: str | None = None,
        date_from: np.datetime64 | None = None,
        date_to: np.datetime64 | None = None,
    ) -> np.ndarray:
        """Doc indices that pass the structured filters. `date_to` is inclusive of that whole day."""
        idx = self.owner_index[owner] if owner else np.arange(len(self.ids))
        if date_from is not None:
            idx = idx[self.dates[idx] >= date_from]
        if date_to is not None:
            idx = idx[self.dates[idx] < date_to + np.timedelta64(1, "D")]
        if sender:
            needle = sender.lower()
            idx = np.array([i for i in idx if needle in self.senders[i].lower()], dtype=np.int64)
        return idx

    def render(self, i: int, max_chars: int = 1500) -> str:
        date = str(self.dates[i])[:10]
        to = ", ".join(self.recipients[i][:4])
        return (
            f"Date: {date}\nFrom: {self.senders[i]}\nTo: {to}\n"
            f"Subject: {self.subjects[i]}\n\n{self.bodies[i][:max_chars]}"
        )


def load_mail() -> tuple[Mail, pd.DataFrame]:
    root = build()
    df = pd.read_parquet(root / "emails.parquet")
    owner_lists: dict[str, list[int]] = {}
    for i, owners in enumerate(df["owners"]):
        for o in owners:
            owner_lists.setdefault(o, []).append(i)
    mail = Mail(
        ids=df["id"].tolist(),
        dates=df["date"].dt.tz_localize(None).to_numpy().astype("datetime64[s]"),
        senders=df["sender"].tolist(),
        recipients=[list(t) for t in df["to"]],
        subjects=df["subject"].tolist(),
        bodies=df["body"].tolist(),
        owner_index={o: np.asarray(v, dtype=np.int64) for o, v in owner_lists.items()},
    )
    questions = pd.read_parquet(root / "questions.parquet")
    questions["gold"] = questions["gold"].map(list)
    return mail, questions
