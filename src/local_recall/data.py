"""Loaders for the BEIR benchmark files (corpus.jsonl, queries.jsonl, qrels/test.tsv)."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Dataset:
    name: str
    doc_ids: list[str]
    doc_texts: list[str]  # title and body joined, the BEIR convention
    queries: dict[str, str]  # only queries that have test judgments
    qrels: dict[str, dict[str, int]]


def download_beir(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download and unzip one BEIR dataset. Skips work that is already done."""
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    with requests.get(BEIR_URL.format(name=name), stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    return target


def load_beir(name: str, split: str = "test", data_dir: Path = DATA_DIR) -> Dataset:
    root = download_beir(name, data_dir)

    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, grade in reader:
            qrels.setdefault(qid, {})[did] = int(grade)

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    with open(root / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            doc_ids.append(row["_id"])
            title = (row.get("title") or "").strip()
            body = (row.get("text") or "").strip()
            doc_texts.append(f"{title} {body}".strip())

    queries: dict[str, str] = {}
    with open(root / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    return Dataset(name=name, doc_ids=doc_ids, doc_texts=doc_texts, queries=queries, qrels=qrels)
