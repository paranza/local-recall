"""Draw the figures in results/figures from the JSON result files.

Every figure is written twice, for light and dark pages. No number is typed by hand.
This module does not import torch or faiss, so it runs in a second or two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
FIGURES = RESULTS / "figures"

BEIR = (("scifact", "SciFact"), ("nfcorpus", "NFCorpus"), ("fiqa", "FiQA"))
SYSTEMS = (("bm25", "BM25"), ("dense", "Dense (exact)"), ("hybrid", "Hybrid (RRF)"),
           ("hybrid_rerank", "Hybrid + cross-encoder"))
LABELLED_EF = (8, 64, 256)
FONTS = ("Segoe UI", "Helvetica Neue", "Arial")
MARGIN_IN = 0.3  # left edge of the title, subtitle and legend, in inches


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    blue: str
    orange: str


THEMES = (
    Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"),
    Theme("dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "#3987e5", "#d95926"),
)


def _load(name: str, results: Path) -> dict | None:
    path = results / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _figure(t: Theme, ncols: int, width: float, height: float, title: str, subtitle: str, **grid):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False, **grid)
    fig.patch.set_facecolor(t.surface)
    left = MARGIN_IN / width
    fig.text(left, 1 - 0.30 / height, title, color=t.ink, fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(left, 1 - 0.62 / height, subtitle, color=t.ink2, fontsize=10, ha="left", va="top")
    for ax in axes[0]:
        ax.set_facecolor(t.surface)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t.axis)
        ax.tick_params(colors=t.muted, labelsize=9, length=0)
        ax.grid(axis="x", color=t.grid, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, list(axes[0])


def _dot_ci(ax, t: Theme, y: float, mean: float, lo: float, hi: float, color: str) -> None:
    ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot([mean], [y], "o", markersize=9, color=color, markeredgecolor=t.surface, markeredgewidth=1.5, zorder=3)


def _key(t: Theme, color: str, label: str, line: bool = True) -> Line2D:
    return Line2D([0], [0], color=color if line else "none", linewidth=2, marker="o", markersize=8,
                  markerfacecolor=color, markeredgecolor=t.surface, label=label)


def _legend(fig, t: Theme, handles: list[Line2D]) -> None:
    width, height = fig.get_size_inches()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(MARGIN_IN / width, 1 - 0.92 / height), ncols=len(handles),
               frameon=False, fontsize=9.5, labelcolor=t.ink2, handletextpad=0.5, columnspacing=2.0, borderaxespad=0)


def _ms(value: float) -> str:
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def quality_by_stage(t: Theme, results: Path):
    """Dot and interval per system, one panel per benchmark. The highest mean in each panel is highlighted."""
    panels = []
    for key, label in BEIR:
        if res := _load(key, results):
            panels.append((label, f"nDCG@10, {res['n_queries']:,} queries", res["systems"], "ndcg@10"))
    if enron := _load("enron", results):
        panels.append(("Enron mailboxes", f"MRR@10, {enron['n_queries']:,} questions", enron["systems"], "mrr@10"))
    if not panels:
        return None
    fig, axes = _figure(t, len(panels), 11.5, 4.0, "No single retriever wins everywhere",
                        "Mean with 95% bootstrap confidence interval. Each panel has its own scale.",
                        gridspec_kw={"left": 0.165, "right": 0.985, "top": 0.58, "bottom": 0.10, "wspace": 0.16})
    for ax, (label, sub, systems, metric) in zip(axes, panels):
        entries = [systems[k][metric] for k, _ in SYSTEMS]
        best = max(range(len(entries)), key=lambda i: entries[i]["mean"])
        lo = min(e["ci95"][0] for e in entries)
        hi = max(e["ci95"][1] for e in entries)
        pad = (hi - lo) * 0.08
        for i, e in enumerate(entries):
            _dot_ci(ax, t, i, e["mean"], *e["ci95"], t.blue if i == best else t.muted)
        top = entries[best]
        ax.annotate(f"{top['mean']:.3f}", (top["mean"], best), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", color=t.ink, fontsize=9.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(len(entries) - 0.5, -0.75)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_yticks(range(len(entries)), [name for _, name in SYSTEMS] if ax is axes[0] else [""] * len(entries))
        ax.tick_params(axis="y", labelsize=10, labelcolor=t.ink2, pad=8)
        ax.set_title(f"{label}\n", color=t.ink, fontsize=11, fontweight="bold", loc="left", pad=7)
        ax.text(0, 1.035, sub, transform=ax.transAxes, color=t.muted, fontsize=9, ha="left", va="bottom")
    _legend(fig, t, [_key(t, t.blue, "Highest mean on this benchmark"), _key(t, t.muted, "Other systems")])
    return fig


def hnsw_tradeoff(t: Theme, results: Path):
    """Recall of the exact top 10 against search latency, for each HNSW setting and for exact search."""
    panels = [(label, res) for key, label in BEIR if (res := _load(key, results)) and res.get("ann_sweep")]
    if not panels:
        return None
    fig, axes = _figure(t, len(panels), 11.5, 4.5, "Exact search is already fast enough at this size",
                        "Approximate search (HNSW) is faster, but by less than two milliseconds, and it loses recall. "
                        "One query on the CPU, search only.",
                        sharey=True, gridspec_kw={"left": 0.07, "right": 0.985, "top": 0.59, "bottom": 0.15, "wspace": 0.07})
    for ax, (label, res) in zip(axes, panels):
        sweep = sorted(res["ann_sweep"], key=lambda a: a["ef_search"])
        xs = [a["p50_ms"] for a in sweep]
        ys = [a["recall_vs_exact@10"] for a in sweep]
        exact = res["latency_cpu"]["dense_search_exact"]["p50_ms"]
        ax.plot(xs, ys, color=t.blue, linewidth=2, solid_joinstyle="round", zorder=2)
        ax.plot(xs, ys, "o", markersize=8, color=t.blue, markeredgecolor=t.surface, markeredgewidth=1.5, zorder=3)
        ax.plot([exact], [1.0], "o", markersize=9, color=t.orange, markeredgecolor=t.surface, markeredgewidth=1.5,
                zorder=4, clip_on=False)
        for a in sweep:
            if a["ef_search"] in LABELLED_EF:
                ax.annotate(f"ef = {a['ef_search']}", (a["p50_ms"], a["recall_vs_exact@10"]), xytext=(7, -10),
                            textcoords="offset points", ha="left", va="top", color=t.ink2, fontsize=9)
        ax.annotate(f"exact, {exact:.1f} ms", (exact, 1.0), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", color=t.ink, fontsize=9, annotation_clip=False)
        ax.set_xscale("log")
        ax.set_xlim(min(xs) * 0.6, max(max(xs), exact) * 2.6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.tick_params(which="both", length=0)
        ax.grid(axis="y", color=t.grid, linewidth=0.8)
        ax.set_title(f"{label} ({res['n_docs']:,} documents)", color=t.ink, fontsize=11, fontweight="bold",
                     loc="left", pad=22)
    axes[0].set_ylim(0.70, 1.0)
    axes[0].set_ylabel("Recall of the exact top 10", color=t.ink2, fontsize=9.5, labelpad=8)
    axes[len(axes) // 2].set_xlabel("Search latency per query, median (ms, log scale)", color=t.ink2, fontsize=9.5, labelpad=8)
    _legend(fig, t, [_key(t, t.blue, "HNSW, ef from 8 to 256"), _key(t, t.orange, "Exact search", line=False)])
    return fig


def agent_vs_single_step(t: Theme, results: Path):
    """The agent loop against retrieving once, on three quality scores and on time."""
    agent = _load("agent", results)
    if not agent:
        return None
    s = agent["summary"]
    rows = (("Cites a correct source email", "answerable", "source_hit"),
            ("Answer F1", "answerable", "answer_f1"),
            ("Says NOT_FOUND when the\nevidence is removed", "evidence_removed", "abstained"))
    systems = (("rag", t.blue, -0.17), ("agent", t.orange, 0.17))
    fig, (left, right) = _figure(
        t, 2, 11.5, 4.2, "With a small local model, the agent loop did not pay off",
        f"{agent['llm']}, {agent['n_answerable']} questions, {agent['n_evidence_removed']} of them repeated with the "
        "evidence removed. Mean with 95% bootstrap confidence interval.",
        gridspec_kw={"left": 0.205, "right": 0.975, "top": 0.62, "bottom": 0.12, "wspace": 0.12, "width_ratios": [2.1, 1]})
    for i, (_, setting, metric) in enumerate(rows):
        for system, color, offset in systems:
            e = s[f"{setting}/{system}"][metric]
            _dot_ci(left, t, i + offset, e["mean"], *e["ci95"], color)
            left.annotate(f"{e['mean']:.2f}", (e["ci95"][1], i + offset), xytext=(7, 0), textcoords="offset points",
                          ha="left", va="center", color=t.ink2, fontsize=9)
    left.set_xlim(0, 1)
    left.set_ylim(len(rows) - 0.5, -0.5)
    left.set_yticks(range(len(rows)), [label for label, _, _ in rows])
    left.tick_params(axis="y", labelsize=10, labelcolor=t.ink2, pad=8)
    left.set_title("Quality (higher is better)", color=t.ink, fontsize=11, fontweight="bold", loc="left", pad=8)

    top = 0.0
    for system, color, offset in systems:
        e = s[f"answerable/{system}"]["seconds"]
        _dot_ci(right, t, offset * 2.2, e["mean"], *e["ci95"], color)
        right.annotate(f"{e['mean']:.1f} s", (e["ci95"][1], offset * 2.2), xytext=(7, 0), textcoords="offset points",
                       ha="left", va="center", color=t.ink2, fontsize=9)
        top = max(top, e["ci95"][1])
    right.set_xlim(0, top * 1.25)
    right.set_ylim(1.0, -1.0)
    right.set_yticks([])
    right.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    right.set_title("Seconds per question (lower is better)", color=t.ink, fontsize=11, fontweight="bold",
                    loc="left", pad=8)
    _legend(fig, t, [_key(t, t.blue, "Single step: retrieve once, then answer"),
                     _key(t, t.orange, f"Agent: search and read tools, up to {agent['max_steps']} steps")])
    return fig


def latency_by_stage(t: Theme, results: Path):
    """Median and 95th percentile time of each step for one query, on the largest benchmark."""
    found = [(label, res) for key, label in BEIR if (res := _load(key, results))]
    if not found:
        return None
    label, res = max(found, key=lambda item: item[1]["n_docs"])
    lat = res["latency_cpu"]
    steps = [("BM25 search, top 100", lat.get("bm25")),
             ("Embed the query (PyTorch fp32)", lat.get("encode_query_torch_fp32")),
             ("Exact vector search", lat.get("dense_search_exact")),
             ("Cross-encoder, 50 candidates", lat.get("rerank_top50"))]
    if hnsw := next((a for a in res.get("ann_sweep", []) if a["ef_search"] == 64), None):
        steps.insert(3, ("HNSW vector search, ef = 64", hnsw))
    steps = [(name, entry) for name, entry in steps if entry]
    fig, (ax,) = _figure(t, 1, 9.0, 3.6, "The re-ranker takes almost all of the time",
                         f"One query on the CPU, {label} ({res['n_docs']:,} documents). The dot is the median and the line "
                         "runs to the 95th percentile.",
                         gridspec_kw={"left": 0.29, "right": 0.955, "top": 0.74, "bottom": 0.17})
    for i, (_, e) in enumerate(steps):
        ax.plot([e["p50_ms"], e["p95_ms"]], [i, i], color=t.blue, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([e["p50_ms"]], [i], "o", markersize=9, color=t.blue, markeredgecolor=t.surface, markeredgewidth=1.5, zorder=3)
        ax.annotate(_ms(e["p50_ms"]), (e["p95_ms"], i), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                    color=t.ink, fontsize=9.5)
    lo = min(e["p50_ms"] for _, e in steps)
    hi = max(e["p95_ms"] for _, e in steps)
    ax.set_xscale("log")
    ax.set_xlim(lo * 0.5, hi * 6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.tick_params(which="both", length=0)
    ax.set_ylim(len(steps) - 0.5, -0.5)
    ax.set_yticks(range(len(steps)), [name for name, _ in steps])
    ax.tick_params(axis="y", labelsize=10, labelcolor=t.ink2, pad=8)
    ax.set_xlabel("Time per query (ms, log scale)", color=t.ink2, fontsize=9.5, labelpad=8)
    return fig


CHARTS = (("quality_by_stage", quality_by_stage), ("hnsw_tradeoff", hnsw_tradeoff),
          ("agent_vs_single_step", agent_vs_single_step), ("latency_by_stage", latency_by_stage))


def main(results: Path = RESULTS, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((name for name in FONTS if name in installed), "DejaVu Sans")
    with plt.rc_context({"font.family": family}):
        for name, draw in CHARTS:
            for theme in THEMES:
                fig = draw(theme, results)
                if fig is None:
                    continue
                path = out_dir / f"{name}_{theme.name}.png"
                fig.savefig(path, dpi=160, facecolor=theme.surface, metadata={"Software": None})
                plt.close(fig)
                written.append(path)
    print(f"wrote {len(written)} figures to {out_dir}")
    return written


if __name__ == "__main__":
    main()
