"""The figures are drawn from the committed result files, in both themes."""

from __future__ import annotations

import pytest

from local_recall import plots


def test_every_chart_is_written_in_both_themes(tmp_path):
    if not (plots.RESULTS / "fiqa.json").exists() or not (plots.RESULTS / "agent.json").exists():
        pytest.skip("result files not present")
    written = plots.main(out_dir=tmp_path)
    names = {path.name for path in written}
    assert names == {f"{chart}_{theme.name}.png" for chart, _ in plots.CHARTS for theme in plots.THEMES}
    assert all(path.stat().st_size > 10_000 for path in written)


def test_missing_results_draw_nothing(tmp_path):
    assert plots.main(results=tmp_path / "empty", out_dir=tmp_path / "figures") == []
