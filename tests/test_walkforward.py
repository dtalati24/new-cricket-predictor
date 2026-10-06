# tests/test_walkforward.py
import json

import numpy as np
import pandas as pd
import pytest

from src.config import MARKET
from src.walkforward import summarize


def _write_year(d, year, chosen, n=40, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.normal(50, 12, n)
    df = pd.DataFrame({"match_id": [f"{year}-{i}" for i in range(n)], "line": 50.5,
                       "actual": actual, "cal_median": actual + rng.normal(0, 9, n),
                       "side": ["over"] * (n // 2) + [None] * (n - n // 2),
                       "pnl": [0.95] * (n // 4) + [-1.0] * (n // 4) + [0.0] * (n - n // 2)})
    df.to_parquet(d / f"{year}_{chosen}_matches.parquet")
    (d / f"{year}.json").write_text(json.dumps({"year": year, "chosen": chosen}))


def test_summarize_pools_chosen_models_and_splits_clean_years(tmp_path):
    _write_year(tmp_path, 2023, "logistic", seed=1)
    _write_year(tmp_path, 2024, "lightgbm", seed=2)
    _write_year(tmp_path, 2025, "logistic", seed=3)
    s = summarize(out_dir=tmp_path, years=(2023, 2024, 2025, 2026))
    assert set(s["per_year"]) == {"2023", "2024", "2025"}
    assert s["pooled"]["n_matches"] == 120
    assert s["pooled_2023_2024"]["n_matches"] == 80
    p = s["pooled"]
    assert p["n_bets"] == 60 and p["roi"] == pytest.approx((0.95 * 30 - 30) / 60)
    assert p["roi_se"] > 0
    assert (tmp_path / "summary.json").exists()


@pytest.mark.skipif(not (MARKET / "overline_comparison.parquet").exists(), reason="market tables not built")
def test_year_rows_match_the_known_2024_count():
    from src.walkforward import year_rows
    r = year_rows(2024)
    assert len(r) == 435
    assert (r.date.dt.year == 2024).all()


def test_failed_comparison_model_is_recorded_and_chosen_model_is_kept(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import src.walkforward as wf

    rng = np.random.default_rng(0)

    def rows(year, n=30):
        return pd.DataFrame({"match_id": [f"{year}-{i}" for i in range(n)],
                             "date": pd.date_range(f"{year}-01-01", periods=n),
                             "pre_match_line": 50.5, "runs_at_6ov": rng.normal(50, 10, n)})

    n_val, n_test = 20, 30
    monkeypatch.setattr(wf, "year_rows", rows)
    monkeypatch.setattr(wf, "check_xi", lambda r, xi: None)
    monkeypatch.setattr(wf.pd, "read_parquet", lambda *a, **k: None)
    monkeypatch.setattr(wf, "MODELS", ("good", "bad"))
    monkeypatch.setattr(wf, "feature_columns", lambda levels: ["a"])
    monkeypatch.setattr(wf, "choose_model", lambda results: "good")
    monkeypatch.setattr(wf, "prepare_cached", lambda cutoff: (
        SimpleNamespace(skill=SimpleNamespace(levels=[])),
        pd.DataFrame({"a": np.arange(10.0), "y": 0, "date": pd.date_range("2000-01-01", periods=10)}),
        None))
    monkeypatch.setattr(wf, "fit_model", lambda name, X, y, dates, cols: name)

    def fake_sim(r, ctx, model, table, n, workers=1):
        if model == "bad" and n == n_test:
            raise RuntimeError("boom")
        return [rng.normal(50, 10, 200) for _ in r.runs_at_6ov]

    monkeypatch.setattr(wf, "simulate_matches", fake_sim)
    rep = wf.run_fold(2024, n_val=n_val, n_test=n_test, out_dir=tmp_path)
    assert rep["chosen"] == "good"
    assert (tmp_path / "2024_good_matches.parquet").exists()
    assert not (tmp_path / "2024_bad_matches.parquet").exists()
    saved = json.loads((tmp_path / "2024.json").read_text())
    assert saved["test"]["good"]["role"] == "chosen"
    assert "boom" in saved["test"]["bad"]["error"]
    assert saved["test"]["bad"]["role"] == "comparison"
