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
    assert s["missing_years"] == [2026]


def test_summarize_reports_a_year_whose_chosen_parquet_is_missing(tmp_path, capsys):
    _write_year(tmp_path, 2023, "logistic", seed=1)
    _write_year(tmp_path, 2024, "lightgbm", seed=2)
    (tmp_path / "2024_lightgbm_matches.parquet").unlink()
    s = summarize(out_dir=tmp_path, years=(2023, 2024))
    assert s["missing_years"] == [2024]
    assert "2024" in capsys.readouterr().out
    assert json.loads((tmp_path / "summary.json").read_text())["missing_years"] == [2024]


@pytest.mark.skipif(not (MARKET / "overline_comparison.parquet").exists(), reason="market tables not built")
def test_year_rows_match_the_known_2024_count():
    from src.walkforward import year_rows
    r = year_rows(2024)
    assert len(r) == 435
    assert (r.date.dt.year == 2024).all()


def _patch_fold(monkeypatch, fail_bad=False):
    """Offline run_fold: two fake models, counts of setup builds and the setups each simulation got."""
    from types import SimpleNamespace

    import src.walkforward as wf

    rng = np.random.default_rng(0)
    calls = {"build": [], "sim": []}

    def rows(year, n=30):
        return pd.DataFrame({"match_id": [f"{year}-{i}" for i in range(n)],
                             "date": pd.date_range(f"{year}-01-01", periods=n),
                             "pre_match_line": 50.5, "runs_at_6ov": rng.normal(50, 10, n)})

    monkeypatch.setattr(wf, "year_rows", rows)
    monkeypatch.setattr(wf, "check_xi", lambda r, xi: None)
    monkeypatch.setattr(wf.pd, "read_parquet", lambda *a, **k: None)
    monkeypatch.setattr(wf, "MODELS", ("good", "bad"))
    monkeypatch.setattr(wf, "feature_columns", lambda levels: ["a"])
    monkeypatch.setattr(wf, "choose_model", lambda results: "good")
    monkeypatch.setattr(wf, "run_table", lambda table: "rt")
    monkeypatch.setattr(wf, "prepare_cached", lambda cutoff: (
        SimpleNamespace(skill=SimpleNamespace(levels=[])),
        pd.DataFrame({"a": np.arange(10.0), "y": 0, "date": pd.date_range("2000-01-01", periods=10)}),
        None))
    monkeypatch.setattr(wf, "fit_model", lambda name, X, y, dates, cols: name)

    def fake_build(r, ctx):
        calls["build"].append(len(r))
        return [object() for _ in range(len(r))]

    def fake_sim(setups, model, table, cols, n, workers=1):
        calls["sim"].append((model, n, setups))
        if fail_bad and model == "bad" and n == 30:
            raise RuntimeError("boom")
        return [rng.normal(50, 10, 200) for _ in setups]

    monkeypatch.setattr(wf, "build_setups", fake_build)
    monkeypatch.setattr(wf, "simulate_setups", fake_sim)
    return wf, calls


def test_failed_comparison_model_is_recorded_and_chosen_model_is_kept(tmp_path, monkeypatch):
    wf, _ = _patch_fold(monkeypatch, fail_bad=True)
    rep = wf.run_fold(2024, n_val=20, n_test=30, out_dir=tmp_path)
    assert rep["chosen"] == "good"
    assert (tmp_path / "2024_good_matches.parquet").exists()
    assert not (tmp_path / "2024_bad_matches.parquet").exists()
    saved = json.loads((tmp_path / "2024.json").read_text())
    assert saved["test"]["good"]["role"] == "chosen"
    assert "boom" in saved["test"]["bad"]["error"]
    assert saved["test"]["bad"]["role"] == "comparison"


def test_setups_are_built_once_per_phase_not_once_per_model(tmp_path, monkeypatch):
    wf, calls = _patch_fold(monkeypatch)
    wf.run_fold(2024, n_val=20, n_test=30, out_dir=tmp_path)
    assert calls["build"] == [30, 30]  # validation year once, test year once
    val = [s for m, n, s in calls["sim"] if n == 20]
    test = [s for m, n, s in calls["sim"] if n == 30]
    assert len(val) == 2 and len(test) == 2
    assert val[0] is val[1] and test[0] is test[1]


def test_models_option_limits_what_is_fitted_and_scored(tmp_path, monkeypatch):
    wf, calls = _patch_fold(monkeypatch)
    rep = wf.run_fold(2024, n_val=20, n_test=30, out_dir=tmp_path, models=("good",))
    assert {m for m, _, _ in calls["sim"]} == {"good"}
    assert set(rep["test"]) == {"good"}
    with pytest.raises(ValueError):
        wf.run_fold(2024, out_dir=tmp_path, models=("nope",))


def test_base_feature_set_is_the_first_models_inputs():
    from src.features.context import CONTEXT
    from src.model.frame import FORM, PP_FORM, SITUATION
    from src.skill.shrink import LEVELS
    from src.walkforward import feature_columns, feature_set_columns
    full, base = feature_set_columns(LEVELS, "full"), feature_set_columns(LEVELS, "base")
    assert full == feature_columns(LEVELS)
    assert set(base) < set(full)
    assert not set(base) & set(CONTEXT)
    assert not any(c.startswith("competition__") for c in base)
    assert not set(base) & set(PP_FORM)
    assert not any(c.startswith(("batter_x_phase__", "bowler_x_phase__")) for c in base)
    assert len(full) - len(base) == len(CONTEXT) + len(PP_FORM) + 3 * 11
    assert (set(FORM) - set(PP_FORM)) | set(SITUATION) <= set(base)
    assert any(c.startswith("hint__") for c in base) and any(c.startswith("batter__") for c in base)
    with pytest.raises(ValueError):
        feature_set_columns(LEVELS, "nope")


def test_base_fold_writes_to_base_folder_and_fits_base_columns(tmp_path, monkeypatch):
    wf, calls = _patch_fold(monkeypatch)
    fit_cols = []
    monkeypatch.setattr(wf, "feature_columns", lambda levels: ["a", "context_x", "competition__dot"])
    monkeypatch.setattr(wf, "CONTEXT", ["context_x"])
    monkeypatch.setattr(wf, "CLASSES", ["dot"])
    monkeypatch.setattr(wf, "fit_model", lambda name, X, y, dates, cols: fit_cols.append(list(cols)) or name)
    rep = wf.run_fold(2024, n_val=20, n_test=30, out_dir=tmp_path, feature_set="base")
    assert (tmp_path / "base" / "2024.json").exists() and not (tmp_path / "2024.json").exists()
    assert rep["feature_set"] == "base" and rep["n_features"] == 1
    assert fit_cols and all(c == ["a"] for c in fit_cols)
    wf.run_fold(2024, n_val=20, n_test=30, out_dir=tmp_path, models=("good",), smoke=True, feature_set="base")
    assert (tmp_path / "base" / "smoke" / "2024.json").exists()
