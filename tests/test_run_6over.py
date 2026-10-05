import numpy as np
import pandas as pd
import pytest

from src.bet.backtest import decide, pnl
import src.run_6over as run
from src.run_6over import check_not_scored, choose_model, final_paths, per_match, resolve_models


def test_per_match_columns_and_betting_rule():
    rows = pd.DataFrame({"match_id": ["a", "b", "c"],
                         "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                         "pre_match_line": [50.5, 50.5, 50.5],
                         "runs_at_6ov": [60.0, 60.0, 40.0]})
    raw = [np.linspace(m - 5, m + 5, 500) for m in (70, 30, 50)]
    sims = [s + 0.5 for s in raw]
    out = per_match(rows, raw, sims)
    assert list(out.columns) == ["match_id", "date", "line", "actual", "raw_median", "cal_median",
                                 "cal_p10", "cal_p90", "p_over", "p_under", "pit", "side", "pnl"]
    assert out.pit.between(0, 1).all()
    assert out.raw_median.tolist() == [np.median(s) for s in raw]
    for r, s in zip(out.itertuples(), sims):
        side = decide(float(np.mean(s > 50.5)), float(np.mean(s < 50.5)))
        assert (r.side if isinstance(r.side, str) else None) == side
        assert r.pnl == pnl(side, 50.5, r.actual)
    assert out.side.tolist()[:2] == ["over", "under"] and pd.isna(out.side[2])


def test_choose_model_ranks_uncalibrated_last():
    ok = lambda mae, crps: {"calibrated": {"mae": mae, "crps": crps}}  # noqa: E731
    results = {"a": {"calibrated": None}, "b": ok(3.0, 5.0), "c": ok(3.0, 4.0), "d": ok(2.0, 9.0)}
    assert choose_model(results) == "d"
    del results["d"]
    assert choose_model(results) == "c"
    assert choose_model({"a": {"calibrated": None}, "b": ok(9.0, 9.0)}) == "b"


def test_final_paths_are_per_model():
    paths = {n: final_paths(n) for n in ("logistic", "lightgbm", "xgboost")}
    assert len({p for pair in paths.values() for p in pair}) == 6
    assert paths["xgboost"][0].name == "6over_final_xgboost.json"
    assert paths["xgboost"][1].name == "6over_final_xgboost_matches.parquet"


def test_check_not_scored_guards_each_model(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REPORTS_DIR", tmp_path)
    check_not_scored("logistic", False)
    final_paths("logistic")[0].write_text("{}")
    with pytest.raises(SystemExit, match="already scored"):
        check_not_scored("logistic", False)
    check_not_scored("logistic", True)
    check_not_scored("xgboost", False)


def test_resolve_models():
    assert resolve_models([], "lightgbm") == ["lightgbm"]
    assert resolve_models(["--all"], "lightgbm") == ["lightgbm", "logistic", "xgboost"]
    assert resolve_models(["--model", "xgboost", "--force"], "logistic") == ["xgboost"]
    with pytest.raises(SystemExit):
        resolve_models(["--model", "nope"], "logistic")
    with pytest.raises(SystemExit):
        resolve_models(["--model"], "logistic")
