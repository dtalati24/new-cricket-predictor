# src/walkforward.py
"""Walk-forward test: for each test year Y, choose and calibrate on Y-1, score Y.

    python -m src.walkforward --year 2024 [--workers 4] [--n-sims-val 2000] [--n-sims-test 10000] [--models logistic,lightgbm,xgboost] [--feature-set full|base] [--smoke]
    python -m src.walkforward --summary [--feature-set base]

One year per call, so each year can be its own GitHub Actions job. 2025-2026
were seen before the context features were designed; the summary reports the
pooled result with and without them (2023-2024 is the cleaner evidence).
"""
from __future__ import annotations

import gc
import json
import sys

import numpy as np
import pandas as pd

from src.bet.backtest import backtest
from src.cache import input_files, load_or_build
from src.config import PARSED, REPORTS_DIR
from src.features.context import CONTEXT
from src.features.outcomes import CLASSES, competition_key, venue_key
from src.market.baseline import comparison_set
from src.model.ballmodel import MODELS, fit as fit_model
from src.model.frame import feature_columns
from src.run_6over import SIX, choose_model, per_match, prepare
from src.sim.calibrate import apply, fit_calibration, summary
from src.sim.simulate import run_table
from src.sim.setup import build_setups, check_xi, simulate_setups
from src.skill.shrink import LEVELS

YEARS = (2023, 2024, 2025, 2026)
FEATURE_SETS = ("full", "base")
OUT = REPORTS_DIR / "walkforward"
SMOKE_TRAIN_ROWS = 200_000
SMOKE_MATCHES = 30


def year_rows(year: int) -> pd.DataFrame:
    r = comparison_set("all")
    r = r[r.market_type == SIX].copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r[r.date.dt.year == year]
    r["venue_key"] = venue_key(r.venue, r.city)
    r["competition_key"] = competition_key(r.event_name)
    return r.sort_values("date", kind="stable").reset_index(drop=True)


def prepare_cached(cutoff: str):
    def build():
        balls, ctx, frame, table = prepare(cutoff)
        return ctx, frame, table
    return load_or_build(cutoff, feature_columns(LEVELS), input_files(), build)


def feature_set_columns(levels, name: str) -> list[str]:
    """"full" = every model input; "base" = the first model's inputs (no context, no competition level)."""
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; choose from {list(FEATURE_SETS)}")
    cols = feature_columns(levels)
    if name == "full":
        return cols
    drop = set(CONTEXT) | {f"competition__{c}" for c in CLASSES}
    return [c for c in cols if c not in drop]


def _fit_all(frame, cutoff, models, cols, smoke):
    tr = frame.date < cutoff
    train = frame.loc[tr, [*cols, "y", "date"]]
    if smoke:
        train = train.tail(SMOKE_TRAIN_ROWS)
    return {n: fit_model(n, train, train.y.to_numpy(), train.date, cols) for n in models}


def _score(name, report, cal, rows, cols, sim_cols, test_cut, n_test, workers, smoke, out_dir, shared):
    """Fit one model on < Y, simulate Y, write its parquet and the report. One model in memory at a time.

    `shared` holds the test-year setups and run table: built from the first load, reused by every model."""
    year = report["year"]
    ctx, frame, table = prepare_cached(test_cut)
    if "setups" not in shared:
        shared["setups"], shared["table"] = build_setups(rows, ctx), run_table(table)
    model = _fit_all(frame, test_cut, [name], cols, smoke)[name]
    del frame
    gc.collect()
    ys = rows.runs_at_6ov.to_numpy(float)
    lines = rows.pre_match_line.to_numpy(float)
    raw = simulate_setups(shared["setups"], model, shared["table"], sim_cols, n_test, workers=workers)
    sims = [apply(cal, s) for s in raw]
    per_match(rows, raw, sims).to_parquet(out_dir / f"{year}_{name}_matches.parquet")
    report["test"][name] = {"role": "chosen" if name == report["chosen"] else "comparison",
                            "calibration": cal.__dict__, "raw": summary(raw, ys),
                            "calibrated": summary(sims, ys), "backtest": backtest(sims, lines, ys)}
    (out_dir / f"{year}.json").write_text(json.dumps(report, indent=2, default=float))
    print(year, "test", name, report["test"][name]["calibrated"], flush=True)


def run_fold(year: int, workers: int = 4, n_val: int = 2000, n_test: int = 10000,
             smoke: bool = False, out_dir=OUT, models=None, feature_set: str = "full") -> dict:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    out_dir = out_dir if feature_set == "full" else out_dir / feature_set
    out_dir = out_dir / "smoke" if smoke else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models = tuple(models) if models else (("logistic",) if smoke else tuple(MODELS))
    if bad := [m for m in models if m not in MODELS]:
        raise ValueError(f"unknown models {bad}; choose from {list(MODELS)}")
    val_cut, test_cut = f"{year - 1}-01-01", f"{year}-01-01"
    val_rows, test_rows = year_rows(year - 1), year_rows(year)
    if smoke:
        val_rows, test_rows = val_rows.head(SMOKE_MATCHES), test_rows.head(SMOKE_MATCHES)
    check_xi(pd.concat([val_rows, test_rows]), pd.read_parquet(PARSED / "playing_xi.parquet"))

    # Phase 1: fit on < Y-1, simulate Y-1, calibrate each model, choose one.
    ctx, frame, table = prepare_cached(val_cut)
    sim_cols = feature_columns(ctx.skill.levels)  # full layout: the static tensor is built for all of it
    cols = feature_set_columns(ctx.skill.levels, feature_set)
    val_setups, val_table = build_setups(val_rows, ctx), run_table(table)
    fitted = _fit_all(frame, val_cut, models, cols, smoke)
    del frame
    gc.collect()
    ys = val_rows.runs_at_6ov.to_numpy(float)
    val, cals = {}, {}
    for n, m in fitted.items():
        sims = simulate_setups(val_setups, m, val_table, sim_cols, n_val, workers=workers)
        val[n] = {"raw": summary(sims, ys)}
        try:
            cals[n] = fit_calibration(sims, ys)
            val[n]["calibrated"] = summary([apply(cals[n], s) for s in sims], ys)
        except ValueError as err:
            val[n]["calibrated"] = None
            val[n]["calibration_error"] = str(err)
        print(year, "val", n, val[n], flush=True)
    if not cals:
        raise SystemExit(f"{year}: no model could be calibrated on {year - 1}")
    chosen = choose_model(val)
    del ctx, fitted, val_setups
    gc.collect()
    report = {"year": year, "chosen": chosen, "smoke": smoke, "n_val_sims": n_val,
              "n_test_sims": n_test, "n_val_matches": len(val_rows),
              "n_test_matches": len(test_rows), "feature_set": feature_set,
              "n_features": len(cols), "val": val, "test": {}}
    (out_dir / f"{year}.json").write_text(json.dumps(report, indent=2, default=float))

    # Phase 2: refit on < Y, simulate Y, apply each model's Y-1 calibration. The chosen
    # model first; comparisons are best-effort, each in a fresh load so memory stays flat.
    lines, ys = test_rows.pre_match_line.to_numpy(float), test_rows.runs_at_6ov.to_numpy(float)
    report["market_mae"] = float(np.mean(np.abs(lines - ys)))
    args = (test_rows, cols, sim_cols, test_cut, n_test, workers, smoke, out_dir, {})
    _score(chosen, report, cals[chosen], *args)
    for n in [k for k in cals if k != chosen]:
        try:
            _score(n, report, cals[n], *args)
        except Exception as err:
            report["test"][n] = {"role": "comparison", "error": repr(err)}
            (out_dir / f"{year}.json").write_text(json.dumps(report, indent=2, default=float))
            print(year, "test", n, "FAILED", repr(err), flush=True)
    return report


def _stats(df: pd.DataFrame) -> dict:
    bets = df[df.side.notna()]
    n = len(bets)
    return {
        "n_matches": int(len(df)),
        "mae_model": float(np.mean(np.abs(df.cal_median - df.actual))),
        "mae_market": float(np.mean(np.abs(df.line - df.actual))),
        "n_bets": int(n),
        "hit_rate": float((bets.pnl > 0).mean()) if n else float("nan"),
        "roi": float(bets.pnl.mean()) if n else float("nan"),
        "roi_se": float(bets.pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
    }


def summarize(out_dir=OUT, years=YEARS) -> dict:
    frames, missing = {}, []
    for y in years:
        meta = out_dir / f"{y}.json"
        path = out_dir / f"{y}_{json.loads(meta.read_text())['chosen']}_matches.parquet" if meta.exists() else None
        if path is not None and path.exists():
            frames[y] = pd.read_parquet(path)
        else:
            missing.append(y)
            print(f"MISSING: no result for {y}", flush=True)
    if not frames:
        raise SystemExit(f"no fold results in {out_dir}")
    clean = [frames[y] for y in frames if y in (2023, 2024)]
    out = {"per_year": {str(y): _stats(f) for y, f in frames.items()},
           "pooled": _stats(pd.concat(frames.values(), ignore_index=True)),
           "pooled_2023_2024": _stats(pd.concat(clean, ignore_index=True)) if clean else None,
           "missing_years": missing}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def _arg(argv, name, default, cast=int):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--summary" in argv:
        fs = _arg(argv, "--feature-set", "full", str)
        print(json.dumps(summarize(out_dir=OUT if fs == "full" else OUT / fs), indent=2, default=float))
    elif "--year" in argv:
        rep = run_fold(_arg(argv, "--year", None), workers=_arg(argv, "--workers", 4),
                       n_val=_arg(argv, "--n-sims-val", 2000),
                       n_test=_arg(argv, "--n-sims-test", 10000), smoke="--smoke" in argv,
                       models=_arg(argv, "--models", None, lambda v: v.split(",")),
                       feature_set=_arg(argv, "--feature-set", "full", str))
        print(json.dumps({k: rep[k] for k in ("year", "chosen", "market_mae")}, default=float))
    else:
        raise SystemExit("usage: python -m src.walkforward --year Y [--workers N] "
                         "[--n-sims-val N] [--n-sims-test N] [--models a,b] [--feature-set full|base] "
                         "[--smoke] | --summary [--feature-set base]")
