# src/run_6over.py
"""The 6-over pipeline.

    python -m src.run_6over dev     train < 2024; choose model, prune, calibrate on 2024
    python -m src.run_6over final   train < 2025; score the 2025+ holdout ONCE

`final` refuses to run twice unless given --force, so the holdout is looked at once.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.bet.backtest import backtest, decide, pnl
from src.config import DEV_END, PARSED, REPORTS_DIR, TRAIN_END, ensure_dirs
from src.features.form import batter_form, bowler_form
from src.features.outcomes import load_balls, log_loss, outcome_table, venue_key
from src.features.players import (
    annotate, batting_order_history, load_xi, player_styles, powerplay_history)
from src.market.baseline import comparison_bar, comparison_set
from src.model.ballmodel import MODELS, fit as fit_model, permutation_importance
from src.model.frame import build_frame, feature_columns, importance_groups
from src.sim.calibrate import Calibration, apply, fit_calibration, pit, summary
from src.sim.setup import Context, check_xi, simulate_matches
from src.skill.shrink import LEVELS, fit as fit_skill, level_report

SIX = "1_INNING_6_OVR_LINE"
DEV_N_SIMS = 2_000      # model choice; median MC error ~0.3 runs per match, averaged over 435
FINAL_N_SIMS = 10_000
DEV_REPORT = REPORTS_DIR / "6over_dev.json"
FINAL_REPORT = REPORTS_DIR / "6over_final.json"
DEV_MATCHES = REPORTS_DIR / "6over_dev_matches.parquet"
FINAL_MATCHES = REPORTS_DIR / "6over_final_matches.parquet"


def market_rows(split: str, start: str | None = None) -> pd.DataFrame:
    r = comparison_set(split)
    r = r[r.market_type == SIX].copy()
    r["date"] = pd.to_datetime(r["date"])
    if start:
        r = r[r.date >= start]
    r["venue_key"] = venue_key(r.venue, r.city)
    return r.sort_values("date", kind="stable").reset_index(drop=True)


def per_match(rows: pd.DataFrame, raw, sims) -> pd.DataFrame:
    """One row per match: raw and calibrated forecast, PIT, and the bet the rule would place."""
    out = []
    for r, rs, s in zip(rows.itertuples(), raw, sims):
        line, y = float(r.pre_match_line), float(r.runs_at_6ov)
        p_over, p_under = float(np.mean(s > line)), float(np.mean(s < line))
        side = decide(p_over, p_under)
        out.append({"match_id": r.match_id, "date": r.date, "line": line, "actual": y,
                    "raw_median": float(np.median(rs)), "cal_median": float(np.median(s)),
                    "cal_p10": float(np.quantile(s, 0.1)), "cal_p90": float(np.quantile(s, 0.9)),
                    "p_over": p_over, "p_under": p_under, "pit": pit(s, y),
                    "side": side, "pnl": pnl(side, line, y)})
    return pd.DataFrame(out)


def choose_model(results: dict) -> str:
    """Best by calibrated (mae, crps); a model whose calibration failed ranks after all others."""
    def key(n):
        c = results[n]["calibrated"]
        return (c is None, c["mae"] if c else 0.0, c["crps"] if c else 0.0)
    return min(results, key=key)


def _pit_histogram(matches: pd.DataFrame) -> list[int]:
    return np.histogram(matches.pit, bins=10, range=(0, 1))[0].tolist()


def prepare(cutoff: str):
    styles = player_styles()
    balls = annotate(load_balls(), styles)
    xi = load_xi(balls)
    ctx = Context(skill=fit_skill(balls, cutoff), styles=styles,
                  bat_form=batter_form(balls), bowl_form=bowler_form(balls),
                  bat_pos=batting_order_history(balls), pp=powerplay_history(balls, xi), xi=xi)
    frame = build_frame(balls, ctx.skill, ctx.bat_form, ctx.bowl_form)
    table = outcome_table(balls[balls.date < cutoff])
    return balls, ctx, frame, table


def dev() -> dict:
    balls, ctx, frame, table = prepare(DEV_END)
    levels = level_report(ctx.skill, balls[balls.date < DEV_END],
                          balls[(balls.date >= DEV_END) & (balls.date < TRAIN_END)])
    print(levels.to_string(index=False), flush=True)
    # The density gate is report-only by user decision (2026-10-05): all levels are kept
    # for a later feature-importance analysis.

    cols = feature_columns(ctx.skill.levels)
    tr = (frame.date < DEV_END).to_numpy()
    va = ((frame.date >= DEV_END) & (frame.date < TRAIN_END)).to_numpy()
    X, y, dates = frame[cols], frame.y.to_numpy(), frame.date
    rows = market_rows("train", start=DEV_END)
    check_xi(rows, ctx.xi)
    ys = rows.runs_at_6ov.to_numpy(float)

    results, fitted = {}, {}
    for name in MODELS:
        m = fit_model(name, X.loc[tr], y[tr], dates[tr], cols)
        sims = simulate_matches(rows, ctx, m, table, DEV_N_SIMS)
        results[name] = {"val_log_loss": log_loss(m.predict(X.loc[va]), y[va]),
                         "raw": summary(sims, ys)}
        try:
            cal = fit_calibration(sims, ys)
            results[name]["calibrated"] = summary([apply(cal, s) for s in sims], ys)  # in-sample
        except ValueError as err:
            results[name]["calibrated"] = None
            results[name]["calibration_error"] = str(err)
        fitted[name] = (m, sims)
        print(name, results[name], flush=True)
    if all(r["calibrated"] is None for r in results.values()):
        raise SystemExit("no model could be calibrated: "
                         + "; ".join(f"{n}: {r['calibration_error']}" for n, r in results.items()))
    best = choose_model(results)
    m, sims = fitted[best]

    groups = importance_groups(ctx.skill.levels)
    imp = permutation_importance(m, X.loc[va], y[va], groups)
    print(imp.to_string(index=False), flush=True)
    drop = {c for g in imp[(imp.ll_increase <= 0) & (imp.group != "all_skill")].group
            for c in groups[g]}
    kept = [c for c in cols if c not in drop]
    if drop:
        pruned = fit_model(best, X.loc[tr], y[tr], dates[tr], kept)
        if log_loss(pruned.predict(X.loc[va]), y[va]) <= results[best]["val_log_loss"] + 1e-4:
            m = pruned
            sims = simulate_matches(rows, ctx, m, table, DEV_N_SIMS)
        else:
            kept = cols

    cal = fit_calibration(sims, ys)
    cal_sims = [apply(cal, s) for s in sims]
    matches = per_match(rows, sims, cal_sims)
    ensure_dirs()
    matches.to_parquet(DEV_MATCHES)
    report = {
        "levels": levels.to_dict(orient="records"),
        "level_names": [lv.name for lv in ctx.skill.levels],
        "n_sims": DEV_N_SIMS,
        "pit_histogram": _pit_histogram(matches),
        "k": ctx.skill.k,
        "models": results,
        "model": best,
        "kept_columns": kept,
        "importance": imp.to_dict(orient="records"),
        "calibration": cal.__dict__,
        "dev_raw": summary(sims, ys),
        # in-sample for the calibration: a sanity check, not a result
        "dev_calibrated": summary(cal_sims, ys),
    }
    DEV_REPORT.write_text(json.dumps(report, indent=2, default=float))
    return report


def final(force: bool = False) -> dict:
    if FINAL_REPORT.exists() and not force:
        raise SystemExit("the holdout was already scored; --force reruns it (say so in any write-up)")
    dev_report = json.loads(DEV_REPORT.read_text())
    # Fail early: cheap checks before the slow prepare().
    rows = market_rows("holdout")
    check_xi(rows, pd.read_parquet(PARSED / "playing_xi.parquet"))
    missing = [c for c in dev_report["kept_columns"] if c not in feature_columns(LEVELS)]
    if missing:
        raise SystemExit(f"dev kept_columns not in the current feature set: {missing}")
    cal = Calibration(**dev_report["calibration"])
    if not 0.5 < cal.s < 2.5:
        raise SystemExit(f"dev calibration spread s={cal.s} is at the edge of its grid")
    balls, ctx, frame, table = prepare(TRAIN_END)
    tr = (frame.date < TRAIN_END).to_numpy()
    m = fit_model(dev_report["model"], frame.loc[tr], frame.y.to_numpy()[tr],
                  frame.date[tr], dev_report["kept_columns"])
    ys = rows.runs_at_6ov.to_numpy(float)
    raw = simulate_matches(rows, ctx, m, table, FINAL_N_SIMS)
    sims = [apply(cal, s) for s in raw]
    matches = per_match(rows, raw, sims)
    ensure_dirs()
    matches.to_parquet(FINAL_MATCHES)
    bar = comparison_bar("holdout").set_index("market_type").loc[SIX]
    report = {
        "model": dev_report["model"],
        "kept_columns": dev_report["kept_columns"],
        "forced": force,
        "levels": [lv.name for lv in ctx.skill.levels],
        "n_sims": FINAL_N_SIMS,
        "pit_histogram": _pit_histogram(matches),
        "raw": summary(raw, ys),
        "calibrated": summary(sims, ys),
        "market_mae": float(bar.market_mae),
        "detectable_gain": float(bar.detectable_gain),
        "backtest": backtest(sims, rows.pre_match_line.to_numpy(float), ys),
    }
    report["beats_bar"] = report["calibrated"]["mae"] < report["market_mae"] - report["detectable_gain"]
    FINAL_REPORT.write_text(json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "dev":
        print(json.dumps(dev(), indent=2, default=float))
    elif step == "final":
        print(json.dumps(final(force="--force" in sys.argv), indent=2, default=float))
    else:
        raise SystemExit("usage: python -m src.run_6over dev|final [--force]")
