"""GATE 4 -- how good is the market itself?

This is the pivot of the whole project. It computes the market's own forecast
error on first-innings over-lines, which is the number any model has to beat.
It is deliberately built before any model exists.

Why this and not P&L
--------------------
On ~3,000 usable innings a 2% ROI edge is undetectable, and the LINE market
P&L convention is not yet confirmed. Forecast error against the line needs no
settlement convention at all and carries far more statistical power per
observation, so it is the headline metric.

Reference points
----------------
* ``UNCONDITIONAL_T20_FIRST_INNINGS_MAE`` (33.4 runs) -- predicting the corpus
  mean every time. The market must be far below this or our pipeline is wrong.
* The market's MAE is then the bar for the model.

Known optimism, stated up front: ``pre_match_line`` is a LAST TRADED price,
not an available quote. A line that traded at 188.5 does not mean 188.5 could
have been taken at our decision time, and Basic-tier data has no back/lay to
check. Every number here inherits that caveat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BETFAIR_DIR,
    MATCH_ODDS_DEFS,
    MIN_PRE_MATCH_TICKS,
    MODELLING_GENDER,
    TRAIN_END,
    OVER_LINE_PLAUSIBLE_BAND,
    TOTAL_RUNS_RUNNER_ID,
    UNRELIABLE_OVER_LINE_TYPES,
    MARKET_DIR,
    OVER_LINE_OVERS,
    REPORTS_DIR,
    UNCONDITIONAL_T20_FIRST_INNINGS_MAE,
    ensure_dirs,
)
from src.market.settle import clean_mask


def build_comparison(
    prices: pd.DataFrame,
    event_match: pd.DataFrame,
    settled: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (match, over-line market) with the line and the truth."""
    matched = event_match.loc[event_match["status"] == "matched", ["event_id", "match_id"]]

    df = prices.merge(matched, on="event_id", how="inner")
    df = df[df["pre_match_line"].notna()]

    # Only the real "Total Runs" runner. 48 markets typed 1ST_INNINGS_RUNS
    # carry a lone "Total Fours"/"Total Sixes" runner instead, whose line
    # counts boundaries rather than runs.
    df = df[df["runner_id"] == TOTAL_RUNS_RUNNER_ID]

    # Only the unambiguous over-line types. Markets typed 1ST_INNINGS_RUNS
    # with bettingType LINE are deliberately excluded -- see the note on
    # AMBIGUOUS_LINE_MARKET_TYPES in config: several coexist per event with
    # no field distinguishing an innings total from a powerplay line.
    df = df[df["market_type"].isin(OVER_LINE_OVERS)]

    # Betfair lists some fixtures as two events (verified: "Zimbabwe v India
    # 2016" and "Zimbabwe v India (1st ODI)" both resolve to match 1007649),
    # so one match can inherit duplicate markets of the same type. Keep the
    # most-traded, which is the one a bet could actually have been struck in.
    df = (
        df.sort_values("n_ticks_pre", ascending=False, kind="stable")
        .drop_duplicates(subset=["match_id", "market_type"], keep="first")
    )

    # T20 with 6-ball overs only. The Hundred is in scope for skill estimation
    # but out of scope for the innings model, and its 100-ball totals are not
    # comparable to a 120-ball line.
    s = settled[(settled["match_type"] == "T20") & (settled["balls_per_over"] == 6)].copy()

    df = df.merge(s, on="match_id", how="inner")
    df["n_overs"] = df["market_type"].map(OVER_LINE_OVERS)
    df = df[df["n_overs"].notna()]
    df["n_overs"] = df["n_overs"].astype(int)

    # Pick the settled truth column matching each market's checkpoint.
    actual = np.full(len(df), np.nan)
    for n in sorted(set(OVER_LINE_OVERS.values())):
        sel = (df["n_overs"] == n).to_numpy()
        if sel.any():
            actual[sel] = df.loc[sel, f"runs_at_{n}ov"].to_numpy()
    df["actual_runs"] = actual

    # Row-wise cleanliness: rain/truncation, and the match must have been
    # scheduled long enough for this checkpoint to exist at all.
    ok = np.zeros(len(df), dtype=bool)
    for n in sorted(set(OVER_LINE_OVERS.values())):
        sel = (df["n_overs"] == n).to_numpy()
        if sel.any():
            ok[sel] = clean_mask(df[sel], n_overs=n).to_numpy()
    df["is_clean"] = ok

    # --- pre-match-only quality gates (see config for the measurement) -----
    # astype(float) matters when df is empty: an empty .map() inherits the
    # string dtype of market_type, and Series.between then raises rather than
    # returning an empty boolean mask.
    lo = df["market_type"].map(lambda m: OVER_LINE_PLAUSIBLE_BAND[m][0]).astype(float)
    hi = df["market_type"].map(lambda m: OVER_LINE_PLAUSIBLE_BAND[m][1]).astype(float)
    df["line_plausible"] = df["pre_match_line"].between(lo, hi)
    df["enough_ticks"] = df["n_ticks_pre"] >= MIN_PRE_MATCH_TICKS
    df["market_reliable"] = ~df["market_type"].isin(UNRELIABLE_OVER_LINE_TYPES)
    df["usable"] = (
        df["is_clean"] & df["line_plausible"] & df["enough_ticks"] & df["market_reliable"]
    )

    df["error"] = df["pre_match_line"] - df["actual_runs"]
    return df


def _detectable_improvement(errors: np.ndarray, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest MAE improvement detectable by a paired test at this n.

    Uses the normal approximation for a paired mean difference; the paired sd
    of |model error| - |market error| is unknown before a model exists, so we
    upper-bound it by the sd of the market's own absolute error, which is
    conservative (a correlated model makes the paired sd smaller).
    """
    n = len(errors)
    if n < 2:
        return float("nan")
    sd = float(np.std(np.abs(errors), ddof=1))
    z_a, z_b = 1.959963985, 0.8416212336
    return (z_a + z_b) * sd / np.sqrt(n)


def summarize(df: pd.DataFrame, mask_col: str = "usable") -> pd.DataFrame:
    clean = df[df[mask_col]]
    rows = []
    for mkt, g in clean.groupby("market_type"):
        err = g["error"].to_numpy()
        act = g["actual_runs"].to_numpy()
        rows.append(
            {
                "market_type": mkt,
                "n": len(g),
                "actual_mean": act.mean(),
                "actual_sd": act.std(ddof=1),
                "line_mean": g["pre_match_line"].mean(),
                "market_bias": err.mean(),
                "market_mae": np.abs(err).mean(),
                "market_rmse": np.sqrt((err**2).mean()),
                # Generous reference: predicting this subset's own mean.
                "mean_only_mae": np.abs(act - act.mean()).mean(),
                "r2_vs_mean": 1 - ((act - g["pre_match_line"]) ** 2).sum() / ((act - act.mean()) ** 2).sum(),
                "corr": np.corrcoef(g["pre_match_line"], act)[0, 1],
                "detectable_mae_gain": _detectable_improvement(err),
            }
        )
    return pd.DataFrame(rows).sort_values("market_type")


def run(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(BETFAIR_DIR / "prices_overline.parquet")
    event_match = pd.read_parquet(MARKET_DIR / "event_match.parquet")
    settled = pd.read_parquet(MARKET_DIR / "first_innings_settled.parquet")

    df = build_comparison(prices, event_match, settled)
    summary = summarize(df)

    if save:
        ensure_dirs()
        df.to_parquet(MARKET_DIR / "overline_comparison.parquet", index=False)
        summary.to_csv(REPORTS_DIR / "gate4_market_baseline.csv", index=False)
    return df, summary


def baseline_table() -> pd.DataFrame:
    """Gate 4 table, one row per usable line type, with the column names the tests use.

    The 15-over line is absent because `market_reliable` is 0 for it, not because
    of any filter here.
    """
    df, _ = run(save=False)
    return summarize(df, "usable").rename(
        columns={"r2_vs_mean": "r2", "detectable_mae_gain": "detectable_gain"}
    )


def _with_first_ball(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `first_ball_ts` per row, via the event's MATCH_ODDS in-play transition.

    An over-line market carries no in-play definitions in the BASIC archive, so the
    first ball has to come from the MATCH_ODDS market on the same event. Where an
    event has several, the earliest transition wins.
    """
    from src.market.window import first_ball_times

    mo = pd.read_parquet(
        BETFAIR_DIR / "match_odds_markets.parquet", columns=["market_id", "event_id"]
    )
    fb = first_ball_times(pd.read_parquet(MATCH_ODDS_DEFS)).merge(mo, on="market_id")
    fb = fb.groupby("event_id", as_index=False).first_ball_ts.min()
    return frame.merge(fb, on="event_id", how="left")


def comparison_set(split: str = "holdout") -> pd.DataFrame:
    """The rows a model and the market are BOTH scored on. Use this, never `usable`.

    Three filters are applied here and are deliberately NOT parameters, because
    each one is something a caller would otherwise be free to forget:

    * ``gender == MODELLING_GENDER``. The measured bar pools men's and women's
      T20, whose scores differ by ~19% and whose time trends diverge. Scoring
      against a pooled bar lets a model look good by fitting one population.
    * ``pre_match_ts < first_ball_ts``. `prices.py` cuts "pre-match" at the
      scheduled marketTime and play often starts earlier, so ~4% of rows carry a
      price stamped AFTER the first ball. The premise of the project is a bet
      placed before the first ball, so those rows are not a like-for-like
      comparison however small the effect.
    * ``usable``, which carries Build 2's line-plausibility, >=2-pre-tick and
      market-reliability gates.

    `split` is the only knob: "train" (date < TRAIN_END), "holdout" (>= TRAIN_END)
    or "all". Never a random split.

    STILL OPEN, and deliberately not fixed here (see docs/gates.md): `usable`
    inherits `clean_mask`, which excludes rain-affected and no-result matches
    wholesale even when the first innings ran its full scheduled length and the
    line would have settled normally. Relaxing it needs a per-checkpoint
    completeness rule, and the obvious form of that rule -- requiring
    ``legal_balls_at_Nov >= n_overs * balls_per_over`` -- selects AGAINST all-out
    innings, which end early but settle fine. Measured, that swings the 20-over
    market bias from +1.27 to -4.64 runs: a manufactured edge, not a fix. Do not
    change this without re-measuring the bias.
    """
    df, _ = run(save=False)
    rows = _with_first_ball(df[df["usable"]])
    rows = rows[rows["gender"] == MODELLING_GENDER]
    rows = rows[rows["pre_match_ts"] < rows["first_ball_ts"]]

    date = pd.to_datetime(rows["date"])
    if split == "train":
        rows = rows[date < TRAIN_END]
    elif split == "holdout":
        rows = rows[date >= TRAIN_END]
    elif split != "all":
        raise ValueError(f"split must be train, holdout or all; got {split!r}")
    return rows


def comparison_bar(split: str = "holdout") -> pd.DataFrame:
    """The market's error on `comparison_set(split)` -- what a model is scored against.

    This is NOT `baseline_table()`. That one reproduces Build 2's published figures
    and exists to prove the port is faithful; it pools genders, both eras and the
    leaked rows, and must never move. This one is the live bar.
    """
    rows = comparison_set(split).assign(_sel=True)
    return summarize(rows, "_sel").rename(
        columns={"r2_vs_mean": "r2", "detectable_mae_gain": "detectable_gain"}
    )


def overline_window_sweep(minutes_grid=(15, 30, 45, 60, 120)) -> pd.DataFrame:
    """Gate 0, finished: how many usable over-line markets last priced inside the window.

    Joins each usable target market to its event's MATCH_ODDS first-ball time (the
    over-line market has no in-play definitions of its own here) and counts those
    whose `pre_match_ts` lies in [first_ball - m, first_ball). Written next to
    Task 2's MATCH_ODDS-proxy sweep as gate0_overline_window_sweep.csv.
    """
    df, _ = run(save=False)
    j = _with_first_ball(df[df["usable"]])
    rows = []
    for mkt, g in j.groupby("market_type"):
        lead = (g["first_ball_ts"] - pd.to_datetime(g["pre_match_ts"], utc=True)).dt.total_seconds() / 60
        for m in minutes_grid:
            rows.append({
                "market_type": mkt,
                "window_minutes": m,
                "n_usable": len(g),
                "n_with_first_ball": int(g["first_ball_ts"].notna().sum()),
                "n_pre_match_ts_in_window": int(((lead > 0) & (lead <= m)).sum()),
                "n_last_priced_after_first_ball": int((lead <= 0).sum()),
            })
    out = pd.DataFrame(rows)
    ensure_dirs()
    out.to_csv(REPORTS_DIR / "gate0_overline_window_sweep.csv", index=False)
    return out


if __name__ == "__main__":
    df, summary = run()
    clean = df[df["is_clean"]]
    usable = df[df["usable"]]
    print(f"over-line markets joined to a match : {len(df):,}")
    print(f"  clean (no rain/truncation)        : {len(clean):,}")
    print(f"  + line plausible, >=2 pre-ticks,")
    print(f"    reliable market type  = USABLE  : {len(usable):,}")
    print(f"  distinct matches (usable)         : {usable['match_id'].nunique():,}")
    print()
    print("--- ALL clean rows, no quality gates (shows what the gates remove) ---")
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print(summarize(df, mask_col="is_clean").round(3).to_string(index=False))
    print()
    print("--- USABLE rows: the Gate 4 result ---")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.round(2).to_string(index=False))
    print()
    print(f"Reference: unconditional full-innings MAE = {UNCONDITIONAL_T20_FIRST_INNINGS_MAE} runs")
    print()
    print("GATE 4: market MAE must be materially below mean_only_mae on every line.")
    print("If it is not, the join or the price extraction is wrong -- not the market.")
    if len(usable):
        print()
        print("by year (usable only):")
        clean = usable.assign(year=pd.to_datetime(clean["market_time"], utc=True).dt.year)
        print(
            clean.groupby(["year", "market_type"])["error"]
            .agg(n="size", mae=lambda e: e.abs().mean(), bias="mean")
            .round(2)
            .to_string()
        )

    print()
    print("--- Gate 0 over-line window sweep (also written to gate0_overline_window_sweep.csv) ---")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(overline_window_sweep().to_string(index=False))
