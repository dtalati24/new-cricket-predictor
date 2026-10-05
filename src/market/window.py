"""Locate the post-toss, pre-first-ball betting window.

The toss has no timestamp anywhere in the data -- Cricsheet records the winner
and the decision but not when it happened. The FIRST BALL does have one: a
Betfair market goes in-play at the first delivery, which appears in the
market-definition stream as the first `in_play == True` row.

Tosses are held roughly 30 minutes before the scheduled start, so the window
is taken as [first_ball - 30min, first_ball). This is an approximation and the
30 minutes is a parameter, not a constant of nature -- Task 2's report sweeps
it so the choice is made on measured tick counts rather than assumed.

A market can go in-play, suspend at the innings break and resume, so only the
FIRST transition is the first ball. A market that never goes in-play (an
abandoned fixture) has no first ball and is excluded rather than defaulted.
"""
from __future__ import annotations

import pandas as pd


def first_ball_times(defs: pd.DataFrame) -> pd.DataFrame:
    """One row per market that ever went in-play, carrying its first such time."""
    live = defs[defs.in_play.astype(bool)]
    if live.empty:
        return pd.DataFrame(columns=["market_id", "first_ball_ts"])
    out = live.groupby("market_id", as_index=False).pt.min()
    return out.rename(columns={"pt": "first_ball_ts"})


def window_bounds(first_ball_ts: pd.Timestamp, minutes: int = 30):
    """Half-open window [toss_estimate, first_ball). The bet precedes the ball."""
    return first_ball_ts - pd.Timedelta(minutes=minutes), first_ball_ts


def ticks_per_market(fb: pd.DataFrame, ticks: pd.DataFrame, minutes: int) -> pd.Series:
    """Ticks inside [first_ball - minutes, first_ball) per market, zeros included.

    `fb` is first_ball_times() output; `ticks` needs `market_id` and `pt`.
    """
    j = ticks.merge(fb, on="market_id")
    lo = j.first_ball_ts - pd.Timedelta(minutes=minutes)
    inside = j[(j.pt >= lo) & (j.pt < j.first_ball_ts)]
    n = inside.groupby("market_id").size()
    return n.reindex(fb.market_id, fill_value=0)


def report(minutes_grid=(15, 30, 45, 60, 120)) -> pd.DataFrame:
    """Gate 0 sweep: how much trading falls inside the window, at several widths.

    CAVEAT: the tick counts are for MATCH_ODDS, a PROXY. The instrument we will
    actually trade is the over-line market, and its tick table
    (prices_overline.parquet) does not exist until Task 4. MATCH_ODDS is the
    most liquid market on the same event, so these counts are an UPPER BOUND on
    over-line liquidity, not an estimate of it. Over-line activity inside the
    window will be lower.

    Columns: markets_with_first_ball (same on every row), then per width the
    markets with >=1 tick, >=MIN_PRE_MATCH_TICKS ticks (the floor below which
    Build 2's line errors arose), and the median tick count among markets that
    have any. Reported as a sweep because the 30-minute assumption is the
    weakest part of the design. The result is also written to
    REPORTS_DIR/gate0_window_sweep.csv so the evidence is recorded.
    """
    from src.config import (MATCH_ODDS_DEFS, MATCH_ODDS_TICKS, MIN_PRE_MATCH_TICKS,
                            REPORTS_DIR, ensure_dirs)

    fb = first_ball_times(pd.read_parquet(MATCH_ODDS_DEFS))
    ticks = pd.read_parquet(MATCH_ODDS_TICKS, columns=["market_id", "pt"])
    rows = []
    for m in minutes_grid:
        n = ticks_per_market(fb, ticks, m)
        rows.append({
            "window_minutes": m,
            "markets_with_first_ball": len(fb),
            "median_window_start": (fb.first_ball_ts - pd.Timedelta(minutes=m)).median(),
            "markets_ge1_tick": int((n >= 1).sum()),
            f"markets_ge{MIN_PRE_MATCH_TICKS}_ticks": int((n >= MIN_PRE_MATCH_TICKS).sum()),
            "median_ticks_where_any": float(n[n >= 1].median()),
        })
    out = pd.DataFrame(rows)
    ensure_dirs()
    out.to_csv(REPORTS_DIR / "gate0_window_sweep.csv", index=False)
    return out


if __name__ == "__main__":
    print(report().to_string(index=False))
