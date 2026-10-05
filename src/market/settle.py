"""Settle first-innings markets from Cricsheet ball-by-ball truth.

This module answers, for every first innings: what did the over-line markets
actually settle at? It is deliberately built BEFORE any model, because the
market baseline computed on top of it (see ``market.baseline``) is the number
the whole project has to beat, and it is knowable in week one.

Scope decisions encoded here
----------------------------
* First innings only. A first innings has no target and no required run rate,
  so it is a far better-posed forecasting problem than a chase.
* ``runs`` means TEAM total including every extra (wides, no-balls, byes,
  leg-byes, penalties). That is what a "Total Runs" market settles on. Summing
  only batter-attributed runs understates a T20 innings by ~10 runs and was a
  confirmed source of bias in the previous build.
* Over-lines are cumulative: ``runs_at_6ov`` is the score at the END of over 6,
  i.e. over index < 6 (Cricsheet's ``over`` is 0-based).
* Super-overs are excluded entirely.

Truncation vs. all-out -- the distinction that matters
-------------------------------------------------------
If a team is bowled out for 80 in 12 overs, the 20-over line settles at 80:
the innings genuinely finished. If instead rain ended the innings at 12 overs,
the 20-over line never resolves and Betfair voids it. These look identical in
a naive ball count, so both are flagged separately and neither is silently
dropped -- the caller decides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MARKET_DIR, OVER_LINE_OVERS, PARSED_DIR, ensure_dirs

# Cricsheet records a rain-affected result in these fields.
DLS_METHODS = {"D/L", "DLS", "VJD"}
NON_RESULTS = {"no result", "abandoned"}

# A first innings is treated as having run its full scheduled length if it got
# within this many legal balls of it. Absorbs the odd short-over oddity in the
# source data without masking a genuine rain truncation.
_BALL_TOLERANCE = 2


def _first_innings_ids(innings: pd.DataFrame) -> pd.DataFrame:
    """One row per match for its first, non-super-over innings."""
    first = innings[(innings["innings_num"] == 1) & (~innings["is_super_over"].astype(bool))]
    dup = first["match_id"].duplicated().sum()
    if dup:
        raise ValueError(f"{dup} matches have more than one first innings row")
    return first


def checkpoint_runs(deliveries: pd.DataFrame, legal_balls: int) -> pd.DataFrame:
    """Runs scored by the first `legal_balls` legal balls of each innings.

    A wide or no-ball is a delivery but NOT a legal ball, so the 6-over mark
    is the 36th legal ball and may be the 38th delivery. Counting deliveries
    instead silently shortens every innings that leaked extras, which biases
    the label downward exactly where the bowling was poor.

    `legal_balls` is returned alongside so callers can exclude innings that
    ended before the checkpoint -- their total is real but it is not a
    6-over total and must not be compared to a 6-over line.

    WARNING for downstream filters: a threshold like ``legal_balls_at_Nov >= 36``
    silently drops all 367 Hundred matches (5-ball overs: 6 overs is 30 legal
    balls). Any such threshold must be derived from ``scheduled_overs`` and
    ``balls_per_over``, never hard-coded.

    NOTE: this is the legal-ball formulation of the rule, kept as its
    independently testable statement. ``build_settlement`` deliberately does
    NOT call it: it indexes by OVER (``over < n_overs``), the only form that
    is correct for the Hundred's 5-ball overs, where 6 overs is 30 legal
    balls, not 36. Do not unify the two.
    """
    d = deliveries.copy()
    d["is_legal"] = d.wide.isna() & d.noball.isna()
    d["legal_idx"] = d.groupby(["match_id", "innings_num"]).is_legal.cumsum()
    keep = d[d.legal_idx <= legal_balls]
    out = keep.groupby(["match_id", "innings_num"], as_index=False).agg(
        runs=("runs_batter", "sum"),
        extras=("runs_extras", "sum"),
        legal_balls=("is_legal", "sum"),
    )
    out["runs"] = out.runs + out.extras
    return out.drop(columns=["extras"])


def build_settlement(
    matches: pd.DataFrame,
    innings: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> pd.DataFrame:
    """Per-match first-innings truth, one row per match.

    Returns every first innings with its settled values and exclusion flags.
    Filtering is the caller's job -- see ``clean_mask``.
    """
    first = _first_innings_ids(innings)

    d = deliveries.merge(
        first[["match_id", "innings_num"]], on=["match_id", "innings_num"], how="inner"
    )

    agg = d.groupby("match_id").agg(
        total_runs=("runs_total", "sum"),
        total_legal_balls=("is_legal", "sum"),
        n_deliveries=("runs_total", "size"),
        wickets=("player_dismissed_id", "count"),
        max_over=("over", "max"),
    )

    # Cumulative score at the end of each over-line checkpoint.
    # Indexed by OVER, not legal-ball count, so it is correct for 5-ball-over
    # matches (the Hundred). See checkpoint_runs: it is the legal-ball form and
    # is intentionally not used here.
    for mkt, n_overs in OVER_LINE_OVERS.items():
        col = f"runs_at_{n_overs}ov"
        if col in agg.columns:
            continue
        agg[col] = (
            d[d["over"] < n_overs].groupby("match_id")["runs_total"].sum().reindex(agg.index).fillna(0)
        ).astype(int)
        # Balls actually bowled inside the checkpoint, to detect truncation.
        agg[f"legal_balls_at_{n_overs}ov"] = (
            d[d["over"] < n_overs].groupby("match_id")["is_legal"].sum().reindex(agg.index).fillna(0)
        ).astype(int)

    out = first.merge(agg, on="match_id", how="left").merge(
        matches[
            [
                "match_id",
                "date",
                "match_type",
                "gender",
                "venue",
                "city",
                "event_name",
                "season",
                "balls_per_over",
                "scheduled_overs",
                "toss_winner",
                "toss_decision",
                "outcome_winner",
                "outcome_result",
                "outcome_method",
            ]
        ],
        on="match_id",
        how="left",
    )

    out["expected_legal_balls"] = out["scheduled_overs"] * out["balls_per_over"]
    out["all_out"] = out["wickets"] >= 10
    out["innings_truncated"] = (~out["all_out"]) & (
        out["total_legal_balls"] < out["expected_legal_balls"] - _BALL_TOLERANCE
    )
    out["rain_affected"] = out["outcome_method"].isin(DLS_METHODS)
    out["no_result"] = out["outcome_result"].isin(NON_RESULTS)

    return out


def clean_mask(settled: pd.DataFrame, n_overs: int | None = None) -> pd.Series:
    """Rows safe to compare against a market line.

    Excludes rain-affected, no-result and truncated innings, all of which
    either void the market or make the settled total incomparable to the line
    the market was quoting. If ``n_overs`` is given, additionally requires the
    match to have been scheduled long enough for that checkpoint to exist.
    """
    m = (
        (~settled["rain_affected"])
        & (~settled["no_result"])
        & (~settled["innings_truncated"])
        & settled["total_legal_balls"].notna()
    )
    if n_overs is not None:
        m &= settled["scheduled_overs"] >= n_overs
    return m


def run(save: bool = True) -> pd.DataFrame:
    matches = pd.read_parquet(PARSED_DIR / "matches.parquet")
    innings = pd.read_parquet(PARSED_DIR / "innings.parquet")
    deliveries = pd.read_parquet(PARSED_DIR / "deliveries.parquet")

    settled = build_settlement(matches, innings, deliveries)

    if save:
        ensure_dirs()
        settled.to_parquet(MARKET_DIR / "first_innings_settled.parquet", index=False)
    return settled


if __name__ == "__main__":
    s = run()
    t20 = s[(s["match_type"] == "T20") & (s["balls_per_over"] == 6)]
    clean = t20[clean_mask(t20)]
    print(f"first innings rows        : {len(s):,}")
    print(f"  T20 (6-ball overs)      : {len(t20):,}")
    print(f"  clean (no rain/trunc)   : {len(clean):,}")
    print()
    print(f"total_runs  mean {clean['total_runs'].mean():.1f}  sd {clean['total_runs'].std():.1f}")
    for n in sorted(set(OVER_LINE_OVERS.values())):
        c = clean[clean["scheduled_overs"] >= n]
        col = f"runs_at_{n}ov"
        print(f"  runs_at_{n:>2}ov  n={len(c):>6,}  mean {c[col].mean():6.1f}  sd {c[col].std():5.1f}")
    print()
    print("excluded:")
    print(f"  rain_affected     {int(t20['rain_affected'].sum()):,}")
    print(f"  no_result         {int(t20['no_result'].sum()):,}")
    print(f"  innings_truncated {int(t20['innings_truncated'].sum()):,}")
