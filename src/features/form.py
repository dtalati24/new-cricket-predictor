# src/features/form.py
"""Recent form as exponentially weighted rates over a player's matches.

Separate from the skill counts on purpose: counts are plain integers for the
shrinkage; recency lives here. Each table holds the state AFTER each date and
is read point-in-time with `as_of`. Each table also carries the same rates over
powerplay balls only (overs 1-6), carried forward through matches without any.
"""
from __future__ import annotations

import pandas as pd

from src.features.outcomes import NOT_LEGAL, is_wicket

HALFLIFE = 10  # matches
# Dismissals not credited to the bowler.
NOT_BOWLER_WICKETS = ("run out", "retired hurt", "retired not out", "retired out",
                      "obstructing the field")


def post_match_ewm(per: pd.DataFrame, key: str, cols: list[str],
                   halflife: float = HALFLIFE) -> pd.DataFrame:
    """`per` has one row per (key, date). EWM over that key's dates, row included."""
    per = per.sort_values([key, "date"], kind="stable").reset_index(drop=True)
    ew = per.groupby(key)[cols].ewm(halflife=halflife).mean().reset_index(level=0, drop=True)
    per[cols] = ew.sort_index()
    return per


def _with_powerplay(all_overs: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in pp.columns if c not in ("player_id", "date")]
    out = all_overs.merge(pp, on=["player_id", "date"], how="outer")
    out = out.sort_values(["player_id", "date"], kind="stable").reset_index(drop=True)
    out[cols] = out.groupby("player_id")[cols].ffill()
    return out


def batter_form(balls: pd.DataFrame) -> pd.DataFrame:
    return _with_powerplay(_batter_rates(balls, "bat_form"), _batter_rates(balls[balls.over < 6], "bat_pp"))


def bowler_form(balls: pd.DataFrame) -> pd.DataFrame:
    return _with_powerplay(_bowler_rates(balls, "bowl_form"), _bowler_rates(balls[balls.over < 6], "bowl_pp"))


def _batter_rates(balls: pd.DataFrame, prefix: str) -> pd.DataFrame:
    faced = balls[balls.outcome != "wide"]
    per = (faced.groupby(["batter_id", "date"])
           .agg(runs=("runs_batter", "sum"), balls=("runs_batter", "size"))
           .reset_index().rename(columns={"batter_id": "player_id"}))
    outs = (balls[is_wicket(balls)].groupby(["player_dismissed_id", "date"]).size()
            .rename("outs").reset_index().rename(columns={"player_dismissed_id": "player_id"}))
    per = per.merge(outs, on=["player_id", "date"], how="left").fillna({"outs": 0})
    per = post_match_ewm(per, "player_id", ["runs", "balls", "outs"])
    per[f"{prefix}_sr"] = per.runs / per.balls
    per[f"{prefix}_out"] = per.outs / per.balls
    return per[["player_id", "date", f"{prefix}_sr", f"{prefix}_out"]]


def _bowler_rates(balls: pd.DataFrame, prefix: str) -> pd.DataFrame:
    b = balls.assign(
        conceded=balls.runs_total - balls.bye.fillna(0) - balls.legbye.fillna(0)
        - balls.penalty.fillna(0),
        legal=~balls.outcome.isin(NOT_LEGAL),
        wkt=balls.wicket_kind.notna() & ~balls.wicket_kind.isin(NOT_BOWLER_WICKETS),
    )
    per = (b.groupby(["bowler_id", "date"])
           .agg(conceded=("conceded", "sum"), legal=("legal", "sum"), wkts=("wkt", "sum"))
           .reset_index().rename(columns={"bowler_id": "player_id"}))
    per = post_match_ewm(per, "player_id", ["conceded", "legal", "wkts"])
    legal = per.legal.where(per.legal > 0)
    per[f"{prefix}_econ"] = per.conceded / legal
    per[f"{prefix}_wkt"] = per.wkts / legal
    return per[["player_id", "date", f"{prefix}_econ", f"{prefix}_wkt"]]
