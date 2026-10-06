# src/features/form.py
"""Recent form as exponentially weighted rates over a player's matches.

Separate from the skill counts on purpose: counts are plain integers for the
shrinkage; recency lives here. Each table holds the state AFTER each date and
is read point-in-time with `as_of`.
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


def batter_form(balls: pd.DataFrame) -> pd.DataFrame:
    faced = balls[balls.outcome != "wide"]
    per = (faced.groupby(["batter_id", "date"])
           .agg(runs=("runs_batter", "sum"), balls=("runs_batter", "size"))
           .reset_index().rename(columns={"batter_id": "player_id"}))
    outs = (balls[is_wicket(balls)].groupby(["player_dismissed_id", "date"]).size()
            .rename("outs").reset_index().rename(columns={"player_dismissed_id": "player_id"}))
    per = per.merge(outs, on=["player_id", "date"], how="left").fillna({"outs": 0})
    per = post_match_ewm(per, "player_id", ["runs", "balls", "outs"])
    per["bat_form_sr"] = per.runs / per.balls
    per["bat_form_out"] = per.outs / per.balls
    return per[["player_id", "date", "bat_form_sr", "bat_form_out"]]


def bowler_form(balls: pd.DataFrame) -> pd.DataFrame:
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
    per["bowl_form_econ"] = per.conceded / legal
    per["bowl_form_wkt"] = per.wkts / legal
    return per[["player_id", "date", "bowl_form_econ", "bowl_form_wkt"]]
