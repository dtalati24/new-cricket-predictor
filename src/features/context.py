# src/features/context.py
"""Match-level context known at the toss, one row per match for the side batting first.

Joined onto every first-innings ball in the training frame and broadcast into the
simulator's static features, so training and simulation see identical values.
History-based columns are read strictly before the match date with `as_of`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOSS_SEASON = ["bat_won_toss", "chose_to_bat", "season"]
CONTEXT = TOSS_SEASON


def first_innings(balls: pd.DataFrame) -> pd.DataFrame:
    b = balls[balls.innings_num == 1]
    cols = ["date", "batting_team", "bowling_team", "venue_key", "competition_key"]
    return b.drop_duplicates("match_id").set_index("match_id")[cols]


def toss_and_season(first: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.set_index("match_id").reindex(first.index)
    d = pd.to_datetime(first.date)
    return pd.DataFrame({
        "bat_won_toss": (m.toss_winner == first.batting_team).astype(float),
        "chose_to_bat": (m.toss_decision == "bat").astype(float),
        "season": d.dt.year + (d.dt.dayofyear - 1) / 366,
    }, index=first.index)


def match_context(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    first = first_innings(balls)
    out = toss_and_season(first, matches)
    return out[CONTEXT].astype(np.float32)
