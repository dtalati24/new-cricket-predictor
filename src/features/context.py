# src/features/context.py
"""Match-level context known at the toss, one row per match for the side batting first.

Joined onto every first-innings ball in the training frame and broadcast into the
simulator's static features, so training and simulation see identical values.
History-based columns are read strictly before the match date with `as_of`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import as_of
from src.features.form import post_match_ewm
from src.features.outcomes import NOT_LEGAL, is_wicket
from src.features.weather import WEATHER

TOSS_SEASON = ["chose_to_bat", "season"]
HISTORY = ["league_pp_recent", "comp_pp_recent", "venue_pp_recent", "venue_pp_n",
           "team_bat_pp", "opp_bat_pp", "opp_bowl_pp_conceded"]
CONTEXT = TOSS_SEASON + HISTORY + WEATHER

PP_LEGAL_BALLS = 36
LEAGUE_WINDOW_DAYS = 90
HALFLIFE_TEAM = 5
HALFLIFE_VENUE = 5
HALFLIFE_COMP = 20
_INNINGS_COLS = ["date", "batting_team", "bowling_team", "venue_key", "competition_key"]


def first_innings(balls: pd.DataFrame) -> pd.DataFrame:
    b = balls[balls.innings_num == 1]
    cols = ["date", "batting_team", "bowling_team", "venue_key", "competition_key"]
    return b.drop_duplicates("match_id").set_index("match_id")[cols]


def toss_and_season(first: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.set_index("match_id").reindex(first.index)
    d = pd.to_datetime(first.date)
    return pd.DataFrame({
        # 1 if the side batting first chose to; 0 if it was put in
        "chose_to_bat": (m.toss_decision == "bat").astype(float),
        "season": d.dt.year + (d.dt.dayofyear - 1) / 366,
    }, index=first.index)


def innings_powerplay(balls: pd.DataFrame) -> pd.DataFrame:
    """Runs in the first 36 legal balls; rain-cut innings (short and not all out) dropped."""
    key = [balls.match_id, balls.innings_num]
    legal = (~balls.outcome.isin(NOT_LEGAL)).astype(int)
    in_pp = (legal.groupby(key).cumsum() - legal) < PP_LEGAL_BALLS
    b = balls.loc[in_pp, ["match_id", "innings_num", *_INNINGS_COLS, "runs_total"]].assign(
        legal=legal[in_pp].to_numpy(), wkt=is_wicket(balls)[in_pp].astype(int).to_numpy())
    agg = {c: (c, "first") for c in _INNINGS_COLS}
    g = b.groupby(["match_id", "innings_num"], as_index=False).agg(
        **agg, pp_runs=("runs_total", "sum"), legal=("legal", "sum"), wkts=("wkt", "sum"))
    keep = (g.legal == PP_LEGAL_BALLS) | (g.wkts >= 10)
    return g.loc[keep, ["match_id", "innings_num", *_INNINGS_COLS, "pp_runs"]].reset_index(drop=True)


def _ewm_table(pp: pd.DataFrame, key: str, halflife: float, name: str) -> pd.DataFrame:
    per = pp.groupby([key, "date"], as_index=False).pp_runs.mean().rename(columns={key: "key"})
    return post_match_ewm(per, "key", ["pp_runs"], halflife=halflife).rename(columns={"pp_runs": name})


def _read(table: pd.DataFrame, keys: pd.Series, dates: pd.Series, col: str) -> pd.Series:
    q = pd.DataFrame({"key": keys.to_numpy(), "date": dates.to_numpy()}, index=keys.index)
    return as_of(table, q, ["key"], [col])[col]


def history_features(balls: pd.DataFrame, first: pd.DataFrame) -> pd.DataFrame:
    pp = innings_powerplay(balls)
    f1 = pp[pp.innings_num == 1]
    out = pd.DataFrame(index=first.index)

    daily = f1.groupby("date").pp_runs.agg(["sum", "count"]).sort_index().cumsum().reset_index()
    daily["key"] = "all"
    all_key = pd.Series("all", index=first.index)
    now = as_of(daily, pd.DataFrame({"key": all_key, "date": first.date}), ["key"], ["sum", "count"]).fillna(0)
    then = as_of(daily, pd.DataFrame({"key": all_key,
                                      "date": first.date - pd.Timedelta(days=LEAGUE_WINDOW_DAYS)}),
                 ["key"], ["sum", "count"]).fillna(0)
    n = now["count"] - then["count"]
    out["league_pp_recent"] = (now["sum"] - then["sum"]) / n.where(n > 0)

    out["comp_pp_recent"] = _read(_ewm_table(f1, "competition_key", HALFLIFE_COMP, "v"),
                                  first.competition_key, first.date, "v")
    out["venue_pp_recent"] = _read(_ewm_table(pp, "venue_key", HALFLIFE_VENUE, "v"),
                                   first.venue_key, first.date, "v")
    cnt = pp.groupby(["venue_key", "date"]).size().rename("n").reset_index().rename(
        columns={"venue_key": "key"})
    cnt["v"] = cnt.groupby("key").n.cumsum()
    out["venue_pp_n"] = _read(cnt, first.venue_key, first.date, "v").fillna(0)

    bat = _ewm_table(pp, "batting_team", HALFLIFE_TEAM, "v")
    out["team_bat_pp"] = _read(bat, first.batting_team, first.date, "v")
    out["opp_bat_pp"] = _read(bat, first.bowling_team, first.date, "v")
    out["opp_bowl_pp_conceded"] = _read(_ewm_table(pp, "bowling_team", HALFLIFE_TEAM, "v"),
                                        first.bowling_team, first.date, "v")
    return out[HISTORY]


def match_context(balls: pd.DataFrame, matches: pd.DataFrame,
                  weather: pd.DataFrame | None = None) -> pd.DataFrame:
    first = first_innings(balls)
    out = toss_and_season(first, matches).join(history_features(balls, first))
    if weather is None:
        out[WEATHER] = np.nan
    else:
        out = out.join(weather.reindex(out.index)[WEATHER])
    return out[CONTEXT].astype(np.float32)
