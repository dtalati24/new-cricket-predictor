# tests/test_context.py
import numpy as np
import pandas as pd
import pytest

from src.features.context import (CONTEXT, HISTORY, first_innings, history_features,
                                  innings_powerplay, toss_and_season)
from src.features.form import HALFLIFE, post_match_ewm
from src.features.outcomes import competition_key, label_outcomes


def _balls():
    return pd.DataFrame({
        "match_id": ["m1", "m1", "m2"], "innings_num": [1, 2, 1],
        "date": pd.to_datetime(["2024-05-02", "2024-05-02", "2024-12-31"]),
        "batting_team": ["A", "B", "C"], "bowling_team": ["B", "A", "D"],
        "venue_key": ["V", "V", "W"], "competition_key": ["X", "X", "Y"]})


def test_first_innings_is_one_row_per_match_for_the_side_batting_first():
    f = first_innings(_balls())
    assert list(f.index) == ["m1", "m2"]
    assert f.loc["m1", "batting_team"] == "A"


def test_toss_flags_separate_chose_to_bat_from_put_in():
    matches = pd.DataFrame({"match_id": ["m1", "m2"], "toss_winner": ["A", "D"],
                            "toss_decision": ["bat", "field"]})
    t = toss_and_season(first_innings(_balls()), matches)
    assert t.loc["m1", "chose_to_bat"] == 1.0
    assert t.loc["m2", "chose_to_bat"] == 0.0


def test_season_is_year_plus_fraction():
    t = toss_and_season(first_innings(_balls()),
                        pd.DataFrame({"match_id": ["m1", "m2"], "toss_winner": ["A", "C"],
                                      "toss_decision": ["bat", "bat"]}))
    assert 2024.0 < t.loc["m1", "season"] < 2024.5
    assert 2024.99 < t.loc["m2", "season"] < 2025.0


def test_competition_key_merges_sponsor_renames():
    k = competition_key(pd.Series(["NatWest T20 Blast", "Vitality Blast", "Vitality Blast Men",
                                   "Indian Premier League", None, "World T20",
                                   "ICC World Twenty20", "ICC Men's T20 World Cup",
                                   "ICC World Twenty20 Qualifier", "Ram Slam T20 Challenge",
                                   "MiWAY T20 Challenge", "CSA T20 Challenge"]))
    wc, wcq, csa = "ICC Men's T20 World Cup", "ICC Men's T20 World Cup Qualifier", "CSA T20 Challenge"
    assert k.tolist() == ["T20 Blast", "T20 Blast", "T20 Blast", "Indian Premier League", "unknown",
                          wc, wc, wc, wcq, csa, csa, csa]


def test_context_starts_with_toss_and_season():
    assert CONTEXT[:2] == ["chose_to_bat", "season"]


def _innings_balls(match_id, innings, date, bat, bowl, venue, comp, runs_per_ball,
                   n_legal=36, wides=0, wkts=0):
    rows = []
    for i in range(wides):
        rows.append({"wide": 1.0, "runs_batter": 0, "runs_total": 1, "wicket_kind": None})
    for i in range(n_legal):
        wk = "bowled" if i < wkts else None
        rows.append({"wide": np.nan, "runs_batter": runs_per_ball, "runs_total": runs_per_ball,
                     "wicket_kind": wk})
    d = pd.DataFrame(rows)
    for c in ("noball", "bye", "legbye"):
        d[c] = np.nan
    d["outcome"] = label_outcomes(d)
    return d.assign(match_id=match_id, innings_num=innings, date=pd.Timestamp(date),
                    batting_team=bat, bowling_team=bowl, venue_key=venue, competition_key=comp)


def test_powerplay_counts_36_legal_balls_and_drops_rain_cut_innings():
    full = _innings_balls("m1", 1, "2024-01-01", "A", "B", "V", "X", 1, n_legal=40, wides=2)
    cut = _innings_balls("m2", 1, "2024-01-02", "A", "B", "V", "X", 1, n_legal=20)
    out_early = _innings_balls("m3", 1, "2024-01-03", "A", "B", "V", "X", 1, n_legal=20, wkts=10)
    pp = innings_powerplay(pd.concat([full, cut, out_early], ignore_index=True))
    by = pp.set_index("match_id").pp_runs
    assert by["m1"] == 36 + 2, "36 legal singles plus the 2 wides bowled inside them"
    assert "m2" not in by.index, "rain-cut: fewer than 36 legal balls and not all out"
    assert by["m3"] == 20, "bowled out inside the powerplay counts"


def test_post_match_ewm_honours_halflife():
    per = pd.DataFrame({"k": ["p", "p"], "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                        "v": [2.0, 0.0]})
    w = 0.5 ** (1 / 5)
    out = post_match_ewm(per, "k", ["v"], halflife=5)
    assert out.v.iloc[1] == pytest.approx(2 * w / (1 + w))
    assert post_match_ewm(per, "k", ["v"]).v.iloc[1] == pytest.approx(
        2 * 0.5 ** (1 / HALFLIFE) / (1 + 0.5 ** (1 / HALFLIFE)))


def test_history_is_point_in_time_and_keyed_correctly():
    balls = pd.concat([
        _innings_balls("m1", 1, "2024-01-01", "A", "B", "V", "X", 1),   # A bats: 36
        _innings_balls("m1", 2, "2024-01-01", "B", "A", "V", "X", 2),   # B bats: 72
        _innings_balls("m2", 1, "2024-01-01", "C", "D", "W", "X", 3),   # same day, other venue
        _innings_balls("m3", 1, "2024-02-01", "A", "B", "V", "X", 1),   # later match
    ], ignore_index=True)
    h = history_features(balls, first_innings(balls))
    assert h.loc["m1"].isna()[["team_bat_pp", "venue_pp_recent", "league_pp_recent"]].all(), \
        "nothing before the first date"
    assert h.loc["m1", "venue_pp_n"] == 0
    assert np.isnan(h.loc["m2", "comp_pp_recent"]), "same-day matches never see each other"
    m3 = h.loc["m3"]
    assert m3.team_bat_pp == 36, "A's own batting powerplay"
    assert m3.opp_bat_pp == 72, "B's batting powerplay"
    assert m3.opp_bowl_pp_conceded == 36, "B's bowlers conceded 36 to A in m1"
    assert m3.venue_pp_recent == pytest.approx((36 + 72) / 2), "both innings at V on one date, averaged"
    assert m3.venue_pp_n == 2
    assert m3.comp_pp_recent == pytest.approx((36 + 108) / 2), "first innings only, both Jan 1 matches"
    assert m3.league_pp_recent == pytest.approx((36 + 108) / 2)
