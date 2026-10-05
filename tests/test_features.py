# tests/test_features.py
import numpy as np
import pandas as pd
import pytest

from src.features.asof import as_of
from src.features.form import HALFLIFE, batter_form, post_match_ewm
from src.features.outcomes import label_outcomes
from src.features.players import (
    batting_hand, batting_positions, bowler_type, powerplay_history)

D = pd.to_datetime


def test_as_of_is_strictly_before():
    t = pd.DataFrame({"player_id": ["a", "a"], "date": D(["2020-01-01", "2020-02-01"]),
                      "x": [1.0, 2.0]})
    q = pd.DataFrame({"player_id": ["a", "a", "a", "b"],
                      "date": D(["2020-01-01", "2020-01-15", "2020-02-01", "2020-03-01"])})
    out = as_of(t, q, ["player_id"], ["x"])
    assert np.isnan(out.x.iloc[0]), "same-day state must not be visible"
    assert out.x.iloc[1] == 1.0
    assert out.x.iloc[2] == 1.0, "Feb 1 state is not visible on Feb 1"
    assert np.isnan(out.x.iloc[3]), "unknown key"
    assert list(out.index) == list(q.index)


def test_as_of_keeps_query_order_when_query_is_unsorted():
    t = pd.DataFrame({"player_id": ["a", "a"], "date": D(["2020-01-01", "2020-02-01"]),
                      "x": [1.0, 2.0]})
    q = pd.DataFrame({"player_id": ["a", "a"], "date": D(["2020-03-01", "2020-01-15"])},
                     index=[7, 3])
    out = as_of(t, q, ["player_id"], ["x"])
    assert out.loc[7, "x"] == 2.0 and out.loc[3, "x"] == 1.0


def test_post_match_ewm_includes_the_row_and_decays():
    per = pd.DataFrame({"player_id": ["p", "p"], "date": D(["2020-01-01", "2020-02-01"]),
                        "v": [2.0, 0.0]})
    out = post_match_ewm(per, "player_id", ["v"])
    w = 0.5 ** (1 / HALFLIFE)
    assert out.v.tolist() == pytest.approx([2.0, 2 * w / (1 + w)])


def test_batter_form_ignores_wides_as_balls_faced():
    b = pd.DataFrame({
        "batter_id": ["p"] * 3, "date": D(["2020-01-01"] * 3),
        "runs_batter": [4, 0, 2], "runs_total": [4, 1, 2],
        "wide": [np.nan, 1.0, np.nan], "noball": [np.nan] * 3,
        "bye": [np.nan] * 3, "legbye": [np.nan] * 3,
        "wicket_kind": [None, None, "caught"], "player_dismissed_id": [None, None, "p"]})
    b["outcome"] = label_outcomes(b)
    f = batter_form(b).iloc[0]
    assert f.bat_form_sr == pytest.approx(6 / 2), "two balls faced, the wide is not one"
    assert f.bat_form_out == pytest.approx(1 / 2)


@pytest.mark.parametrize("style,expected", [
    ("Right arm Fast medium", "pace_right"),
    ("Right-arm medium-fast", "pace_right"),
    ("Left arm Fast", "pace_left"),
    ("Right arm Offbreak", "spin_right"),
    ("Legbreak Googly", "spin_right"),
    ("Slow Left arm Orthodox", "spin_left"),
    ("Left arm Wrist spin", "spin_left"),
    ("Right arm Slow", "spin_right"),
    ("Right arm Slow medium", "pace_right"),
    ("Right arm Medium, Right arm Offbreak", "pace_right"),
    ("Right arm Bowler", "unknown"),
    ("(Unknown Arm) Slow", "unknown"),
    (None, "unknown"),
    (float("nan"), "unknown"),
])
def test_bowler_type(style, expected):
    assert bowler_type(style) == expected


@pytest.mark.parametrize("style,expected", [
    ("Right hand Bat", "right"), ("Left-hand bat", "left"),
    ("Left-arm medium", "unknown"), (None, "unknown")])
def test_batting_hand(style, expected):
    assert batting_hand(style) == expected


def test_batting_positions_follow_first_appearance():
    b = pd.DataFrame({"match_id": ["m"] * 4, "innings_num": [1] * 4,
                      "date": D(["2020-01-01"] * 4),
                      "batter_id": ["A", "A", "C", "B"],
                      "non_striker_id": ["B", "B", "B", "C"]})
    p = batting_positions(b).set_index("player_id").bat_pos
    assert p.to_dict() == {"A": 1, "B": 2, "C": 3}


def test_powerplay_history_counts_appearances_without_bowling_as_zero():
    balls = pd.DataFrame({"match_id": ["m1"] * 3, "over": [0, 0, 2],
                          "bowler_id": ["p", "p", "p"]})
    xi = pd.DataFrame({"match_id": ["m1", "m2"], "player_id": ["p", "p"],
                       "date": D(["2020-01-01", "2020-02-01"])})
    h = powerplay_history(balls, xi).set_index("date").pp_overs
    w = 0.5 ** (1 / HALFLIFE)
    assert h.iloc[0] == pytest.approx(2.0)
    assert h.iloc[1] == pytest.approx(2 * w / (1 + w))
