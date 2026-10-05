# tests/test_frame.py
import numpy as np
import pandas as pd
import pytest

from src.features.outcomes import CLASSES, label_outcomes
from src.model.frame import (
    FORM, SITUATION, build_frame, feature_columns, importance_groups, situation,
    skill_features)
from src.skill.shrink import LEVELS, Level, SkillModel
from src.skill.counts import cumulative_counts


def _innings(outcomes, batters):
    """One first innings; each outcome is (wide, noball, runs_batter, runs_total, wicket_kind)."""
    rows = []
    for (wide, nb, rb, rt, wk), bat in zip(outcomes, batters):
        rows.append({"match_id": "m", "batter_id": bat, "wide": wide, "noball": nb,
                     "bye": np.nan, "legbye": np.nan, "runs_batter": rb, "runs_total": rt,
                     "wicket_kind": wk})
    d = pd.DataFrame(rows)
    d["outcome"] = label_outcomes(d)
    return d


def test_situation_is_the_state_before_each_delivery():
    n = np.nan
    d = _innings([(n, n, 1, 1, None),        # legal single
                  (1.0, n, 0, 1, None),      # wide
                  (n, n, 4, 4, None),        # legal four
                  (n, n, 0, 0, "bowled"),    # wicket
                  (n, 1.0, 0, 1, None),      # no-ball
                  (n, n, 0, 0, None),        # free hit
                  (n, n, 0, 0, None)],
                 ["A", "B", "B", "B", "C", "C", "C"])
    s = situation(d)
    assert s.runs.tolist() == [0, 1, 2, 6, 6, 7, 7]
    assert s.wickets.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert s.ball_in_over.tolist() == [0, 1, 1, 2, 3, 3, 4], "wide and no-ball are not legal"
    assert s.free_hit.tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert s.batter_balls.tolist() == [0, 0, 0, 1, 0, 1, 2], "a wide is not a ball faced"


def test_over_rolls_after_six_legal_balls():
    n = np.nan
    d = _innings([(n, n, 0, 0, None)] * 8, ["A"] * 8)
    s = situation(d)
    assert s.over.tolist() == [0] * 6 + [1, 1]
    assert s.ball_in_over.tolist() == [0, 1, 2, 3, 4, 5, 0, 1]


def test_feature_columns_count_with_six_levels():
    assert len(feature_columns(LEVELS)) == 6 * 11 + 11 + len(FORM) + len(SITUATION)


def test_hint_rows_sum_to_one():
    levels = (Level("batter", ("batter_id",), None), Level("bowler", ("bowler_id",), None))
    balls = pd.DataFrame({
        "batter_id": ["b"] * 5, "bowler_id": ["w"] * 5,
        "date": pd.to_datetime(["2020-01-01"] * 5),
        "outcome": pd.Categorical(["4", "0", "1", "6", "0"], categories=CLASSES)})
    skill = SkillModel(np.full(11, 1 / 11), {"batter": 5.0, "bowler": 5.0},
                       {lv.name: cumulative_counts(balls, list(lv.group_cols)) for lv in levels},
                       levels)
    q = pd.DataFrame({"batter_id": ["b"], "bowler_id": ["w"],
                      "date": pd.to_datetime(["2020-02-01"])})
    f = skill_features(skill, q)
    hint = f[[f"hint__{c}" for c in CLASSES]].to_numpy()
    assert hint.sum() == pytest.approx(1.0, abs=1e-5)


def test_importance_groups_cover_every_feature_once_plus_all_skill():
    g = importance_groups(LEVELS)
    singles = [c for name, cols in g.items() if name != "all_skill" for c in cols]
    assert sorted(singles) == sorted(feature_columns(LEVELS))
    assert len(g["all_skill"]) == 6 * 11 + 11


def test_situation_resets_at_each_match():
    n = np.nan
    one = _innings([(n, n, 1, 1, None), (n, n, 0, 0, "bowled"), (n, n, 4, 4, None),
                    (n, 1.0, 0, 1, None)],           # last ball of match 1 is a no-ball
                   ["A", "A", "A", "A"])
    two = _innings([(n, n, 2, 2, None), (n, n, 0, 0, None)], ["A", "A"])
    two["match_id"] = "m2"
    s = situation(pd.concat([one, two], ignore_index=True))
    assert s.runs.tolist() == [0, 1, 1, 5, 0, 2]
    assert s.wickets.tolist() == [0, 0, 1, 1, 0, 0]
    assert s.over.tolist() == [0] * 6
    assert s.ball_in_over.tolist() == [0, 1, 2, 3, 0, 1]
    assert s.batter_balls.tolist() == [0, 1, 2, 3, 0, 1]
    assert s.free_hit.tolist() == [0, 0, 0, 0, 0, 0], "free hit must not leak into match 2"


def test_build_frame_on_two_matches():
    levels = (Level("batter", ("batter_id",), None), Level("bowler", ("bowler_id",), None))
    n = np.nan
    rows = []
    for mid, day in (("m1", "2020-03-01"), ("m2", "2020-03-02")):
        for inn in (1, 2):
            for wide, nb, rb, rt, wk in [(n, n, 1, 1, None), (1.0, n, 0, 1, None),
                                         (n, n, 4, 4, None), (n, n, 0, 0, "bowled")]:
                rows.append({"match_id": mid, "innings_num": inn, "date": pd.Timestamp(day),
                             "batter_id": "b", "bowler_id": "w", "wide": wide, "noball": nb,
                             "bye": n, "legbye": n, "runs_batter": rb, "runs_total": rt,
                             "wicket_kind": wk})
    balls = pd.DataFrame(rows)
    balls["outcome"] = label_outcomes(balls)
    hist = pd.DataFrame({"batter_id": ["b"] * 3, "bowler_id": ["w"] * 3,
                         "date": pd.to_datetime(["2020-01-01"] * 3),
                         "outcome": pd.Categorical(["4", "0", "1"], categories=CLASSES)})
    skill = SkillModel(np.full(11, 1 / 11), {"batter": 5.0, "bowler": 5.0},
                       {lv.name: cumulative_counts(hist, list(lv.group_cols)) for lv in levels},
                       levels)
    bat_form = pd.DataFrame({"player_id": ["b", "b"],
                             "date": pd.to_datetime(["2020-02-01", "2020-03-01"]),
                             "bat_form_sr": [100.0, 999.0], "bat_form_out": [0.1, 0.9]})
    bowl_form = pd.DataFrame({"player_id": ["w", "w"],
                              "date": pd.to_datetime(["2020-02-01", "2020-03-01"]),
                              "bowl_form_econ": [7.0, 99.0], "bowl_form_wkt": [0.2, 0.9]})
    f = build_frame(balls, skill, bat_form, bowl_form)
    assert len(f) == 8 and f.match_id.tolist() == ["m1"] * 4 + ["m2"] * 4, "innings 1 only"
    assert list(f.columns) == feature_columns(levels) + ["y", "date", "match_id"]
    assert f.y.tolist() == balls[balls.innings_num == 1].outcome.cat.codes.tolist()
    assert f.y.dtype == np.int8
    m1, m2 = f[f.match_id == "m1"], f[f.match_id == "m2"]
    # m1 is dated 2020-03-01: the 2020-03-01 form row is same-day, so only the 2020-02-01 row shows
    assert (m1.bat_form_sr == 100.0).all() and (m1.bowl_form_econ == 7.0).all()
    assert (m2.bat_form_sr == 999.0).all() and (m2.bowl_form_wkt == np.float32(0.9)).all()
    assert m2.runs.tolist() == [0, 1, 2, 6]
