# tests/test_frame.py
import numpy as np
import pandas as pd
import pytest

from src.features.outcomes import CLASSES, label_outcomes
from src.model.frame import (
    FORM, SITUATION, feature_columns, importance_groups, situation, skill_features)
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
