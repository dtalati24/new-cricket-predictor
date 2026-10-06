# tests/test_context.py
import numpy as np
import pandas as pd

from src.features.context import CONTEXT, first_innings, toss_and_season
from src.features.outcomes import competition_key


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
                                   "Indian Premier League", None]))
    assert k.tolist() == ["T20 Blast", "T20 Blast", "T20 Blast", "Indian Premier League", "unknown"]


def test_context_starts_with_toss_and_season():
    assert CONTEXT[:2] == ["chose_to_bat", "season"]
