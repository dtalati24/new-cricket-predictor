# tests/test_outcomes.py
import numpy as np
import pandas as pd
import pytest

from src.config import PARSED
from src.features.outcomes import (
    CLASSES, label_outcomes, log_loss, outcome_table, venue_key)


def _balls(**cols):
    n = len(next(iter(cols.values())))
    base = {"runs_batter": [0] * n, "runs_total": [0] * n,
            "wide": [np.nan] * n, "noball": [np.nan] * n,
            "bye": [np.nan] * n, "legbye": [np.nan] * n,
            "wicket_kind": [None] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_eleven_classes_in_fixed_order():
    assert CLASSES == ["0", "1", "2", "3", "4", "6", "5plus",
                       "wicket", "wide", "noball", "bye"]


def test_stumping_off_a_wide_is_a_wide():
    d = _balls(wide=[1.0], runs_total=[1], wicket_kind=["stumped"])
    assert label_outcomes(d).iloc[0] == "wide"


def test_run_out_off_a_noball_is_a_noball():
    d = _balls(noball=[1.0], runs_total=[1], wicket_kind=["run out"])
    assert label_outcomes(d).iloc[0] == "noball"


def test_wicket_beats_runs_off_the_bat():
    d = _balls(runs_batter=[1], runs_total=[1], wicket_kind=["run out"])
    assert label_outcomes(d).iloc[0] == "wicket"


def test_retired_hurt_is_not_a_wicket():
    d = _balls(wicket_kind=["retired hurt"])
    assert label_outcomes(d).iloc[0] == "0"


def test_leg_bye_is_bye_not_a_run():
    d = _balls(legbye=[1.0], runs_total=[1])
    assert label_outcomes(d).iloc[0] == "bye"


def test_odd_bat_runs_go_to_5plus_but_six_stays_six():
    d = _balls(runs_batter=[5, 7, 6], runs_total=[5, 7, 6])
    assert label_outcomes(d).tolist() == ["5plus", "5plus", "6"]


def test_outcome_table_probabilities_sum_to_one_per_class():
    d = _balls(wide=[1.0, 5.0, 1.0, np.nan], runs_total=[1, 5, 1, 4],
               runs_batter=[0, 0, 0, 4])
    t = outcome_table(d)
    assert np.allclose(t.groupby("outcome", observed=True).p.sum(), 1.0)
    w = t[t.outcome == "wide"].set_index("runs").p
    assert w[1] == pytest.approx(2 / 3)
    assert w[5] == pytest.approx(1 / 3)


def test_stumping_off_a_wide_keeps_its_wicket_in_the_table():
    d = _balls(wide=[1.0], runs_total=[1], wicket_kind=["stumped"])
    t = outcome_table(d)
    assert t.outcome.tolist() == ["wide"] and t.wicket.tolist() == [1]


def test_venue_key_drops_city_suffix_but_splits_ambiguous_names():
    v = pd.Series(["Wankhede Stadium, Mumbai", "Wankhede Stadium",
                   "County Ground", "County Ground"])
    c = pd.Series(["Mumbai", None, "Taunton", "Hove"])
    k = venue_key(v, c).tolist()
    assert k[0] == k[1] == "Wankhede Stadium"
    assert k[2] != k[3]


def test_log_loss_of_uniform_is_log_11():
    p = np.full((3, 11), 1 / 11)
    assert log_loss(p, [0, 5, 10]) == pytest.approx(np.log(11))


def test_log_loss_clips_zero_probability():
    p = np.zeros((1, 11)); p[0, 1] = 1.0
    assert np.isfinite(log_loss(p, [0]))


@pytest.mark.skipif(not (PARSED / "deliveries.parquet").exists(),
                    reason="parsed tables not built")
def test_load_balls_is_mens_t20_innings_one_and_two():
    from src.features.outcomes import load_balls
    d = load_balls()
    assert 2_400_000 < len(d) < 2_420_000
    assert set(d.innings_num.unique()) == {1, 2}
    assert d.date.is_monotonic_increasing
    assert d.outcome.notna().all()
