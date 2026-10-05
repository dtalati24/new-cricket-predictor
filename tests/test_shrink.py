import numpy as np
import pandas as pd
import pytest

from src.features.outcomes import CLASSES
from src.skill.counts import cumulative_counts
from src.skill.shrink import (
    LEVELS, ConcentrationAtBoundError, Level, SkillModel, estimates, fit, fit_k, shrink)


def test_shrink_is_the_posterior_mean():
    out = shrink(np.array([[8.0, 2.0]]), np.array([[0.5, 0.5]]), k=10.0)
    np.testing.assert_allclose(out, [[0.65, 0.35]])


def test_a_group_with_no_data_returns_the_parent():
    parent = np.array([[0.3, 0.7]])
    np.testing.assert_allclose(shrink(np.zeros((1, 2)), parent, k=50.0), parent)


def test_rows_sum_to_one():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(20, 11)).astype(float)
    np.testing.assert_allclose(shrink(counts, np.full((20, 11), 1 / 11), 30.0).sum(1), 1.0)


def test_fit_recovers_a_known_concentration():
    rng = np.random.default_rng(42)
    true_k, parent = 80.0, np.array([0.4, 0.35, 0.25])
    counts = np.array([rng.multinomial(120, rng.dirichlet(true_k * parent))
                       for _ in range(600)], dtype=float)
    k_hat = fit_k(counts, np.tile(parent, (600, 1)))
    assert 0.5 * true_k < k_hat < 2.0 * true_k


def test_bound_pinned_fit_raises():
    """Groups with no scatter around the parent push k to its upper bound."""
    counts = np.tile(np.array([50.0, 50.0]), (500, 1))
    with pytest.raises(ConcentrationAtBoundError):
        fit_k(counts, np.full((500, 2), 0.5))


def test_estimates_use_only_earlier_dates():
    balls = pd.DataFrame({
        "batter_id": ["b1"] * 20,
        "date": pd.to_datetime(["2020-01-01"] * 10 + ["2020-02-01"] * 10),
        "outcome": pd.Categorical(["4"] * 10 + ["6"] * 10, categories=CLASSES)})
    lv = Level("batter", ("batter_id",), None)
    skill = SkillModel(league=np.full(11, 1 / 11), k={"batter": 1.0},
                       cum={"batter": cumulative_counts(balls, ["batter_id"])}, levels=(lv,))
    q = pd.DataFrame({"batter_id": ["b1"] * 3,
                      "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])})
    e = estimates(skill, q)["batter"]
    i4, i6 = CLASSES.index("4"), CLASSES.index("6")
    np.testing.assert_allclose(e[0], 1 / 11)
    assert e[1, i4] == pytest.approx((10 + 1 / 11) / 11)
    assert e[1, i6] == pytest.approx((1 / 11) / 11)
    assert e[2, i6] == pytest.approx((10 + 1 / 11) / 21)


def _synthetic(seed=0, n_batters=300, balls_each=300):
    """Batters scatter around the league (k=60); batter x type around the batter (k=200)."""
    rng = np.random.default_rng(seed)
    league = rng.dirichlet(np.full(11, 20.0))
    rows = []
    for b in range(n_batters):
        p_b = rng.dirichlet(60 * league)
        for t in ("pace_right", "spin_left"):
            p_bt = rng.dirichlet(200 * p_b)
            draws = rng.choice(11, size=balls_each // 2, p=p_bt)
            rows.append(pd.DataFrame({"batter_id": f"b{b}", "bowler_type": t, "o": draws}))
    d = pd.concat(rows, ignore_index=True)
    d["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 300, len(d)), "D")
    d["outcome"] = pd.Categorical(np.array(CLASSES)[d.o], categories=CLASSES)
    return d


def test_fit_runs_the_chain_and_orders_k_sensibly():
    levels = (Level("batter", ("batter_id",), None),
              Level("batter_x_bowlertype", ("batter_id", "bowler_type"), "batter"))
    skill = fit(_synthetic(), cutoff="2021-01-01", levels=levels)
    assert skill.k["batter"] < skill.k["batter_x_bowlertype"], \
        "the crossed level scatters less, so it must shrink harder"


def test_levels_are_the_spec_hierarchy():
    by = {lv.name: lv for lv in LEVELS}
    assert set(by) == {"batter", "bowler", "batter_x_bowlertype", "bowler_x_battinghand",
                       "batter_x_venue", "bowler_x_venue"}
    assert by["batter_x_venue"].parent == "batter"
    assert by["bowler_x_venue"].parent == "bowler"
