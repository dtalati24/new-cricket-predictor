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
    assert set(by) == {"batter", "bowler", "venue", "competition", "batter_x_bowlertype", "bowler_x_battinghand",
                       "batter_x_venue", "bowler_x_venue", "batter_x_phase", "bowler_x_phase"}
    assert by["venue"].parent is None
    assert by["venue"].group_cols == ("venue_key",)
    assert by["competition"].parent is None
    assert by["competition"].group_cols == ("competition_key",)
    assert by["batter_x_venue"].parent == "batter"
    assert by["bowler_x_venue"].parent == "bowler"
    assert by["batter_x_phase"] == Level("batter_x_phase", ("batter_id", "phase"), "batter", by_phase=True)
    assert by["bowler_x_phase"] == Level("bowler_x_phase", ("bowler_id", "phase"), "bowler", by_phase=True)
    assert not any(lv.by_phase for lv in LEVELS if lv.name not in ("batter_x_phase", "bowler_x_phase"))


def test_child_estimate_uses_the_leave_own_out_parent():
    ks, kb = 7.0, 5.0
    league = np.full(11, 1 / 11)
    balls = pd.DataFrame({
        "batter_id": ["b1"] * 10 + ["b2"] * 4,
        "bowler_type": ["pace"] * 6 + ["spin"] * 4 + ["pace"] * 4,
        "date": pd.to_datetime(["2020-01-01"] * 14),
        "outcome": pd.Categorical(["4"] * 6 + ["6"] * 4 + ["1"] * 4, categories=CLASSES)})
    levels = (Level("batter", ("batter_id",), None),
              Level("batter_x_bowlertype", ("batter_id", "bowler_type"), "batter"))
    skill = SkillModel(
        league=league, k={"batter": kb, "batter_x_bowlertype": ks},
        cum={"batter": cumulative_counts(balls, ["batter_id"]),
             "batter_x_bowlertype": cumulative_counts(balls, ["batter_id", "bowler_type"])},
        levels=levels)
    q = pd.DataFrame({"batter_id": ["b1", "b1", "b2"],
                      "bowler_type": ["pace", "spin", "pace"],
                      "date": pd.to_datetime(["2020-02-01"] * 3)})
    e = estimates(skill, q)
    i4, i6, i1 = CLASSES.index("4"), CLASSES.index("6"), CLASSES.index("1")
    # b1 batter-level: 6 fours, 4 sixes. Full (not leave-own-out) estimate is returned.
    assert e["batter"][0, i4] == pytest.approx((6 + kb / 11) / (10 + kb))
    # b1 x pace (6 fours): parent = batter counts minus these = 4 sixes, shrunk to league.
    par = (np.eye(11)[i6] * 4 + kb * league) / (4 + kb)
    exp = (np.eye(11)[i4] * 6 + ks * par) / (6 + ks)
    np.testing.assert_allclose(e["batter_x_bowlertype"][0], exp)
    # b2 x pace holds ALL of b2's balls: nothing left over, parent is the league.
    exp2 = (np.eye(11)[i1] * 4 + ks * league) / (4 + ks)
    np.testing.assert_allclose(e["batter_x_bowlertype"][2], exp2)


def test_grandchild_chains_are_rejected():
    levels = (Level("a", ("batter_id",), None), Level("b", ("batter_id", "bowler_type"), "a"),
              Level("c", ("batter_id", "bowler_type", "venue_key"), "b"))
    balls = pd.DataFrame({"batter_id": ["x"], "bowler_type": ["p"], "venue_key": ["v"],
                          "date": pd.to_datetime(["2020-01-01"]),
                          "outcome": pd.Categorical(["4"], categories=CLASSES)})
    with pytest.raises(ValueError):
        fit(balls, "2021-01-01", levels)


def test_phase_level_shrinks_toward_the_phase_adjusted_overall_record():
    kb, kp = 5.0, 8.0
    league = np.full(11, 1 / 11)
    pp = np.linspace(1, 2, 11)
    pp = pp / pp.sum()                      # league powerplay rates
    balls = pd.DataFrame({                  # b1: 4 sixes, all outside the powerplay
        "batter_id": ["b1"] * 4, "phase": ["middle"] * 4,
        "date": pd.to_datetime(["2020-01-01"] * 4),
        "outcome": pd.Categorical(["6"] * 4, categories=CLASSES)})
    levels = (Level("batter", ("batter_id",), None),
              Level("batter_x_phase", ("batter_id", "phase"), "batter", by_phase=True))
    skill = SkillModel(league=league, k={"batter": kb, "batter_x_phase": kp},
                       cum={lv.name: cumulative_counts(balls, list(lv.group_cols)) for lv in levels},
                       levels=levels, phase_league={"pp": pp, "middle": league, "death": league})
    q = pd.DataFrame({"batter_id": ["b1", "new"], "phase": ["pp", "pp"],
                      "date": pd.to_datetime(["2020-02-01"] * 2)})
    e = estimates(skill, q)["batter_x_phase"]
    i6 = CLASSES.index("6")
    overall = (np.eye(11)[i6] * 4 + kb * league) / (4 + kb)     # no pp balls, so nothing left out
    exp = overall * pp / league
    np.testing.assert_allclose(e[0], exp / exp.sum())
    np.testing.assert_allclose(e[1], pp, err_msg="no history at all: the league powerplay average")


def test_unknown_phase_raises():
    levels = (Level("batter", ("batter_id",), None),
              Level("batter_x_phase", ("batter_id", "phase"), "batter", by_phase=True))
    balls = pd.DataFrame({"batter_id": ["b1"], "phase": ["pp"], "date": pd.to_datetime(["2020-01-01"]),
                          "outcome": pd.Categorical(["4"], categories=CLASSES)})
    skill = SkillModel(np.full(11, 1 / 11), {"batter": 5.0, "batter_x_phase": 5.0},
                       {lv.name: cumulative_counts(balls, list(lv.group_cols)) for lv in levels},
                       levels, phase_league={"pp": np.full(11, 1 / 11)})
    q = pd.DataFrame({"batter_id": ["b1"], "phase": ["death"], "date": pd.to_datetime(["2020-02-01"])})
    with pytest.raises(ValueError):
        estimates(skill, q)


def test_fit_learns_phase_levels():
    """Batters differ by phase around a phase-shifted overall rate; k is fitted, not at a bound."""
    rng = np.random.default_rng(1)
    league = rng.dirichlet(np.full(11, 20.0))
    shift = {"pp": np.linspace(0.7, 1.3, 11), "middle": np.ones(11)}
    rows = []
    for b in range(300):
        p_b = rng.dirichlet(60 * league)
        for ph, f in shift.items():
            base = p_b * f / (p_b * f).sum()
            draws = rng.choice(11, size=150, p=rng.dirichlet(150 * base))
            rows.append(pd.DataFrame({"batter_id": f"b{b}", "phase": ph, "o": draws}))
    d = pd.concat(rows, ignore_index=True)
    d["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 300, len(d)), "D")
    d["outcome"] = pd.Categorical(np.array(CLASSES)[d.o], categories=CLASSES)
    levels = (Level("batter", ("batter_id",), None),
              Level("batter_x_phase", ("batter_id", "phase"), "batter", by_phase=True))
    skill = fit(d, cutoff="2021-01-01", levels=levels)
    assert set(skill.phase_league) == {"pp", "middle"}
    np.testing.assert_allclose(sum(skill.phase_league.values()) / 2, skill.league, atol=0.01)
    assert 30 < skill.k["batter_x_phase"] < 1500
