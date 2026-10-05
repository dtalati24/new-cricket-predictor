import numpy as np
import pytest

from src.bet.backtest import THRESHOLD, backtest, decide, pnl
from src.sim.calibrate import Calibration, apply, coverage, crps, fit_calibration, pit


def test_crps_of_a_point_mass_is_absolute_error():
    assert crps(np.full(1000, 60.0), 50.0) == pytest.approx(10.0)


def test_crps_prefers_a_well_spread_forecast_to_a_confident_wrong_one():
    rng = np.random.default_rng(0)
    assert crps(rng.normal(50, 12, 5000), 50.0) < crps(np.full(5000, 80.0), 50.0)


def test_pit_counts_ties_as_half():
    assert pit(np.array([1, 2, 2, 3]), 2) == pytest.approx(0.5)


def test_coverage_of_p10_p90():
    s = [np.arange(100.0)] * 4
    assert coverage(s, [50, 5, 95, 50]) == pytest.approx(0.5)


def test_calibration_removes_a_constant_bias():
    rng = np.random.default_rng(1)
    truth = rng.normal(50, 10, 300)
    # every forecast 15 runs low on average; per-match noise keeps the 80% target reachable
    sims = [rng.normal(t - 15 + rng.normal(0, 7), 10, 2000) for t in truth]
    cal = fit_calibration(sims, truth)
    fixed = [apply(cal, s) for s in sims]
    assert abs(np.mean([np.median(f) for f in fixed]) - truth.mean()) < 1.0


def test_apply_is_identity_for_the_neutral_calibration():
    s = np.array([40.0, 50.0, 70.0])
    np.testing.assert_allclose(apply(Calibration(0.0, 1.0, 1.0), s), s)


def test_threshold_breaks_even_after_commission():
    assert THRESHOLD == pytest.approx(1 / 1.95)
    assert THRESHOLD * 0.95 - (1 - THRESHOLD) == pytest.approx(0.0)


def test_decide():
    assert decide(0.55, 0.45) == "over"
    assert decide(0.45, 0.55) == "under"
    assert decide(0.51, 0.49) is None


def test_pnl_win_lose_push():
    assert pnl("over", 50.5, 60) == pytest.approx(0.95)
    assert pnl("over", 50.5, 40) == -1.0
    assert pnl("under", 50.0, 50) == 0.0
    assert pnl(None, 50.5, 60) == 0.0


def test_backtest_totals():
    sims = [np.full(100, 70.0), np.full(100, 30.0), np.full(100, 50.0)]
    # over wins; under loses; under loses
    out = backtest(sims, np.array([50.5, 50.5, 50.5]), np.array([60, 60, 60]))
    assert out["n_bets"] == 3
    assert out["pnl"] == pytest.approx(0.95 - 1.0 - 1.0)
    assert out["hit_rate"] == pytest.approx(1 / 3)


def test_calibration_raises_when_the_coverage_target_is_unreachable():
    rng = np.random.default_rng(2)
    meds = rng.normal(50, 10, 200)
    sims = [np.full(100, m) for m in meds]          # point masses: no spread multiplier widens them
    ys = meds + rng.normal(0, 10, 200)
    with pytest.raises(ValueError, match="80% coverage"):
        fit_calibration(sims, ys)
