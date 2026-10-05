# tests/test_simulate.py
import numpy as np
import pandas as pd
import pytest

from src.features.outcomes import CLASSES
from src.sim.simulate import MatchSetup, bowler_plan, run_table, simulate

N_STATIC = 1


def _table():
    rows = [(c, int(c), 0) for c in ["0", "1", "2", "3", "4", "6"]]
    rows += [("5plus", 5, 0), ("wicket", 0, 1), ("wide", 1, 0), ("noball", 1, 0), ("bye", 1, 0)]
    t = pd.DataFrame(rows, columns=["outcome", "runs", "wicket"])
    t["n"] = 1
    t["p"] = 1.0
    return run_table(t)


def _setup():
    # static[b, w, 0] = batting position, so tests can see who is on strike
    static = np.repeat(np.arange(11, dtype=float)[:, None, None], 11, axis=1)
    return MatchSetup([f"b{i}" for i in range(11)], [f"w{i}" for i in range(11)],
                      np.ones(11), static)


def _always(cls):
    i = CLASSES.index(cls)
    def predict(X):
        p = np.zeros((len(X), 11)); p[:, i] = 1.0
        return p
    return predict


def test_always_singles_scores_36():
    assert (simulate(_setup(), _always("1"), _table(), n_sims=50) == 36).all()


def test_always_four_scores_144():
    assert (simulate(_setup(), _always("4"), _table(), n_sims=50) == 144).all()


def test_all_out_stops_the_innings():
    assert (simulate(_setup(), _always("wicket"), _table(), n_sims=50) == 0).all()


def test_wides_alone_never_finish():
    with pytest.raises(RuntimeError):
        simulate(_setup(), _always("wide"), _table(), n_sims=5)


def test_wides_do_not_use_up_legal_balls():
    calls = {"n": 0}
    def predict(X):
        cls = "wide" if calls["n"] % 2 == 0 else "1"
        calls["n"] += 1
        return _always(cls)(X)
    # 36 legal singles plus 36 wides of 1 run each
    assert (simulate(_setup(), predict, _table(), n_sims=20) == 72).all()


def test_noball_makes_the_next_delivery_a_free_hit():
    seen = []
    def predict(X):
        seen.append(X[0, N_STATIC + 5])
        return _always("noball" if len(seen) == 1 else "0")(X)
    simulate(_setup(), predict, _table(), n_sims=3)
    assert seen[:3] == [0, 1, 0]


def test_strike_rotates_on_singles_and_at_end_of_over():
    seen = []
    def predict(X):
        seen.append(X[0, 0])
        return _always("0")(X)
    simulate(_setup(), predict, _table(), n_sims=1)
    assert seen[:6] == [0] * 6 and seen[6:12] == [1] * 6, "dots: openers swap only at the over"
    seen.clear()
    def singles(X):
        seen.append(X[0, 0])
        return _always("1")(X)
    simulate(_setup(), singles, _table(), n_sims=1)
    assert seen[:7] == [0, 1, 0, 1, 0, 1, 1], "single on the 6th ball: same batter keeps strike"


def test_new_batter_comes_in_at_the_fall_of_a_wicket():
    seen = []
    def predict(X):
        seen.append(X[0, 0])
        return _always("wicket" if len(seen) == 1 else "0")(X)
    simulate(_setup(), predict, _table(), n_sims=1)
    assert seen[1] == 2


def test_bowler_plan_respects_the_rules():
    # 6 bowlers, not 5: with exactly 5 x 4 = 20 overs a random order can corner itself
    plan = bowler_plan(np.ones(6), n_sims=500, n_overs=20, rng=np.random.default_rng(0))
    assert (plan[:, 1:] != plan[:, :-1]).all(), "no bowler bowls consecutive overs"
    counts = np.stack([(plan == b).sum(axis=1) for b in range(6)], axis=1)
    assert (counts <= 4).all()


def test_bowler_plan_raises_when_the_rules_cannot_be_met():
    with pytest.raises(RuntimeError):
        bowler_plan(np.ones(2), n_sims=1, n_overs=10, rng=np.random.default_rng(0))


def test_bowler_plan_follows_the_weights():
    plan = bowler_plan(np.array([3.0, 1.0, 1e-6]), n_sims=4000, n_overs=1,
                       rng=np.random.default_rng(0))
    share = (plan[:, 0] == 0).mean()
    assert 0.72 < share < 0.78
