# tests/test_speed.py
import numpy as np
import pandas as pd
import pytest

import src.sim.simulate as simmod
from src.features.outcomes import CLASSES
from src.sim.setup import simulate_setups
from src.sim.simulate import MatchSetup, run_table

COLS = ["s0", "over", "ball_in_over", "wickets", "runs", "batter_balls", "free_hit"]


def _table():
    rows = [(c, int(c), 0) for c in ["0", "1", "2", "3", "4", "6"]]
    rows += [("5plus", 5, 0), ("wicket", 0, 1), ("wide", 1, 0), ("noball", 1, 0), ("bye", 1, 0)]
    t = pd.DataFrame(rows, columns=["outcome", "runs", "wicket"])
    t["n"] = 1
    t["p"] = 1.0
    return run_table(t)


def _setup(shift):
    static = np.repeat(np.arange(11, dtype=float)[:, None, None], 11, axis=1) + shift
    return MatchSetup([f"b{i}" for i in range(11)], [f"w{i}" for i in range(11)],
                      np.ones(11), static)


class FakeModel:
    """Deterministic, row-dependent probabilities; picklable for the process pool."""
    name = "fake"
    est = None

    def __init__(self):
        self.rows_seen = 0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.rows_seen += len(X)
        a = X.to_numpy(float)
        z = np.sin(a @ np.arange(1, a.shape[1] + 1)[:, None] * np.arange(1, 12)[None, :])
        p = np.exp(z)
        p[:, CLASSES.index("wide")] *= 0.1
        return p / p.sum(axis=1, keepdims=True)


def test_dedupe_gives_identical_results_with_fewer_predictions(monkeypatch):
    setups, table = [_setup(0.0)], _table()
    m1 = FakeModel()
    monkeypatch.setattr(simmod, "DEDUPE", True)
    a = simulate_setups(setups, m1, table, COLS, n_sims=500)
    m2 = FakeModel()
    monkeypatch.setattr(simmod, "DEDUPE", False)
    b = simulate_setups(setups, m2, table, COLS, n_sims=500)
    np.testing.assert_array_equal(a[0], b[0])
    assert m1.rows_seen < m2.rows_seen, "identical early states are predicted once"


def test_parallel_equals_serial():
    setups, table = [_setup(s) for s in (0.0, 0.5, 1.0)], _table()
    serial = simulate_setups(setups, FakeModel(), table, COLS, n_sims=300, workers=1)
    parallel = simulate_setups(setups, FakeModel(), table, COLS, n_sims=300, workers=2)
    for s, p in zip(serial, parallel):
        np.testing.assert_array_equal(s, p)
