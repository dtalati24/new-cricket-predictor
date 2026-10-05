# src/sim/simulate.py
"""The Monte Carlo innings, vectorised across simulations.

Each step asks the ball model for 11 probabilities for every live simulation,
draws a class, then draws what that delivery added (runs, wicket) from the
historical table for that class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.outcomes import CLASSES

WIDE, NOBALL = CLASSES.index("wide"), CLASSES.index("noball")
MAX_DELIVERIES = 400


def sample_rows(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = np.cumsum(p, axis=1)
    u = rng.random((len(p), 1)) * c[:, -1:]
    return np.minimum((c < u).sum(axis=1), p.shape[1] - 1)


def bowler_plan(weights, n_sims: int, n_overs: int, rng: np.random.Generator,
                max_overs: int = 4) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if (weights > 0).sum() < 2:
        raise ValueError("need at least two bowlers with positive weight")
    plan = np.empty((n_sims, n_overs), dtype=int)
    used = np.zeros((n_sims, len(weights)), dtype=int)
    rows = np.arange(n_sims)
    for o in range(n_overs):
        w = np.broadcast_to(weights, used.shape).copy()
        w[used >= max_overs] = 0
        if o:
            w[rows, plan[:, o - 1]] = 0
        if (w.sum(axis=1) == 0).any():
            raise RuntimeError(f"no bowler can legally bowl over {o + 1}")
        pick = sample_rows(w, rng)
        plan[:, o] = pick
        used[rows, pick] += 1
    return plan


@dataclass
class RunTable:
    runs: list[np.ndarray]
    wicket: list[np.ndarray]
    p: list[np.ndarray]


def run_table(table: pd.DataFrame) -> RunTable:
    runs, wkt, p = [], [], []
    for c in CLASSES:
        t = table[table.outcome == c]
        if t.empty:
            raise ValueError(f"no history for class {c!r}")
        runs.append(t.runs.to_numpy(int))
        wkt.append(t.wicket.to_numpy(int))
        p.append(t.p.to_numpy(float) / t.p.sum())
    return RunTable(runs, wkt, p)


@dataclass
class MatchSetup:
    batters: list[str]               # batting order
    bowlers: list[str]
    bowler_weights: np.ndarray
    static: np.ndarray               # (n_bat, n_bowl, n_static)


def simulate(setup: MatchSetup, predict, table: RunTable, n_sims: int = 10_000,
             legal_balls: int = 36, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_bat = len(setup.batters)
    max_wkts = min(10, n_bat - 1)
    plan = bowler_plan(setup.bowler_weights, n_sims, -(-legal_balls // 6), rng)
    runs = np.zeros(n_sims, dtype=int)
    wkts = np.zeros(n_sims, dtype=int)
    legal = np.zeros(n_sims, dtype=int)
    striker = np.zeros(n_sims, dtype=int)
    non_striker = np.ones(n_sims, dtype=int)
    next_in = np.full(n_sims, 2)
    faced = np.zeros((n_sims, n_bat), dtype=int)
    free_hit = np.zeros(n_sims, dtype=int)
    for _ in range(MAX_DELIVERIES):
        act = np.flatnonzero((legal < legal_balls) & (wkts < max_wkts))
        if act.size == 0:
            return runs
        over = legal[act] // 6
        bat = striker[act]
        X = np.column_stack([setup.static[bat, plan[act, over]], over, legal[act] % 6,
                             wkts[act], runs[act], faced[act, bat], free_hit[act]])
        cls = sample_rows(predict(X), rng)
        r = np.zeros(act.size, dtype=int)
        w = np.zeros(act.size, dtype=int)
        for c in np.unique(cls):
            m = cls == c
            j = rng.choice(len(table.p[c]), size=int(m.sum()), p=table.p[c])
            r[m] = table.runs[c][j]
            w[m] = table.wicket[c][j]
        is_legal = (cls != WIDE) & (cls != NOBALL)
        runs[act] += r
        faced[act, bat] += cls != WIDE
        free_hit[act] = cls == NOBALL
        legal[act] += is_legal
        wkts[act] += w
        # Runs actually run decide the strike; the 1-run penalty on a wide or
        # no-ball is not run.
        swap = ((r - ~is_legal) % 2 == 1) & (w == 0)
        # ponytail: the striker is always the one out and the new batter takes
        # strike; non-striker run-outs (~3% of wickets) are not modelled.
        out = act[w == 1]
        striker[out] = next_in[out]
        next_in[out] += 1
        end_over = is_legal & (legal[act] % 6 == 0)
        flip = act[swap ^ end_over]
        striker[flip], non_striker[flip] = non_striker[flip], striker[flip].copy()
    raise RuntimeError(f"innings unfinished after {MAX_DELIVERIES} deliveries")
