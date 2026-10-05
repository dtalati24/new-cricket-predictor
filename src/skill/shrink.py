"""Empirical-Bayes shrinkage of 11-class outcome rates.

    estimate = (counts + k * parent) / (n + k)

One k per level, fitted by maximising the Dirichlet-Multinomial marginal
likelihood of the training-slice counts given the parent. Levels:

    league -> batter -> batter x bowler_type
    league -> bowler -> bowler x batting_hand
    batter x venue (parent batter), bowler x venue (parent bowler)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.features.outcomes import CLASSES, log_loss
from src.skill.counts import counts_before, cumulative_counts, totals

LOG_K_BOUNDS = (np.log(0.1), np.log(1e6))
_BOUND_TOL = 1e-2
MIN_MEDIAN_BALLS = 30


class ConcentrationAtBoundError(RuntimeError):
    """k landed on its search bound: the fit failed, it did not 'shrink hard'."""


def _clean(parent: np.ndarray) -> np.ndarray:
    p = np.clip(parent, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


def dm_loglik(log_k: float, counts: np.ndarray, parent: np.ndarray) -> float:
    k = np.exp(log_k)
    a = k * parent
    n = counts.sum(axis=1)
    return float(np.sum(gammaln(k) - gammaln(n + k)
                        + (gammaln(counts + a) - gammaln(a)).sum(axis=1)))


def fit_k(counts: np.ndarray, parent: np.ndarray) -> float:
    parent = _clean(parent)
    res = minimize_scalar(lambda lk: -dm_loglik(lk, counts, parent),
                          bounds=LOG_K_BOUNDS, method="bounded", options={"xatol": 1e-4})
    if min(res.x - LOG_K_BOUNDS[0], LOG_K_BOUNDS[1] - res.x) < _BOUND_TOL:
        raise ConcentrationAtBoundError(f"k={np.exp(res.x):.4g} is at its search bound")
    return float(np.exp(res.x))


def shrink(counts: np.ndarray, parent: np.ndarray, k: float) -> np.ndarray:
    return (counts + k * _clean(parent)) / (counts.sum(axis=1, keepdims=True) + k)


@dataclass(frozen=True)
class Level:
    name: str
    group_cols: tuple[str, ...]
    parent: str | None          # None = league vector


LEVELS = (
    Level("batter", ("batter_id",), None),
    Level("bowler", ("bowler_id",), None),
    Level("batter_x_bowlertype", ("batter_id", "bowler_type"), "batter"),
    Level("bowler_x_battinghand", ("bowler_id", "batting_hand"), "bowler"),
    Level("batter_x_venue", ("batter_id", "venue_key"), "batter"),
    Level("bowler_x_venue", ("bowler_id", "venue_key"), "bowler"),
)


@dataclass
class SkillModel:
    league: np.ndarray                 # (11,), training slice
    k: dict[str, float]                # fitted on the training slice
    cum: dict[str, pd.DataFrame]       # running counts over ALL balls; read strictly before
    levels: tuple[Level, ...]


def _parent_of(lv: Level, by_name: dict[str, Level]) -> Level | None:
    """The parent Level, checked to be a top level (own parent = league). Deeper chains unsupported."""
    if lv.parent is None:
        return None
    par = by_name[lv.parent]
    if par.parent is not None:
        raise ValueError(f"{lv.name}: parent {par.name} has its own parent; unsupported")
    return par


def leave_own_out_parent(parent_counts: np.ndarray, own_counts: np.ndarray,
                         league: np.ndarray, k_parent: float) -> np.ndarray:
    """A child's parent vector: the parent group's estimate with the child's own balls removed.

    The parent's counts include the child's balls; leaving them in makes the child look
    unscattered around its parent (k -> bound). fit() and estimates() both use this, so the
    fitted k is applied against the same kind of parent. Parent's own parent is the league.
    """
    return shrink(parent_counts - own_counts, np.tile(league, (len(own_counts), 1)), k_parent)


def fit(balls: pd.DataFrame, cutoff: str, levels: tuple[Level, ...] = LEVELS) -> SkillModel:
    """k from balls before `cutoff`; running counts from all balls (as_of keeps them honest)."""
    train = balls[balls.date < cutoff]
    league = train.outcome.value_counts().reindex(CLASSES, fill_value=0).to_numpy(float)
    league = league / league.sum()
    by_name = {lv.name: lv for lv in levels}
    for lv in levels:
        _parent_of(lv, by_name)  # reject unsupported chains before any fitting
    seen: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}  # keys, counts
    k: dict[str, float] = {}
    for lv in levels:
        cols = list(lv.group_cols)
        tot = totals(train, cols)
        counts = tot[CLASSES].to_numpy(float)
        par = _parent_of(lv, by_name)
        if par is None:
            parent = np.tile(league, (len(tot), 1))
        else:
            pkeys, pcounts = seen[par.name]
            pcols = list(par.group_cols)
            idx = tot[pcols].merge(pkeys.assign(_i=np.arange(len(pkeys))),
                                   on=pcols, how="left")["_i"].to_numpy()
            parent = leave_own_out_parent(pcounts[idx], counts, league, k[par.name])
        k[lv.name] = fit_k(counts, parent)
        seen[lv.name] = (tot[cols].reset_index(drop=True), counts)
    cum = {lv.name: cumulative_counts(balls, list(lv.group_cols)) for lv in levels}
    return SkillModel(league, k, cum, tuple(levels))


def estimates(skill: SkillModel, query: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-level estimates. A child shrinks toward its leave-own-out parent (as k was fitted);
    out[parent] itself stays the full parent estimate."""
    by_name = {lv.name: lv for lv in skill.levels}
    out: dict[str, np.ndarray] = {}
    for lv in skill.levels:
        counts = counts_before(skill.cum[lv.name], query, list(lv.group_cols))
        par = _parent_of(lv, by_name)
        if par is None:
            parent = np.tile(skill.league, (len(query), 1))
        else:
            pcounts = counts_before(skill.cum[par.name], query, list(par.group_cols))
            parent = leave_own_out_parent(pcounts, counts, skill.league, skill.k[par.name])
        out[lv.name] = shrink(counts, parent, skill.k[lv.name])
    return out


def level_report(skill: SkillModel, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """The density gate: median group size >= 30 and held-out log loss beats the parent."""
    est = estimates(skill, val)
    y = val.outcome.cat.codes.to_numpy()
    league_ll = log_loss(np.tile(skill.league, (len(val), 1)), y)
    rows = []
    for lv in skill.levels:
        n = totals(train, list(lv.group_cols))[CLASSES].sum(axis=1)
        parent_ll = league_ll if lv.parent is None else log_loss(est[lv.parent], y)
        ll = log_loss(est[lv.name], y)
        rows.append({"level": lv.name, "n_groups": len(n), "median_balls": float(n.median()),
                     "k": skill.k[lv.name], "log_loss": ll, "parent_log_loss": parent_ll,
                     "gain": parent_ll - ll})
    out = pd.DataFrame(rows)
    out["passes"] = (out.median_balls >= MIN_MEDIAN_BALLS) & (out.gain > 0)
    return out
