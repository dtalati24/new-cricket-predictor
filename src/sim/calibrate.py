"""Scoring a simulated distribution, and the one correction fitted on 2024.

corrected = a + b * median + s * (samples - median)
a, b by least squares of actual on simulated median; s chosen so 80% of
actuals land inside the corrected p10-p90.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kstest


def crps(samples: np.ndarray, y: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n ** 2)


def pit(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples)
    return float(np.mean(s < y) + 0.5 * np.mean(s == y))


def coverage(sims, ys, lo: float = 0.1, hi: float = 0.9) -> float:
    inside = [np.quantile(s, lo) <= y <= np.quantile(s, hi) for s, y in zip(sims, ys)]
    return float(np.mean(inside))


@dataclass
class Calibration:
    a: float
    b: float
    s: float


def apply(cal: Calibration, samples: np.ndarray) -> np.ndarray:
    m = np.median(samples)
    return cal.a + cal.b * m + cal.s * (np.asarray(samples, dtype=float) - m)


def fit_calibration(sims, ys) -> Calibration:
    ys = np.asarray(ys, dtype=float)
    med = np.array([np.median(s) for s in sims])
    b, a = np.polyfit(med, ys, 1)
    grid = np.linspace(0.5, 2.5, 201)
    cov = np.array([coverage([apply(Calibration(a, b, s), x) for x in sims], ys) for s in grid])
    s = float(grid[np.argmin(np.abs(cov - 0.8))])
    if s in (grid[0], grid[-1]):
        raise ValueError(f"80% coverage target not reachable inside [{grid[0]}, {grid[-1]}]: s={s}")
    return Calibration(float(a), float(b), s)


def summary(sims, ys) -> dict:
    ys = np.asarray(ys, dtype=float)
    med = np.array([np.median(s) for s in sims])
    pits = np.array([pit(s, y) for s, y in zip(sims, ys)])
    return {
        "n": int(len(ys)),
        "mae": float(np.mean(np.abs(med - ys))),
        "crps": float(np.mean([crps(s, y) for s, y in zip(sims, ys)])),
        "bias": float(np.mean([np.mean(s) for s in sims]) - ys.mean()),
        "coverage": coverage(sims, ys),
        "ks_p": float(kstest(pits, "uniform").pvalue),
    }
