"""Bet the 6-over line when the simulated probability beats evens after commission.

Line markets pay about even money. With commission c on winnings, a bet at
probability p breaks even when p * (1 - c) = 1 - p, i.e. p = 1 / (2 - c).
"""
from __future__ import annotations

import numpy as np

COMMISSION = 0.05
THRESHOLD = 1 / (2 - COMMISSION)


def decide(p_over: float, p_under: float):
    if p_over > THRESHOLD:
        return "over"
    if p_under > THRESHOLD:
        return "under"
    return None


def pnl(side, line: float, actual: float) -> float:
    if side is None or actual == line:
        return 0.0
    won = actual > line if side == "over" else actual < line
    return (1 - COMMISSION) if won else -1.0


def backtest(sims, lines, actuals) -> dict:
    results = []
    for s, line, y in zip(sims, lines, actuals):
        side = decide(float(np.mean(s > line)), float(np.mean(s < line)))
        if side is not None:
            results.append(pnl(side, line, y))
    r = np.array(results)
    return {
        "n_matches": int(len(lines)),
        "n_bets": int(len(r)),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "pnl": float(r.sum()),
        "roi": float(r.mean()) if len(r) else float("nan"),
    }
