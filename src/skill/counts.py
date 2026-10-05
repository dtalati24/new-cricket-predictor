"""Outcome counts per group: training totals, and post-date running totals.

Plain integer counts only. The Dirichlet-Multinomial fit reads a group's
total as its quantity of evidence; weighting balls breaks that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import as_of
from src.features.outcomes import CLASSES


def _wide(g) -> pd.DataFrame:
    w = g.size().unstack("outcome", fill_value=0).reindex(columns=CLASSES, fill_value=0)
    w.columns = list(CLASSES)
    return w


def totals(balls: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return _wide(balls.groupby(group_cols + ["outcome"], observed=True)).reset_index()


def cumulative_counts(balls: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    w = _wide(balls.groupby(group_cols + ["date", "outcome"], observed=True))
    return w.groupby(level=group_cols).cumsum().reset_index()


def counts_before(cum: pd.DataFrame, query: pd.DataFrame, group_cols: list[str]) -> np.ndarray:
    return as_of(cum, query, group_cols, CLASSES).fillna(0).to_numpy(float)
