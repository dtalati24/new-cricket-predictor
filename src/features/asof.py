# src/features/asof.py
"""Point-in-time lookup: what was known about a key strictly before a date.

Every history table here (skill counts, form, batting position, bowling share)
holds one row per (key, date) with the state AFTER that date's matches.
`as_of` reads, for each query row, the latest state from a date strictly
before the query's date. Same-day matches never see each other: Cricsheet has
no start times, so a double-header cannot be ordered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def as_of(table: pd.DataFrame, query: pd.DataFrame, by: list[str],
          cols: list[str]) -> pd.DataFrame:
    q = query[by + ["date"]].copy()
    q["_row"] = np.arange(len(q))
    t = table[by + ["date"] + cols].copy()
    for f in (q, t):               # merge_asof needs identical key dtypes
        f["date"] = f["date"].astype("datetime64[ns]")
        for c in by:
            f[c] = f[c].astype(str)
    out = pd.merge_asof(q.sort_values("date", kind="stable"),
                        t.sort_values("date", kind="stable"),
                        on="date", by=by, allow_exact_matches=False)
    out = out.sort_values("_row").reset_index(drop=True)
    out.index = query.index
    return out[cols]
