"""Fast check of a per-ball model change, no simulation: minutes instead of hours.

For test year Y: fit on first-innings balls before Y, then score Y's powerplay
balls (overs 1-6). Reports per-ball log loss and the expected-runs bias per
innings: predicted sum of P(class) x E[runs | class] minus the same for the
realised classes. Negative bias = the model runs low.

    python -m src.quick_eval --years 2024,2025 --models lightgbm,logistic --halflives none,8,4,2,1
"""
from __future__ import annotations

import json
import sys

import numpy as np

from src.config import REPORTS_DIR
from src.features.outcomes import CLASSES, log_loss
from src.model.ballmodel import fit
from src.model.frame import feature_columns
from src.walkforward import _arg, prepare_cached, year_rows

OUT = REPORTS_DIR / "quick_eval.json"


def class_runs(table) -> np.ndarray:
    """E[runs | class] from the training outcome table, in CLASSES order."""
    r = (table.runs * table.p).groupby(table.outcome, observed=True).sum()
    return r.reindex(CLASSES).to_numpy(float)


def evaluate(year: int, models, halflives) -> list[dict]:
    cutoff = f"{year}-01-01"
    ctx, frame, table = prepare_cached(cutoff)
    cols = feature_columns(ctx.skill.levels)
    tr = (frame.date < cutoff).to_numpy()
    te = ((frame.date >= cutoff) & (frame.date < f"{year + 1}-01-01") & (frame.over < 6)).to_numpy()
    X, y, v = frame[cols], frame.y.to_numpy(), class_runs(table)
    test = frame.loc[te, ["match_id"]].assign(real=v[y[te]])
    lined = test.match_id.isin(set(year_rows(year).match_id))
    out = []
    for name in models:
        for h in halflives:
            m = fit(name, X.loc[tr], y[tr], frame.date[tr], cols, halflife=h)
            p = m.predict(X.loc[te])
            per = test.assign(pred=p @ v).groupby("match_id")[["pred", "real"]].sum()
            per_lined = test[lined].assign(pred=p[lined.to_numpy()] @ v).groupby("match_id")[["pred", "real"]].sum()
            row = {"year": year, "model": name, "halflife": h, "log_loss": log_loss(p, y[te]),
                   "bias_all": float((per.pred - per.real).mean()), "n_innings": len(per),
                   "bias_lined": float((per_lined.pred - per_lined.real).mean()), "n_lined": len(per_lined)}
            print(json.dumps(row), flush=True)
            out.append(row)
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    years = _arg(argv, "--years", [2024, 2025], lambda s: [int(x) for x in s.split(",")])
    models = _arg(argv, "--models", ["lightgbm", "logistic"], lambda s: s.split(","))
    halflives = _arg(argv, "--halflives", [None, 8.0, 4.0, 2.0, 1.0],
                     lambda s: [None if x == "none" else float(x) for x in s.split(",")])
    rows = [r for yr in years for r in evaluate(yr, models, halflives)]
    OUT.write_text(json.dumps(rows, indent=2))
