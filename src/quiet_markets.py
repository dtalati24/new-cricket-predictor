"""Split walk-forward results by how actively the 6-over line traded before the first ball.

    python -m src.quiet_markets --extract                 # once: price paths -> data/ticks/line_ticks.parquet
    python -m src.quiet_markets data/reports/walkforward_gh_pp [lightgbm]

"Quiet" = at most QUIET_MAX line changes in the 30 minutes before the first
ball. Basic-tier data only records a tick when the last traded line changes,
so this counts changes, not trades. See docs/quant-review-2026-11-14.md.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from src.config import TICKS

LINE_TICKS = TICKS / "line_ticks.parquet"
SIX = "1_INNING_6_OVR_LINE"
QUIET_MAX = 6
WINDOW_MIN = 30


def _scan(job):
    from src.betfair.manifest import resolve_rel
    from src.betfair.parse import read_mcm_lines
    mid, rel = job
    return [(mid, msg["pt"], float(r["ltp"]))
            for msg in read_mcm_lines(resolve_rel(rel)) for mc in msg.get("mc") or [] if mc.get("id") == mid
            for r in mc.get("rc") or [] if r.get("ltp") is not None]


def extract(workers: int = 8) -> pd.DataFrame:
    """Every last-traded-line change for each 6-over market in the comparison set."""
    from concurrent.futures import ProcessPoolExecutor
    from src.betfair.manifest import load_manifest
    from src.market.baseline import comparison_set
    c = comparison_set("all")
    rel = load_manifest().set_index("market_id").market_file_rel
    jobs = [(m, rel[m]) for m in c[c.market_type == SIX].market_id]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for out in ex.map(_scan, jobs, chunksize=20) for r in out]
    t = pd.DataFrame(rows, columns=["market_id", "pt", "ltp"])
    t["pt"] = pd.to_datetime(t.pt, unit="ms", utc=True)
    t.to_parquet(LINE_TICKS)
    return t


def changes_before(ticks: pd.DataFrame, first_ball: pd.DataFrame, minutes: int = WINDOW_MIN) -> pd.Series:
    """Line changes per market inside [first_ball - minutes, first_ball); zeros included."""
    j = ticks.merge(first_ball[["market_id", "first_ball_ts"]], on="market_id")
    inside = j[(j.pt < j.first_ball_ts) & (j.pt >= j.first_ball_ts - pd.Timedelta(minutes=minutes))]
    return inside.groupby("market_id").size().reindex(first_ball.market_id, fill_value=0)


def split(results_dir: str, model: str = "lightgbm") -> pd.DataFrame:
    """Per year and quiet/busy: matches, bets, ROI and the least-squares weight on the model.

    Weight = how far to move from the line toward the model's median: 0 means the
    line already holds everything the model knows, 1 means the line adds nothing.
    """
    from src.market.baseline import comparison_set
    files = sorted(glob.glob(f"{results_dir}/**/*_{model}_matches.parquet", recursive=True))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    c = comparison_set("all")
    c = c[c.market_type == SIX][["match_id", "market_id", "first_ball_ts"]]
    d = d.merge(c, on="match_id")
    n = changes_before(pd.read_parquet(LINE_TICKS), c).rename("changes").reset_index()
    d = d.merge(n, on="market_id")
    d["year"] = pd.to_datetime(d.date).dt.year.astype(str)
    d["group"] = np.where(d.changes <= QUIET_MAX, "quiet", "busy")

    def stats(x):
        gap, res, b = x.cal_median - x.line, x.actual - x.line, x[x.side.notna()]
        w = (gap * res).sum() / (gap ** 2).sum()
        return pd.Series({"matches": len(x), "bets": len(b), "roi": b.pnl.mean(),
                          "roi_se": b.pnl.std(ddof=1) / np.sqrt(len(b)), "model_weight": w,
                          "weight_se": np.sqrt(((res - w * gap) ** 2).sum() / (len(x) - 1) / (gap ** 2).sum()),
                          "mae_line": res.abs().mean(), "mae_model": (x.cal_median - x.actual).abs().mean()})

    both = pd.concat([d, d.assign(year="all")])
    return both.groupby(["year", "group"]).apply(stats, include_groups=False).round(3)


if __name__ == "__main__":
    if "--extract" in sys.argv:
        print(len(extract()), "line changes written to", LINE_TICKS)
    elif len(sys.argv) > 1:
        print(split(sys.argv[1], *sys.argv[2:3]).to_string())
    else:
        raise SystemExit(__doc__)
