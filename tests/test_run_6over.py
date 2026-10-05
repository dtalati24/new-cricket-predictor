import numpy as np
import pandas as pd

from src.bet.backtest import decide, pnl
from src.run_6over import per_match


def test_per_match_columns_and_betting_rule():
    rows = pd.DataFrame({"match_id": ["a", "b", "c"],
                         "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                         "pre_match_line": [50.5, 50.5, 50.5],
                         "runs_at_6ov": [60.0, 60.0, 40.0]})
    raw = [np.linspace(m - 5, m + 5, 500) for m in (70, 30, 50)]
    sims = [s + 0.5 for s in raw]
    out = per_match(rows, raw, sims)
    assert list(out.columns) == ["match_id", "date", "line", "actual", "raw_median", "cal_median",
                                 "cal_p10", "cal_p90", "p_over", "p_under", "pit", "side", "pnl"]
    assert out.pit.between(0, 1).all()
    assert out.raw_median.tolist() == [np.median(s) for s in raw]
    for r, s in zip(out.itertuples(), sims):
        side = decide(float(np.mean(s > 50.5)), float(np.mean(s < 50.5)))
        assert (r.side if isinstance(r.side, str) else None) == side
        assert r.pnl == pnl(side, 50.5, r.actual)
    assert out.side.tolist()[:2] == ["over", "under"] and pd.isna(out.side[2])
