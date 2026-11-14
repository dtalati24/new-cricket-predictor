import pandas as pd

from src.quiet_markets import changes_before


def test_changes_before_counts_only_the_window_and_keeps_zeros():
    fb = pd.DataFrame({"market_id": ["a", "b"], "first_ball_ts": pd.to_datetime(["2024-01-01 12:00"] * 2, utc=True)})
    ticks = pd.DataFrame({"market_id": ["a"] * 4, "pt": pd.to_datetime(
        ["2024-01-01 11:00", "2024-01-01 11:30", "2024-01-01 11:59", "2024-01-01 12:00"], utc=True)})
    n = changes_before(ticks, fb, minutes=30)
    assert n["a"] == 2          # 11:30 and 11:59; 11:00 is too early, 12:00 is the first ball
    assert n["b"] == 0
