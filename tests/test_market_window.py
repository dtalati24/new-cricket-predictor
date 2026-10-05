import pandas as pd
from src.market.window import first_ball_times, window_bounds, ticks_per_market


def test_first_ball_is_the_first_in_play_transition():
    """A market can go in-play, suspend and resume many times. We want the first."""
    defs = pd.DataFrame({
        "market_id": ["1.1", "1.1", "1.1", "1.2"],
        "pt": pd.to_datetime([
            "2026-01-01 09:00", "2026-01-01 10:00",
            "2026-01-01 11:00", "2026-01-02 14:00"], utc=True),
        "in_play": [False, True, True, True],
        "status": ["OPEN", "OPEN", "SUSPENDED", "OPEN"],
    })
    out = first_ball_times(defs).set_index("market_id")
    assert out.loc["1.1", "first_ball_ts"] == pd.Timestamp("2026-01-01 10:00", tz="UTC")
    assert out.loc["1.2", "first_ball_ts"] == pd.Timestamp("2026-01-02 14:00", tz="UTC")


def test_market_that_never_goes_in_play_is_absent():
    """An abandoned match has no first ball. It must not get a fabricated one."""
    defs = pd.DataFrame({
        "market_id": ["1.3"],
        "pt": pd.to_datetime(["2026-01-01 09:00"], utc=True),
        "in_play": [False],
        "status": ["OPEN"],
    })
    assert len(first_ball_times(defs)) == 0


def test_window_bounds_is_half_open_and_ends_at_first_ball():
    """The bet must be placed strictly before the first ball."""
    fb = pd.Timestamp("2026-01-01 14:00", tz="UTC")
    lo, hi = window_bounds(fb, minutes=30)
    assert lo == pd.Timestamp("2026-01-01 13:30", tz="UTC")
    assert hi == fb


def test_ticks_per_market_counts_half_open_window():
    """Tick at window start counts; tick at first ball does not; earlier ones don't."""
    fb = pd.DataFrame({
        "market_id": ["1.1", "1.2"],
        "first_ball_ts": pd.to_datetime(["2026-01-01 14:00", "2026-01-01 14:00"], utc=True),
    })
    ticks = pd.DataFrame({
        "market_id": ["1.1"] * 4,
        "pt": pd.to_datetime([
            "2026-01-01 13:29", "2026-01-01 13:30",
            "2026-01-01 13:59", "2026-01-01 14:00"], utc=True),
    })
    n = ticks_per_market(fb, ticks, minutes=30)
    assert n.loc["1.1"] == 2      # 13:30 in, 13:59 in; 13:29 and 14:00 out
    assert n.loc["1.2"] == 0      # market with no ticks still gets a row
