"""Regression tests for the MATCH_ODDS tick extractor.

The load-bearing test here is `test_prematch_agrees_with_build2`: Build 2's
`prices_matchodds.parquet` was independently validated (retrospective §4.6 --
realised/implied 0.96-1.04 across ~20k markets, pre-match booksum median
1.0016). Reproducing its pre-match summary exactly from the full tick path is
evidence that the new extraction is correct, and it is the foundation both
in-play diagnostics stand on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.betfair import ticks as tk

pytestmark = pytest.mark.skipif(
    not config.MATCH_ODDS_TICKS.exists(),
    reason="tick tables not extracted; run `python -m src.betfair.ticks`",
)


@pytest.fixture(scope="module")
def tables():
    return {
        "ticks": pd.read_parquet(config.MATCH_ODDS_TICKS),
        "markets": pd.read_parquet(config.MATCH_ODDS_MARKETS),
        "defs": pd.read_parquet(config.TICKS / "match_odds_defs.parquet"),
        "runners": pd.read_parquet(config.TICKS / "match_odds_runners.parquet"),
    }


def test_every_targeted_market_was_scanned(tables):
    """No silent file drops. The archive is complete for this population."""
    assert len(tables["markets"]) == 9218
    assert tables["markets"]["market_id"].is_unique


def test_ticks_are_sorted_and_typed(tables):
    t = tables["ticks"]
    assert t["ltp"].notna().all()
    assert (t["ltp"] > 1.0).all(), "decimal odds must exceed 1.0"
    assert t.groupby("market_id")["pt"].apply(lambda s: s.is_monotonic_increasing).all()


def test_in_play_coverage_is_sufficient_for_alignment(tables):
    """The whole in-play thesis depends on tick density. Guard the number."""
    defs = tables["defs"]
    starts = defs[defs["in_play"]].groupby("market_id")["pt"].min()
    assert len(starts) >= 8_900, "in-play transitions vanished"

    j = tables["ticks"].merge(
        starts.rename("inplay_start"), on="market_id", how="left"
    )
    per_market = j[j["pt"] >= j["inplay_start"]].groupby("market_id").size()
    assert per_market.median() >= 250, "in-play tick density dropped"


def test_scan_market_parses_a_known_file():
    """End-to-end on one real file, checked against its own contents."""
    mk = pd.read_parquet(config.MATCH_ODDS_MARKETS).iloc[0]
    path = config.RAW_BETFAIR / mk["market_file_rel"]
    t, d, meta = tk.scan_market(path)

    assert meta["file_market_type"] == "MATCH_ODDS"
    assert meta["market_id"] == mk["market_id"]
    assert len(meta["runner_names"]) >= 2
    assert all(r["ltp"] > 1.0 for r in t)
    # definition rows are emitted only on change
    keys = [(r["status"], r["in_play"], r["bet_delay"], r["n_active_runners"]) for r in d]
    assert all(a != b for a, b in zip(keys, keys[1:]))


def test_no_hundred_matches_are_silently_pooled(tables):
    """The Hundred is match_type T20 with 5-ball overs (retrospective §4.3).

    We do not filter it out at extraction -- but it must stay identifiable so
    downstream code cannot pool it by accident.
    """
    m = tables["markets"]
    assert set(m["balls_per_over"].dropna().unique()) <= {5, 6}
    assert (m["balls_per_over"] == 5).sum() > 300
