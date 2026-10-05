import pandas as pd
import pytest
from src import config
from src.config import MIN_PRE_MATCH_TICKS, MODELLING_GENDER, TRAIN_END
from src.market.baseline import baseline_table, comparison_bar, comparison_set

_NEEDS = (
    config.BETFAIR_DIR / "prices_overline.parquet",
    config.MARKET_DIR / "event_match.parquet",
    config.MARKET_DIR / "first_innings_settled.parquet",
    config.MARKET_DIR / "overline_comparison.parquet",
)
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in _NEEDS),
    reason="market tables not built; run the pipeline in README 'Running it'",
)

# Pinned to Build 2's exact published figures (archive gate4_market_baseline.csv).
# The pipeline is deterministic on a fixed archive snapshot and our regenerated tables
# are DataFrame.equals to the archive's, so there is no reason for any slack beyond
# float formatting. A future legitimate archive refresh SHOULD fail this test loudly
# and force a deliberate re-measurement of the bar -- do not widen these tolerances.
# detectable_gain is pinned too: it is the number the project's success criterion is
# stated in (a gain below it is not detectable), so it must not drift silently.
EXPECTED = {
    "1_INNING_6_OVR_LINE":  {"n": 2506, "mae": 9.490422984836393,  "gain": 0.42794931348235143},
    "1_INNING_20_OVR_LINE": {"n": 2532, "mae": 24.386255924170616, "gain": 1.1169210397785752},
}

@pytest.mark.parametrize("market_type", list(EXPECTED))
def test_bar_reproduces_build2_figures(market_type):
    """The bar is a measured fact. If we cannot reproduce it, the port is wrong."""
    t = baseline_table().set_index("market_type")
    exp = EXPECTED[market_type]
    assert t.loc[market_type, "n"] == exp["n"], "sample size drifted"
    assert abs(t.loc[market_type, "market_mae"] - exp["mae"]) < 0.01, "MAE drifted"
    assert abs(t.loc[market_type, "detectable_gain"] - exp["gain"]) < 0.001, "detectable gain drifted"

def test_15_over_line_is_excluded():
    """Its market R2 is -0.58 -- worse than a constant. It must never be a target."""
    t = baseline_table()
    assert "1_INNING_15_OVR_LINE" not in set(t.market_type)

def test_single_tick_markets_are_filtered():
    """Every catastrophic line error in Build 2 had exactly one pre-match tick.

    Build 2 keeps single-tick rows in the table and flags them (`usable`), so the
    invariant is that none reach the measurement, not that none exist.
    """
    c = pd.read_parquet(config.MARKET_DIR / "overline_comparison.parquet")
    used = c[c.market_type.isin(EXPECTED) & c.usable]
    assert (used.n_ticks_pre >= MIN_PRE_MATCH_TICKS).all(), "a single-tick market survived the filter"

def test_overline_comparison_row_count():
    """Spec section 9: the joined over-line table has 7,853 rows."""
    c = pd.read_parquet(config.MARKET_DIR / "overline_comparison.parquet")
    assert len(c) == 7853


# --- comparison_set: the rows a model and the market are both scored on -------
#
# These are the LIVE bar, not the Build 2 checksum above. They move whenever the
# archive is refreshed or a filter changes, and that is the point: a silent change
# to the comparison set is the single easiest way to manufacture fake edge, so the
# figures are pinned and a drift must be explained rather than absorbed.
HOLDOUT = {
    "1_INNING_6_OVR_LINE":  {"n": 764, "mae": 10.4804, "gain": 0.8482},
    "1_INNING_20_OVR_LINE": {"n": 762, "mae": 26.9567, "gain": 2.0957},
}


@pytest.mark.parametrize("market_type", list(HOLDOUT))
def test_holdout_bar_is_pinned(market_type):
    """The number a model is actually scored against, men's 2025+ and leak-free."""
    b = comparison_bar("holdout").set_index("market_type")
    exp = HOLDOUT[market_type]
    assert b.loc[market_type, "n"] == exp["n"], "comparison set size drifted"
    assert abs(b.loc[market_type, "market_mae"] - exp["mae"]) < 0.01, "holdout MAE drifted"
    assert abs(b.loc[market_type, "detectable_gain"] - exp["gain"]) < 0.001, "gain drifted"


def test_comparison_set_applies_all_three_filters():
    """Men's only, strictly pre-first-ball, and usable. None of these is optional."""
    rows = comparison_set("all")
    assert (rows.gender == MODELLING_GENDER).all(), "a women's match reached the comparison set"
    assert (rows.pre_match_ts < rows.first_ball_ts).all(), "a post-first-ball price survived"
    assert rows.usable.all(), "an unusable market reached the comparison set"


def test_train_and_holdout_partition_all_exactly():
    """No row counted twice, none dropped -- a leaky split would overstate any gain."""
    tr, ho, al = (set(comparison_set(s).market_id) for s in ("train", "holdout", "all"))
    assert tr.isdisjoint(ho), "a market appears in both train and holdout"
    assert tr | ho == al, "train + holdout does not reconstitute the full set"


def test_the_split_is_by_date_not_at_random():
    """Time-ordered holdout everywhere. Train must end strictly before it begins."""
    tr = pd.to_datetime(comparison_set("train").date)
    ho = pd.to_datetime(comparison_set("holdout").date)
    assert tr.max() < pd.Timestamp(TRAIN_END) <= ho.min()


def test_an_unknown_split_raises_rather_than_returning_everything():
    """A typo must not silently score a model on its own training data."""
    with pytest.raises(ValueError, match="split must be"):
        comparison_set("test")


def test_the_build2_checksum_is_not_the_live_bar():
    """baseline_table() pools genders and eras; comparison_bar() does not.

    If these ever agree, one of them has stopped doing its job.
    """
    pooled = baseline_table().set_index("market_type").loc["1_INNING_6_OVR_LINE"]
    live = comparison_bar("holdout").set_index("market_type").loc["1_INNING_6_OVR_LINE"]
    assert pooled.n != live.n
    assert abs(pooled.market_mae - live.market_mae) > 0.5
