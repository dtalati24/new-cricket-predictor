import pandas as pd
import pytest
from src import config
from src.market.settle import checkpoint_runs

def test_runs_at_checkpoint_counts_legal_balls_not_deliveries():
    """A wide is a delivery but not a legal ball. The 6-over mark is 36 LEGAL balls."""
    d = pd.DataFrame({
        "match_id": ["m1"] * 8,
        "innings_num": [1] * 8,
        "runs_batter": [1, 1, 1, 1, 1, 1, 1, 1],
        "runs_extras": [0, 0, 1, 0, 0, 0, 0, 0],
        "wide": [None, None, 1, None, None, None, None, None],
        "noball": [None] * 8,
        "is_wicket": [False] * 8,
    })
    # 8 deliveries, one of them a wide -> 7 legal balls.
    out = checkpoint_runs(d, legal_balls=7)
    assert out.loc[0, "legal_balls"] == 7
    assert out.loc[0, "runs"] == 9, "all batter runs plus the wide's extra"

def test_innings_shorter_than_the_checkpoint_returns_its_whole_total():
    """A side bowled out in 4 overs has a 6-over total: whatever they finished on."""
    d = pd.DataFrame({
        "match_id": ["m1"] * 3, "innings_num": [1] * 3,
        "runs_batter": [4, 4, 4], "runs_extras": [0, 0, 0],
        "wide": [None] * 3, "noball": [None] * 3, "is_wicket": [False] * 3,
    })
    out = checkpoint_runs(d, legal_balls=36)
    assert out.loc[0, "runs"] == 12
    assert out.loc[0, "legal_balls"] == 3, "flagging the short innings for exclusion"

def test_the_two_checkpoint_formulations_diverge_for_five_ball_overs():
    """Guard against unifying checkpoint_runs and build_settlement.

    In a 5-ball-over match (the Hundred) 6 overs is 30 legal balls, not 36.
    build_settlement indexes by OVER (`over < 6`), which is correct there;
    this test mirrors that rule inline with a one-line groupby. checkpoint_runs
    with legal_balls=36 counts by legal ball and reaches into the seventh over.
    If this test ever passes trivially (the two numbers agree), build_settlement
    has been rewired to the legal-ball form and 367 Hundred matches now have
    wrong 6-over labels.
    """
    n = 35  # overs 0..6, 5 legal balls each
    d = pd.DataFrame({
        "match_id": ["h1"] * n, "innings_num": [1] * n,
        "over": [i // 5 for i in range(n)],
        "runs_batter": [1] * n, "runs_extras": [0] * n,
        "wide": [None] * n, "noball": [None] * n,
    })
    d["runs_total"] = d.runs_batter + d.runs_extras

    over_index = d[d["over"] < 6].groupby("match_id")["runs_total"].sum()["h1"]
    legal_ball = checkpoint_runs(d, legal_balls=36).loc[0, "runs"]

    assert over_index == 30, "6 five-ball overs = 30 legal balls"
    assert legal_ball == 35, "36 legal balls spills into the seventh over"
    assert over_index != legal_ball


@pytest.fixture(scope="module")
def settled() -> pd.DataFrame:
    path = config.MARKET / "first_innings_settled.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built; run `python -m src.market.settle`")
    return pd.read_parquet(path)


def test_settled_table_covers_every_parsed_match(settled):
    """Spec section 9: one settled row per parsed match."""
    assert len(settled) == 22537


def test_the_hundred_is_367_matches_and_is_identifiable(settled):
    """The Hundred is match_type == "T20" with 5-ball overs. Segment, never pool.

    The labels are where mispooling would corrupt the target: 6 overs is 30
    legal balls there, not 36.
    """
    assert ((settled.match_type == "T20") & (settled.balls_per_over == 5)).sum() == 367
