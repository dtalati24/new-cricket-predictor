"""Tests for the Betfair-runner -> Cricsheet-team assignment.

The unit tests pin the behaviours the mapping has to get right on names that
actually appear in the archive. The integration tests run against the built
artefact and assert the coverage and the independent winner-agreement check,
so a silent regression in the mapping cannot pass unnoticed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.betfair import runner_teams as rt

MAP_PATH = config.TICKS / "runner_team_map.parquet"


# --------------------------------------------------------------------------
# Unit
# --------------------------------------------------------------------------
def test_normalise_strips_accents_punctuation_and_expands_ampersand():
    assert rt.normalise("St Kitts & Nevis Patriots") == "st kitts and nevis patriots"
    assert rt.normalise("Wellington Firebirds!") == "wellington firebirds"
    assert rt.normalise(None) == ""


def test_tokens_drop_competition_noise_but_never_return_empty():
    assert "wbbl" not in rt.tokens("Brisbane Heat WBBL")
    assert rt.tokens("Brisbane Heat WBBL") == ["brisbane", "heat"]
    # A name made only of noise tokens must still yield tokens, otherwise the
    # similarity score silently becomes 0 for both candidates.
    assert rt.tokens("Women") == ["women"]


@pytest.mark.parametrize(
    "r1, r2, team_a, team_b",
    [
        # Betfair short forms against Cricsheet's full franchise names.
        ("Delhi", "Punjab", "Delhi Daredevils", "Kings XI Punjab"),
        # Competition suffix on both runners.
        ("Brisbane Heat WBBL", "Perth Scorchers WBBL",
         "Brisbane Heat", "Perth Scorchers"),
        # Gender qualifier that Cricsheet does not carry in the team field.
        ("India Women", "Australia Women", "India", "Australia"),
        # Ampersand and spelled-out "and".
        ("St Kitts & Nevis Patriots", "Guyana Amazon Warriors",
         "St Kitts and Nevis Patriots", "Guyana Amazon Warriors"),
        # Franchise rename: Cricsheet uses the new name, Betfair the old city.
        ("Bengaluru", "Chennai", "Royal Challengers Bengaluru",
         "Chennai Super Kings"),
    ],
)
def test_assign_market_resolves_real_archive_name_pairs(r1, r2, team_a, team_b):
    res = rt.assign_market(r1, r2, team_a, team_b)
    assert res["team_1"] == team_a
    assert res["team_2"] == team_b
    assert res["margin"] >= rt.MIN_MARGIN
    assert res["min_sim"] >= rt.MIN_SIM


def test_assign_market_is_order_invariant():
    a = rt.assign_market("Delhi", "Punjab", "Delhi Daredevils", "Kings XI Punjab")
    b = rt.assign_market("Punjab", "Delhi", "Delhi Daredevils", "Kings XI Punjab")
    assert a["team_1"] == b["team_2"]
    assert a["team_2"] == b["team_1"]
    assert a["margin"] == pytest.approx(b["margin"])


def test_indistinguishable_runners_produce_no_margin():
    """Two identical runner names cannot be assigned; margin must be zero."""
    res = rt.assign_market("Hampshire", "Hampshire", "Hampshire", "Sussex")
    assert res["margin"] == pytest.approx(0.0)
    assert res["margin"] < rt.MIN_MARGIN


def test_unrelated_names_fail_the_similarity_floor():
    res = rt.assign_market("Kolkata", "Mumbai", "Sydney Sixers", "Perth Scorchers")
    assert res["min_sim"] < rt.MIN_SIM


def test_squad_qualifier_mismatch_is_detected():
    # The archive really does carry "Bangladesh U19" joined to a senior
    # "Bangladesh" fixture; that is a bad event join, not a name variant.
    assert rt._squad_mismatch("Bangladesh U19", "Bangladesh")
    assert rt._squad_mismatch("New Zealand A", "New Zealand")
    assert not rt._squad_mismatch("Bangladesh", "Bangladesh")
    assert not rt._squad_mismatch("India Women", "India")


# --------------------------------------------------------------------------
# Integration -- against the built artefact
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mapping() -> pd.DataFrame:
    if not MAP_PATH.exists():
        pytest.skip(f"{MAP_PATH} not built")
    return pd.read_parquet(MAP_PATH)


def test_mapping_covers_every_runner_row(mapping):
    runners = pd.read_parquet(config.TICKS / "match_odds_runners.parquet")
    assert len(mapping) == len(runners)
    assert mapping["market_id"].nunique() == runners["market_id"].nunique()


def test_runner_coverage_is_at_least_99_percent(mapping):
    assert mapping["resolved"].mean() >= 0.99


def test_resolved_markets_map_two_runners_onto_the_two_distinct_teams(mapping):
    markets = pd.read_parquet(config.MATCH_ODDS_MARKETS)
    res = mapping[mapping["resolved"]].merge(
        markets[["market_id", "team_a", "team_b"]], on="market_id", how="left"
    )
    g = res.groupby("market_id")
    # Exactly two resolved runners per market, and they take different teams.
    assert (g.size() == 2).all()
    assert (g["team"].nunique() == 2).all()
    both = g.agg(teams=("team", lambda s: set(s)), a=("team_a", "first"),
                 b=("team_b", "first"))
    assert (both["teams"] == both.apply(lambda r: {r["a"], r["b"]}, axis=1)).all()


def test_winner_agreement_against_independent_cricsheet_outcome(mapping):
    """Betfair's WINNER flag and Cricsheet's outcome_winner are independent
    of the name-based mapping, so their agreement is a genuine external check.
    """
    stats = rt.validate(mapping)
    assert stats["checkable_markets"] > 8000
    assert stats["winner_agreement"] >= 0.99


def test_unresolved_rows_carry_a_reason_and_no_team(mapping):
    bad = mapping[~mapping["resolved"]]
    assert bad["team"].isna().all()
    assert (bad["reason"].astype(str).str.len() > 0).all()
