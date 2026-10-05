# tests/test_ingest_parse.py
import pandas as pd
import pytest
from src.config import PARSED

pytestmark = pytest.mark.skipif(
    not (PARSED / "deliveries.parquet").exists() or not (PARSED / "matches.parquet").exists(),
    reason="parsed tables not built; run `python -m src.ingest.parse`",
)


def test_deliveries_table_shape():
    """The parsed corpus must cover every match JSON on disk."""
    d = pd.read_parquet(PARSED / "deliveries.parquet")
    m = pd.read_parquet(PARSED / "matches.parquet")
    assert len(m) == 22537, f"expected 22537 matches, got {len(m)}"
    assert 11_000_000 < len(d) < 12_000_000, f"ball count out of range: {len(d)}"


def test_no_duplicate_deliveries():
    """(match_id, innings_num, over, ball_seq) is unique: ball_seq numbers every
    delivery in an over, wides and no-balls included, and restarts each over."""
    d = pd.read_parquet(PARSED / "deliveries.parquet")
    assert d.match_id.notna().all()
    assert d.innings_num.notna().all()
    assert d.duplicated(["match_id", "innings_num", "over", "ball_seq"]).sum() == 0


def test_wides_and_wickets_present():
    """Both must exist, or the outcome categorisation downstream is meaningless."""
    d = pd.read_parquet(PARSED / "deliveries.parquet")
    assert d.wide.notna().sum() > 100_000, "no wides parsed"
    # Build 2 stores dismissals as wicket_kind (null = no wicket), not is_wicket.
    assert d.wicket_kind.notna().sum() > 300_000, "no wickets parsed"
