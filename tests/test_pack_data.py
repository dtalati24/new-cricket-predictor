# tests/test_pack_data.py
import pytest

from src.pack_data import BUNDLE_PATTERNS, bundle_files


def test_missing_pattern_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/parsed"):
        bundle_files(root=tmp_path)


def test_bundle_covers_what_the_pipeline_reads():
    joined = " ".join(BUNDLE_PATTERNS)
    for needed in ("data/parsed", "data/market", "data/ticks", "player_meta.rda",
                   "hourly.parquet", "venues.csv"):
        assert needed in joined
