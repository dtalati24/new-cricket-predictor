# tests/test_cache.py
from src.cache import cache_key, load_or_build


def test_key_changes_with_columns_and_file_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1")
    k1 = cache_key("2024-01-01", ["a", "b"], [f])
    assert cache_key("2024-01-01", ["a", "b", "c"], [f]) != k1
    f.write_text("x = 2")
    assert cache_key("2024-01-01", ["a", "b"], [f]) != k1
    assert cache_key("2025-01-01", ["a", "b"], [f]) != cache_key("2024-01-01", ["a", "b"], [f])


def test_load_or_build_builds_once(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return {"v": 42}

    f = tmp_path / "a.py"
    f.write_text("x")
    a = load_or_build("2024-01-01", ["c"], [f], build, cache_dir=tmp_path / "cache")
    b = load_or_build("2024-01-01", ["c"], [f], build, cache_dir=tmp_path / "cache")
    assert a == b == {"v": 42}
    assert len(calls) == 1
