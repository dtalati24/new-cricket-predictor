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


def test_corrupt_pickle_rebuilds(tmp_path):
    from src.cache import cache_key
    f = tmp_path / "a.py"
    f.write_text("x")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"2024-01-01-{cache_key('2024-01-01', ['c'], [f])[:12]}.pkl").write_bytes(b"garbage")
    calls = []

    def build():
        calls.append(1)
        return {"v": 7}

    assert load_or_build("2024-01-01", ["c"], [f], build, cache_dir=cache) == {"v": 7}
    assert len(calls) == 1
    assert load_or_build("2024-01-01", ["c"], [f], build, cache_dir=cache) == {"v": 7}
    assert len(calls) == 1


def test_input_files_cover_context_code():
    from src.cache import input_files
    names = {p.as_posix().split("/src/")[-1] for p in input_files() if "/src/" in p.as_posix()}
    assert {"run_6over.py", "sim/setup.py", "config.py"} <= names
