# src/cache.py
"""Cache prepared data per training cutoff, keyed by everything that shapes it.

The key covers the cutoff, the feature column list, and every input file:
source files by content, data files by size and modification time. Change any
of them and the cache rebuilds; delete data/model/cache to force a rebuild.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from src.config import MODEL, PARSED, PLAYER_META_RDA, ROOT, TICKS

CACHE_DIR = MODEL / "cache"


def input_files() -> list[Path]:
    files = sorted((ROOT / "src" / "features").glob("*.py")) + sorted(
        (ROOT / "src" / "skill").glob("*.py")) + [ROOT / "src" / "model" / "frame.py"]
    files += sorted(PARSED.glob("*.parquet")) + [TICKS / "match_odds_markets.parquet", PLAYER_META_RDA]
    for p in (ROOT / "data" / "raw" / "weather" / "hourly.parquet",
              ROOT / "data" / "raw" / "weather" / "venues.csv",
              ROOT / "src" / "features" / "venue_overrides.csv"):
        if p.exists():
            files.append(p)
    return files


def cache_key(cutoff: str, columns: list[str], files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(cutoff.encode())
    h.update("\x1f".join(columns).encode())
    for f in files:
        f = Path(f)
        h.update(str(f).encode())
        if f.suffix in (".py", ".csv"):
            h.update(hashlib.sha256(f.read_bytes()).digest())
        else:
            st = f.stat()
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def load_or_build(cutoff: str, columns: list[str], files: list[Path], build,
                  cache_dir: Path = CACHE_DIR):
    path = Path(cache_dir) / f"{cutoff}-{cache_key(cutoff, columns, files)[:12]}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    tmp.replace(path)
    return obj
