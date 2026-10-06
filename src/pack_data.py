# src/pack_data.py
"""Pack the ~80 MB of processed tables the pipeline reads into one tarball.

    python -m src.pack_data      writes data_bundle.tar.gz at the repo root

Betfair-derived tables are licensed: the bundle goes to GitHub only as a
release asset on this private repository, never into git history.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

from src.config import ROOT

BUNDLE_PATTERNS = [
    "data/parsed/*.parquet",
    "data/market/*.parquet",
    "data/ticks/*.parquet",
    "data/raw/cricsheet/player_meta.rda",
    "data/raw/weather/hourly.parquet",
    "data/raw/weather/venues.csv",
]


def bundle_files(root: Path = ROOT) -> list[Path]:
    files = []
    for pat in BUNDLE_PATTERNS:
        hits = sorted(Path(root).glob(pat))
        if not hits:
            raise FileNotFoundError(f"no files match {pat} under {root}")
        files += hits
    return files


def pack(out: Path = ROOT / "data_bundle.tar.gz") -> Path:
    with tarfile.open(out, "w:gz") as tar:
        for f in bundle_files():
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    pack()
