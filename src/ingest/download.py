"""Fetch the raw Cricsheet inputs into ``data/raw/cricsheet/``.

Two artefacts are needed:

* ``all_json.zip`` -- every match Cricsheet publishes, one JSON per match.
  The zip is kept as the canonical input and read *in place* by
  :mod:`ingest.parse`; extracting 22.5k small files onto Windows costs far
  more wall-clock time than ``zipfile`` spends decompressing them on the fly,
  and leaves a second copy of the corpus to drift out of sync.
* ``people.csv`` -- the player register mapping the stable Cricsheet
  ``identifier`` to a display name and to external keys (notably
  ``key_cricinfo``, which is the only bridge to batting-hand / bowling-style
  metadata; Cricsheet's own match JSON carries no such field).

Everything is cached: a file that already exists is left alone unless
``force=True``. Paths come from :mod:`config` -- nothing here invents its own.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.config import (
    CRICSHEET_MATCHES_DIR,
    CRICSHEET_NAMES_CSV,
    CRICSHEET_PEOPLE_CSV,
    CRICSHEET_RAW_DIR,
    CRICSHEET_ZIP,
)

ALL_JSON_URL = "https://cricsheet.org/downloads/all_json.zip"
PEOPLE_CSV_URL = "https://cricsheet.org/register/people.csv"
NAMES_CSV_URL = "https://cricsheet.org/register/names.csv"


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Stream ``url`` to ``dest``, atomically. No-op if ``dest`` exists."""
    if dest.exists() and not force:
        return dest
    import requests  # imported lazily: parsing must not require the network stack

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def download_all_json(force: bool = False) -> Path:
    """Download the full match archive. Returns the path to the zip."""
    return _download(ALL_JSON_URL, CRICSHEET_ZIP, force=force)


def download_register(force: bool = False) -> tuple[Path, Path]:
    """Download the people/names register CSVs."""
    people = _download(PEOPLE_CSV_URL, CRICSHEET_PEOPLE_CSV, force=force)
    names = _download(NAMES_CSV_URL, CRICSHEET_NAMES_CSV, force=force)
    return people, names


def extract_all_json(force: bool = False) -> Path:
    """Optional: explode the zip into loose JSON files.

    Not part of the normal pipeline -- :mod:`ingest.parse` reads the zip
    directly. Provided for ad-hoc inspection of individual matches.
    """
    CRICSHEET_MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    marker = CRICSHEET_MATCHES_DIR / ".extracted"
    if marker.exists() and not force:
        return CRICSHEET_MATCHES_DIR
    with zipfile.ZipFile(CRICSHEET_ZIP) as zf:
        zf.extractall(CRICSHEET_MATCHES_DIR)
    marker.write_text("ok", encoding="utf-8")
    return CRICSHEET_MATCHES_DIR


def raw_status() -> dict[str, object]:
    """Describe what is on disk, without downloading anything."""
    n_zip = 0
    if CRICSHEET_ZIP.exists():
        with zipfile.ZipFile(CRICSHEET_ZIP) as zf:
            n_zip = sum(1 for n in zf.namelist() if n.endswith(".json"))
    n_loose = (
        sum(1 for _ in CRICSHEET_MATCHES_DIR.glob("*.json"))
        if CRICSHEET_MATCHES_DIR.is_dir()
        else 0
    )
    return {
        "raw_dir": str(CRICSHEET_RAW_DIR),
        "zip_exists": CRICSHEET_ZIP.exists(),
        "zip_json_files": n_zip,
        "loose_json_files": n_loose,
        "people_csv_exists": CRICSHEET_PEOPLE_CSV.exists(),
        "names_csv_exists": CRICSHEET_NAMES_CSV.exists(),
    }


def main(force: bool = False) -> None:
    download_all_json(force=force)
    download_register(force=force)
    for key, value in raw_status().items():
        print(f"{key:>20}: {value}")


if __name__ == "__main__":
    main()
