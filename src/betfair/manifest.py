"""Full-archive scan of the Betfair BASIC download into one market manifest.

Reads every event's cheap ``{eventId}.bz2`` summary file (not the per-market
price logs) and writes one row per market to
``BETFAIR_DIR / "market_manifest.parquet"``.

Relative paths, not absolute ones
---------------------------------
``event_dir_rel`` and ``market_file_rel`` are stored RELATIVE to
``config.BETFAIR_RAW_DIR``, with forward slashes. The previous build froze
absolute paths into its manifest; the moment the archive moved to another
drive every row in the table pointed at nothing and the manifest had to be
rebuilt from scratch to recover. Resolve a stored path with
:func:`resolve_rel` (or :func:`market_path` / :func:`event_dir_path`), which
re-anchors it on whatever ``BETFAIR_RAW_DIR`` says today.

Dating
------
``market_time`` comes from the market's own ``marketTime`` field and never
from the archive folder path. A market filed under ``2024/Apr/1`` can have a
``marketTime`` days away; the folder is a download-batch artefact, not a
fixture date.

Usage
-----
    python -m betfair.manifest              # full scan -> parquet
    python -m betfair.manifest --verify     # scan, then check known totals
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.config import BETFAIR_DIR, BETFAIR_RAW_DIR, ensure_dirs

from src.betfair.parse import (
    SUMMARY_FILENAME_RE,
    event_summary_path,
    market_file_name,
    scan_event_summary,
)

MANIFEST_PATH = BETFAIR_DIR / "market_manifest.parquet"

MANIFEST_COLUMNS = (
    "market_id",
    "event_id",
    "event_name",
    "market_type",
    "betting_type",
    "market_time",
    "country_code",
    "n_runners",
    "number_of_winners",
    "event_dir_rel",
    "market_file_rel",
)

# Totals measured directly off the raw archive. The scan asserts against
# these; if they stop reproducing, the archive changed (or the scan broke) --
# investigate, never edit the expectation to match.
EXPECTED_N_EVENTS = 26_084
EXPECTED_N_MARKETS = 371_641
EXPECTED_TYPE_COUNTS = {
    "INNINGS_RUNS": 60_139,
    "MATCH_ODDS": 25_945,
    "1ST_INNINGS_RUNS": 25_533,
    "TIED_MATCH": 24_135,
    "COMPLETED_MATCH": 21_850,
    "TOP_BATSMAN": 8_542,
    "1_INNING_20_OVR_LINE": 6_179,
    "1_INNING_6_OVR_LINE": 5_589,
    "1_INNING_15_OVR_LINE": 5_368,
    "1_INNING_10_OVR_LINE": 4_609,
}


# --------------------------------------------------------------------------
# relative <-> absolute path helpers
# --------------------------------------------------------------------------
def to_rel(path) -> str:
    """Express ``path`` relative to ``BETFAIR_RAW_DIR``, forward-slashed."""
    return Path(path).resolve().relative_to(Path(BETFAIR_RAW_DIR).resolve()).as_posix()


def resolve_rel(rel: str) -> Path:
    """Re-anchor a manifest-relative path on today's ``BETFAIR_RAW_DIR``.

    The inverse of :func:`to_rel`. This is the ONLY supported way to turn a
    manifest path back into something you can open.
    """
    return Path(BETFAIR_RAW_DIR).joinpath(*str(rel).split("/"))


def event_dir_path(row_or_rel) -> Path:
    """Absolute event directory for a manifest row (or an ``event_dir_rel``)."""
    rel = row_or_rel if isinstance(row_or_rel, str) else row_or_rel["event_dir_rel"]
    return resolve_rel(rel)


def market_path(row_or_rel) -> Path:
    """Absolute per-market file for a manifest row (or a ``market_file_rel``)."""
    rel = row_or_rel if isinstance(row_or_rel, str) else row_or_rel["market_file_rel"]
    return resolve_rel(rel)


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------
def iter_event_dirs(root=None):
    """Yield every absolute event directory (one holding a ``{eventId}.bz2``)."""
    root = os.fspath(root or BETFAIR_RAW_DIR)
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(SUMMARY_FILENAME_RE.match(f) for f in filenames):
            yield dirpath


def scan_event_dir(event_dir) -> list:
    """Manifest rows for one event directory, read from its summary file.

    Returns ``[]`` for a directory with no readable summary rather than
    raising: one unreadable event must not abort a 26k-event scan. Errors are
    surfaced by the caller's row count, not swallowed silently at the top
    level -- see :func:`build_manifest`, which reports the shortfall.
    """
    summary = event_summary_path(event_dir)
    if not os.path.exists(summary):
        return []
    try:
        definitions = scan_event_summary(summary)
    except Exception:  # noqa: BLE001 - third-party archive, keep scanning
        return []

    event_rel = to_rel(event_dir)
    rows = []
    for d in definitions:
        rows.append(
            (
                d.market_id,
                d.event_id,
                d.event_name,
                d.market_type,
                d.betting_type,
                d.market_time,
                d.country_code,
                d.n_runners,
                d.number_of_winners,
                event_rel,
                f"{event_rel}/{market_file_name(d.market_id)}",
            )
        )
    return rows


def build_manifest(root=None, out_path=None, workers: int = 8, progress: bool = True):
    """Scan the whole archive and write the manifest parquet. Returns the frame."""
    import pandas as pd

    ensure_dirs()
    out_path = Path(out_path) if out_path is not None else MANIFEST_PATH
    event_dirs = sorted(iter_event_dirs(root))
    n_events = len(event_dirs)
    if progress:
        print(f"scanning {n_events:,} event summary files", flush=True)

    rows = []
    n_empty = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(scan_event_dir, event_dirs, chunksize=50), 1):
            if not res:
                n_empty += 1
            rows.extend(res)
            if progress and i % 2000 == 0:
                print(f"  {i:,}/{n_events:,} events, {len(rows):,} markets", flush=True)

    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    df["market_time"] = pd.to_datetime(df["market_time"], utc=True, errors="coerce")
    df["n_runners"] = df["n_runners"].astype("int32")
    df["number_of_winners"] = pd.to_numeric(
        df["number_of_winners"], errors="coerce"
    ).astype("Int32")
    df = df.sort_values(["market_time", "event_id", "market_id"], kind="stable")
    df = df.reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    if progress:
        print(
            f"\nwrote {len(df):,} market rows from {n_events:,} events "
            f"({n_empty:,} events yielded nothing) -> {out_path}",
            flush=True,
        )
    return df


def load_manifest(path=None):
    """Read the manifest parquet."""
    import pandas as pd

    return pd.read_parquet(Path(path) if path is not None else MANIFEST_PATH)


def verify(df) -> int:
    """Check the scan against the measured archive totals. Returns failure count."""
    failures = 0
    n_events = df["event_dir_rel"].nunique()
    for label, got, want in (
        ("events", n_events, EXPECTED_N_EVENTS),
        ("markets", len(df), EXPECTED_N_MARKETS),
    ):
        ok = got == want
        failures += 0 if ok else 1
        print(f"{'OK ' if ok else 'FAIL'} {label:<22} got {got:>9,}  want {want:>9,}")

    counts = df["market_type"].value_counts()
    for mtype, want in EXPECTED_TYPE_COUNTS.items():
        got = int(counts.get(mtype, 0))
        ok = got == want
        failures += 0 if ok else 1
        print(f"{'OK ' if ok else 'FAIL'} {mtype:<22} got {got:>9,}  want {want:>9,}")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument(
        "--reuse", action="store_true", help="verify an existing manifest, do not rescan"
    )
    args = ap.parse_args(argv)

    df = load_manifest(args.out) if args.reuse else build_manifest(
        out_path=args.out, workers=args.workers
    )
    if args.verify or args.reuse:
        print()
        failures = verify(df)
        if failures:
            print(f"\n{failures} verification check(s) FAILED", flush=True)
            return 1
        print("\nall verification checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
