"""Pre-match price extraction for the target Betfair market types.

Reads the manifest built by :mod:`betfair.manifest`, opens the per-market
price logs for the over-line, ladder and benchmark market types, and writes
three parquets into ``config.BETFAIR_DIR``:

* ``prices_overline.parquet``   -- one row per ``LINE`` market: the traded
  total in RUNS. PRIMARY target.
* ``prices_ladder.parquet``     -- one row per ``ASIAN_HANDICAP_SINGLE_LINE``
  rung, the market's implied CDF of the first-innings total.
* ``prices_matchodds.parquet``  -- one row per ``MATCH_ODDS`` runner. This is
  the sharpness benchmark: MATCH_ODDS is the liquid, well-behaved market, so
  if the extraction method looks broken there, the method is broken, not the
  thin market you were hoping to trade.

Tables are routed by ``bettingType``, NOT by ``marketType``
----------------------------------------------------------
``marketType`` does not determine what ``ltp`` means, and trusting it silently
mixes runs into an odds column. Measured over the full archive:

* ``1ST_INNINGS_RUNS`` is TWO different markets wearing one name. 21,431 of
  its 25,533 markets (83.9%) are ``bettingType == "LINE"`` with a single
  "Total Runs" runner whose ltp is a RUN TOTAL. Only 4,102 (16.1%) are the
  ``ASIAN_HANDICAP_SINGLE_LINE`` rung ladder (19-55 rungs) whose ltp is
  decimal odds. Routing that market type wholesale into a "ladder" table puts
  values like 165.5 into a column named ``pre_match_odds``, and anything that
  then de-vigs it produces an implied probability of 0.6%.
* Even ``MATCH_ODDS`` is not safe by name: 2 of its 25,945 markets are
  ``bettingType == "LINE"``.

So: ``LINE`` -> overline table, ``ASIAN_HANDICAP_SINGLE_LINE`` -> ladder
table, ``ODDS`` (benchmark types only) -> matchodds table. Every table keeps
its ``market_type`` column, so the original per-type population is always one
filter away.

On the overline table, prefer ``runner_id == TOTAL_RUNS_RUNNER_ID`` (15316):
48 markets typed ``1ST_INNINGS_RUNS`` actually carry a lone "Total Fours" or
"Total Sixes" runner, so their line counts boundaries, not runs. Their runner
id is what gives them away.

What "pre-match" means here
---------------------------
The pre-match value of a runner is the ``ltp`` of its last tick whose message
timestamp ``pt`` is **strictly before** that market's own ``marketTime``. A
tick landing exactly on ``marketTime`` is in-play, not pre-match. If no tick
qualifies, the pre-match value is NULL. There is no fallback: not the first
in-play tick, not the market's opening price, not a neighbouring rung. A
missing pre-match price is information (nobody traded this before the off)
and inventing one destroys it.

There is deliberately no ``closing_price`` column
-------------------------------------------------
The previous build had one, and it was a trap. It was the last traded price
anywhere in the file, and 62.5% of those timestamps fall AFTER the scheduled
off -- median +30.6 minutes into the match. By then the market has watched
the powerplay. A "closing price" of 1.06 on the eventual winner is not a
forecast, it is the result. Any CLV or calibration measured against it is
measuring the outcome and will look spectacular for no reason.

Where a last-tick-anywhere value is still useful (sanity checks, settlement
reconstruction) it is exposed on the over-line table ONLY, under the
deliberately ugly names ``last_line_any`` / ``last_ts_any``. Those columns
are in-play contaminated. Never use them as a target, a feature, or a
"closing line".

Tick counts are outcome-contaminated too
----------------------------------------
``n_ticks_total`` counts the whole file, in-play included. Runners that trade
again after their pre-match tick win far more often than those that do not
(21.0% vs 4.0% in the validated top-batsman sample) purely because a runner
still in contention keeps trading. ``n_ticks_total`` is retained for data
diagnostics -- how much of the archive is empty, which market types are dead
-- and must NEVER be used as a model feature or as a sample filter.
``n_ticks_pre`` is safe by construction (it is bounded by the off) and is the
one to use when you need a liquidity proxy.

Basic tier carries no volume, so a price backed by one 2-pound trade and one
backed by 50k are indistinguishable in every column here.

Sentinel market times are excluded
----------------------------------
60 markets in the archive (17 of them in target types, all in event
``2023/Dec/5/32462412``) carry a placeholder ``marketTime`` of 2099-07-03 or
2099-01-01. Left alone these are the ``closing_price`` trap arriving by the
back door: with a cutoff in 2099 *every* tick counts as "pre-match", so the
pre-match price silently becomes the last in-play price. They are undatable,
so they cannot be split by season or joined to a fixture either. They are
dropped, and the count is reported rather than swallowed.

Usage
-----
    python -m betfair.prices                 # extract all three tables
    python -m betfair.prices --coverage      # extract, then report coverage
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.config import (
    BENCHMARK_MARKET_TYPES,
    BETFAIR_DIR,
    LADDER_MARKET_TYPES,
    OVER_LINE_MARKET_TYPES,
    TOTAL_RUNS_RUNNER_ID,
    ensure_dirs,
)

from src.betfair.manifest import load_manifest, resolve_rel
from src.betfair.parse import parse_market_file

OVERLINE_PATH = BETFAIR_DIR / "prices_overline.parquet"
LADDER_PATH = BETFAIR_DIR / "prices_ladder.parquet"
MATCHODDS_PATH = BETFAIR_DIR / "prices_matchodds.parquet"

OVERLINE_COLUMNS = (
    "market_id",
    "event_id",
    "market_type",
    "market_time",
    "runner_id",
    "pre_match_line",
    "pre_match_ts",
    "n_ticks_pre",
    "n_ticks_total",
    "last_line_any",
    "last_ts_any",
)
LADDER_COLUMNS = (
    "market_id",
    "event_id",
    "market_time",
    "runner_id",
    "runner_name",
    "hc",
    "final_status",
    "pre_match_odds",
    "pre_match_ts",
    "n_ticks_pre",
    "n_ticks_total",
)
MATCHODDS_COLUMNS = (
    "market_id",
    "event_id",
    "market_time",
    "runner_id",
    "runner_name",
    "final_status",
    "pre_match_odds",
    "pre_match_ts",
    "n_ticks_pre",
)

_OVERLINE = frozenset(OVER_LINE_MARKET_TYPES)
_LADDER = frozenset(LADDER_MARKET_TYPES)
_BENCHMARK = frozenset(BENCHMARK_MARKET_TYPES)
TARGET_MARKET_TYPES = _OVERLINE | _LADDER | _BENCHMARK

# ``bettingType`` values, which are what actually decide the meaning of ltp.
BETTING_TYPE_LINE = "LINE"
BETTING_TYPE_LADDER = "ASIAN_HANDICAP_SINGLE_LINE"
BETTING_TYPE_ODDS = "ODDS"

# Any marketTime at or beyond this is a Betfair placeholder, not a fixture.
# Real cricket in this archive runs 2015..2026; the sentinels sit in 2099.
SENTINEL_MARKET_TIME_YEAR = 2090


def route(market_type, betting_type):
    """Which price table a market belongs in, decided by ``bettingType``.

    Returns ``"overline"``, ``"ladder"``, ``"matchodds"``, or ``None`` for a
    market this pipeline does not extract. ``market_type`` only gates which
    markets are in scope at all -- it never decides what ``ltp`` means.
    """
    if market_type not in TARGET_MARKET_TYPES:
        return None
    if betting_type == BETTING_TYPE_LADDER:
        return "ladder"
    if betting_type == BETTING_TYPE_LINE:
        return "overline"
    if betting_type == BETTING_TYPE_ODDS and market_type in _BENCHMARK:
        return "matchodds"
    return None


def is_sentinel_time(dt) -> bool:
    """True for a placeholder ``marketTime`` that cannot date a fixture."""
    return dt is None or dt.year >= SENTINEL_MARKET_TIME_YEAR


def _split_ticks(ticks, cutoff):
    """(pre_value, pre_ts, n_pre, n_total, last_value, last_ts) for one runner.

    ``cutoff`` is the market's ``marketTime``; the pre-match split is strict
    (``ts < cutoff``). ``ticks`` need not be sorted -- the archive is written
    in stream order, but this does not rely on it.
    """
    n_total = len(ticks)
    if n_total == 0:
        return None, None, 0, 0, None, None

    pre_ltp = pre_ts = None
    last_ltp = last_ts = None
    n_pre = 0
    for t in ticks:
        if t.ts < cutoff:
            n_pre += 1
            if pre_ts is None or t.ts >= pre_ts:
                pre_ts, pre_ltp = t.ts, t.ltp
        if last_ts is None or t.ts >= last_ts:
            last_ts, last_ltp = t.ts, t.ltp
    return pre_ltp, pre_ts, n_pre, n_total, last_ltp, last_ts


def extract_market(path, market_type=None, event_id=None, market_time=None):
    """Extract price rows from one market file.

    Returns ``(kind, rows)`` where ``kind`` is ``"overline"``, ``"ladder"``,
    ``"matchodds"`` or ``None``. The table is chosen by the file's own
    ``bettingType`` (see :func:`route`), not by ``market_type``.

    ``market_time`` from the manifest is used as the cutoff when supplied,
    falling back to the market file's own definition. A market with no
    market time, or with a 2099 placeholder, has no defensible pre-match
    boundary and yields no rows rather than a guessed one.
    """
    parsed = parse_market_file(path)
    d = parsed.definition
    cutoff = market_time or d.market_time
    event_id = event_id or d.event_id
    market_type = market_type or d.market_type

    kind = route(market_type, d.betting_type)
    if kind is None or is_sentinel_time(cutoff):
        return kind, []

    by_runner = defaultdict(list)
    for t in parsed.ticks:
        by_runner[t.runner_id].append(t)

    if kind == "overline":
        # Verified: exactly one runner, "Total Runs", id 15316, and its ltp is
        # the traded line IN RUNS, not odds. Prefer that id when present so a
        # malformed definition cannot silently swap in some other runner.
        runner_ids = [r.id for r in d.runners]
        if TOTAL_RUNS_RUNNER_ID in runner_ids:
            rid = TOTAL_RUNS_RUNNER_ID
        elif runner_ids:
            rid = runner_ids[0]
        elif by_runner:
            rid = next(iter(by_runner))
        else:
            rid = TOTAL_RUNS_RUNNER_ID
        pre, pre_ts, n_pre, n_tot, last, last_ts = _split_ticks(by_runner.get(rid, []), cutoff)
        return kind, [
            (
                d.market_id,
                event_id,
                market_type,
                cutoff,
                rid,
                pre,
                pre_ts,
                n_pre,
                n_tot,
                last,
                last_ts,
            )
        ]

    if kind == "ladder":
        rows = []
        # One row per declared rung, priced or not. A rung nobody traded is a
        # real observation about the ladder and is kept with a null price.
        seen = set()
        for r in d.runners:
            seen.add(r.id)
            ticks = by_runner.get(r.id, [])
            pre, pre_ts, n_pre, n_tot, _l, _lt = _split_ticks(ticks, cutoff)
            hc = r.hc
            if hc is None and ticks:
                # Fall back to the hc carried on the ticks themselves; on a
                # ladder the handicap IS the bet and must not go missing.
                hc = next((t.hc for t in ticks if t.hc is not None), None)
            rows.append(
                (
                    d.market_id,
                    event_id,
                    cutoff,
                    r.id,
                    r.name,
                    hc,
                    r.status,
                    pre,
                    pre_ts,
                    n_pre,
                    n_tot,
                )
            )
        # A runner that only ever appears in rc updates (never in a definition
        # this file saw) still traded; keep it rather than dropping the price.
        for rid, ticks in by_runner.items():
            if rid in seen:
                continue
            pre, pre_ts, n_pre, n_tot, _l, _lt = _split_ticks(ticks, cutoff)
            rows.append(
                (
                    d.market_id,
                    event_id,
                    cutoff,
                    rid,
                    None,
                    next((t.hc for t in ticks if t.hc is not None), None),
                    None,
                    pre,
                    pre_ts,
                    n_pre,
                    n_tot,
                )
            )
        return kind, rows

    rows = []
    for r in d.runners:
        pre, pre_ts, n_pre, _n_tot, _l, _lt = _split_ticks(by_runner.get(r.id, []), cutoff)
        rows.append(
            (d.market_id, event_id, cutoff, r.id, r.name, r.status, pre, pre_ts, n_pre)
        )
    return kind, rows


def _extract_event(job):
    """Worker: extract every target market of one event directory."""
    markets = job
    out = {"overline": [], "ladder": [], "matchodds": []}
    for market_file_rel, market_type, event_id, market_time in markets:
        path = resolve_rel(market_file_rel)
        if not os.path.exists(path):
            continue
        try:
            kind, rows = extract_market(path, market_type, event_id, market_time)
        except Exception:  # noqa: BLE001 - third-party archive; keep scanning
            continue
        if kind:
            out[kind].extend(rows)
    return out


def build_prices(manifest=None, workers: int = 8, progress: bool = True, out_dir=None):
    """Extract all three price tables and write them. Returns the three frames."""
    import pandas as pd

    ensure_dirs()
    out_dir = Path(out_dir) if out_dir is not None else Path(BETFAIR_DIR)
    df = load_manifest() if manifest is None else manifest
    tgt = df[df["market_type"].isin(TARGET_MARKET_TYPES)]
    n_target = len(tgt)
    sentinel = tgt["market_time"].isna() | (tgt["market_time"].dt.year >= SENTINEL_MARKET_TIME_YEAR)
    tgt = tgt[~sentinel]
    if progress:
        print(
            f"extracting {len(tgt):,} of {n_target:,} target markets "
            f"({int(sentinel.sum()):,} dropped: placeholder marketTime, undatable)",
            flush=True,
        )
        print(
            "  routing by bettingType: "
            + ", ".join(
                f"{bt}={n:,}" for bt, n in tgt["betting_type"].value_counts().items()
            ),
            flush=True,
        )

    # Group by event so one worker touches one directory: better locality, and
    # far less per-task pickling than one job per market.
    jobs = defaultdict(list)
    for row in tgt.itertuples(index=False):
        jobs[row.event_dir_rel].append(
            (row.market_file_rel, row.market_type, row.event_id, row.market_time.to_pydatetime())
        )
    jobs = list(jobs.values())

    acc = {"overline": [], "ladder": [], "matchodds": []}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_extract_event, jobs, chunksize=20), 1):
            for k, v in res.items():
                acc[k].extend(v)
            if progress and i % 2000 == 0:
                print(
                    f"  {i:,}/{len(jobs):,} events  "
                    f"overline={len(acc['overline']):,} ladder={len(acc['ladder']):,} "
                    f"matchodds={len(acc['matchodds']):,}",
                    flush=True,
                )

    frames = {}
    for kind, cols, path in (
        ("overline", OVERLINE_COLUMNS, out_dir / OVERLINE_PATH.name),
        ("ladder", LADDER_COLUMNS, out_dir / LADDER_PATH.name),
        ("matchodds", MATCHODDS_COLUMNS, out_dir / MATCHODDS_PATH.name),
    ):
        f = pd.DataFrame(acc[kind], columns=list(cols))
        for c in ("market_time", "pre_match_ts", "last_ts_any"):
            if c in f.columns:
                f[c] = pd.to_datetime(f[c], utc=True, errors="coerce")
        f = f.sort_values(["market_time", "market_id"], kind="stable").reset_index(drop=True)
        f.to_parquet(path, index=False)
        frames[kind] = f
        if progress:
            print(f"wrote {len(f):,} rows -> {path}", flush=True)
    return frames["overline"], frames["ladder"], frames["matchodds"]


def coverage_report(overline, ladder, matchodds):
    """Print per-market-type coverage: % of markets with a pre-match value.

    "ever traded" is the share of markets with at least one tick anywhere in
    the file, in-play included -- a data-availability diagnostic, not a
    modelling statistic. It is unavailable for the matchodds table, which by
    design carries no in-play tick count.
    """
    header = f"{'market_type':<26}{'markets':>9}{'pre-match':>11}{'pct':>8}{'ever traded':>13}{'pct':>8}"

    def line(label, n_markets, n_pre, n_any=None):
        pct_pre = 100.0 * n_pre / n_markets if n_markets else 0.0
        row = f"{label:<26}{n_markets:>9,}{n_pre:>11,}{pct_pre:>7.1f}%"
        if n_any is None:
            row += f"{'n/a':>13}{'':>8}"
        else:
            pct_any = 100.0 * n_any / n_markets if n_markets else 0.0
            row += f"{n_any:>13,}{pct_any:>7.1f}%"
        print(row)

    print("\nLINE markets -- pre_match_line is a TOTAL IN RUNS (prices_overline)")
    print(header)
    for mt, g in overline.groupby("market_type", sort=True):
        line(mt, len(g), int(g["pre_match_line"].notna().sum()),
             int((g["n_ticks_total"] > 0).sum()))

    print("\nASIAN_HANDICAP_SINGLE_LINE ladders -- pre_match_odds is DECIMAL ODDS "
          "(prices_ladder)")
    print(header)
    if ladder.empty:
        line("1ST_INNINGS_RUNS", 0, 0, 0)
    else:
        per_market = ladder.groupby("market_id").agg(
            market_type=("market_id", "size"),
            has_pre=("pre_match_odds", lambda s: bool(s.notna().any())),
            has_any=("n_ticks_total", lambda s: bool((s > 0).any())),
        )
        line("1ST_INNINGS_RUNS", len(per_market), int(per_market.has_pre.sum()),
             int(per_market.has_any.sum()))

    print("\nODDS benchmark -- pre_match_odds is DECIMAL ODDS (prices_matchodds)")
    print(header)
    if matchodds.empty:
        line("MATCH_ODDS", 0, 0)
    else:
        per_market = matchodds.groupby("market_id").agg(
            has_pre=("pre_match_odds", lambda s: bool(s.notna().any())),
        )
        line("MATCH_ODDS", len(per_market), int(per_market.has_pre.sum()))

    print(f"\nladder rungs: {len(ladder):,} rows, "
          f"{int(ladder['pre_match_odds'].notna().sum()):,} with a pre-match price, "
          f"{int(ladder['hc'].notna().sum()):,} carrying hc "
          f"({int(ladder['hc'].isna().sum()):,} missing)")
    if not ladder.empty:
        settled = ladder["final_status"].isin(("WINNER", "LOSER")).sum()
        print(f"ladder rungs settled WINNER/LOSER: {int(settled):,}")
    n_odd = int((overline["runner_id"] != TOTAL_RUNS_RUNNER_ID).sum())
    print(f"overline rows whose runner is NOT 'Total Runs' (id {TOTAL_RUNS_RUNNER_ID}): "
          f"{n_odd:,} -- their line is not a run total; filter on runner_id")
    print("\nNOTE: last_line_any / last_ts_any and n_ticks_total are in-play "
          "contaminated. Diagnostics only -- never features, never filters.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="read existing parquets")
    args = ap.parse_args(argv)

    import pandas as pd

    if args.reuse:
        ov = pd.read_parquet(OVERLINE_PATH)
        la = pd.read_parquet(LADDER_PATH)
        mo = pd.read_parquet(MATCHODDS_PATH)
    else:
        ov, la, mo = build_prices(workers=args.workers)
    if args.coverage or args.reuse:
        coverage_report(ov, la, mo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
