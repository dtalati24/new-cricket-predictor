"""Extract full timestamped MATCH_ODDS price paths from the raw Betfair archive.

Build 2's `prices_matchodds.parquet` collapsed each market to a single
pre-match summary row, which discarded the in-play price path entirely. The
raw stream files retain it: every line is a market-change message carrying a
publish time `pt` (epoch ms) and either a `marketDefinition` (status / inPlay
transitions) or `rc` entries (per-runner last-traded-price changes).

Four tables are written under data/ticks:

  match_odds_ticks.parquet    one row per runner price change
  match_odds_defs.parquet     one row per market-definition change
                              (status, inPlay, betDelay)
  match_odds_markets.parquet  one row per market: Betfair metadata joined to
                              the Cricsheet match it was matched to
  match_odds_runners.parquet  per-runner name and settlement

The defs table is what makes the innings break findable: cricket match-odds
markets suspend at the break, so a status transition bracketing a multi-minute
gap is a candidate break marker that needs no ball-level timestamps.

CAVEATS
-------
* `ltp` is a *last traded price*, not an available quote. BASIC tier carries
  no back/lay ladder and no volume. Nothing here says a price was gettable.
* `pt` is Betfair's publish time, not an exchange match time. Betfair applies
  a bet delay (`betDelay`, typically 5s in play), so in-play ticks lag the
  real-world event by at least that much on top of broadcast latency.
* A market can be re-opened and re-suspended many times; do not assume a
  single suspension per market.
* Per retrospective trap 5, some markets carry a placeholder `marketTime` of
  2099-07-03. Filter on the manifest's market_time, not the in-file value,
  when defining "pre-match".
* Per retrospective trap 7, tick *counts* are outcome-contaminated. Counts of
  ticks after a point in time leak the result; never use them as a feature or
  a filter without arguing why the cut is safe.
"""
from __future__ import annotations

import bz2
import json
from pathlib import Path

import pandas as pd

from src import config


def _iter_lines(path: Path):
    with bz2.open(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def scan_market(path: Path) -> tuple[list[dict], list[dict], dict]:
    """Return (tick rows, market-definition change rows, market metadata)."""
    ticks: list[dict] = []
    defs: list[dict] = []
    meta: dict = {}
    runner_names: dict[int, str] = {}
    final_status: dict[int, str] = {}
    last_def = None

    for line in _iter_lines(path):
        msg = json.loads(line)
        pt = msg.get("pt")
        for mc in msg.get("mc", []):
            mid = mc.get("id")
            md = mc.get("marketDefinition")
            if md is not None:
                if not meta:
                    meta = {
                        "market_id": mid,
                        "file_event_id": md.get("eventId"),
                        "file_market_type": md.get("marketType"),
                        "file_betting_type": md.get("bettingType"),
                        "file_market_time": md.get("marketTime"),
                        "market_base_rate": md.get("marketBaseRate"),
                        "turn_in_play_enabled": md.get("turnInPlayEnabled"),
                    }
                meta["suspend_time"] = md.get("suspendTime")
                meta["settled_time"] = md.get("settledTime")
                meta["final_market_status"] = md.get("status")
                for r in md.get("runners", []):
                    rid = r.get("id")
                    if r.get("name"):
                        runner_names[rid] = r["name"]
                    if r.get("status") in ("WINNER", "LOSER", "REMOVED"):
                        final_status[rid] = r["status"]
                key = (
                    md.get("status"),
                    md.get("inPlay"),
                    md.get("betDelay"),
                    md.get("numberOfActiveRunners"),
                )
                if key != last_def:
                    defs.append(
                        {
                            "market_id": mid,
                            "pt": pt,
                            "status": key[0],
                            "in_play": key[1],
                            "bet_delay": key[2],
                            "n_active_runners": key[3],
                        }
                    )
                    last_def = key
            for rc in mc.get("rc", []):
                if "ltp" in rc:
                    ticks.append(
                        {
                            "market_id": mid,
                            "runner_id": rc.get("id"),
                            "pt": pt,
                            "ltp": rc["ltp"],
                        }
                    )

    meta["runner_names"] = runner_names
    meta["runner_final_status"] = final_status
    return ticks, defs, meta


def target_markets() -> pd.DataFrame:
    """MATCH_ODDS markets for events joined to a Cricsheet T20/IT20 match."""
    mf = pd.read_parquet(config.LIVE_MARKET_MANIFEST)
    em = pd.read_parquet(config.LIVE_EVENT_MATCH)
    matches = pd.read_parquet(config.LIVE_MATCHES)

    joined = em.dropna(subset=["match_id"]).merge(
        matches, on="match_id", how="inner", suffixes=("_bf", "")
    )
    t20 = joined[joined["match_type"].isin(["T20", "IT20"])]
    keep = mf[
        (mf["market_type"] == "MATCH_ODDS") & (mf["event_id"].isin(t20["event_id"]))
    ]
    cols = [
        "event_id", "match_id", "date", "match_type", "gender", "venue", "city",
        "event_name", "season", "balls_per_over", "scheduled_overs", "team_a",
        "team_b", "toss_winner", "toss_decision", "outcome_winner",
        "outcome_result", "outcome_method", "date_offset", "loose_match",
    ]
    return keep.merge(t20[cols], on="event_id", how="left")


def run(limit: int | None = None) -> dict:
    targets = target_markets()
    if limit:
        targets = targets.head(limit)

    all_ticks: list[dict] = []
    all_defs: list[dict] = []
    metas: list[dict] = []
    missing = 0

    for i, row in enumerate(targets.itertuples(index=False), start=1):
        path = config.RAW_BETFAIR / row.market_file_rel
        if not path.exists():
            missing += 1
            continue
        ticks, defs, meta = scan_market(path)
        all_ticks.extend(ticks)
        all_defs.extend(defs)
        meta["n_ticks"] = len(ticks)
        metas.append(meta)
        if i % 1000 == 0:
            print(f"  {i}/{len(targets)} markets, {len(all_ticks):,} ticks", flush=True)

    ticks_df = pd.DataFrame(all_ticks)
    ticks_df["pt"] = pd.to_datetime(ticks_df["pt"], unit="ms", utc=True)
    ticks_df["runner_id"] = ticks_df["runner_id"].astype("int64")
    ticks_df["ltp"] = ticks_df["ltp"].astype("float32")
    ticks_df = ticks_df.sort_values(["market_id", "pt", "runner_id"]).reset_index(
        drop=True
    )

    defs_df = pd.DataFrame(all_defs)
    defs_df["pt"] = pd.to_datetime(defs_df["pt"], unit="ms", utc=True)

    drop = ("runner_names", "runner_final_status")
    meta_df = pd.DataFrame(
        [{k: v for k, v in m.items() if k not in drop} for m in metas]
    )
    for col in ("file_market_time", "suspend_time", "settled_time"):
        meta_df[col] = pd.to_datetime(meta_df[col], utc=True, format="ISO8601")

    runner_rows = []
    for m in metas:
        for rid, name in m["runner_names"].items():
            runner_rows.append(
                {
                    "market_id": m["market_id"],
                    "runner_id": int(rid),
                    "runner_name": name,
                    "final_status": m["runner_final_status"].get(rid),
                }
            )
    runners_df = pd.DataFrame(runner_rows)

    markets = targets.merge(meta_df, on="market_id", how="inner")

    config.TICKS.mkdir(parents=True, exist_ok=True)
    ticks_df.to_parquet(config.MATCH_ODDS_TICKS, index=False)
    markets.to_parquet(config.MATCH_ODDS_MARKETS, index=False)
    defs_df.to_parquet(config.TICKS / "match_odds_defs.parquet", index=False)
    runners_df.to_parquet(config.TICKS / "match_odds_runners.parquet", index=False)

    return {
        "markets_targeted": len(targets),
        "markets_scanned": len(metas),
        "markets_missing_file": missing,
        "ticks": len(ticks_df),
        "def_changes": len(defs_df),
        "runners": len(runners_df),
    }


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for key, val in run(n).items():
        print(f"{key}: {val:,}")
