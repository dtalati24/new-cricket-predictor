"""Normalize raw Cricsheet match JSON into the parquet tables everything
downstream is built from.

Five tables land in ``PARSED_DIR``:

``matches``      one row per match (context, toss, outcome, ball-format)
``innings``      one row per innings (batting/bowling side, super-over flag)
``deliveries``   one row per ball bowled -- the base fact table
``playing_xi``   one row per named player per match
``people``       the Cricsheet register (identifier -> name, cricinfo key)

Three properties of this parse are load-bearing, and each was a real bug in
the previous build:

1. **Players are identified by registry id, never by name.** Cricsheet's own
   register has 274 names shared by two or more distinct human beings (five
   different "Rashid Khan"s, and so on). ``info.registry.people`` gives the
   per-match name -> stable identifier map; every player reference in the
   delivery table is resolved through it. ``batter_id`` / ``bowler_id`` /
   ``non_striker_id`` / ``player_dismissed_id`` are the only valid join keys.

2. **``balls_per_over`` is carried per match, never assumed to be 6.** 367
   matches (The Hundred, recorded as ``match_type`` T20) have
   ``balls_per_over == 5`` with ``overs == 20`` -- a 100-ball innings. Any
   consumer converting overs to balls must multiply by this column.

3. **The playing XI comes from ``info.players``, not from the deliveries.**
   Roughly 83% of innings are not bowled out, so the set of players who
   actually faced a ball systematically understates how many wickets a side
   had in hand; a simulator seeded from the delivery table alone truncates
   simulated innings badly.

Deliberately **not** emitted: the raw ``actual_delivery`` string. It is
present on every delivery, but it is the scorecard ball reference rather than
a position, and it repeats -- an illegal ball carries the same value as the
legal ball that follows it (a wide at "18.4" is followed by another "18.4"),
so it is neither a key nor a count. ``over`` plus ``ball_seq`` (1-based
position within the over, illegal balls included) is the unambiguous locator,
and ``is_legal`` supports legal-ball counting.
"""

from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    CRICSHEET_MATCHES_DIR,
    CRICSHEET_PEOPLE_CSV,
    CRICSHEET_ZIP,
    PARSED_DIR,
    ensure_dirs,
)

# Matches parsed per parquet row-group flush. The delivery table is ~11M rows;
# materialising all of it in Python before writing costs several GB, so it is
# written incrementally instead.
CHUNK_MATCHES = 2000

DELIVERY_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("innings_num", pa.int16()),
        ("over", pa.int16()),
        ("ball_seq", pa.int16()),
        ("batter_id", pa.string()),
        ("bowler_id", pa.string()),
        ("non_striker_id", pa.string()),
        ("runs_batter", pa.int16()),
        ("runs_extras", pa.int16()),
        ("runs_total", pa.int16()),
        ("wide", pa.float32()),
        ("noball", pa.float32()),
        ("bye", pa.float32()),
        ("legbye", pa.float32()),
        ("penalty", pa.float32()),
        ("wicket_kind", pa.string()),
        ("player_dismissed_id", pa.string()),
        ("is_legal", pa.bool_()),
    ]
)

_DELIVERY_COLUMNS = [f.name for f in DELIVERY_SCHEMA]


# --------------------------------------------------------------------------
# raw source discovery
# --------------------------------------------------------------------------
def raw_source() -> tuple[str, Path]:
    """Pick where match JSON is read from.

    Prefers the zip: it is the artefact the download step produces, it cannot
    half-exist the way a partially extracted directory can, and reading it in
    place beats walking 22.5k loose files on Windows. Falls back to an
    extracted directory when only that is present.
    """
    if CRICSHEET_ZIP.exists():
        return "zip", CRICSHEET_ZIP
    if CRICSHEET_MATCHES_DIR.is_dir() and any(CRICSHEET_MATCHES_DIR.glob("*.json")):
        return "dir", CRICSHEET_MATCHES_DIR
    raise FileNotFoundError(
        f"no Cricsheet match JSON found: neither {CRICSHEET_ZIP} nor "
        f"*.json in {CRICSHEET_MATCHES_DIR}. Run ingest.download first."
    )


def iter_raw_matches(limit: int | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(match_id, parsed_json)`` for every match; id = filename stem."""
    kind, src = raw_source()
    if kind == "zip":
        with zipfile.ZipFile(src) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".json"))
            if limit is not None:
                names = names[:limit]
            for name in names:
                yield Path(name).stem, json.loads(zf.read(name))
    else:
        paths = sorted(src.glob("*.json"))
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# per-match parsing
# --------------------------------------------------------------------------
def parse_match(
    match_id: str, data: dict[str, Any]
) -> tuple[dict, list[dict], list[tuple], list[dict]]:
    """Split one match JSON into (match, innings, deliveries, playing_xi) rows.

    Delivery rows are plain tuples in ``DELIVERY_SCHEMA`` column order -- at
    ~11M rows the per-row dict overhead dominates memory use.
    """
    info = data.get("info", {})
    # The per-match name -> stable identifier map. Present in every match in
    # the current corpus; the .get chain keeps a future gap from exploding.
    registry: dict[str, str] = (info.get("registry") or {}).get("people") or {}

    teams = info.get("teams") or []
    team_a = teams[0] if len(teams) > 0 else None
    team_b = teams[1] if len(teams) > 1 else None

    toss = info.get("toss") or {}
    outcome = info.get("outcome") or {}
    event = info.get("event") or {}
    innings_list = data.get("innings") or []

    scheduled_overs = info.get("overs")
    match_row = {
        "match_id": match_id,
        "date": (info.get("dates") or [None])[0],
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "event_name": event.get("name") if isinstance(event, dict) else None,
        # Cricsheet mixes "2016/17" strings with bare-year ints; normalize to
        # str so the column has a single arrow type.
        "season": str(info["season"]) if info.get("season") is not None else None,
        "team_a": team_a,
        "team_b": team_b,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        # Rain and abandonment survive here: a match decided by "D/L", or one
        # with result "no result", has a truncated or void score that must be
        # excluded before fitting anything on innings totals.
        "outcome_winner": outcome.get("winner"),
        "outcome_result": outcome.get("result"),
        "outcome_method": outcome.get("method"),
        "balls_per_over": info.get("balls_per_over"),
        "scheduled_overs": float(scheduled_overs) if scheduled_overs is not None else None,
        "n_innings": len(innings_list),
    }

    playing_xi_rows = [
        {
            "match_id": match_id,
            "team": team,
            "player_name": player_name,
            "player_id": registry.get(player_name),
            "position": position,
        }
        for team, player_names in (info.get("players") or {}).items()
        for position, player_name in enumerate(player_names or [], start=1)
    ]

    innings_rows: list[dict] = []
    delivery_rows: list[tuple] = []

    for innings_num, innings in enumerate(innings_list, start=1):
        batting_team = innings.get("team")
        # Cricsheet matches have exactly two sides, so the bowling side is
        # whichever of info.teams is not batting.
        bowling_team = None
        if batting_team is not None and len(teams) == 2:
            others = [t for t in teams if t != batting_team]
            bowling_team = others[0] if len(others) == 1 else None

        innings_rows.append(
            {
                "match_id": match_id,
                "innings_num": innings_num,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "is_super_over": bool(innings.get("super_over", False)),
                "declared": bool(innings.get("declared", False)),
                # A forfeited innings carries no overs at all; the row is kept
                # so innings numbering stays aligned with the match.
                "forfeited": bool(innings.get("forfeited", False)),
            }
        )

        for over in innings.get("overs") or []:
            over_num = over.get("over")
            for ball_seq, delivery in enumerate(over.get("deliveries") or [], start=1):
                runs = delivery.get("runs") or {}
                extras = delivery.get("extras") or {}
                wickets = delivery.get("wickets") or []
                wide = extras.get("wides")
                noball = extras.get("noballs")

                if wickets:
                    # Two dismissals on one ball do occur (a run out off a
                    # no-ball) but are vanishingly rare; the first entry is
                    # the striker's own dismissal.
                    first = wickets[0]
                    kind = first.get("kind")
                    dismissed_id = registry.get(first.get("player_out"))
                else:
                    kind = None
                    dismissed_id = None

                delivery_rows.append(
                    (
                        match_id,
                        innings_num,
                        over_num,
                        ball_seq,
                        registry.get(delivery.get("batter")),
                        registry.get(delivery.get("bowler")),
                        registry.get(delivery.get("non_striker")),
                        runs.get("batter", 0),
                        runs.get("extras", 0),
                        runs.get("total", 0),
                        wide,
                        noball,
                        extras.get("byes"),
                        extras.get("legbyes"),
                        extras.get("penalty"),
                        kind,
                        dismissed_id,
                        wide is None and noball is None,
                    )
                )

    return match_row, innings_rows, delivery_rows, playing_xi_rows


# --------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------
def parse_register() -> pd.DataFrame:
    """Load ``people.csv``, keeping the ESPNcricinfo key.

    ``key_cricinfo`` is the join column used later to backfill batting hand
    and bowling style from an external source -- neither the Cricsheet match
    JSON nor its register carries those fields.
    """
    with open(CRICSHEET_PEOPLE_CSV, encoding="utf-8", newline="") as fh:
        rows = [
            {
                "identifier": row["identifier"],
                "name": row["name"],
                "unique_name": row["unique_name"],
                "cricinfo_id": row.get("key_cricinfo") or None,
            }
            for row in csv.DictReader(fh)
        ]
    return pd.DataFrame(rows, columns=["identifier", "name", "unique_name", "cricinfo_id"])


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _deliveries_table(rows: list[tuple]) -> pa.Table:
    columns = list(zip(*rows)) if rows else [()] * len(_DELIVERY_COLUMNS)
    return pa.Table.from_arrays(
        [pa.array(col, type=field.type) for col, field in zip(columns, DELIVERY_SCHEMA)],
        schema=DELIVERY_SCHEMA,
    )


def run(limit: int | None = None, verbose: bool = True) -> dict[str, int]:
    """Parse the whole corpus to parquet. Returns row counts per table."""
    ensure_dirs()
    kind, src = raw_source()
    if verbose:
        print(f"reading match JSON from {kind}: {src}")

    match_rows: list[dict] = []
    innings_rows: list[dict] = []
    xi_rows: list[dict] = []
    pending: list[tuple] = []
    n_deliveries = 0
    failures: list[tuple[str, str]] = []

    writer = pq.ParquetWriter(
        PARSED_DIR / "deliveries.parquet", DELIVERY_SCHEMA, compression="zstd"
    )
    try:
        for i, (match_id, data) in enumerate(iter_raw_matches(limit=limit), start=1):
            try:
                m, inn, dlv, xi = parse_match(match_id, data)
            except Exception as exc:  # noqa: BLE001 -- collected, then reported
                failures.append((match_id, repr(exc)))
                continue
            match_rows.append(m)
            innings_rows.extend(inn)
            xi_rows.extend(xi)
            pending.extend(dlv)

            if i % CHUNK_MATCHES == 0:
                n_deliveries += len(pending)
                writer.write_table(_deliveries_table(pending))
                pending = []
                if verbose:
                    print(f"  {i:>6,} matches / {n_deliveries:>10,} deliveries")
        if pending:
            n_deliveries += len(pending)
            writer.write_table(_deliveries_table(pending))
    finally:
        writer.close()

    matches = pd.DataFrame(match_rows)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["balls_per_over"] = matches["balls_per_over"].astype("int16")
    matches["scheduled_overs"] = matches["scheduled_overs"].astype("float32")
    matches["n_innings"] = matches["n_innings"].astype("int16")

    innings = pd.DataFrame(innings_rows)
    innings["innings_num"] = innings["innings_num"].astype("int16")

    playing_xi = pd.DataFrame(xi_rows)
    playing_xi["position"] = playing_xi["position"].astype("int16")

    people = parse_register()

    matches.to_parquet(PARSED_DIR / "matches.parquet", index=False)
    innings.to_parquet(PARSED_DIR / "innings.parquet", index=False)
    playing_xi.to_parquet(PARSED_DIR / "playing_xi.parquet", index=False)
    people.to_parquet(PARSED_DIR / "people.parquet", index=False)

    if failures:
        print(f"parse: {len(failures)} match(es) failed:")
        for name, err in failures[:20]:
            print(f"  {name}: {err}")

    return {
        "matches": len(matches),
        "innings": len(innings),
        "deliveries": n_deliveries,
        "playing_xi": len(playing_xi),
        "people": len(people),
        "failures": len(failures),
    }


def summary_report() -> str:
    """Human-readable sanity report over the parsed tables."""
    matches = pd.read_parquet(PARSED_DIR / "matches.parquet")
    innings = pd.read_parquet(PARSED_DIR / "innings.parquet")
    playing_xi = pd.read_parquet(PARSED_DIR / "playing_xi.parquet")
    people = pd.read_parquet(PARSED_DIR / "people.parquet")
    deliveries = pd.read_parquet(
        PARSED_DIR / "deliveries.parquet",
        columns=["match_id", "innings_num", "runs_total", "wide", "batter_id"],
    )

    t20_ids = set(matches.loc[matches["match_type"] == "T20", "match_id"])
    t20_inn = innings[innings["match_id"].isin(t20_ids) & ~innings["is_super_over"]]
    t20_dlv = deliveries[deliveries["match_id"].isin(t20_ids)].merge(
        t20_inn[["match_id", "innings_num"]], on=["match_id", "innings_num"], how="inner"
    )
    first = t20_dlv[t20_dlv["innings_num"] == 1].groupby("match_id")["runs_total"].sum()

    lines = [
        "=" * 66,
        "cricsheet ingest summary",
        "=" * 66,
        f"matches          {len(matches):>12,}",
        f"innings          {len(innings):>12,}",
        f"deliveries       {len(deliveries):>12,}",
        f"playing_xi       {len(playing_xi):>12,}",
        f"people           {len(people):>12,}",
        "",
        "matches by type:",
    ]
    for mt, n in matches["match_type"].value_counts().items():
        lines.append(f"  {mt:<8} {n:>8,}")
    bpo5 = matches[matches["balls_per_over"] == 5]
    lines += [
        "",
        f"date range       {matches['date'].min():%Y-%m-%d} .. {matches['date'].max():%Y-%m-%d}",
        f"balls_per_over   {dict(matches['balls_per_over'].value_counts())}",
        f"  bpo==5         {len(bpo5):,} matches, types={sorted(bpo5['match_type'].unique())}, "
        f"overs={sorted(bpo5['scheduled_overs'].dropna().unique())}, "
        f"gender={dict(bpo5['gender'].value_counts())}",
        f"outcome_result   {dict(matches['outcome_result'].value_counts(dropna=False))}",
        f"outcome_method   {dict(matches['outcome_method'].value_counts(dropna=False))}",
        "",
        f"T20 non-super-over innings   {len(t20_inn):>10,}",
        f"T20 first innings            {int((t20_inn['innings_num'] == 1).sum()):>10,}",
        f"T20 legal deliveries (ex-wd) {int(t20_dlv['wide'].isna().sum()):>10,}",
        f"T20 1st-innings totals       n={len(first):,} mean={first.mean():.1f} "
        f"sd={first.std():.1f} p10={first.quantile(0.10):.0f} p90={first.quantile(0.90):.0f}",
        "",
        f"unresolved batter_id on deliveries: {int(deliveries['batter_id'].isna().sum()):,}",
        f"unresolved player_id on playing_xi: {int(playing_xi['player_id'].isna().sum()):,}",
        "=" * 66,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
    print()
    print(summary_report())
