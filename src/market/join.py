"""Join Betfair events to Cricsheet matches.

Sits between the two ingest layers and produces the crosswalk every later
stage needs: Betfair ``event_id`` -> Cricsheet ``match_id``.

Two adaptations are made here rather than by editing the salvaged matcher in
``betfair.match_to_cricsheet``, so that module stays byte-identical to the
version its 26 tests were written against:

1. **Column names.** The matcher expects ``team1``/``team2``; the parsed
   Cricsheet table uses ``team_a``/``team_b``.
2. **Date tolerance.** The matcher filters on an exact date string. Betfair
   ``marketTime`` is UTC while Cricsheet dates are local match dates, so a
   late-evening fixture west of UTC (Caribbean, Americas) legitimately lands
   on the neighbouring UTC day. We therefore retry at -1 and +1 days and
   record the offset used, rather than silently widening the window.

Dates ALWAYS come from ``marketTime``, never from the archive folder path --
a market's ``marketTime`` can differ from its folder date by several days.
"""

from __future__ import annotations

import pandas as pd

from src.betfair.match_to_cricsheet import match_event
from src.config import BETFAIR_DIR, MARKET_DIR, PARSED_DIR, ensure_dirs

# Offsets tried in order. 0 first so an exact hit always wins.
_DATE_OFFSETS = (0, -1, 1)


def adapt_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Rename/retype the parsed match table into the matcher's expected shape."""
    missing = {"team_a", "team_b", "date", "gender", "venue", "match_id"} - set(matches.columns)
    if missing:
        raise KeyError(f"matches table is missing {sorted(missing)}")
    out = matches.rename(columns={"team_a": "team1", "team_b": "team2"}).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def event_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-market manifest to one row per Betfair event.

    ``event_name`` can drift across a single event's markets, so the most
    common spelling is used. ``market_time`` uses the earliest market on the
    event, which is the closest available proxy for the scheduled start.
    """
    g = manifest.groupby("event_id")
    out = pd.DataFrame(
        {
            "event_name": g["event_name"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
            "market_time": g["market_time"].min(),
            "n_markets": g.size(),
        }
    ).reset_index()
    return out


def join_events(manifest: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Resolve every Betfair event to a Cricsheet match, with an audit trail."""
    adapted = adapt_matches(matches)
    by_date = {d: g for d, g in adapted.groupby("date", sort=False)}
    _EMPTY = adapted.iloc[0:0]
    events = event_table(manifest)

    rows = []
    for ev in events.itertuples(index=False):
        if pd.isna(ev.market_time) or not ev.event_name:
            rows.append(
                {"event_id": ev.event_id, "status": "no_market_time", "match_id": None,
                 "date_offset": None, "loose_match": None, "note": None}
            )
            continue

        base = pd.Timestamp(ev.market_time)
        result = None
        used_offset = None
        for off in _DATE_OFFSETS:
            probe = (base + pd.Timedelta(days=off)).strftime("%Y-%m-%d")
            # Pass only that day's matches. match_event re-filters by date
            # internally (a no-op here), but scanning the full 22.5k-row table
            # once per probe would be ~1.7B comparisons across the archive.
            r = match_event(ev.event_name, probe, by_date.get(probe, _EMPTY))
            if r.status == "matched":
                result, used_offset = r, off
                break
            # Keep the first non-trivial failure so the audit trail is useful.
            if result is None or (result.status == "no_candidates" and r.status != "no_candidates"):
                result, used_offset = r, off
        rows.append(
            {
                "event_id": ev.event_id,
                "status": result.status,
                "match_id": result.match_id,
                "date_offset": used_offset if result.status == "matched" else None,
                "loose_match": bool(getattr(result, "loose_match", False)),
                "note": result.note,
            }
        )

    joined = events.merge(pd.DataFrame(rows), on="event_id", how="left")
    return joined


def run(save: bool = True) -> pd.DataFrame:
    manifest = pd.read_parquet(BETFAIR_DIR / "market_manifest.parquet")
    matches = pd.read_parquet(PARSED_DIR / "matches.parquet")
    joined = join_events(manifest, matches)
    if save:
        ensure_dirs()
        joined.to_parquet(MARKET_DIR / "event_match.parquet", index=False)
    return joined


if __name__ == "__main__":
    j = run()
    n = len(j)
    print(f"events: {n:,}")
    print(j["status"].value_counts().to_string())
    m = j[j["status"] == "matched"]
    print(f"\nmatched: {len(m):,} ({100*len(m)/n:.1f}%)")
    print(f"  loose_match: {int(m['loose_match'].sum()):,}")
    print("  date_offset used:")
    print(m["date_offset"].value_counts().sort_index().to_string())
    dup = m["match_id"].duplicated().sum()
    print(f"\ndistinct cricsheet matches: {m['match_id'].nunique():,} (duplicate event->match: {dup:,})")
