"""Resolve a Betfair event (team names + date) to a Cricsheet match_id.

There is no shared identifier between Betfair's historical data and
Cricsheet, so matching is done on:

1. team names (both orderings), after normalizing known naming drift
   (verified against the real downloaded data -- e.g. Betfair/Cricsheet both
   use "Royal Challengers Bangalore" for some events and "Royal Challengers
   Bengaluru" for others, sometimes even within the same season; Betfair was
   also observed to occasionally typo "Kolkata Knight Rider" without the
   trailing S, abbreviate to "RC Bangalore"/"RC Bengaluru", use bare city
   names like "Delhi v Punjab" instead of franchise names, and occasionally
   truncate a long event name mid-word e.g. "...Royal Challengers Bengalu").
   Exact-match failures fall back to a word-containment/prefix comparison
   (see `_loose_team_name_match`) that catches the bare-city and truncation
   cases; results resolved this way are flagged `loose_match=True` on
   `MatchResult` so they stay auditable rather than silently accepted, plus
2. the match date (Cricsheet's `date` column vs. the Betfair market's date,
   derived from `marketTime`), plus
3. venue as a tiebreaker when multiple Cricsheet matches share the same
   date + team pair (rare, but possible on a multi-match tournament day).

Ambiguous or unresolved cases are reported, not silently guessed at -- see
`MatchResult.status`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

# Canonical-name aliases, derived from names actually observed in the
# downloaded Betfair archive and/or in data/parsed/matches.parquet. Keys and
# values are both lowercased for comparison; the alias collapses onto a
# single canonical string so that e.g. "Royal Challengers Bangalore" and
# "Royal Challengers Bengaluru" compare equal.
TEAM_NAME_ALIASES: dict[str, str] = {
    "royal challengers bangalore": "royal challengers bengaluru",
    "royal challengers bengaluru": "royal challengers bengaluru",
    "rc bangalore": "royal challengers bengaluru",
    "rc bengaluru": "royal challengers bengaluru",
    "kolkata knight rider": "kolkata knight riders",
    "kolkata knight riders": "kolkata knight riders",
    "delhi daredevils": "delhi capitals",  # franchise renamed 2019
    "delhi capitals": "delhi capitals",
    "kings xi punjab": "punjab kings",  # franchise renamed 2021
    "punjab kings": "punjab kings",
    "rising pune supergiant": "rising pune supergiants",
    "rising pune supergiants": "rising pune supergiants",
    # Country abbreviation: verified real no_candidates case -- Betfair uses
    # "UAE", Cricsheet uses the full "United Arab Emirates". Not fixable by
    # word-containment since "uae" isn't literally a substring of "united
    # arab emirates".
    "uae": "united arab emirates",
    "united arab emirates": "united arab emirates",
    # English T20 Blast brand name vs county name: verified real
    # no_candidates case -- Betfair's bare "Birmingham" is Cricsheet's
    # "Warwickshire" (Birmingham Bears is Warwickshire's short-format
    # team). Most other counties' Blast brand names retain the county name
    # as a word (e.g. "Sussex Sharks" still contains "Sussex", caught by
    # the loose-match fallback already) -- Warwickshire is the one
    # confirmed exception. Caution: bare "Birmingham" is theoretically
    # ambiguous with "Birmingham Phoenix" (The Hundred), but in the archive
    # Phoenix fixtures always spell out "Phoenix"/"Bears" explicitly (this
    # alias only fires on the literal bare string "Birmingham", not on
    # "Birmingham Phoenix"/"Birmingham Bears", which are left to the
    # loose-match fallback instead) -- of ~47 Birmingham-related event
    # names found, only 2 were bare, both Blast-context.
    "birmingham": "warwickshire",
    "warwickshire": "warwickshire",
}

_WOMENS_SUFFIX_RE = re.compile(r"\s+(women|w|wbbl)$", re.IGNORECASE)


def normalize_team_name(name: str) -> str:
    """Canonicalize a team name for matching purposes.

    Lowercases, strips whitespace, normalizes "&" to "and" (verified real
    case: Betfair's "Trinidad and Tobago" vs Cricsheet's "Trinidad & Tobago
    Red Steel" only differ by this), rewrites the one confirmed
    city-spelling variant (Bangalore/Bengaluru -- same city, same IPL
    franchise, just two spellings; safe to always collapse since no other
    real team uses either word), then collapses known full-name aliases.
    Does NOT strip a "Women"/"W" suffix by default -- that's handled
    separately by `split_gender_suffix` since men's and women's teams
    sharing a club name (e.g. "Royal Challengers Bengaluru") must NOT be
    matched to each other.
    """
    n = " ".join(name.strip().split()).lower()
    n = n.replace(" & ", " and ")
    n = n.replace("bangalore", "bengaluru")
    return TEAM_NAME_ALIASES.get(n, n)


def split_gender_suffix(name: str) -> tuple[str, bool]:
    """Split a trailing Women/W/WBBL marker off a team name.

    Returns (base_name, is_womens). E.g. "Royal Challengers Bangalore W" ->
    ("Royal Challengers Bangalore", True); "Brisbane Heat WBBL" -> ("Brisbane
    Heat", True) (WBBL = Women's Big Bash League, seen verbatim appended to
    both team names in real event names for that competition).
    """
    m = _WOMENS_SUFFIX_RE.search(name.strip())
    if m:
        return name[: m.start()].strip(), True
    return name.strip(), False


_TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")
# e.g. "Australia v India - 3rd Test", "UAE v Ireland - 1st ODI" (verified:
# 35 real event names use this " - <ordinal/Only> <format>" suffix instead
# of parentheses).
_TRAILING_DASH_SUFFIX_RE = re.compile(
    r"\s*-\s*(\d+(st|nd|rd|th)|Only)\s+\S+\s*$", re.IGNORECASE
)


def parse_event_name(event_name: str) -> tuple[str, str] | None:
    """Split a Betfair eventName like "Team A v Team B" into (team_a, team_b).

    Betfair frequently appends a series-position annotation, either in
    parentheses -- e.g. "New Zealand v Australia (1st T20)" or "Ireland v
    Australia (Only ODI)" (verified: ~24% of real event names in the archive
    have this) -- or after a dash, e.g. "Australia v India - 3rd Test"
    (verified: 35 real event names). Both are stripped before splitting on
    " v " -- otherwise the second team name would come out as e.g.
    "Australia (1st T20)" or "India - 3rd Test" and never match anything.

    Returns None if the event name doesn't look like a two-team match (e.g.
    tournament/group names such as "ICC Women's World Cup").
    """
    if not event_name or " v " not in event_name:
        return None
    cleaned = _TRAILING_PAREN_RE.sub("", event_name)
    cleaned = _TRAILING_DASH_SUFFIX_RE.sub("", cleaned)
    parts = cleaned.split(" v ")
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    if not a or not b:
        return None
    return a, b


@dataclass
class MatchResult:
    status: str  # "matched" | "no_candidates" | "ambiguous" | "unparseable_event_name"
    match_id: str | None = None
    candidates: list[str] | None = None  # match_ids, for ambiguous/no_candidates diagnostics
    normalized_team_a: str | None = None
    normalized_team_b: str | None = None
    is_womens: bool | None = None
    note: str | None = None
    loose_match: bool = False  # True if resolved only via the word-containment/prefix fallback


_MIN_LOOSE_TOKEN_LEN = 4  # guard against short strings matching too eagerly


def _loose_team_name_match(betfair_norm: str, cricsheet_norm: str) -> bool:
    """Looser-than-exact team-name comparison, used only as a fallback after
    an exact match fails. Handles two verified real patterns that exact
    matching cannot:

    - Betfair uses a bare city/abbreviated name instead of the full
      franchise name, e.g. "Delhi" for "Delhi Capitals", "Punjab" for
      "Kings XI Punjab" -- caught by whole-word containment.
    - Betfair's event name is truncated (a fixed-length field cut off
      mid-word), e.g. "Royal Challengers Bengalu" for "...Bengaluru" --
      caught by a prefix check.

    Both sides must be at least `_MIN_LOOSE_TOKEN_LEN` characters to avoid
    short strings matching too much (e.g. "II" or "A").
    """
    if betfair_norm == cricsheet_norm:
        return True
    if len(betfair_norm) < _MIN_LOOSE_TOKEN_LEN or len(cricsheet_norm) < _MIN_LOOSE_TOKEN_LEN:
        return False
    if re.search(rf"\b{re.escape(betfair_norm)}\b", cricsheet_norm) or re.search(
        rf"\b{re.escape(cricsheet_norm)}\b", betfair_norm
    ):
        return True
    # Truncation guard: require a slightly longer minimum so short prefixes
    # (e.g. "Ind" vs "India") don't spuriously match unrelated names.
    if len(betfair_norm) >= 6 and len(cricsheet_norm) >= 6:
        if cricsheet_norm.startswith(betfair_norm) or betfair_norm.startswith(cricsheet_norm):
            return True
    return False


def _loose_pair_match(norm_a: str, norm_b: str, row_team1: str, row_team2: str) -> bool:
    return (_loose_team_name_match(norm_a, row_team1) and _loose_team_name_match(norm_b, row_team2)) or (
        _loose_team_name_match(norm_a, row_team2) and _loose_team_name_match(norm_b, row_team1)
    )


def match_event(
    event_name: str,
    event_date: str | datetime,
    matches_df: pd.DataFrame,
    venue_hint: str | None = None,
) -> MatchResult:
    """Resolve one Betfair event to a Cricsheet match_id.

    `matches_df` is expected to look like data/parsed/matches.parquet: one
    row per match, columns include match_id, date (str 'YYYY-MM-DD'), team1,
    team2, venue.

    Matching rule: normalize both Betfair team names, normalize both
    Cricsheet team names for every match on the same date, and require the
    normalized {team_a, team_b} sets to be equal (order-independent) AND the
    women's/men's flags to agree (derived from a trailing "Women"/"W" suffix
    on the Betfair side; Cricsheet distinguishes via its own team names,
    which for women's teams also carry a Women-ish marker in this dataset --
    see note below). If more than one Cricsheet match matches on
    date+teams, venue (if provided) is used as a tiebreaker; if still
    ambiguous, status="ambiguous" and no match_id is returned.
    """
    parsed = parse_event_name(event_name)
    if parsed is None:
        return MatchResult(status="unparseable_event_name", note=f"could not split eventName: {event_name!r}")

    raw_a, raw_b = parsed
    base_a, women_a = split_gender_suffix(raw_a)
    base_b, women_b = split_gender_suffix(raw_b)
    is_womens = women_a or women_b
    norm_a = normalize_team_name(base_a)
    norm_b = normalize_team_name(base_b)
    target = {norm_a, norm_b}

    if isinstance(event_date, datetime):
        date_str = event_date.strftime("%Y-%m-%d")
    else:
        date_str = str(event_date)[:10]

    same_day = matches_df[matches_df["date"] == date_str]
    # Cricsheet's own team names don't carry a "Women"/"W" marker (e.g. WPL
    # rows use plain "Royal Challengers Bangalore"), so gender must be
    # matched via Cricsheet's separate `gender` column, not via team-name
    # text. Without this, a same-day men's and women's fixture between the
    # same two countries (which does happen, e.g. bilateral series played
    # concurrently) would be ambiguous or silently cross-matched.
    if "gender" in same_day.columns:
        wanted_gender = "female" if is_womens else "male"
        same_day = same_day[same_day["gender"] == wanted_gender]
    if same_day.empty:
        return MatchResult(
            status="no_candidates",
            normalized_team_a=norm_a,
            normalized_team_b=norm_b,
            is_womens=is_womens,
            note=f"no Cricsheet matches on {date_str}",
        )

    def row_team_set(row) -> set[str]:
        return {normalize_team_name(row["team1"]), normalize_team_name(row["team2"])}

    candidates = same_day[same_day.apply(lambda r: row_team_set(r) == target, axis=1)]
    loose_match = False

    if candidates.empty:
        # Fall back to word-containment/prefix matching (see
        # _loose_team_name_match) -- catches bare city names ("Delhi" for
        # "Delhi Capitals") and truncated event names ("Royal Challengers
        # Bengalu" for "...Bengaluru"), both verified as real patterns in
        # the archive. Only accepted if it resolves to exactly one same-day
        # candidate; anything else is reported, not guessed at.
        loose_candidates = same_day[
            same_day.apply(
                lambda r: _loose_pair_match(
                    norm_a, norm_b, normalize_team_name(r["team1"]), normalize_team_name(r["team2"])
                ),
                axis=1,
            )
        ]
        if len(loose_candidates) == 1:
            candidates = loose_candidates
            loose_match = True
        elif len(loose_candidates) > 1:
            return MatchResult(
                status="ambiguous",
                candidates=loose_candidates["match_id"].astype(str).tolist(),
                normalized_team_a=norm_a,
                normalized_team_b=norm_b,
                is_womens=is_womens,
                note=(
                    f"{len(loose_candidates)} Cricsheet matches on {date_str} loosely match teams "
                    f"{{{norm_a!r}, {norm_b!r}}} (word-containment/prefix fallback); no exact match"
                ),
            )
        else:
            return MatchResult(
                status="no_candidates",
                normalized_team_a=norm_a,
                normalized_team_b=norm_b,
                is_womens=is_womens,
                note=f"no Cricsheet match on {date_str} with teams {{{norm_a!r}, {norm_b!r}}}",
            )

    if len(candidates) == 1:
        return MatchResult(
            status="matched",
            match_id=str(candidates.iloc[0]["match_id"]),
            normalized_team_a=norm_a,
            normalized_team_b=norm_b,
            is_womens=is_womens,
            loose_match=loose_match,
            note="matched via word-containment/prefix fallback (no exact team-name match)" if loose_match else None,
        )

    # Multiple same-day, same-teams matches (rare -- e.g. a bilateral series
    # with two matches technically dated the same day is not expected, but a
    # tournament could plausibly have same team pair on same day across
    # different formats/groups). Try venue as a tiebreaker.
    if venue_hint:
        vnorm = " ".join(venue_hint.strip().split()).lower()
        venue_matches = candidates[
            candidates["venue"].fillna("").str.lower().str.strip().apply(lambda v: v in vnorm or vnorm in v)
        ]
        if len(venue_matches) == 1:
            return MatchResult(
                status="matched",
                match_id=str(venue_matches.iloc[0]["match_id"]),
                normalized_team_a=norm_a,
                normalized_team_b=norm_b,
                is_womens=is_womens,
                note="disambiguated by venue",
            )

    return MatchResult(
        status="ambiguous",
        candidates=candidates["match_id"].astype(str).tolist(),
        normalized_team_a=norm_a,
        normalized_team_b=norm_b,
        is_womens=is_womens,
        note=f"{len(candidates)} Cricsheet matches share date {date_str} and teams {{{norm_a!r}, {norm_b!r}}}",
    )
