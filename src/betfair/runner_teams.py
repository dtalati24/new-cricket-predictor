"""Map every MATCH_ODDS runner to the Cricsheet team it represents.

Why this is not a name lookup
-----------------------------
Betfair runner names and Cricsheet team names disagree constantly and in
several different ways at once:

    Betfair "Delhi"                  Cricsheet "Delhi Daredevils"
    Betfair "Brisbane Heat WBBL"     Cricsheet "Brisbane Heat"
    Betfair "India Women"            Cricsheet "India"
    Betfair "St Kitts & Nevis ..."   Cricsheet "St Kitts and Nevis Patriots"

72.8% of runners match a team name exactly; the remaining 27.2% do not, and a
global name dictionary would have to be hand-maintained forever. Worse, the
retrospective (§2, "Registry-id player resolution") records that 274 Cricsheet
*player* names are shared by more than one real person. The general lesson --
name joins are hostile -- applies to teams too: "Delhi" was Delhi Daredevils
until 2018 and Delhi Capitals after, and Cricsheet uses both.

So this module does not join on names. It solves a **two-by-two assignment
problem per market**. A MATCH_ODDS market has exactly two active runners, and
the Cricsheet match it was joined to has exactly two teams. There are only two
possible assignments. We score both and keep the one that wins by a margin.
That turns a hard open-vocabulary matching problem into a binary choice with a
built-in ambiguity measure, and it is robust to any renaming that does not
also rename the opponent to something more similar.

Independent validation
----------------------
The mapping is built from *names only*. Betfair independently records which
runner was the WINNER, and Cricsheet independently records `outcome_winner`.
Agreement between the two is therefore a genuine out-of-model check on the
assignment, and it is computed in `validate()` and reported. It is deliberately
NOT used to build or repair the mapping.

Output: data/ticks/runner_team_map.parquet, one row per (market_id, runner_id).
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from src import config

OUT = config.TICKS / "runner_team_map.parquet"

# Suffixes Betfair appends that carry competition/gender, not team identity.
# They are dropped only when doing so leaves a non-empty name, and dropping
# them can never *create* a match with the wrong team, because both candidate
# teams in a market are in the same competition and the same gender.
_NOISE_TOKENS = {
    "women", "womens", "w", "ladies",
    "wbbl", "wcpl", "wpl", "wbb",
    "xi", "cc", "cricket", "club", "team",
}

_ABBREV = {
    "utd": "united",
    "intl": "international",
    "int": "international",
    "uae": "unitedarabemirates",
    "usa": "unitedstatesofamerica",
    "png": "papuanewguinea",
    "rsa": "southafrica",
    "nz": "newzealand",
    "sa": "southafrica",
    "wi": "westindies",
    "st": "saint",
}


def normalise(name: str) -> str:
    """Lowercase, de-accent, expand '&', strip punctuation."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(name: str, drop_noise: bool = True) -> list[str]:
    toks = [_ABBREV.get(t, t) for t in normalise(name).split()]
    if drop_noise:
        kept = [t for t in toks if t not in _NOISE_TOKENS]
        if kept:
            return kept
    return toks


def _token_match(a: str, b: str) -> bool:
    """True when two tokens plausibly denote the same word.

    Equality, or one being a prefix of the other with at least three
    characters -- which covers "chall"/"challengers", "hyd"/"hyderabad",
    "superchargers"/"supercharg". Three characters is deliberately
    conservative: shorter prefixes start matching unrelated cities.
    """
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    return False


def similarity(runner: str, team: str) -> float:
    """Symmetric token-coverage score in [0, 1], with a character tiebreak."""
    rt, tt = tokens(runner), tokens(team)
    if not rt or not tt:
        return 0.0
    r_hit = sum(any(_token_match(r, t) for t in tt) for r in rt)
    t_hit = sum(any(_token_match(r, t) for r in rt) for t in tt)
    dice = (r_hit + t_hit) / (len(rt) + len(tt))
    # Character-level similarity on the joined token strings, used only to
    # break ties between two structurally equal candidates.
    chars = SequenceMatcher(None, "".join(rt), "".join(tt)).ratio()
    return 0.9 * dice + 0.1 * chars


def assign_market(name_1: str, name_2: str, team_a: str, team_b: str) -> dict:
    """Score both assignments of two runners onto two teams.

    Returns the winning assignment plus the margin over the alternative. A
    margin of 0 means the names cannot distinguish the two teams at all.
    """
    s = {
        ("1", "a"): similarity(name_1, team_a),
        ("1", "b"): similarity(name_1, team_b),
        ("2", "a"): similarity(name_2, team_a),
        ("2", "b"): similarity(name_2, team_b),
    }
    straight = s[("1", "a")] + s[("2", "b")]
    swapped = s[("1", "b")] + s[("2", "a")]
    if straight >= swapped:
        return {
            "team_1": team_a, "team_2": team_b,
            "score": straight / 2, "margin": (straight - swapped) / 2,
            "min_sim": min(s[("1", "a")], s[("2", "b")]),
        }
    return {
        "team_1": team_b, "team_2": team_a,
        "score": swapped / 2, "margin": (swapped - straight) / 2,
        "min_sim": min(s[("1", "b")], s[("2", "a")]),
    }


# A mapping is accepted when the winning assignment beats its mirror by at
# least MIN_MARGIN and both runners individually clear MIN_SIM. Both are
# needed: margin alone accepts "Delhi"/"Kolkata" vs two teams neither of which
# is either; min_sim alone accepts a market where both teams score 0.9.
MIN_MARGIN = 0.05
MIN_SIM = 0.35

# Squad-level qualifiers. Betfair distinguishes "Bangladesh U19" from
# "Bangladesh"; Cricsheet's team field for a senior fixture does not. A market
# whose runner carries one of these while the joined Cricsheet team does not
# is evidence that the *event join*, not the name match, is wrong -- it is a
# different fixture entirely. Such markets are marked unresolved.
_SQUAD_QUALIFIERS = ("u19", "u23", "under19", "development", "emerging",
                     "invitational", "legends", "academy", "presidents")


def _squad_mismatch(runner: str, team: str) -> bool:
    rt, tt = set(tokens(runner, drop_noise=False)), set(tokens(team, drop_noise=False))
    for q in _SQUAD_QUALIFIERS:
        if q in rt and q not in tt:
            return True
    # A single trailing "a" token ("New Zealand A") is the A-team marker.
    r_raw = tokens(runner, drop_noise=False)
    t_raw = tokens(team, drop_noise=False)
    if len(r_raw) > 1 and r_raw[-1] == "a" and (not t_raw or t_raw[-1] != "a"):
        return True
    return False


def build() -> pd.DataFrame:
    runners = pd.read_parquet(config.TICKS / "match_odds_runners.parquet")
    markets = pd.read_parquet(config.MATCH_ODDS_MARKETS)

    meta = markets.set_index("market_id")[
        ["match_id", "team_a", "team_b", "outcome_winner"]
    ]
    rows = []
    for mid, grp in runners.groupby("market_id", sort=True):
        if mid not in meta.index:
            continue
        info = meta.loc[mid]
        grp = grp.sort_values("runner_id")
        active = grp[grp["final_status"] != "REMOVED"]
        # Markets with a third selection (e.g. "The Draw", or a duplicated
        # team entry) are not two-runner match odds and are recorded as
        # unresolved rather than force-fitted.
        pool = active if len(active) == 2 else grp

        def emit(r, team, res, ok, reason):
            rows.append(
                {
                    "market_id": mid, "runner_id": r.runner_id,
                    "runner_name": r.runner_name, "final_status": r.final_status,
                    "match_id": info.match_id, "team": team if ok else None,
                    "score": res.get("score", float("nan")),
                    "margin": res.get("margin", float("nan")),
                    "min_sim": res.get("min_sim", float("nan")),
                    "resolved": bool(ok), "reason": reason,
                }
            )

        if len(pool) != 2:
            for r in grp.itertuples(index=False):
                emit(r, None, {}, False, f"n_runners={len(grp)}")
            continue

        r1, r2 = pool.itertuples(index=False)
        res = assign_market(r1.runner_name, r2.runner_name, info.team_a, info.team_b)
        ok = res["margin"] >= MIN_MARGIN and res["min_sim"] >= MIN_SIM
        reason = ""
        if not ok:
            reason = "low_margin" if res["margin"] < MIN_MARGIN else "low_similarity"
        elif _squad_mismatch(r1.runner_name, res["team_1"]) or _squad_mismatch(
            r2.runner_name, res["team_2"]
        ):
            ok, reason = False, "squad_level_mismatch"
        for r, team in ((r1, res["team_1"]), (r2, res["team_2"])):
            emit(r, team, res, ok, reason)
        # Runners excluded from the two-runner pool (Betfair sometimes voids a
        # truncated duplicate of a runner name) are recorded as unresolved so
        # the output covers every runner in the source table.
        for r in grp.itertuples(index=False):
            if r.runner_id not in (r1.runner_id, r2.runner_id):
                emit(r, None, {}, False, "voided_duplicate_runner")
    return pd.DataFrame(rows)


def validate(mapping: pd.DataFrame) -> dict:
    """Check the name-only mapping against the two independent outcome fields."""
    markets = pd.read_parquet(config.MATCH_ODDS_MARKETS)
    m = mapping.merge(
        markets[["market_id", "outcome_winner", "outcome_result"]],
        on="market_id", how="left",
    )
    # Only settled two-runner markets with a decided Cricsheet result can be
    # checked: dead heats, voids and no-results carry no usable label.
    chk = m[
        m["resolved"]
        & m["final_status"].isin(["WINNER", "LOSER"])
        & m["outcome_winner"].notna()
    ].copy()
    n_win = chk.groupby("market_id")["final_status"].apply(lambda s: (s == "WINNER").sum())
    chk = chk[chk["market_id"].isin(n_win[n_win == 1].index)]
    win = chk[chk["final_status"] == "WINNER"]
    agree = (win["team"] == win["outcome_winner"]).mean()
    return {
        "runners_total": len(mapping),
        "runners_resolved": int(mapping["resolved"].sum()),
        "coverage_runners": float(mapping["resolved"].mean()),
        "markets_total": mapping["market_id"].nunique(),
        "markets_fully_resolved": int(
            mapping.groupby("market_id")["resolved"].all().sum()
        ),
        "checkable_markets": int(win["market_id"].nunique()),
        "winner_agreement": float(agree),
        "disagreements": int((win["team"] != win["outcome_winner"]).sum()),
    }


def run() -> dict:
    mapping = build()
    config.TICKS.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(OUT, index=False)
    stats = validate(mapping)
    return stats


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k}: {v}")
