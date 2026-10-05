# src/features/players.py
"""Player attributes known at the toss: style, usual batting position, and how
much each bowler bowls in the powerplay. Histories are post-date states read
point-in-time with `as_of`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import PARSED, PLAYER_META_RDA
from src.features.form import post_match_ewm

_SPIN_WORDS = ("offbreak", "legbreak", "orthodox", "wrist")


def bowler_type(style) -> str:
    """pace/spin x arm from Cricinfo's free-text style. First listed style wins."""
    if not isinstance(style, str):
        return "unknown"
    s = style.split(",")[0].strip().lower().replace("-", " ")
    if "left" in s:
        arm = "left"
    elif "right" in s or s.startswith("legbreak"):
        arm = "right"
    else:
        return "unknown"
    if any(w in s for w in _SPIN_WORDS) or s.endswith("slow"):
        return f"spin_{arm}"
    if "medium" in s or "fast" in s:
        return f"pace_{arm}"
    return "unknown"


def batting_hand(style) -> str:
    if not isinstance(style, str) or "bat" not in style.lower():
        return "unknown"
    s = style.lower()
    return "left" if s.startswith("left") else "right" if s.startswith("right") else "unknown"


def player_styles() -> pd.DataFrame:
    import pyreadr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # pyreadr warns casting the dob column
        meta = next(iter(pyreadr.read_r(str(PLAYER_META_RDA)).values()))
    meta = meta.dropna(subset=["cricsheet_id"]).drop_duplicates("cricsheet_id")
    return pd.DataFrame({
        "player_id": meta.cricsheet_id.astype(str).to_numpy(),
        "bowler_type": [bowler_type(s) for s in meta.bowling_style],
        "batting_hand": [batting_hand(s) for s in meta.batting_style],
    })


def annotate(balls: pd.DataFrame, styles: pd.DataFrame) -> pd.DataFrame:
    s = styles.set_index("player_id")
    return balls.assign(
        bowler_type=balls.bowler_id.map(s.bowler_type).fillna("unknown"),
        batting_hand=balls.batter_id.map(s.batting_hand).fillna("unknown"),
    )


def batting_positions(balls: pd.DataFrame) -> pd.DataFrame:
    """Order of first appearance (striker before non-striker on the same ball)."""
    seq = np.arange(len(balls), dtype=float)
    cols = ["match_id", "innings_num", "date"]
    a = balls[cols].assign(player_id=balls.batter_id.to_numpy(), seq=seq)
    b = balls[cols].assign(player_id=balls.non_striker_id.to_numpy(), seq=seq + 0.5)
    first = (pd.concat([a, b]).sort_values("seq", kind="stable")
             .drop_duplicates(["match_id", "innings_num", "player_id"]))
    first["bat_pos"] = first.groupby(["match_id", "innings_num"]).cumcount() + 1
    return first[["match_id", "innings_num", "date", "player_id", "bat_pos"]].reset_index(drop=True)


def batting_order_history(balls: pd.DataFrame) -> pd.DataFrame:
    per = batting_positions(balls).groupby(["player_id", "date"], as_index=False).bat_pos.mean()
    return post_match_ewm(per, "player_id", ["bat_pos"])


def load_xi(balls: pd.DataFrame) -> pd.DataFrame:
    xi = pd.read_parquet(PARSED / "playing_xi.parquet")
    dates = balls[["match_id", "date"]].drop_duplicates("match_id")
    return xi.merge(dates, on="match_id")[["match_id", "date", "team", "player_id", "position"]]


def powerplay_history(balls: pd.DataFrame, xi: pd.DataFrame) -> pd.DataFrame:
    """EWM of overs bowled in overs 1-6 per appearance; appearances with none count 0."""
    pp = (balls[balls.over < 6].groupby(["match_id", "bowler_id"]).over.nunique()
          .rename("pp_overs").reset_index().rename(columns={"bowler_id": "player_id"}))
    app = xi[["match_id", "player_id", "date"]].merge(
        pp, on=["match_id", "player_id"], how="left").fillna({"pp_overs": 0})
    per = app.groupby(["player_id", "date"], as_index=False).pp_overs.mean()
    return post_match_ewm(per, "player_id", ["pp_overs"])
