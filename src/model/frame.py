# src/model/frame.py
"""One row per first-innings delivery: skill, form and the situation before the ball.

The simulator builds exactly these columns, in exactly this order:
static columns (skill, hint, form) then SITUATION.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import as_of
from src.features.context import CONTEXT
from src.features.outcomes import CLASSES, NOT_LEGAL, is_wicket
from src.skill.shrink import Level, SkillModel, estimates

SITUATION = ["over", "ball_in_over", "wickets", "runs", "batter_balls", "free_hit"]
PP_FORM = ["bat_pp_sr", "bat_pp_out", "bowl_pp_econ", "bowl_pp_wkt"]
FORM = ["bat_form_sr", "bat_form_out", "bowl_form_econ", "bowl_form_wkt"] + PP_FORM


def skill_columns(levels: tuple[Level, ...]) -> list[str]:
    return ([f"{lv.name}__{c}" for lv in levels for c in CLASSES]
            + [f"hint__{c}" for c in CLASSES])


def static_columns(levels: tuple[Level, ...]) -> list[str]:
    return skill_columns(levels) + FORM + CONTEXT


def feature_columns(levels: tuple[Level, ...]) -> list[str]:
    return static_columns(levels) + SITUATION


def skill_features(skill: SkillModel, query: pd.DataFrame) -> pd.DataFrame:
    est = estimates(skill, query)
    bat = est.get("batter_x_bowlertype", est.get("batter"))
    bowl = est.get("bowler_x_battinghand", est.get("bowler"))
    hint = bat * bowl / skill.league            # batter rate x bowler rate / league rate
    hint = hint / hint.sum(axis=1, keepdims=True)
    blocks = [est[lv.name] for lv in skill.levels] + [hint]
    return pd.DataFrame(np.hstack(blocks).astype(np.float32), index=query.index,
                        columns=skill_columns(skill.levels))


def form_features(query: pd.DataFrame, bat_form: pd.DataFrame,
                  bowl_form: pd.DataFrame) -> pd.DataFrame:
    qb = query[["batter_id", "date"]].rename(columns={"batter_id": "player_id"})
    qw = query[["bowler_id", "date"]].rename(columns={"bowler_id": "player_id"})
    a = as_of(bat_form, qb, ["player_id"], [c for c in FORM if c.startswith("bat_")])
    b = as_of(bowl_form, qw, ["player_id"], [c for c in FORM if c.startswith("bowl_")])
    return pd.concat([a, b], axis=1).astype(np.float32)


def situation(b: pd.DataFrame) -> pd.DataFrame:
    """State before each delivery. `b` is first-innings balls in delivery order."""
    m = b.match_id
    legal = (~b.outcome.isin(NOT_LEGAL)).astype(int)
    wkt = is_wicket(b).astype(int)
    faced = (b.outcome != "wide").astype(int)
    legal_before = legal.groupby(m).cumsum() - legal
    return pd.DataFrame({
        "over": legal_before // 6,
        "ball_in_over": legal_before % 6,
        "wickets": wkt.groupby(m).cumsum() - wkt,
        "runs": b.runs_total.astype(int).groupby(m).cumsum() - b.runs_total.astype(int),
        "batter_balls": faced.groupby([m, b.batter_id]).cumsum() - faced,
        "free_hit": (b.outcome == "noball").groupby(m).shift(1, fill_value=False).astype(int),
    }, index=b.index).astype(np.float32)


def build_frame(balls: pd.DataFrame, skill: SkillModel, bat_form: pd.DataFrame,
                bowl_form: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """`balls` must be annotated (bowler_type, batting_hand). `context`: index match_id."""
    b = balls[balls.innings_num == 1]
    ctx = pd.DataFrame(context.reindex(b.match_id.to_numpy())[CONTEXT].to_numpy(np.float32),
                       index=b.index, columns=CONTEXT)
    frame = pd.concat([skill_features(skill, b), form_features(b, bat_form, bowl_form),
                       ctx, situation(b)], axis=1)[feature_columns(skill.levels)]
    frame["y"] = b.outcome.cat.codes.astype(np.int8)
    frame["date"] = b.date
    frame["match_id"] = b.match_id
    return frame.reset_index(drop=True)


def importance_groups(levels: tuple[Level, ...]) -> dict[str, list[str]]:
    g = {lv.name: [f"{lv.name}__{c}" for c in CLASSES] for lv in levels}
    g["hint"] = [f"hint__{c}" for c in CLASSES]
    g.update({c: [c] for c in FORM + CONTEXT + SITUATION})
    g["all_skill"] = skill_columns(levels)
    return g
