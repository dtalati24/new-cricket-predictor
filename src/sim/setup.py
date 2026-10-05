# src/sim/setup.py
"""Everything known at the toss, turned into a MatchSetup for the simulator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.asof import as_of
from src.model.frame import feature_columns, form_features, skill_features, static_columns
from src.sim.simulate import MatchSetup, run_table, simulate
from src.skill.shrink import SkillModel

# Every XI member gets this floor weight so the bowler rules can always be met;
# someone with no powerplay history is still picked almost never.
WEIGHT_FLOOR = 1e-6


@dataclass
class Context:
    skill: SkillModel
    styles: pd.DataFrame
    bat_form: pd.DataFrame
    bowl_form: pd.DataFrame
    bat_pos: pd.DataFrame
    pp: pd.DataFrame
    xi: pd.DataFrame


def check_xi(rows: pd.DataFrame, xi: pd.DataFrame) -> None:
    have = set(zip(xi.match_id, xi.team))
    bad = [r.match_id for r in rows.itertuples()
           if (r.match_id, r.batting_team) not in have or (r.match_id, r.bowling_team) not in have]
    if bad:
        raise ValueError(f"{len(bad)} matches lack a playing XI: {bad[:10]}")


def match_setup(row, ctx: Context) -> MatchSetup:
    xi = ctx.xi[ctx.xi.match_id == row.match_id]
    bat = xi[xi.team == row.batting_team].sort_values("position")
    bowl = xi[xi.team == row.bowling_team].sort_values("position")
    # Usual position, point-in-time; players with no history go last, in team-sheet order.
    pos = as_of(ctx.bat_pos, bat, ["player_id"], ["bat_pos"]).bat_pos
    bat = bat.assign(key=pos.fillna(100 + bat.position)).sort_values("key", kind="stable")
    w = as_of(ctx.pp, bowl, ["player_id"], ["pp_overs"]).pp_overs.fillna(0).to_numpy()
    batters, bowlers = bat.player_id.tolist(), bowl.player_id.tolist()
    s = ctx.styles.set_index("player_id")
    pairs = pd.DataFrame({"batter_id": np.repeat(batters, len(bowlers)),
                          "bowler_id": np.tile(bowlers, len(batters))})
    pairs["bowler_type"] = pairs.bowler_id.map(s.bowler_type).fillna("unknown")
    pairs["batting_hand"] = pairs.batter_id.map(s.batting_hand).fillna("unknown")
    pairs["venue_key"] = row.venue_key
    pairs["date"] = pd.Timestamp(row.date)
    static = pd.concat([skill_features(ctx.skill, pairs),
                        form_features(pairs, ctx.bat_form, ctx.bowl_form)], axis=1)
    static = static[static_columns(ctx.skill.levels)].to_numpy(np.float32)
    return MatchSetup(batters, bowlers, w + WEIGHT_FLOOR,
                      static.reshape(len(batters), len(bowlers), -1))


def simulate_matches(rows: pd.DataFrame, ctx: Context, model, table: pd.DataFrame,
                     n_sims: int, legal_balls: int = 36) -> list[np.ndarray]:
    cols = feature_columns(ctx.skill.levels)
    predict = lambda X: model.predict(pd.DataFrame(X, columns=cols))  # noqa: E731
    rt = run_table(table)
    out = []
    for i, row in enumerate(rows.itertuples()):
        out.append(simulate(match_setup(row, ctx), predict, rt, n_sims, legal_balls, seed=i))
        if (i + 1) % 50 == 0:
            print(f"  simulated {i + 1}/{len(rows)} matches", flush=True)
    return out
