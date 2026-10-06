# src/sim/setup.py
"""Everything known at the toss, turned into a MatchSetup for the simulator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.asof import as_of
from src.features.context import CONTEXT
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
    context: pd.DataFrame


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
    crow = ctx.context.loc[row.match_id, CONTEXT].to_numpy(np.float32)
    cblock = pd.DataFrame(np.tile(crow, (len(pairs), 1)), index=pairs.index, columns=CONTEXT)
    static = pd.concat([skill_features(ctx.skill, pairs),
                        form_features(pairs, ctx.bat_form, ctx.bowl_form), cblock], axis=1)
    static = static[static_columns(ctx.skill.levels)].to_numpy(np.float32)
    return MatchSetup(batters, bowlers, w + WEIGHT_FLOOR,
                      static.reshape(len(batters), len(bowlers), -1))


_WORKER: dict = {}


def _single_thread(model) -> None:
    """One model per process: stop each process's model from using every core."""
    if getattr(model, "name", "") in ("lightgbm", "xgboost"):
        model.est.set_params(n_jobs=1)


def _dedupe(model) -> bool | None:
    """Logistic predicts faster than the dedupe costs; None leaves the module flag in charge."""
    return False if getattr(model, "name", "") == "logistic" else None


def _init_worker(model, table, cols, n_sims, legal_balls) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _single_thread(model)
    _WORKER.update(model=model, table=table, cols=cols, n_sims=n_sims, legal_balls=legal_balls)


def _run_one(item):
    i, setup = item
    w = _WORKER
    predict = lambda X: w["model"].predict(pd.DataFrame(X, columns=w["cols"]))  # noqa: E731
    return simulate(setup, predict, w["table"], w["n_sims"], w["legal_balls"], seed=i,
                    dedupe=_dedupe(w["model"]))


def simulate_setups(setups, model, table, cols, n_sims: int, legal_balls: int = 36,
                    workers: int = 1) -> list[np.ndarray]:
    """Simulate each setup; match i uses seed i, so any worker count gives the same output."""
    out: list[np.ndarray] = []
    if workers > 1:
        import threadpoolctl  # noqa: F401  # fail here, not as a BrokenProcessPool
    if workers <= 1:
        predict = lambda X: model.predict(pd.DataFrame(X, columns=cols))  # noqa: E731
        items = (simulate(s, predict, table, n_sims, legal_balls, seed=i,
                          dedupe=_dedupe(model))
                 for i, s in enumerate(setups))
        for i, r in enumerate(items):
            out.append(r)
            if (i + 1) % 50 == 0:
                print(f"  simulated {i + 1}/{len(setups)} matches", flush=True)
        return out
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(model, table, cols, n_sims, legal_balls)) as ex:
        for i, r in enumerate(ex.map(_run_one, enumerate(setups), chunksize=2)):
            out.append(r)
            if (i + 1) % 50 == 0:
                print(f"  simulated {i + 1}/{len(setups)} matches", flush=True)
    return out


def simulate_matches(rows: pd.DataFrame, ctx: Context, model, table: pd.DataFrame,
                     n_sims: int, legal_balls: int = 36, workers: int = 1) -> list[np.ndarray]:
    setups = [match_setup(row, ctx) for row in rows.itertuples()]
    return simulate_setups(setups, model, run_table(table), feature_columns(ctx.skill.levels),
                           n_sims, legal_balls, workers)
