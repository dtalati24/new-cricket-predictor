# Powerplay records — design

**2026-10-08.** User-approved idea: add batter and bowler powerplay records on
top of their all-overs records, shrunk toward a league powerplay average.

Why: every skill level today pools all 20 overs. The 6-over market is all
powerplay, where fielding restrictions change the job: some openers and some
new-ball bowlers are much better (or worse) in overs 1–6 than overall.

## 1. Phase

`phase` on every ball, from the 0-based Cricsheet `over`: `pp` (overs 0–5),
`middle` (6–14), `death` (15+). Added in `load_balls`. The simulator only plays
the powerplay, so every simulated ball has `phase = "pp"`.

## 2. Two new shrinkage levels

    Level("batter_x_phase", ("batter_id", "phase"), "batter", by_phase=True)
    Level("bowler_x_phase", ("bowler_id", "phase"), "bowler", by_phase=True)

Each gives all 11 outcome rates in that phase (so average, strike rate,
boundary and dot rates in one vector).

**What they shrink toward (user request: include the league average).** The
parent is the player's leave-own-out all-overs estimate, rescaled to the phase
by the league's phase effect:

    parent = normalise( player_overall_loo * league_phase / league_all )

where `league_phase` is the league's 11 outcome rates in that phase and
`league_all` its all-overs rates, both from the training slice. So:

- a player with lots of powerplay balls: mostly their own powerplay record;
- a player with few powerplay balls but a long career: their overall record,
  adjusted to how everyone scores in the powerplay;
- a player with no history at all: the league powerplay average (their overall
  estimate is already the league vector, times the phase effect).

`SkillModel` gains `phase_league: dict[str, np.ndarray]` (phase -> 11 rates).
`fit` and `estimates` apply the same rescaling, so k is fitted against the same
kind of parent it is used with. `level_report` is unchanged.

## 3. Powerplay form

Next to the all-overs form, EWM (same halflife, 10 matches) over the player's
previous matches, counting only powerplay balls:

- batter: `bat_pp_sr` (runs per ball), `bat_pp_out` (dismissals per ball);
- bowler: `bowl_pp_econ` (runs conceded per legal ball), `bowl_pp_wkt`
  (bowler-credited wickets per legal ball).

Same definitions as the existing `batter_form` / `bowler_form`, on the
powerplay subset; added to `FORM`. The powerplay columns live in the same form
tables: outer-merged on (player, date) and forward-filled within each player,
so a match with no powerplay balls carries the previous powerplay state.

## 4. Wiring

- `src/features/outcomes.py`: `phase` column.
- `src/skill/shrink.py`: `Level.by_phase`, `SkillModel.phase_league`, the
  rescaled parent, two new `LEVELS` entries.
- `src/features/form.py`: powerplay form tables, merged into the existing form
  tables so `form_features` reads them in the same lookup.
- `src/model/frame.py`: `FORM` gains the four columns.
- `src/sim/setup.py`: `pairs["phase"] = "pp"`.
- The A/B `base` feature set stays the first model's inputs: it also drops the
  two new levels and the powerplay form.

## 5. Testing

- Rescaled parent: a player with no history gets exactly the league powerplay
  rates; a player with only non-powerplay history gets their overall estimate
  times the phase effect, renormalised.
- k for the new levels is fitted (not at its bound) on a synthetic set where
  players differ by phase.
- Powerplay form counts only overs 0–5 and is point-in-time.
- Setup builds `phase = "pp"` and the full static layout.

Then `python -m src.quick_eval --years 2024,2025 --models lightgbm,logistic
--halflives none` against the current model's numbers. A GitHub walk-forward
run only if it helps.
