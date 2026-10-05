# First-innings 6-over simulator — design

**2026-10-05.** Target: Betfair's `1_INNING_6_OVR_LINE`, men's T20, bet placed
after the toss and before the first ball. The 20-over line comes later from the
same simulation; it is out of scope until the 6-over line works.

## 1. The idea in one paragraph

Shrinkage turns each player's sparse record into a stable estimate of what they
do on a ball. Those estimates go into an ML model together with the match
situation and recent form, and the model outputs a probability for each of 11
ball outcomes. A Monte Carlo simulator draws balls from that model to play the
first 6 overs 10,000 times. The spread of simulated totals, after one
calibration correction, gives P(over the line), and we bet when that beats the
price after commission.

## 2. Scope of data

- **Men's T20 and IT20, 6-ball overs only.** 2.41M balls, 10,534 matches,
  2005-2026. Tests, ODIs, women's matches and the Hundred (5-ball overs) are
  excluded everywhere.
- **Skill counts** use both innings. A batter's identity is the same in a chase.
- **ML training rows** are first-innings balls only (1.28M), because chase
  behaviour differs and we only ever simulate a first innings.
- **Impact player (IPL 2023+).** 1,325 team-games list 12 players. We simulate
  with the XI named at the toss and ignore the substitute: the side batting
  first has already picked its batters by then.

## 3. Ball outcomes: 11 classes

`0, 1, 2, 3, 4, 6, 5plus, wicket, wide, noball, bye`

When one delivery qualifies for several classes, the first match in this order
wins:

1. `wide` (counts in this section are over the full corpus, all formats: 198,123)
2. `noball` (57,248)
3. `wicket` (real dismissals only; `retired hurt`, `retired not out` are not wickets)
4. `bye` (byes and leg byes, 142,171)
5. runs off the bat: `0, 1, 2, 3, 4, 6`, and `5plus` for 5/7/8 (1,522 balls)

The order keeps the legal-ball count right: a stumping off a wide is still not a
legal ball.

**Runs within a class are drawn from history,** from the training slice only:

- `wide`: extra runs (1 run 91%, 5 runs 4%), plus the rare stumping/run-out on a wide
- `noball`: runs off the bat on top of the 1 (0 66%, 1 18%, 4 9%, 6 2%), plus the rare run-out
- `bye`: 1, 2, 3 or 4 runs
- `wicket`: runs completed before a run-out (7,781 wicket balls carry bat runs)
- `5plus`: 5 almost always

`wide` and `noball` do not use up a legal ball. A `noball` makes the next
delivery a free hit, which is an input to the ML model.

Penalty runs (188 balls) are ignored.

## 4. Shrinkage layer

### What it does

For each player (and player-by-context group) it estimates the 11 outcome
probabilities, blending the group's own record with a parent:

```
estimate = (counts + k * parent) / (n + k)
```

`k` is how many balls of evidence the parent is worth. A group with few balls
lands near its parent; a group with thousands lands near its own record.

### Levels

```
league -> batter -> batter x bowler_type
league -> bowler -> bowler x batting_hand
league -> venue
batter x venue   (parent: batter)
bowler x venue   (parent: bowler)
```

`bowler_type` is pace/spin crossed with arm, from `bowling_style` in `player_meta.rda` (16,101 players; spellings vary, e.g. "Right arm Medium" and "Right-arm medium", so they are normalised first). Batter vs
an individual bowler is excluded: its groups have a median of ~5 balls, so
shrinkage returns the parent and the level adds nothing.

### Fitting `k`

One scalar `k` per level, fitted by maximising the Dirichlet-Multinomial
marginal likelihood of that level's counts given its parent (one-dimensional,
`scipy.optimize.minimize_scalar` over `log k`). Fitted on the training slice
only.

### Rules that must hold

- **Point in time.** Counts for match M use only balls from matches on dates
  strictly before M's date. Two matches on the same date never see each other.
  This has a regression test.
- **Counts are plain integers.** No recency weighting inside counts; it breaks
  the `k` fit (Build 1 pinned `k` at its bound on 8 of 10 levels this way).
  Recency lives in the form features.
- **A `k` at its search bound raises an error.** It never returns silently.
- **Density gate (report-only).** Each level's median group size, k and held-out log-loss gain versus its parent are measured and reported. By user decision (2026-10-05) no level is dropped on this basis; all levels are kept and judged later by feature importance. The first dev run measured: batter_x_bowlertype median 17 balls, gain +0.0038; bowler_x_battinghand gain −0.0002; batter_x_venue gain −0.0003; bowler_x_venue gain +0.00002.

## 5. ML model

### Inputs, one row per first-innings ball

- **Skill:** the shrunk 11-class vectors for batter, batter x bowler type,
  bowler, bowler x batting hand, venue, batter x venue, bowler x venue.
- **Combined hint:** batter rate x bowler rate / league rate, per class
  (renormalised). Saves the trees from learning the obvious interaction slowly.
- **Situation:** over, legal ball in over, wickets down, runs so far, balls the
  batter has faced this innings, free hit.
- **Form:** batter and bowler recent scoring rate and dismissal rate,
  exponentially weighted over recent matches, point-in-time.

### Output

11 probabilities summing to 1.

### Bake-off

LightGBM, XGBoost, and multinomial logistic regression (the floor). Ranked
first by per-ball log loss on the validation year, but **chosen** on the
simulated 6-over results in that year (MAE and CRPS), because per-ball and
innings-level accuracy can disagree.

### Feature importance

Permutation importance on the validation year: shuffle one input at a time,
measure the log-loss increase.

- Inputs that do not beat their shuffled copy are dropped, and the model is
  retrained to confirm nothing got worse.
- **Shrinkage check:** if shuffling all skill inputs together costs nothing,
  the shrinkage layer is not earning its place. That is reported, not hidden.

## 6. Simulator

Per match, at the toss:

1. **Batting order:** XI sorted by each player's average batting position over
   their recent matches (from the order they came in, point-in-time). Players
   with no history go to the bottom.
2. **Bowlers:** for each over, sample from the bowling XI weighted by each
   bowler's historical share of overs in that phase. Rules: max 4 overs each,
   no bowler in consecutive overs. Players who have never bowled get zero
   weight.
3. **Balls:** ask the model for the 11 probabilities, draw an outcome, draw its
   runs (section 3), update score, wickets, striker (odd runs and end of over
   swap ends), free hit.
4. Stop at 36 legal balls or 10 wickets. Record the total.

10,000 innings per match. More draws do not help: sampling error at 10,000 is
far below the model's own error.

## 7. Calibration

On the validation year, measure bias (mean simulated minus actual), p10-p90
coverage, and the PIT histogram. Fit one correction, `a + b * simulated` plus a
spread multiplier to hit 80% p10-p90 coverage. Apply it unchanged to the
holdout. Only escalate to isotonic recalibration if the PIT histogram is still
clearly not flat.

## 8. Time splits

| step | train on | evaluate on |
|---|---|---|
| model choice, feature pruning, calibration fit | up to 2023 | 2024 |
| final run, once | up to 2024 | 2025 onward |

The 2025+ slice is `src.market.baseline.comparison_set("holdout")`: men's,
usable markets, last price strictly before the first ball, 764 rows. It is
looked at once. `CALIB_END` in `src/config.py` assumed a different split and is
unused by this design.

## 9. Success

**The bar:** market MAE **10.48** on the 764 holdout rows. Beating it by less
than **0.85 runs** is indistinguishable from noise.

Reported on the holdout:

- MAE of the simulated median vs actual 6-over runs
- CRPS
- bias, p10-p90 coverage, PIT histogram
- **betting backtest:** line markets pay about evens, so with 5% commission bet
  over when P(over) > 51.3%, under when P(under) > 51.3%. Report bets, hit
  rate and profit. Line price is the last price before the first ball.

## 10. Code layout

```
src/features/outcomes.py   11-class labelling, extra-run tables
src/features/players.py    bowler type, batting hand, batting order, bowling shares
src/features/form.py       point-in-time recent form
src/skill/counts.py        point-in-time outcome counts
src/skill/shrink.py        k fit, shrinkage, the level chain
src/model/frame.py         builds the per-ball training table
src/model/ballmodel.py     the three models, log loss, permutation importance
src/sim/simulate.py        the Monte Carlo innings
src/sim/calibrate.py       the output correction
src/bet/backtest.py        P(over) vs line, commission, results
```

Each step writes its table under `data/` so later steps do not recompute it.

## 11. Testing

TDD on every module. The tests that matter most:

- point-in-time counts exclude the match itself and same-day matches
- the 11-class precedence (wicket on a wide is `wide`, etc.)
- shrinkage formula, and that a bound-pinned `k` raises
- the simulator never exceeds 36 legal balls or 10 wickets, respects the
  bowler rules, and reproduces a known total when given a deterministic model
- the backtest's commission threshold
