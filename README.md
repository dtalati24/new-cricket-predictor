# new-cricket-predictor

Pricing Betfair's men's T20 first-innings 6-over run line, betting after the
toss and before the first ball.

## The model

1. **Skill (empirical-Bayes shrinkage).** Each delivery is one of 11 outcomes
   (0, 1, 2, 3, 4, 6, 5+, wicket, wide, no-ball, bye). Outcome rates are shrunk
   toward a parent: batter, bowler, venue and competition toward the league;
   batter vs bowler type, bowler vs batting hand, batter at venue and bowler at
   venue toward the player's own rate (leave-own-out); batter and bowler in
   each phase (powerplay, middle, death) toward the player's own rate rescaled
   by the league's phase effect. Shrinkage strength is fitted by marginal
   likelihood, one value per level.
2. **Per-ball model.** Multinomial logistic regression, LightGBM or XGBoost
   predicts the 11 outcome probabilities from the skill estimates, a batter ×
   bowler hint, recent form (all overs and powerplay only), the match
   situation, and match context:
   - toss (did the side batting first choose to),
   - season, and league-wide powerplay scoring over the last 90 days,
   - recent powerplay scoring at the venue and in the competition,
   - the batting team's recent powerplay runs, and the opposition's recent
     powerplay runs scored and conceded,
   - weather at the venue (temperature, humidity, dew point, cloud, wind, rain)
     and the local start hour, from Open-Meteo and Betfair start times.
   Every history feature uses only matches before the one being priced.
3. **Monte Carlo.** 10,000 simulated first-innings powerplays per match, with
   extras, free hits, strike rotation and bowler rules (max 4 overs, no
   consecutive overs).
4. **Calibration.** One shift-and-stretch correction fitted on the previous
   year, then the bet rule: back over/under when the simulated probability
   beats evens after 5% commission (above 51.3%).

## What's here

```
src/ingest/, src/betfair/, src/market/   data pipeline: Cricsheet + Betfair -> parquet tables
src/features/   outcomes, point-in-time lookup, form, players, match context, weather
src/skill/      shrinkage counts and fits
src/model/      per-ball feature table, the three per-ball models
src/sim/        match setup, Monte Carlo, calibration and scoring
src/bet/        bet rule and backtest
src/run_6over.py    original dev run + single 2025+ holdout
src/walkforward.py  walk-forward test, one year per call
docs/           specs, plans, data notes (gates.md)
```

`data/` is gitignored (~5.7 GB) and lives only on the dev machine.

## The bar to beat

Market MAE on men's T20, 2025+ holdout: **10.48 runs at 6 overs, 26.96 at 20**.
`src.market.baseline.comparison_bar("holdout")` reproduces it.

## Results: 6-over holdout (2025+, 764 matches, scored once)

Market MAE 10.48 runs; beating it needs a gain of more than 0.85 runs.
Logistic (chosen in advance on 2024): MAE 10.68, backtest +0.3% ROI over 692 bets.
LightGBM (comparison): MAE 10.49, +1.3% ROI over 695 bets.
XGBoost (comparison): MAE 10.57, +0.0% ROI over 698 bets.
All three are calibrated (80% interval covers ~81% of outcomes, PIT flat) but
none beats the market; the ROIs are within one standard error (~±3.7%) of zero.
Details: `data/reports/6over_final_{model}.json`.

The holdout above used the first version of the model, without the match
context features (toss, trend, competition, venue/team/opposition powerplay
form, weather).

## Results: walk-forward 2024, with match context (2026-10-07)

First year of the walk-forward test, run on GitHub: models fitted on matches
before 2023, chosen and calibrated on 2023, refitted before 2024, then scored
on 435 unseen 2024 matches. Market MAE on those matches: 10.26 runs.

| model | role | MAE | bias | 80% range covers | bets | won | ROI |
|---|---|---|---|---|---|---|---|
| logistic | chosen on 2023 | 10.03 | +1.3 | 78% | 396 | 56.8% | +10.8% |
| LightGBM | comparison | 10.10 | +0.0 | 78% | 392 | 55.4% | +8.0% |
| XGBoost | comparison | 10.13 | +0.2 | 79% | 391 | 56.0% | +9.2% |

One standard error on ROI is about ±4.9%, so +10.8% is roughly two standard
errors above zero: suggestive, not proven. The 0.23-run MAE gain is below the
0.85 runs needed to be distinguishable from noise on accuracy alone.

**Compared with the first model on the same 435 matches.** The first model
scored 2024 during its development run: MAE 10.00 and +10.7% ROI over 405 bets,
but with its calibration fitted on 2024 itself, which flatters it. Before
calibration it missed by 10.61 runs and ran 3.25 runs low; the new logistic
model misses by 10.45 and runs 2.68 low. So the match-context features help
modestly, and the new 2024 number is the honest out-of-sample one. 2024 also
looks like a good year for this approach in general: the first model managed
only +0.3% ROI on 2025+, where the market was harder to beat (MAE 10.48).

**Caveats.** One year only; 2025 and 2026 are below, 2023 is not run yet.
Weather uses observed values at the start hour, while a bettor at the toss has
a forecast, which flatters the model slightly. The 2025–2026 results were seen
before the context features were designed, so 2023–2024 are the cleaner test.

Details: `data/reports/walkforward_gh/walkforward-2024/2024.json` and the
per-match parquet files beside it.

## Results: old inputs vs new inputs, 2025–2026 (2026-10-08)

A 1:1 test of whether the match-context features help. Each year was run twice
on GitHub with the same training data, cutoffs and simulation seeds: `full`
(126 inputs, with match context) and `base` (98 inputs: the first model's
skill, form and situation inputs, without context or the competition level).
Scored on the same 764 matches (442 in 2025, 322 in 2026). Market MAE 10.48.
Differences below are new minus old, paired match by match, ± one standard error.

| model | miss before calibration | miss after calibration | profit per match | ROI new | ROI old |
|---|---|---|---|---|---|
| logistic | −0.15 ± 0.06 | +0.02 ± 0.06 | +0.00 ± 0.02 | −0.4% | −0.5% |
| LightGBM | −0.25 ± 0.08 | +0.04 ± 0.09 | +0.03 ± 0.03 | +4.0% | +0.9% |
| XGBoost | −0.26 ± 0.08 | +0.05 ± 0.08 | +0.05 ± 0.03 | +7.2% | +1.3% |

ROI is over about 690 bets per arm, ± 3.7%. Both arms chose the same model each
year: logistic for 2025 (new −0.3%, old +1.1%) and LightGBM for 2026 (new
+3.7%, old −0.8%).

What it shows:

- **The new inputs make the raw simulator better.** It misses by 0.25 runs less
  for the boosted models, three standard errors, and the result repeats in both
  years. It also mostly fixes the simulator running low (boosted raw bias about
  −0.4 runs, against −2.3 with the old inputs in 2025).
- **Calibration erases the gain on accuracy.** Each calibration is learned on
  the previous year, when the model ran low, so it shifts the new version up
  too far. After calibration the two arms are tied, and neither beats the market
  on average miss (10.58–10.65 vs 10.48).
- **Betting leans toward the new inputs**, mainly for XGBoost (+0.05 per match
  staked, about 1.8 standard errors). Not proven, and picking the best of three
  models after the fact flatters it.

Removing the calibration shift (keeping the median as simulated) was checked
on 2024 and 2025 from the saved results: it helps the boosted models in 2025
(10.42 vs 10.50), hurts slightly in 2024 (10.16 vs 10.10), and logistic needs it
(raw bias about −4.7 runs). A simple "bet the side the uncalibrated median
points to" rule did no better than the current rule. So the calibration stays
as it is. An exact test of a spread-only calibration would need the raw
simulated scores, which the runs don't save.

Squared error tells the same story (2025–2026, 764 matches; 180 ≈ a 13.4-run
typical miss): market 180; LightGBM and XGBoost new 179 calibrated / 178 raw,
old 178 / 187; logistic new 182 / 203, old 182 / 209. The model-vs-market
difference has a standard error of about ±7, so all of these tie the market.

Details: `data/reports/walkforward_gh_ab/walkforward-{year}-{full,base}/`.

## The model knows something the market doesn't (2026-10-08)

Tying the market on average miss does not mean no edge: the model and the
market miss on different matches. When the model disagrees with the line, the
real score tends to move the model's way. The correlation between
(model − line) and (actual − line) is 0.18 in 2024, and 0.17–0.19 on the
unseen 2025–2026 matches (new-input models, calibrated median). On 764
matches that is about 5 standard errors from zero.

0.18 sounds small, but most of a 6-over score is luck (about 13 runs of
match-to-match noise), so even a perfect forecaster would have a low
correlation with the market's errors. For example, if the line were off from
the true expected score by 4 runs on a typical match, a perfect forecaster
would reach only about 0.3. A correlation of 0.18 is what produces the ~55%
hit rate the backtest shows.

Practical use: mixing the model with the line beats the line alone. A weight on
the model fitted on 2024 (0.52–0.62 depending on model) and applied unchanged
to 2025–2026 gives MAE 10.39–10.44 vs the market's 10.48, and squared error
174–175 vs 180. Not significant on MAE alone (±0.09), but it points the same
way as the correlation and the betting results.

## Tried and rejected: weighting recent matches (2026-10-08)

Idea: count recent training matches more (weight halving every 1, 2, 4 or 8
years) to fix the model running low as powerplay scoring rises. Checked with
`python -m src.quick_eval` (no simulation: fit on balls before year Y, score
Y's powerplay balls on log loss and expected runs per innings vs the realised
outcomes). Results in `data/reports/quick_eval_recency.json`.

- **LightGBM:** unweighted has the best log loss in both 2024 and 2025. In
  2024 weighting makes running low worse (−1.50 runs per innings unweighted,
  −1.98 with a 1-year halflife); in 2025 it moves the bias toward zero but
  predicts the balls worse.
- **Logistic:** a 4–8 year halflife improves log loss very slightly (1.6473 →
  1.6466 in 2025) and running low by 0.1–0.6 runs, but logistic stays far
  behind LightGBM (1.6326) and still runs about 4.4 runs low.

So recency weighting stays off (the `halflife` option remains, default none).

## Powerplay records (2026-10-08)

Every skill level used to pool all 20 overs. Added (spec:
`docs/superpowers/specs/2026-10-08-powerplay-records-design.md`):

- **Two shrinkage levels, batter × phase and bowler × phase** (phase = overs
  1–6, 7–15, 16–20). Each player's phase record is shrunk toward their overall
  record rescaled by the league's phase effect (league rates in that phase /
  league rates overall), so a player with no history gets the league powerplay
  average. The simulator always uses the powerplay version.
- **Powerplay form:** recent powerplay strike rate and dismissal rate (batter),
  economy and wicket rate (bowler), next to the all-overs form.

The `base` feature set (the first model's inputs) leaves these out too.

**Result (quick check, `data/reports/quick_eval_powerplay.json`),** before →
after on next-year powerplay balls:

| model | year | log loss | runs low per innings |
|---|---|---|---|
| LightGBM | 2024 | 1.61410 → 1.61411 | −1.50 → −1.53 |
| LightGBM | 2025 | 1.63255 → 1.63251 | −0.59 → −0.26 |
| logistic | 2024 | 1.62726 → 1.62030 | −3.70 → −1.29 |
| logistic | 2025 | 1.64717 → 1.64134 | −4.43 → −1.81 |

Logistic gains a lot: as a linear model it cannot combine "over number" with a
player's overall record by itself, so the powerplay-specific records give it
something new, and most of its running low goes away. LightGBM was already
learning that combination from the existing inputs, so it does not change. The
best per-ball model (LightGBM) is therefore no better, and logistic is still
behind it. The levels stay in.

## Checking the predicted spread (PIT)

The simulator outputs a spread of possible 6-over scores, not one number. PIT
(probability integral transform) checks whether that spread is honest:

1. For each match, simulate 10,000 possible 6-over scores.
2. When the real score is known, find the share of simulated scores below it
   (e.g. 7,000 of 10,000 below a real 58 gives 70%). That share is the match's PIT.
3. Repeat for every match.
4. If the spread is honest, these shares are evenly spread from 0% to 100%.
   A pile-up near 0% and 100% means the spread is too narrow (overconfident);
   a pile-up near 100% alone means the model runs low.

On the 764-match 2025+ holdout (logistic model, after calibration) the counts
per 10% bucket were 71, 66, 77, 77, 75, 81, 87, 81, 81, 68: close to flat
(Kolmogorov-Smirnov p = 0.48), and the 80% range held the real score 82% of
the time.

## Walk-forward on GitHub

The full walk-forward test is slow, so it can run on GitHub instead of this PC.
It needs the processed tables, which are licensed and so never go into git.

1. `python -m src.pack_data` packs them into `data_bundle.tar.gz` (about 80 MB).
2. Upload it once as a release on this private repo:
   `gh release create data-v1 data_bundle.tar.gz --notes "Processed tables (licensed; private)"`
3. On GitHub, open Actions, pick "walk-forward", press Run workflow. The
   workflow only appears there after this branch is merged to `main`.

Each test year is its own job on GitHub's 2-core runners. The 2024 run took
3.5 hours with all three models at 10,000 sims, about an hour of it installing
packages on Python 3.14; the workflow now uses Python 3.13, so expect about
2.5 hours per year. Setting `n_sims_test` to 5000 halves the simulation time
and raises the per-match noise from about 0.13 to about 0.19 runs.
Each year's results download as an artifact, and `walkforward-summary` holds the
pooled numbers. The free private-repo quota is 2,000 minutes a month, so that is
about one full run a month.

To compare the first model's inputs against the new match-context inputs on the
same matches, set `feature_sets` to `full,base`. Each year then runs twice (same
training data and seeds); the `base` results land in a `base/` subfolder and get
their own summary.

## Rebuilding the tables

```bash
pip install -r requirements.txt
python -m src.ingest.download
python -m src.ingest.parse
python -m src.betfair.manifest
python -m src.betfair.prices
python -m src.market.join
python -m src.market.settle
python -m src.market.baseline
python -m pytest
```
