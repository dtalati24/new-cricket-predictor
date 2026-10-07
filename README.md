# new-cricket-predictor

Fresh start on pricing Betfair T20 first-innings over/under lines (6-over and
20-over), betting after the toss and before the first ball.

Carried over from `cricket-predictor`: the data pipeline and its tables. No
model yet.

## What's here

```
src/ingest/   Cricsheet JSON -> data/parsed/*.parquet (22,537 matches, 11.4M balls)
src/betfair/  Betfair BASIC archive -> data/ticks/ (market manifest, prices)
src/market/   first-ball window, 6/10/20-over labels, market lines joined to outcomes
src/config.py all paths and data-quality constants
docs/gates.md measured numbers and the traps found in the data
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

Each test year is its own job on GitHub's 2-core runners. How long it takes
depends on which model wins that year (logistic is fastest, LightGBM slowest).
Do a first run with one year (`years: 2024`) to measure it, then choose
`n_sims_test` and `models` so each year fits the 6-hour job limit. Setting
`n_sims_test` to 5000 halves the simulation time and raises the per-match noise
from about 0.13 to about 0.19 runs.
Each year's results download as an artifact, and `walkforward-summary` holds the
pooled numbers. The free private-repo quota is 2,000 minutes a month, so that is
about one full run a month.

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
