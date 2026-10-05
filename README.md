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
