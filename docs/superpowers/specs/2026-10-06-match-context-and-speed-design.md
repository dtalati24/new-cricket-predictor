# Match-context features, faster runs, walk-forward test — design

**2026-10-06.** Follows `2026-10-05-first-innings-6over-design.md`, whose
pipeline is built and scored once on the 2025+ holdout:

| model | calibrated MAE | market MAE | backtest (≈700 bets) |
|---|---|---|---|
| logistic (pre-registered) | 10.68 | 10.48 | +0.3% ROI |
| lightgbm | 10.49 | 10.48 | +1.3% ROI |

No model beats the market by the 0.85-run detectable gain. The raw simulator
runs low and the gap grows with time (−3.2 runs on 2024, −4.7 on 2025+), which
points at rising scoring the model cannot see. This design adds match-level
context the per-ball model lacks, makes runs 3–5× faster, and replaces the
single holdout with a walk-forward test over every year that has Betfair lines.

Excluded by user decision: Betfair match-odds prices as a feature.

## 1. New match-level features

Every feature here is one value per match (or per innings), attached to every
ball of the first innings in the training frame and to every row the simulator
builds. All history-based features are point-in-time: read with
`src.features.asof.as_of`, so a match on date D sees only dates strictly before D.

### 1.1 Toss

Known at the toss, no history needed.

- `chose_to_bat` — 1 if the batting-first team chose to bat (`toss_decision == "bat"`); 0 if it was put in.
  There is no separate "won the toss" column: the side batting first won the toss exactly when the
  winner chose to bat, so it would be identical (checked: 0 differences in 10,534 matches).

### 1.2 Scoring trend and competition

- `season` — match year as a number (e.g. 2025.4 = year + day-of-year / 366).
- `league_pp_recent` — mean first-innings 6-over runs over all men's T20
  matches in the 90 days before this match.
- **Competition shrinkage level.** A new top level in `LEVELS`:
  `Level("competition", ("competition_key",), None)`, shrunk toward the league
  like `venue`. `competition_key` is `event_name` with a small rename map for
  sponsor changes (`NatWest T20 Blast`, `Vitality Blast`, `Vitality Blast Men`
  → `T20 Blast`; extend the map for any rename found among the 40 most frequent
  event names) and `"unknown"` for the 62 matches with no event name.
- `comp_pp_recent` — EWM of first-innings 6-over runs over the competition's
  previous matches (halflife 20 matches).

### 1.3 Venue, team and opposition recent powerplay

"Powerplay runs" = runs in the first 36 legal balls of an innings. An innings
counts only if it reached 36 legal balls or was bowled out before (rain-cut
innings are excluded, as in `first_innings_settled`).

- `venue_pp_recent` — EWM over the venue's previous innings of either side
  batting (halflife 5 innings), and `venue_pp_n` — count of those innings
  (venue sample size; small counts mean the EWM is noisy).
- `team_bat_pp` — batting team's EWM of its own powerplay runs over its
  previous innings, either innings of the match (halflife 5).
- `opp_bat_pp` — the bowling team's EWM of its own batting powerplay runs
  (halflife 5). User hypothesis: a high-scoring opposition pushes the side
  batting first to attack.
- `opp_bowl_pp_conceded` — the bowling team's EWM of powerplay runs it has
  conceded (halflife 5). The direct measure of the attack this innings faces.

Teams are keyed by name as Cricsheet records it. Franchise renames start a new
history; acceptable, since the EWM forgets quickly.

### 1.4 Weather and time of day

- **Start time.** Betfair `match_odds_markets.market_time` (UTC) for the 8,873
  matches that have one; NaN otherwise (mostly pre-2015).
- `start_hour_local` — local hour of the scheduled start (from the venue's UTC
  offset), and `is_night` = 1 if `start_hour_local >= 17`. NaN without a start
  time.
- **Weather** at the venue, from Open-Meteo's historical archive (free, no key;
  user approved 2026-10-06): `temperature_2m`, `relative_humidity_2m`,
  `dew_point_2m`, `cloud_cover`, `wind_speed_10m`, `precipitation`. Values at
  the start hour when a start time exists, else the 14:00–20:00 local mean for
  the match date.

**Venue locations.** Each `venue_key` gets a city: the most common non-null
`city` across its matches (935 matches have no city). Cities are geocoded with
Open-Meteo's geocoding API. Results go to `data/raw/weather/venues.csv`
(`venue_key, city, latitude, longitude, timezone, source`). A hand-edited
`data/raw/weather/venue_overrides.csv` fixes failures and ambiguous names; it is
checked into git because it is small and hand-made. Venues still without
coordinates get NaN weather. Report the share of training balls with weather.

**Download.** One archive request per venue covering all its match dates
(hourly), cached as `data/raw/weather/{venue_slug}.parquet`; about 375 requests.
Up to 4 requests in flight at once (user asked to speed it up, 2026-10-07): at
~6 s each that is ~2,400/hour, still within the free tier (5,000/hour). Re-runs
reuse the cache.

## 2. Faster runs

Measured on the holdout: logistic ~3 h, LightGBM ~7.3 h, XGBoost ~6–8 h at
10,000 sims × 764 matches, with the worker process at ~2.5 of 8 cores and
0.7 GB RAM free.

1. **Free memory before simulating.** After the model is fitted, drop the
   training frame and ball table; keep only what match setup needs (skill
   model, history tables, XI table, run table).
2. **Parallel matches.** Simulate matches in a process pool (default 4
   workers, `--workers` flag). Each match keeps its own seed (`seed = row
   index`), so results are identical to a serial run regardless of worker count.
3. **Deduplicate predictions.** Within a step, rows that are exactly identical
   (same striker, bowler and situation) are predicted once (`np.unique(...,
   axis=0, return_inverse=True)`). Probabilities are unchanged, so results are
   bit-identical.
4. **Cache prepared data per cutoff.** `prepare(cutoff)` output (skill model,
   frame, history tables, run table) is saved under
   `data/model/cache/{cutoff}-{hash}/`, where the hash covers `LEVELS`, the
   feature column list and the source data files' modification times. A stale
   hash rebuilds; deleting the folder always rebuilds.

Target: a 3–5× faster end-to-end run.

## 3. Walk-forward test

Replaces the single 2025+ holdout as the verdict.

For each test year Y in 2023, 2024, 2025, 2026 (2026 is partial):

1. Fit each candidate model (logistic, LightGBM, XGBoost) on balls before
   year Y−1; simulate year Y−1's market matches (2,000 sims); fit each model's
   calibration on Y−1; choose the model by calibrated MAE on Y−1 (CRPS
   tie-break).
2. Refit the chosen model on balls before Y; simulate year Y's market matches
   (10,000 sims); apply its Y−1 calibration; score against the market.
3. Also score all three models on Y with their own Y−1 calibrations, as
   comparisons.

2022 is skipped as a test year: it would need 2021 as its calibration year, and
Betfair 6-over lines (needed to choose and calibrate) start in 2022.

Report per year and pooled across years: MAE vs market, CRPS, bias, coverage,
PIT, and the backtest (bets, hit rate, ROI, with a ±1 standard-error band on
ROI). Pooled test sample ≈ 1,650 matches.

**Honesty note.** The 2025+ results above were seen before these features were
chosen, so 2025–2026 are no longer untouched. The pooled number is reported
with and without 2025–2026; 2023–2024 are the cleaner evidence.

## 4. Testing

- Each history feature: point-in-time (a same-day value is invisible, an
  earlier one visible), and per-key reset.
- Toss flags on hand-built matches.
- Weather join: start-hour value used when a start time exists, daily window
  otherwise; NaN when a venue has no coordinates.
- Dedupe: predictions with and without dedupe are identical on a random
  batch with duplicated rows.
- Parallel: a 2-worker run of three tiny matches equals the serial run exactly.
- Cache: a changed feature list produces a different cache hash.

## 5. Order of work

1. Speed (section 2). It makes everything after it cheaper.
2. Toss, trend, competition, venue/team/opposition features (sections 1.1–1.3).
3. Weather and start time (section 1.4), including the download.
4. Walk-forward run and write-up (section 3).
