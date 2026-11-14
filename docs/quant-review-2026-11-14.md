# Quant review of the model, the results and the plan (2026-11-14)

Scope: read the code path from data to bet, re-analysed the saved walk-forward
results (2024–2026, 1,199 matches, powerplay-records run), and pulled the full
price history of every 6-over line market out of the raw Betfair files to see
what the "line" in the backtest really is.

Reproduce the headline table with:

    python -m src.quiet_markets --extract
    python -m src.quiet_markets data/reports/walkforward_gh_pp lightgbm

## 1. Bottom line

1. **The backtest profit is not a general edge on the 6-over line. All of it
   comes from quiet markets.** In markets whose line changed 6 times or fewer
   in the 30 minutes before the first ball (40% of matches), LightGBM shows
   +21% ± 5% ROI, positive in every year. In the other 60% (busy markets) it
   shows −2% ± 4%, and the model adds nothing to the line.
2. **Whether the quiet-market profit can actually be collected is unknown.**
   The backtest bets at the last traded line. Our data has no order book, so
   we cannot see what line and what amount were on offer. In a quiet market
   the last trade may be a price nobody could get again.
3. **In busy markets (IPL, World Cup, most internationals) the market is
   better than the model** by about 0.5 runs of average miss. More player
   statistics will not change that.
4. **So do not buy the £499 key to bet every match.** First run the untouched
   2023 test (started, section 7), then record real order books for free and
   see what is available in quiet markets (section 8).
5. **The realistic route to money is the one you named: providing prices
   (market making) in quiet line markets, across many markets per match.**
   That is where the data says the mispricing is, and it is the part big
   players ignore because each market is small.

## 2. What the "line" in the backtest is

- `pre_match_line` = the last traded line before the scheduled start (and
  before the first ball). It is a trade that happened, not a price on offer.
- Basic-tier data records a tick only when the last traded line changes, about
  once a minute at most. No volume, no back/lay quotes.
- Median 6-over market: 8 line changes in the last 30 minutes, 3 in the last
  10, and none at all in the last 2 minutes for half the markets.
- On these markets Betfair applies a 3-second bet delay once play starts and
  none before (from the market definitions).

## 3. Finding: the edge is only in quiet markets

"Changes" = number of times the last traded line changed in the 30 minutes
before the first ball. "Model weight" = how far to move from the line toward
the model's forecast to get the best forecast (0 = the line already contains
everything the model knows; 1 = the line adds nothing to the model).

LightGBM, 2024–2026, current bet rule:

| changes in last 30 min | matches | model weight | ROI | line MAE | model MAE |
|---|---|---|---|---|---|
| 0–3 | 149 | 1.14 ± 0.12 | +29% ± 8% | 11.07 | 9.42 |
| 4–6 | 325 | 0.57 ± 0.19 | +18% ± 6% | 10.38 | 10.16 |
| 7–9 | 350 | 0.23 ± 0.16 | +1% ± 6% | 10.32 | 10.66 |
| 10+ | 375 | −0.10 ± 0.14 | −5% ± 5% | 10.22 | 10.98 |

Quiet (0–6) against busy (7+), by model:

| model | quiet ROI (bets) | busy ROI (bets) | quiet weight | busy weight |
|---|---|---|---|---|
| LightGBM | +21.4% ± 4.6% (432) | −2.2% ± 3.8% (656) | 0.94 ± 0.11 | 0.04 ± 0.11 |
| XGBoost | +17.9% ± 4.6% (435) | −3.1% ± 3.8% (668) | 0.98 ± 0.11 | 0.08 ± 0.11 |
| logistic | +10.2% ± 4.6% (437) | −0.3% ± 3.8% (671) | 0.90 ± 0.11 | 0.10 ± 0.12 |

LightGBM by year: quiet +27%, +19%, +17% (2024, 2025, 2026); busy −4%, 0%, −2%.

Checks on this:

- **Not a few freak lines.** Dropping matches where model and line differ by
  more than 10 runs leaves quiet ROI at +21% (LightGBM).
- **Known before the bet.** Counting only changes up to 5 minutes before the
  ball gives the same split (+21% quiet, −4% busy).
- **Same at earlier bet times.** Betting at the line as it stood 5, 10 or 15
  minutes before the ball (simple rule: model at least 1.5 runs from the
  line): quiet +21% to +24%, busy −7% to −10%.
- **Quiet markets start late.** Their first trade is a median 22 minutes
  before the ball (busy: 4.4 hours). So these are post-toss prices, not stale
  pre-toss ones.
- **Where they are.** Share of matches that are quiet: Super Smash 74%, T20
  Blast 68%, Bangladesh PL 59%, Major League Cricket 54%, ILT20 44%, Lanka PL
  43%, PSL 40%, SA20 38%, CPL 38%, BBL 28%, IPL 15%, T20 World Cup 9%.
- **The market's own 20-over line disagrees with quiet 6-over lines, and is
  right.** Scale the 20-over line to a 6-over number and compare with the
  6-over line: in quiet markets the gap predicts the result (coefficient
  0.88 ± 0.21); in busy markets it does not (0.17 ± 0.20). So part of this
  "edge" needs no model at all: the quiet 6-over price is inconsistent with a
  better price a few clicks away. With both in the regression the model still
  matters more (model 0.59 ± 0.17, 20-over gap 0.45 ± 0.24).

Caveat: this split was found after the fact. I looked at about five ways of
cutting the data (staleness, total ticks, first-trade time, last-30-minute
activity, competition). The activity pattern is monotonic, has the same sign in
all three years and all three models, and the quiet/busy difference in model
weight is about 6 standard errors. It still needs a clean test: section 7.

## 4. Can the quiet-market profit be collected? Two readings

**Reading A: the price is simply poor and was on offer.** Few people trade
these markets, the line is set by a basic pricing bot or a casual bettor, and
someone with a decent model can take it. Evidence for: the prints do not
bounce (the previous print does not help predict the result, and averaging the
last three prints leaves the model weight at 1.06); prints 5–15 minutes
earlier show the same edge.

**Reading B: the last trade sits at the edge of a wide spread.** A market
maker offers, say, "under 50.5 or over 58.5" around a fair 54.5. A casual
bettor takes over 58.5. The backtest sees 58.5, says "under is great" and
books a win, but that was the market maker's side of the trade. A taker could
only get under 50.5. Evidence for: a weight of about 1 means the print holds
no information beyond the model, which is implausibly bad for a real two-sided
quote; unders make far more than overs in quiet markets (+34% on 180 bets
against +12% on 252), which fits casual money backing overs at lines that are
too high; the 20-over market "knows better" than the 6-over print.

Probably both are partly true. What it means either way:

- If A: there is a real taker edge, but small stakes only.
- If B: the profit belongs to whoever posts the prices. That is market making,
  and the backtest number is roughly what a maker earns per filled bet.
- Sensitivity (LightGBM, quiet markets): getting a line 1 run worse than the
  print still leaves +18%; 2 runs worse +12%; 3 runs worse +3% (all ± 4.6%).
  Across all matches, 1 run worse takes the whole +7% to +2.5%.

Only an order book answers this. Section 8, step 1.

## 5. Busy markets: the market beats the model

- Result = 0.98 × line + 0.09 × model (busy markets). The line is efficient
  against everything the model knows.
- The model misses by 10.83 runs on average, the line by 10.27.
- The model's forecasts are too bunched together: across busy matches the
  model's forecasts have a spread (sd) of 5.5 runs, the line's 6.3, and when
  the line is high the model is usually below it (correlation 0.53). The market
  knows match-specific things the model does not: the pitch on the day, team
  news and intent, late money.
- Betting busy markets with the current rule loses about 2% a bet (commission
  on coin flips).

## 6. Other pitfalls, ranked

1. **Test years are used up.** 2025–2026 have now been looked at through
   five or more model versions and through this review. Any further tuning on
   them is fitting noise. 2023 was never run and is the last clean year.
2. **The rule bets 92% of matches.** With a 51.3% threshold most bets are
   near coin flips. That is fine when the edge is real and harmful when it is
   not (busy markets).
3. **Choosing one model a year is noise.** LightGBM and XGBoost disagree with
   the line in almost the same way (correlation 0.97). Average them and stop
   choosing. Averaged: +7.9% overall, +21.6% quiet, −1.0% busy.
4. **Small money even if real.** About 180–200 quiet 6-over markets a year. At
   a true 10% ROI and £20 matched per bet that is about £380 a year; at £100 a
   bet, £1,900. The live key is £499. One market type cannot pay for the work.
5. **Live inputs are worse than backtest inputs.** Cricsheet lags 2–11 days,
   so recent form is missing; weather is a forecast; batting order is assumed
   from history; the per-ball model is only refitted once a year in the test.
6. **Settlement rules unconfirmed** (all out, rain, reduced overs). The
   backtest also drops rain-affected matches, which uses knowledge of the
   future; harmless only if Betfair voids those bets.
7. **Commission.** The backtest uses 5%. Betfair's "Basic" rewards plan is
   reported to charge 2% (UNCONFIRMED, check in the account settings). That
   is worth about 1.5 points of ROI.
8. **Bankroll.** At a 57% hit rate, 8–10 losses in a row will happen. Flat
   small stakes until there are 300 real bets, then at most a quarter of the
   Kelly stake, capped by what the book offers.

Checked and fine: no look-ahead in the simulator (batting order and powerplay
bowlers come from history before the match, not from the match itself);
features are point-in-time; lines that traded after the first ball are
excluded.

Tried in this review and rejected: re-centring the model on a rolling window
of the last 100–400 matches instead of the previous calendar year (no gain:
10.44–10.57 against 10.46–10.54); using an average of recent prints as the
line (no change).

## 7. Clean test on 2023 (running)

GitHub run 43607293748, started 2026-11-14: year 2023, all three models, 455
matches, calibrated on 2022. No model result for 2023 has ever been looked at.

Written before the result:

- **Prediction 1:** quiet-market ROI is above zero and above busy-market ROI
  (LightGBM and XGBoost).
- **Prediction 2:** model weight above 0.5 in quiet markets and below 0.3 in
  busy markets.
- **If either fails:** the quiet-market finding is a fluke of 2024–2026 and
  there is no demonstrated edge anywhere.
- Weak spot: the 2023 calibration is fitted on only 165 matches from 2022.

Score it with `python -m src.quiet_markets <results folder> lightgbm`.

**Result (scored 2026-11-14, `data/reports/walkforward_gh_2023/`): partly
confirmed, and smaller than 2024–2026 suggested.**

| model | quiet ROI (bets) | busy ROI (bets) | quiet weight | busy weight |
|---|---|---|---|---|
| LightGBM | +7.1% ± 6.7% (213) | −18.9% ± 6.8% (202) | 0.33 ± 0.17 | −0.23 ± 0.24 |
| XGBoost | +10.7% ± 6.6% (215) | −19.1% ± 6.7% (205) | 0.33 ± 0.17 | −0.28 ± 0.24 |
| logistic | +9.2% ± 6.6% (216) | −7.6% ± 7.1% (190) | 0.37 ± 0.17 | 0.27 ± 0.27 |

- **Prediction 1 held:** quiet ROI is above zero for all three models and far
  above busy ROI (a gap of 26–30 points for the boosted models, about 3
  standard errors). But quiet ROI by itself is only 1.1–1.6 standard errors
  above zero in 2023.
- **Prediction 2 half failed:** the busy weight is below 0.3 as predicted, but
  the quiet weight is 0.33, not above 0.5. By the rule written above, that is
  a fail.
- **Reading:** the quiet/busy split is real in direction (four years out of
  four, quiet beats busy). Its size was overstated by 2024–2026: the quiet
  line in 2023 held real information (MAE 9.15 against the model's 9.4) and
  the quiet edge was about a third to a half as big. Pooled over 2023–2026,
  LightGBM quiet ROI is about +17% ± 4% and busy about −6% ± 3%; treat the
  true quiet figure as nearer +7–10% at the last traded price, before any
  cost of getting a worse line.
- 2023 overall: the model lost to the market (MAE 9.64–10.0 against 9.47) and
  made +1% (logistic, chosen) to −6% (LightGBM). The boosted models ran 1.7–1.9
  runs low; the calibration came from only 165 matches of 2022.
- Section 4's question (can the quiet price actually be had) is now more
  important, not less: at +7–10%, a line 1–2 runs worse removes the edge.

## 8. Plan

Each step has a cost, a time and a stop rule.

**Step 1. Record real order books (free delayed key; about 1 day to build,
then 3–4 weeks of recording).** For every men's T20 with a line market, from
an hour before the start to the end of the first innings: the best three
lines on each side with the amounts, total matched, and each trade print, for
every line market on the event (6, 10, 15, 20 overs, both innings). This
answers, with real numbers: how wide the spread is in quiet markets, how much
money is there, where prints land relative to the quotes (Reading A or B), and
how much trades in total. It also starts the data set that in-play work needs,
which cannot be bought.
Faster alternative to check: one month of Betfair's Advanced historical data
(£49) if its files carry best back/lay lines for line markets. Look at a free
sample file first.

**Step 2. Live pricing script (about half a day).** XIs, toss, venue, forecast
in; fair line and the chance of over at any line out. Use the average of
LightGBM and XGBoost. Log every price with a timestamp.

**Step 3. Paper-trade two ways for the recording period.**
(a) Taker: bet at the line actually on offer, quiet markets only.
(b) Maker: post model ± k runs and count a fill when a later print trades
through it.

**Gate.** At least 100 quiet-market paper bets. Continue to real money (£499
key, £5–10 stakes, quiet markets only) if the taker ROI at real quotes is
above +5% or the maker fills are profitable. Otherwise stop pre-match 6-over
betting.

**Step 4. Breadth (about 1–2 weeks of work).** Extend the simulator from 6
overs to the full innings and to the second innings. One simulation then
prices every line market on the match consistently (6/10/15/20 overs, innings
runs, both innings). The quiet-market test can be repeated at once on the
20-over line (1,833 markets already in the data). Ten quiet markets a match at
small size is how this becomes worth doing.

**Step 5. In-play, at over breaks only.** Between overs there are 60–90
seconds, the next bowler is known, and nobody has a speed advantage worth
having. Restart the simulation from the live score and re-quote. Needs the
recorded data from step 1 to test first. Do not try to react ball by ball
against people with faster feeds.

Model changes worth making along the way (all cheap):

1. Average LightGBM and XGBoost; drop the yearly model choice.
2. Make the weight on the model depend on market activity (about 0 in busy
   markets, 0.6–1 in quiet ones), fitted on 2024–2025 only.
3. Feed the same event's 20-over line and match odds in as inputs at bet time.
4. Fix the bunching: the model's forecasts should spread as wide as the busy
   market's. Candidates: a stronger venue and pitch-recency effect, team-level
   strength, and letting the calibration stretch forecasts by competition.

Not worth more time: more player statistics, recency weighting, rolling
calibration, disagreement thresholds, and trying to beat the IPL closing line
before the match with a stats model.

## 9. Looking for an edge in busy markets (tests run 2026-11-14)

All tests use prices and results only (2022–2026, 1,819 matches) unless they
say "model" (2024–2026, LightGBM and XGBoost averaged). Where possible the
signal was found on 2022–2024 and checked on 2025–2026. This work was cut
short by a usage limit; the scripts are not in the repo yet.

**Tested and found nothing in busy markets (the line already has it):**

- Team form against the line (last 5 results minus line, batting and bowling
  side), venue form, tournament drift, line level.
- Line movement before the match (last 60, 36, 15, 5 minutes).
- Match odds (is the batting side the favourite), toss decision.
- The same event's 20-over line.
- In-play 6-over line: after 4, 8, 12, 16, 20 minutes the line has reacted
  fully to what has happened (no momentum, no over-reaction).
- The minutes after the toss: trades made 0–6 minutes after the toss are
  already efficient against the model (weight 0.2–0.3, ROI about 0). The big
  "edge" against older prints is stale or junk prints, not bettable.
- Chase powerplay line at the innings break: it has fully absorbed the first
  innings (total and powerplay); nothing left.
- Blind overs: 51.6% over 1,044 busy matches, about +0.7% ROI. Nothing.

**One lead: the 20-over line under-reacts to the powerplay.** Timed exactly
from the moment Betfair suspends the 6-over market:

- Truth: each run the powerplay beats its closing line by adds about 1.22
  runs to the final total. The 20-over line moves about 1.06.
- So the first 20-over price after the powerplay is short by about 0.16 runs
  per run of surprise (t = 3.5 on 1,423 matches; 0.12 in 2022–2024, 0.26 in
  2025–2026; 0.16 in busy 20-over markets).
- It fades: 0.14 three minutes later, 0.05 by 40 minutes, zero by 60.
- As a bet (follow the powerplay when it beat or missed the line by 10+
  runs, at the first print a minute after the powerplay): +7% ± 4% over 586
  bets, positive every year, but nearly all from overs after good powerplays
  (+13% ± 6%; unders −6%), and a line 1 run worse leaves +3%.
- Verdict: real in the regression, thin as a bet. It is the one place the
  data shows a busy market leaving something, and it needs no speed.

**What the busy market knows that the model does not:**

- **Team strength in internationals.** In busy internationals the model
  under-predicts favourites badly: a 70/30 favourite scores about 3 runs more
  than the model says (t = 4.9); the line has this right. In leagues (IPL and
  others) there is no such effect. Cause: player records are pooled across
  opposition of very different strength. A crude fix fitted on 2024 (add
  match odds) did not help on 2025–2026, so it needs doing properly inside
  the skill model, not as a patch.
- **Not mainly the batting order or bowlers.** Both assumed openers are right
  in 62% of matches; the model loses to busy lines by the same margin whether
  they are right or wrong. Knowing the powerplay bowlers helps a little.
- In quiet league markets the line over-rates the favourite (t = −1.9) and the
  model is right: part of where the quiet-market edge comes from.

**Conclusion for busy markets.** No public-data signal beats a busy closing
line before the match; I tested about fifteen. The realistic routes are:

1. In play, with the full-innings simulator started from the live score:
   first target is the 20-over line right after the powerplay (test it
   against the prints with the exact timing above before building anything
   live).
2. Fix the model's international weakness (opposition-strength adjustment),
   which stops it losing there but will not by itself beat the line.
3. Information the crowd lacks (pitch reading, earlier team news) or speed.
   Neither is available from our data.
