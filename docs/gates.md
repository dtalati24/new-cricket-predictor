# Gate evidence, Phase 1

The spec says no phase starts until the previous gate passes with a logged
number. This is where those numbers live, in git. Before this file they were in
commit messages, in gitignored `data/reports/*.csv` and in a session ledger.

Everything under "Results" was measured and is settled. Everything under "Open
issues" is unresolved and is recorded so Phase 2 does not walk into it. Nothing
in the second group has been decided.

Scope: Phase 1 Tasks 1-4 (ingest, post-toss window, checkpoint labels, Betfair
layer and the reproduced bar). Nothing here is a claim about a model or about
profitability. No model exists.

Source files are named in each section. The CSVs are gitignored and
regenerable; the numbers below are copied from them.

## Results

### Gate 0: the betting window exists

Gate 0 asks whether there is a tradeable post-toss, pre-first-ball window at
all. The kill threshold was fewer than 500 markets with a first ball. There are
8,978 of 9,218 MATCH_ODDS markets that ever go in-play, so Gate 0 passes by a
wide margin. It was measured twice.

**MATCH_ODDS proxy** (`data/reports/gate0_window_sweep.csv`). Window is the
half-open interval `[first_ball - m, first_ball)`.

| window | markets with >=1 tick | markets with >=2 ticks | median ticks (markets with any) |
|---|---|---|---|
| 15 min | 8,834 | 8,764 | 17 |
| 30 min | 8,874 | 8,833 | 35 |
| 45 min | 8,890 | 8,858 | 46 |
| 60 min | 8,900 | 8,873 | 53 |
| 120 min | 8,912 | 8,888 | 74 |

This is an **upper bound** on over-line liquidity, not a measurement of it.
MATCH_ODDS is the most liquid market on any event, and the over-line markets
did not exist in the tables yet when this was run. The proxy exists because the
plan's Task 2 had no over-line data to count.

**Over-line markets** (`data/reports/gate0_overline_window_sweep.csv`). Counts
usable markets whose last pre-match price falls inside `[first_ball - m,
first_ball)`, with first ball taken from the event's MATCH_ODDS market. This is
the measurement the plan never asked for and the one that actually matters.

| line | usable | 15 min | 30 min | 45 min | 120 min | last priced after first ball |
|---|---|---|---|---|---|---|
| 6 over | 2,506 | 2,335 | 2,393 | 2,399 | 2,401 | 94 |
| 20 over | 2,532 | 2,404 | 2,422 | 2,425 | 2,426 | 101 |
| 10 over (not a target) | 815 | 691 | 750 | 754 | 771 | 25 |

So 95.5% of the usable 6-over set and 95.7% of the 20-over set carry a price
inside the 30-minute window, and about 93% inside 15 minutes. Counts saturate by
45 minutes: going from 45 to 120 adds 2 markets at 6 over. The last pre-match
trade clusters close to the first ball. The 10-over line appears in the sweep
because the code reports every over-line type; it is not a target.

### Gate 4 (the bar): the market's own forecast error

`data/reports/gate4_market_baseline.csv`. The name is Build 2's gate numbering;
in the current spec this is part of Gate 0's "reproduces the bar" requirement.
Usable rows only: rain-affected, no-result and truncated matches excluded, at
least 2 pre-match ticks, line inside the plausible band. Every usable row is a
T20 with 6-ball overs, men's and women's together. The pooling is an
unresolved design question, see Open issue 4.

| line | n | market MAE | mean-only MAE | R² | bias | detectable gain |
|---|---|---|---|---|---|---|
| 6 over | 2,506 | 9.4904 | 10.8419 | 0.2086 | +0.20 | 0.4279 |
| 20 over | 2,532 | 24.3863 | 29.0073 | 0.2761 | +1.16 | 1.1169 |
| 10 over (not a target) | 815 | 14.3969 | 15.6188 | 0.1046 | +1.14 | 1.2236 |

This is **digit-for-digit identical to Build 2's** `gate4_market_baseline.csv`
on all three rows and all eleven columns. The regenerated `market_manifest`,
`event_match`, `matches` and `deliveries` tables are byte-identical to the
archive's; the price tables and `overline_comparison` are `DataFrame.equals` to
it. The port reproduces Build 2, it does not approximate it.

Other counts the spec names, now pinned by tests: 22,537 parsed matches,
11,422,319 deliveries, 22,537 rows in `first_innings_settled` of which 367 are
the Hundred (`match_type == "T20"` with 5-ball overs), and 7,853 rows in
`overline_comparison`.

The 15-over line has no row because it is excluded by `market_reliable = 0`
(its market R² is -0.58), not by a filter in `baseline_table()`.

The detectable gain is the smallest MAE improvement distinguishable from noise
at the sample size (alpha 0.05, power 0.80). A model that beats the market by
less than that has not shown anything.

### The first-ball leak

About 4% of usable rows carry a "pre-match" price stamped at or after the first
ball. `prices.py` cuts at the scheduled `marketTime`, and play often starts
earlier. Measured by joining each over-line market's event to its MATCH_ODDS
first-ball time (all 5,038 usable target rows joined, none missing):

| line | usable | late rows | share |
|---|---|---|---|
| 6 over | 2,506 | 94 | 3.8% |
| 20 over | 2,532 | 101 | 4.0% |

Market MAE with and without those rows:

| line | all rows | leak-free rows | change |
|---|---|---|---|
| 6 over | 9.4904 (n 2,506) | 9.5269 (n 2,412) | +0.0365 |
| 20 over | 24.3863 (n 2,532) | 24.4017 (n 2,431) | +0.0154 |

The leak flatters the market: the late rows are ones where the market had
already seen live play. So removing it makes the bar slightly easier, not
harder. The change is 8.5% of the 6-over detectable gain and 1.4% of the
20-over one, in the direction that makes the project look harder than it is.

The bar stands as published, because reproducing Build 2's number is what the
regression test is for. **Phase 2 must enforce `pre_match_ts < first_ball` in
its own comparison set.** Scoring a model against a price that had seen live
play is not a like-for-like comparison, however small the effect.

### Era dependence

The pooled market MAE rises over time. Market MAE by calendar year of the
match, usable rows, men's and women's pooled:

| year | 6 over n | 6 over MAE | 20 over n | 20 over MAE |
|---|---|---|---|---|
| 2022 | 266 | 8.05 | 278 | 21.61 |
| 2023 | 630 | 9.19 | 645 | 24.70 |
| 2024 | 598 | 9.35 | 607 | 22.95 |
| 2025 | 565 | 10.09 | 555 | 25.31 |
| 2026 | 447 | 10.20 | 447 | 26.46 |

The spec's holdout is time-ordered, train to 2024-12-31 and test 2025 onward
(`TRAIN_END`, `CALIB_END` in `src/config.py`). Pooled against that holdout:

| set | line | n | market MAE | mean-only MAE | R² | bias | detectable gain |
|---|---|---|---|---|---|---|---|
| pooled 2022-2026 | 6 over | 2,506 | 9.4904 | 10.8419 | 0.2086 | +0.1955 | 0.4279 |
| pooled 2022-2026 | 20 over | 2,532 | 24.3863 | 29.0073 | 0.2761 | +1.1623 | 1.1169 |
| holdout 2025+ | 6 over | 1,012 | 10.1403 | 11.3314 | 0.1683 | +0.0919 | 0.7171 |
| holdout 2025+ | 20 over | 1,002 | 25.8224 | 29.0520 | 0.1905 | +1.4341 | 1.8269 |

Two things move at once. The pooled market MAE is higher on the holdout (10.14
against 9.49 at 6 over, 25.82 against 24.39 at 20 over) and its R² drops, so it
is easier to beat. But n falls to about 40% of the pooled set, so the detectable gain grows
by 68% (0.4279 to 0.7171, and 1.1169 to 1.8269).

Net at 6 over: a model has to reach roughly 10.14 - 0.72 = 9.42 on the holdout,
against 9.49 - 0.43 = 9.06 pooled. The holdout target is easier in absolute
terms and needs a bigger measured gain to be believed. The pooled row is kept
as the port-fidelity checksum. It is not the number a model is scored against.

**Why the pooled MAE rises: two effects, separated.** The pooled table above
mixes men's and women's matches, and the mix changes. Women's matches are
easier for the market in absolute terms (see Open issue 4), and their share of
usable 6-over and 20-over rows falls after 2022. A falling women's share pushes
the pooled MAE up even if neither market changed.

| year | women's | men's | women's share |
|---|---|---|---|
| 2022 | 190 | 354 | 34.9% |
| 2023 | 328 | 947 | 25.7% |
| 2024 | 299 | 906 | 24.8% |
| 2025 | 205 | 915 | 18.3% |
| 2026 | 208 | 686 | 23.3% |

The share is not monotonic: it recovers in 2026. The 2025+ holdout is 20.5%
women's (413 of 2,014 rows) against 24.4% for the pooled set (1,230 of 5,038).
Splitting market MAE by gender within each year removes the composition effect:

| line | set | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| 6 over | men's MAE | 8.36 | 9.48 | 10.13 | 10.39 | 10.56 |
| 6 over | women's MAE | 7.46 | 8.35 | 6.95 | 8.77 | 9.06 |
| 20 over | men's MAE | 22.18 | 25.36 | 24.24 | 26.11 | 28.20 |
| 20 over | women's MAE | 20.57 | 22.81 | 19.11 | 21.60 | 20.59 |

What this shows:

- **The men's market is deteriorating, and that is not a composition
  artefact.** At 6 over its MAE rises every year, 8.36 to 10.56 (+26%). At 20
  over it rises from 22.18 to 28.20 (+27%), but not every year: 2024 dips to
  24.24 from 25.36 in 2023.
- **The women's market is roughly flat.** At 20 over it ends where it started
  (20.57 in 2022, 20.59 in 2026) with swings in between. At 6 over it drifts up
  noisily, 7.46 to 9.06, with a dip to 6.95 in 2024. Neither line shows a clean
  trend.
- The falling women's share amplifies the pooled trend, so the pooled figure
  slightly overstates how much the market as a whole has worsened. The
  within-gender men's trend is real and is the dominant effect. The finding
  that the pooled MAE rises stands, with the cause now attributed: mostly the
  men's market getting worse, plus a composition shift on top.

What to do about the two populations having different trends is not decided,
see Open issue 4.

## The live bar: what a model is actually scored against

`baseline.comparison_bar(split)` reports the market's error on
`baseline.comparison_set(split)` -- men's only, strictly pre-first-ball, and
`usable`. This is the number Gate 5 should use. It is **not**
`baseline_table()`, which pools genders, both eras and the leaked rows and
exists only to prove the port reproduces Build 2.

| line | split | n | market MAE | mean-only MAE | R² | bias | detectable gain |
|---|---|---|---|---|---|---|---|
| 6 over | train (<2025) | 1,055 | 9.6147 | — | 0.120 | +0.10 | 0.6605 |
| 6 over | **holdout (2025+)** | **764** | **10.4804** | 11.2467 | 0.114 | −0.27 | **0.8482** |
| 6 over | all | 1,819 | 9.9783 | 10.8410 | 0.138 | −0.06 | 0.5237 |
| 20 over | train (<2025) | 1,071 | 24.4589 | — | 0.199 | +0.57 | 1.7708 |
| 20 over | **holdout (2025+)** | **762** | **26.9567** | 29.1601 | 0.145 | +1.52 | **2.0957** |
| 20 over | all | 1,833 | 25.4973 | 28.3341 | 0.187 | +0.97 | 1.3546 |

Train and holdout partition `all` exactly (1,055 + 764 = 1,819), and the split
is by date, never at random.

**The holdout slice is close to the limit of what it can resolve.** At 6 over
the market beats predicting the subset mean by 0.77 runs (10.4804 against
11.2467) while the detectable gain is 0.8482 — so on men's 2025+ 6-over lines
the market's own edge over a constant is *below* the detection threshold, and
R² is 0.114 against 0.21 pooled. The 20-over slice is only marginally better:
a 2.20-run edge against a 2.0957 threshold. Model selection therefore has to
happen on the train split or the pooled set, with the holdout kept for a single
final confirmation. Tuning against a 764-row holdout would be fitting noise.

## Open issues for Phase 2

These are unresolved. They are listed so they get decided, not because anyone
has decided them.

1. **Spec Gate 5 compares unlike things.** It says beat 9.49 by 0.43 "on the
   time-ordered holdout", but 9.49 and 0.43 are pooled 2022-2026 figures,
   measured partly on years a model trains on. The live holdout figures are in
   the table above: 10.4804 and 0.8482 at 6 over. Gate 5 should name those, and
   should also specify a **paired** test on the same rows, which is more
   powerful than comparing two separately measured MAEs — `_detectable_improvement`
   already documents its sd as a conservative upper bound for exactly that test.
   Changing a gate is the user's decision and nothing here has changed it.

2. **`clean_mask` drops rows it did not need to, and the obvious fix is worse
   than the problem.** It excludes `rain_affected` and `no_result` matches
   wholesale. Around 64 rows at 6 over and 68 at 20 over are matches where the
   first innings ran its full scheduled length and the line would have settled
   normally, with the rain landing on the chase. Including them moves the
   6-over bar from 9.4904 to 9.4821, about 2% of the detectable gain. So the
   comparison set is conditioned on second-innings outcomes.

   **Measured warning.** The natural fix — replace the wholesale exclusion with
   a per-checkpoint completeness rule, `legal_balls_at_Nov >= n_overs *
   balls_per_over` — is wrong, and wrong in the dangerous direction. It selects
   against all-out innings, which end before the checkpoint's ball count but
   settle perfectly well. Measured, it swings the 20-over market bias from
   **+1.27 to −4.64 runs**: the surviving subset is one where teams outscored
   the line, which is a manufactured edge, not a correction. Any real fix has to
   treat an innings that ended because the side was dismissed as complete (which
   is what `clean_mask`'s own `all_out` term does), and must be checked by
   re-measuring the bias. `clean_mask` is ported code and stays untouched;
   `comparison_set` deliberately does not attempt this.

## Decisions taken after the Phase 1 review

3. **The comparison set is now defined once, in code.** `baseline.comparison_set(split)`
   applies three filters that are deliberately *not* parameters, because each is
   something a caller would otherwise be free to forget: `gender ==
   MODELLING_GENDER`, `pre_match_ts < first_ball_ts`, and `usable`. `split` is
   the only knob ("train", "holdout", "all"), an unknown value raises rather
   than silently returning everything, and seven tests pin the behaviour
   including that train and holdout partition exactly. Per-checkpoint
   completeness is **not** included, for the reason in item 2.

4. **Men's T20 only.** `config.MODELLING_GENDER = "male"`. The pooling question
   below is settled: women's matches are excluded rather than segmented or
   carried as a feature.

   Of 5,853 usable rows across all line types, 4,514 were men's and 1,339
   women's (24%). The two populations differ materially:

   | line | set | n | market MAE | mean first-innings runs |
   |---|---|---|---|---|
   | 6 over | women's | 612 | 8.08 | 40.03 |
   | 6 over | men's | 1,894 | 9.95 | 49.14 |
   | 20 over | women's | 618 | 20.98 | 146.60 |
   | 20 over | men's | 1,914 | 25.49 | 171.10 |

   In relative terms the market is equally sharp on both — MAE is about 20% of
   mean runs either way — so this was never a claim that the market misprices
   women's cricket. The reasons to exclude were that the absolute MAEs are not
   comparable across the two sets, that the pooled 9.4904 is a blend sitting
   between the subgroup figures, and that the *trends* diverge (see Era
   dependence): the men's market has worsened by about a quarter since 2022
   while the women's has been flat, so pooling conflates a deteriorating market
   with a stable one.

   **What it costs.** The 2025+ holdout falls from 1,012 rows to 764, which
   raises the 6-over detectable gain from 0.7171 to 0.8482 and the 20-over from
   1.8269 to 2.0957. That is a real loss of statistical power, taken knowingly
   in exchange for scoring against one population instead of a blend.

   **Still open within this decision:** whether the skill layer (Tasks 5-9)
   should also train on men's deliveries only. The comparison set is men's, but
   a shrinkage hierarchy fitted across both genders would pool player skill
   across populations with different scoring levels. Nothing has decided that.

## Rulings made during execution

Short, so a future reader does not relitigate them from the code.

**`checkpoint_runs` and `build_settlement` coexist on purpose.** The plan said
to extract `build_settlement`'s checkpoint logic into `checkpoint_runs`. They
are not equivalent. `build_settlement` sums `runs_total` where `over < n_overs`,
indexing by over, which works for any `balls_per_over`. `checkpoint_runs` takes
the first N legal balls and is called with 36 for 6 overs, which hard-codes
6-ball overs. They agree everywhere except the 367 Hundred matches, where 6
overs is 30 legal balls. The spec says to segment the Hundred and never pool
it, and the plan's "extract it" cannot be done without mislabelling those
matches. So `checkpoint_runs` was added with its own tests, `build_settlement`
was left alone, and a test pins that the two diverge on 5-ball overs. The
regenerated settlement matches Build 2's on all 22,537 matches and on the 367
Hundred matches separately.

**Do not filter on `legal_balls_at_Nov >= 36`.** It would silently drop all 367
Hundred matches. Any completeness threshold has to come from `scheduled_overs`
and `balls_per_over`. There is a warning at `checkpoint_runs` in
`src/market/settle.py`.

**The single-tick test filters on `usable` first.** The plan's
`test_single_tick_markets_are_filtered` asserted that no single-tick market
appears in the comparison table. Build 2 keeps those rows and flags them
(`enough_ticks`, `usable`), and the 2,506 and 2,532 the bar is built on are
exactly the `usable` counts. So the invariant is that none reach the
measurement, not that none exist. The test checks `usable` rows. Dropping the
rows instead would have meant editing ported `baseline.py` logic.

**`baseline_table()` has no 15-over filter.** The 15-over line already has
`market_reliable = 0`, hence `usable = 0`, on all 621 of its rows, so
`summarize()` emits no 15-over row. An explicit filter would hide where the
exclusion happens. `test_15_over_line_is_excluded` checks the flag works and
fails loudly if that changes.

**The bar test is pinned tightly.** `n` is exact, MAE is within 0.01 and
detectable gain within 0.001. The first version allowed 0.5 on MAE, which is
bigger than the whole 0.43 effect being detected. A legitimate archive refresh
should fail this test and force a deliberate re-measurement.

**Pipeline order differs from the plan's Task 4.** The plan lists
`src.betfair.match_to_cricsheet` (a library module with no `__main__`) and omits
`src.market.join`, which is what writes `event_match.parquet`. The working order
is in the README.

**`src/betfair/odds.py` was not ported.** Nothing in Phase 1 imports it. It is
890 lines whose only consumer in Build 2 is its own test. Phase 2 can copy it
from the archive.

**Live tables, not archive ones.** `src/config.py` names Build 2's tables with a
`B2_` prefix. Anything that feeds the pipeline reads the live equivalents under
`data/`. `src/betfair/ticks.py` used to read three archive tables; it now reads
the live ones.

**`pythonpath = ["."]`, not `["src"]`.** With `src/` on the path, the archive's
bare `from config import ...` resolves and silently loads a second copy of the
module. With only the repo root on the path it raises `ImportError`.
