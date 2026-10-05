# First-innings 6-over simulator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict the distribution of first-innings runs at 6 overs for men's T20 matches, from the toss, and measure it against Betfair's 6-over line.

**Architecture:** Shrinkage turns sparse player records into stable 11-class outcome estimates; those feed a per-ball classifier together with match situation and form; a vectorised Monte Carlo simulator plays 6 overs thousands of times; one calibration correction is fitted on 2024 and applied to the 2025+ holdout, which is scored once.

**Tech Stack:** Python 3.14, pandas 3, numpy 2, scipy (`minimize_scalar`, `gammaln`, `kstest`), scikit-learn, LightGBM, XGBoost, pyreadr, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-first-innings-6over-design.md`

## Global Constraints

- Data scope: `match_type in {"T20","IT20"}`, `gender == "male"`, `balls_per_over == 6`, `innings_num in {1,2}`, not super overs.
- 11 classes, this exact order: `["0","1","2","3","4","6","5plus","wicket","wide","noball","bye"]`.
- Precedence: wide > noball > wicket > bye > runs off the bat. `retired hurt` and `retired not out` are not wickets.
- Point in time: anything used for a match on date D comes only from dates strictly before D. All history goes through `src.features.asof.as_of`.
- Shrinkage counts are plain integers. No recency weighting inside counts.
- A fitted `k` at its search bound raises `ConcentrationAtBoundError`. Never returns silently.
- Splits: `DEV_END = "2024-01-01"`, `TRAIN_END = "2025-01-01"`. Dev trains before 2024 and evaluates on 2024. Final trains before 2025 and scores 2025+ once.
- The holdout is `src.market.baseline.comparison_set("holdout")` filtered to `market_type == "1_INNING_6_OVR_LINE"` (764 rows). Bar: market MAE 10.48, detectable gain 0.85.
- Commission 5%, so the bet threshold is `1 / 1.95 = 0.5128`.
- Metrics: MAE, CRPS, bias, p10–p90 coverage, PIT (KS p-value). Never accuracy.
- Run every command from the repo root `C:\Users\dtala\new-cricket-predictor`. Tests that need built data skip when it is missing.

## File map

```
src/config.py                 + DEV_END (Task 1)
src/features/outcomes.py      classes, labelling, outcome table, log loss, load_balls   (Task 1)
src/features/asof.py          strictly-before lookup                                    (Task 2)
src/features/form.py          EWM helper, batter and bowler form                        (Task 2)
src/features/players.py       styles, batting order, powerplay share, XI loader         (Task 2)
src/skill/counts.py           totals and cumulative counts                              (Task 3)
src/skill/shrink.py           k fit, shrink, levels, SkillModel, level report           (Task 3)
src/model/frame.py            feature columns, situation, per-ball frame                (Task 4)
src/model/ballmodel.py        three models, permutation importance                      (Task 5)
src/sim/simulate.py           bowler plan, run table, the Monte Carlo innings           (Task 6)
src/sim/calibrate.py          CRPS, PIT, coverage, calibration                          (Task 7)
src/bet/backtest.py           bet rule and P&L                                          (Task 7)
src/sim/setup.py              per-match setup from the XI, simulate many matches        (Task 8)
src/run_6over.py              dev and final drivers                                     (Tasks 8, 9)
```

---

### Task 1: Setup, ball loading and the 11 outcome classes

**Files:**
- Modify: `src/config.py` (the split block)
- Create: `src/features/__init__.py`, `src/features/outcomes.py`
- Test: `tests/test_outcomes.py`

**Interfaces:**
- Produces: `CLASSES: list[str]`, `NOT_LEGAL: tuple[str, str]`, `is_wicket(d) -> pd.Series[bool]`, `label_outcomes(d) -> pd.Series` (categorical over `CLASSES`), `outcome_table(d) -> pd.DataFrame[outcome, runs, wicket, n, p]`, `log_loss(probs: np.ndarray, codes) -> float`, `venue_key(venue, city) -> pd.Series`, `load_balls() -> pd.DataFrame` (adds `batting_team, bowling_team, date, venue, city, venue_key, outcome`; sorted in delivery order).
- `config.DEV_END = "2024-01-01"`.

- [ ] **Step 1: Install XGBoost**

Run: `pip install "xgboost>=2.0"`
Expected: installs (it is already in `requirements.txt`). Check: `python -c "import xgboost; print(xgboost.__version__)"`.

- [ ] **Step 2: Replace the split block in `src/config.py`**

Replace this exact block:

```python
# --- global modelling constants --------------------------------------------
# Temporal splits. THREE slices, not two: a calibration layer fitted on the
# test slice is simply cheating, and the previous build had no calibration
# layer to misplace. Boundaries are provisional until Gate 3 reports the real
# year distribution of matched markets.
# Set from the real Gate 3/4 distribution of usable over-line matches:
#   2022 282 | 2023 670 | 2024 629 | 2025 596 | 2026 470  (total 2,647)
# Clean calendar-year boundaries give train 1,581 (59.7%) / calib 596 (22.5%)
# / test 470 (17.8%) -- close to a 60/20/20 split without a mid-year cut.
TRAIN_END = "2025-01-01"
CALIB_END = "2026-01-01"
# test slice = date >= CALIB_END
```

with:

```python
# --- time splits (spec 2026-10-05 section 8) --------------------------------
# Development: train before DEV_END; choose the model, prune features and fit
# calibration on DEV_END..TRAIN_END (2024). Final: retrain before TRAIN_END and
# score the 2025+ holdout once. market.baseline also splits on TRAIN_END.
DEV_END = "2024-01-01"
TRAIN_END = "2025-01-01"
```

Then run `python -m pytest -q` — expected: 45 passed (nothing used `CALIB_END`).

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_outcomes.py
import numpy as np
import pandas as pd
import pytest

from src.config import PARSED
from src.features.outcomes import (
    CLASSES, label_outcomes, log_loss, outcome_table, venue_key)


def _balls(**cols):
    n = len(next(iter(cols.values())))
    base = {"runs_batter": [0] * n, "runs_total": [0] * n,
            "wide": [np.nan] * n, "noball": [np.nan] * n,
            "bye": [np.nan] * n, "legbye": [np.nan] * n,
            "wicket_kind": [None] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_eleven_classes_in_fixed_order():
    assert CLASSES == ["0", "1", "2", "3", "4", "6", "5plus",
                       "wicket", "wide", "noball", "bye"]


def test_stumping_off_a_wide_is_a_wide():
    d = _balls(wide=[1.0], runs_total=[1], wicket_kind=["stumped"])
    assert label_outcomes(d).iloc[0] == "wide"


def test_run_out_off_a_noball_is_a_noball():
    d = _balls(noball=[1.0], runs_total=[1], wicket_kind=["run out"])
    assert label_outcomes(d).iloc[0] == "noball"


def test_wicket_beats_runs_off_the_bat():
    d = _balls(runs_batter=[1], runs_total=[1], wicket_kind=["run out"])
    assert label_outcomes(d).iloc[0] == "wicket"


def test_retired_hurt_is_not_a_wicket():
    d = _balls(wicket_kind=["retired hurt"])
    assert label_outcomes(d).iloc[0] == "0"


def test_leg_bye_is_bye_not_a_run():
    d = _balls(legbye=[1.0], runs_total=[1])
    assert label_outcomes(d).iloc[0] == "bye"


def test_odd_bat_runs_go_to_5plus_but_six_stays_six():
    d = _balls(runs_batter=[5, 7, 6], runs_total=[5, 7, 6])
    assert label_outcomes(d).tolist() == ["5plus", "5plus", "6"]


def test_outcome_table_probabilities_sum_to_one_per_class():
    d = _balls(wide=[1.0, 5.0, 1.0, np.nan], runs_total=[1, 5, 1, 4],
               runs_batter=[0, 0, 0, 4])
    t = outcome_table(d)
    assert np.allclose(t.groupby("outcome", observed=True).p.sum(), 1.0)
    w = t[t.outcome == "wide"].set_index("runs").p
    assert w[1] == pytest.approx(2 / 3)
    assert w[5] == pytest.approx(1 / 3)


def test_stumping_off_a_wide_keeps_its_wicket_in_the_table():
    d = _balls(wide=[1.0], runs_total=[1], wicket_kind=["stumped"])
    t = outcome_table(d)
    assert t.outcome.tolist() == ["wide"] and t.wicket.tolist() == [1]


def test_venue_key_drops_city_suffix_but_splits_ambiguous_names():
    v = pd.Series(["Wankhede Stadium, Mumbai", "Wankhede Stadium",
                   "County Ground", "County Ground"])
    c = pd.Series(["Mumbai", None, "Taunton", "Hove"])
    k = venue_key(v, c).tolist()
    assert k[0] == k[1] == "Wankhede Stadium"
    assert k[2] != k[3]


def test_log_loss_of_uniform_is_log_11():
    p = np.full((3, 11), 1 / 11)
    assert log_loss(p, [0, 5, 10]) == pytest.approx(np.log(11))


def test_log_loss_clips_zero_probability():
    p = np.zeros((1, 11)); p[0, 1] = 1.0
    assert np.isfinite(log_loss(p, [0]))


@pytest.mark.skipif(not (PARSED / "deliveries.parquet").exists(),
                    reason="parsed tables not built")
def test_load_balls_is_mens_t20_innings_one_and_two():
    from src.features.outcomes import load_balls
    d = load_balls()
    assert 2_400_000 < len(d) < 2_420_000
    assert set(d.innings_num.unique()) == {1, 2}
    assert d.date.is_monotonic_increasing
    assert d.outcome.notna().all()
```

- [ ] **Step 4: Run to see it fail**

Run: `python -m pytest tests/test_outcomes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 5: Implement**

`src/features/__init__.py`: empty file.

```python
# src/features/outcomes.py
"""Ball outcomes: the 11 classes the model predicts, and what each adds to the score.

Precedence when one delivery fits several classes: wide, noball, wicket, bye,
then runs off the bat. Legality decides first so the legal-ball count stays
right: a stumping off a wide is not a legal ball, so it is `wide`, and its
wicket is carried in `outcome_table`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PARSED

CLASSES = ["0", "1", "2", "3", "4", "6", "5plus", "wicket", "wide", "noball", "bye"]
NOT_LEGAL = ("wide", "noball")
NOT_OUT_KINDS = ("retired hurt", "retired not out")
# Ground names shared by different venues (measured: "County Ground" spans 8
# cities). These get the city appended; every other ground is keyed by name
# alone, because city spellings drift (Bangalore/Bengaluru).
AMBIGUOUS_VENUES = ("County Ground", "Gymkhana Club Ground", "National Stadium", "Queen's Park")


def venue_key(venue: pd.Series, city: pd.Series) -> pd.Series:
    name = venue.str.split(",").str[0].str.strip()
    return name.where(~name.isin(AMBIGUOUS_VENUES),
                      name + "|" + city.fillna("").astype(str))


def is_wicket(d: pd.DataFrame) -> pd.Series:
    return d.wicket_kind.notna() & ~d.wicket_kind.isin(NOT_OUT_KINDS)


def label_outcomes(d: pd.DataFrame) -> pd.Series:
    bat = d.runs_batter.astype(int)
    out = np.select(
        [d.wide.notna().to_numpy(), d.noball.notna().to_numpy(), is_wicket(d).to_numpy(),
         ((d.bye.fillna(0) + d.legbye.fillna(0)) > 0).to_numpy(),
         bat.isin([5, 7, 8]).to_numpy()],
        ["wide", "noball", "wicket", "bye", "5plus"],
        default=bat.astype(str).to_numpy(dtype=object),
    )
    return pd.Series(pd.Categorical(out, categories=CLASSES), index=d.index)


def outcome_table(d: pd.DataFrame) -> pd.DataFrame:
    """P(runs, wicket | class). `runs` is everything the delivery adds to the total.

    Pass the training slice only.
    """
    t = pd.DataFrame({"outcome": label_outcomes(d),
                      "runs": d.runs_total.astype(int).to_numpy(),
                      "wicket": is_wicket(d).astype(int).to_numpy()})
    g = t.groupby(["outcome", "runs", "wicket"], observed=True).size().rename("n").reset_index()
    g["p"] = g.n / g.groupby("outcome", observed=True).n.transform("sum")
    return g


def log_loss(probs: np.ndarray, codes) -> float:
    """Mean negative log probability of the realised class, clipped at 1e-15."""
    codes = np.asarray(codes, dtype=int)
    p = np.clip(np.asarray(probs)[np.arange(len(codes)), codes], 1e-15, 1.0)
    return float(-np.log(p).mean())


def load_balls() -> pd.DataFrame:
    """Men's T20/IT20, 6-ball overs, innings 1-2, in delivery order, labelled."""
    m = pd.read_parquet(PARSED / "matches.parquet")
    m = m[m.match_type.isin(["T20", "IT20"]) & (m.gender == "male") & (m.balls_per_over == 6)]
    inn = pd.read_parquet(PARSED / "innings.parquet")
    inn = inn[inn.innings_num.isin([1, 2]) & ~inn.is_super_over]
    d = pd.read_parquet(PARSED / "deliveries.parquet")
    d = d[d.match_id.isin(set(m.match_id))]
    d = d.merge(inn[["match_id", "innings_num", "batting_team", "bowling_team"]],
                on=["match_id", "innings_num"])
    d = d.merge(m[["match_id", "date", "venue", "city"]], on="match_id")
    d["date"] = pd.to_datetime(d["date"])
    d["venue_key"] = venue_key(d["venue"], d["city"])
    d["outcome"] = label_outcomes(d)
    d = d.sort_values(["date", "match_id", "innings_num", "over", "ball_seq"], kind="stable")
    return d.reset_index(drop=True)
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_outcomes.py -q`
Expected: PASS, 13 tests.

- [ ] **Step 7: Commit**

```bash
git add src/config.py src/features tests/test_outcomes.py
git commit -m "feat: 11-class ball outcomes, outcome run table, men's T20 ball loader"
```

---

### Task 2: Point-in-time lookup, form and player features

**Files:**
- Create: `src/features/asof.py`, `src/features/form.py`, `src/features/players.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `is_wicket`, `NOT_LEGAL` from Task 1; `PARSED`, `PLAYER_META_RDA` from `src.config`.
- Produces:
  - `as_of(table, query, by: list[str], cols: list[str]) -> pd.DataFrame` (index = `query.index`, NaN where nothing earlier)
  - `HALFLIFE = 10`; `post_match_ewm(per, key: str, cols: list[str]) -> pd.DataFrame`
  - `batter_form(balls) -> DataFrame[player_id, date, bat_form_sr, bat_form_out]`
  - `bowler_form(balls) -> DataFrame[player_id, date, bowl_form_econ, bowl_form_wkt]`
  - `bowler_type(style) -> str` (one of `pace_right, pace_left, spin_right, spin_left, unknown`), `batting_hand(style) -> str` (`right, left, unknown`)
  - `player_styles() -> DataFrame[player_id, bowler_type, batting_hand]`
  - `annotate(balls, styles) -> balls` with `bowler_type`, `batting_hand` added
  - `batting_positions(balls) -> DataFrame[match_id, innings_num, date, player_id, bat_pos]`
  - `batting_order_history(balls) -> DataFrame[player_id, date, bat_pos]`
  - `load_xi(balls) -> DataFrame[match_id, date, team, player_id, position]`
  - `powerplay_history(balls, xi) -> DataFrame[player_id, date, pp_overs]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py
import numpy as np
import pandas as pd
import pytest

from src.features.asof import as_of
from src.features.form import HALFLIFE, batter_form, post_match_ewm
from src.features.outcomes import label_outcomes
from src.features.players import (
    batting_hand, batting_positions, bowler_type, powerplay_history)

D = pd.to_datetime


def test_as_of_is_strictly_before():
    t = pd.DataFrame({"player_id": ["a", "a"], "date": D(["2020-01-01", "2020-02-01"]),
                      "x": [1.0, 2.0]})
    q = pd.DataFrame({"player_id": ["a", "a", "a", "b"],
                      "date": D(["2020-01-01", "2020-01-15", "2020-02-01", "2020-03-01"])})
    out = as_of(t, q, ["player_id"], ["x"])
    assert np.isnan(out.x.iloc[0]), "same-day state must not be visible"
    assert out.x.iloc[1] == 1.0
    assert out.x.iloc[2] == 1.0, "Feb 1 state is not visible on Feb 1"
    assert np.isnan(out.x.iloc[3]), "unknown key"
    assert list(out.index) == list(q.index)


def test_as_of_keeps_query_order_when_query_is_unsorted():
    t = pd.DataFrame({"player_id": ["a", "a"], "date": D(["2020-01-01", "2020-02-01"]),
                      "x": [1.0, 2.0]})
    q = pd.DataFrame({"player_id": ["a", "a"], "date": D(["2020-03-01", "2020-01-15"])},
                     index=[7, 3])
    out = as_of(t, q, ["player_id"], ["x"])
    assert out.loc[7, "x"] == 2.0 and out.loc[3, "x"] == 1.0


def test_post_match_ewm_includes_the_row_and_decays():
    per = pd.DataFrame({"player_id": ["p", "p"], "date": D(["2020-01-01", "2020-02-01"]),
                        "v": [2.0, 0.0]})
    out = post_match_ewm(per, "player_id", ["v"])
    w = 0.5 ** (1 / HALFLIFE)
    assert out.v.tolist() == pytest.approx([2.0, 2 * w / (1 + w)])


def test_batter_form_ignores_wides_as_balls_faced():
    b = pd.DataFrame({
        "batter_id": ["p"] * 3, "date": D(["2020-01-01"] * 3),
        "runs_batter": [4, 0, 2], "runs_total": [4, 1, 2],
        "wide": [np.nan, 1.0, np.nan], "noball": [np.nan] * 3,
        "bye": [np.nan] * 3, "legbye": [np.nan] * 3,
        "wicket_kind": [None, None, "caught"], "player_dismissed_id": [None, None, "p"]})
    b["outcome"] = label_outcomes(b)
    f = batter_form(b).iloc[0]
    assert f.bat_form_sr == pytest.approx(6 / 2), "two balls faced, the wide is not one"
    assert f.bat_form_out == pytest.approx(1 / 2)


@pytest.mark.parametrize("style,expected", [
    ("Right arm Fast medium", "pace_right"),
    ("Right-arm medium-fast", "pace_right"),
    ("Left arm Fast", "pace_left"),
    ("Right arm Offbreak", "spin_right"),
    ("Legbreak Googly", "spin_right"),
    ("Slow Left arm Orthodox", "spin_left"),
    ("Left arm Wrist spin", "spin_left"),
    ("Right arm Slow", "spin_right"),
    ("Right arm Slow medium", "pace_right"),
    ("Right arm Medium, Right arm Offbreak", "pace_right"),
    ("Right arm Bowler", "unknown"),
    ("(Unknown Arm) Slow", "unknown"),
    (None, "unknown"),
    (float("nan"), "unknown"),
])
def test_bowler_type(style, expected):
    assert bowler_type(style) == expected


@pytest.mark.parametrize("style,expected", [
    ("Right hand Bat", "right"), ("Left-hand bat", "left"),
    ("Left-arm medium", "unknown"), (None, "unknown")])
def test_batting_hand(style, expected):
    assert batting_hand(style) == expected


def test_batting_positions_follow_first_appearance():
    b = pd.DataFrame({"match_id": ["m"] * 4, "innings_num": [1] * 4,
                      "date": D(["2020-01-01"] * 4),
                      "batter_id": ["A", "A", "C", "B"],
                      "non_striker_id": ["B", "B", "B", "C"]})
    p = batting_positions(b).set_index("player_id").bat_pos
    assert p.to_dict() == {"A": 1, "B": 2, "C": 3}


def test_powerplay_history_counts_appearances_without_bowling_as_zero():
    balls = pd.DataFrame({"match_id": ["m1"] * 3, "over": [0, 0, 2],
                          "bowler_id": ["p", "p", "p"]})
    xi = pd.DataFrame({"match_id": ["m1", "m2"], "player_id": ["p", "p"],
                       "date": D(["2020-01-01", "2020-02-01"])})
    h = powerplay_history(balls, xi).set_index("date").pp_overs
    w = 0.5 ** (1 / HALFLIFE)
    assert h.iloc[0] == pytest.approx(2.0)
    assert h.iloc[1] == pytest.approx(2 * w / (1 + w))
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.features.asof'`.

- [ ] **Step 3: Implement**

```python
# src/features/asof.py
"""Point-in-time lookup: what was known about a key strictly before a date.

Every history table here (skill counts, form, batting position, bowling share)
holds one row per (key, date) with the state AFTER that date's matches.
`as_of` reads, for each query row, the latest state from a date strictly
before the query's date. Same-day matches never see each other: Cricsheet has
no start times, so a double-header cannot be ordered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def as_of(table: pd.DataFrame, query: pd.DataFrame, by: list[str],
          cols: list[str]) -> pd.DataFrame:
    q = query[by + ["date"]].copy()
    q["_row"] = np.arange(len(q))
    t = table[by + ["date"] + cols].copy()
    for f in (q, t):               # merge_asof needs identical key dtypes
        f["date"] = f["date"].astype("datetime64[ns]")
        for c in by:
            f[c] = f[c].astype(str)
    out = pd.merge_asof(q.sort_values("date", kind="stable"),
                        t.sort_values("date", kind="stable"),
                        on="date", by=by, allow_exact_matches=False)
    out = out.sort_values("_row").reset_index(drop=True)
    out.index = query.index
    return out[cols]
```

```python
# src/features/form.py
"""Recent form as exponentially weighted rates over a player's matches.

Separate from the skill counts on purpose: counts are plain integers for the
shrinkage; recency lives here. Each table holds the state AFTER each date and
is read point-in-time with `as_of`.
"""
from __future__ import annotations

import pandas as pd

from src.features.outcomes import NOT_LEGAL, is_wicket

HALFLIFE = 10  # matches
# Dismissals not credited to the bowler.
NOT_BOWLER_WICKETS = ("run out", "retired hurt", "retired not out", "retired out",
                      "obstructing the field")


def post_match_ewm(per: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """`per` has one row per (key, date). EWM over that key's dates, row included."""
    per = per.sort_values([key, "date"], kind="stable").reset_index(drop=True)
    ew = per.groupby(key)[cols].ewm(halflife=HALFLIFE).mean().reset_index(level=0, drop=True)
    per[cols] = ew.sort_index()
    return per


def batter_form(balls: pd.DataFrame) -> pd.DataFrame:
    faced = balls[balls.outcome != "wide"]
    per = (faced.groupby(["batter_id", "date"])
           .agg(runs=("runs_batter", "sum"), balls=("runs_batter", "size"))
           .reset_index().rename(columns={"batter_id": "player_id"}))
    outs = (balls[is_wicket(balls)].groupby(["player_dismissed_id", "date"]).size()
            .rename("outs").reset_index().rename(columns={"player_dismissed_id": "player_id"}))
    per = per.merge(outs, on=["player_id", "date"], how="left").fillna({"outs": 0})
    per = post_match_ewm(per, "player_id", ["runs", "balls", "outs"])
    per["bat_form_sr"] = per.runs / per.balls
    per["bat_form_out"] = per.outs / per.balls
    return per[["player_id", "date", "bat_form_sr", "bat_form_out"]]


def bowler_form(balls: pd.DataFrame) -> pd.DataFrame:
    b = balls.assign(
        conceded=balls.runs_total - balls.bye.fillna(0) - balls.legbye.fillna(0)
        - balls.penalty.fillna(0),
        legal=~balls.outcome.isin(NOT_LEGAL),
        wkt=balls.wicket_kind.notna() & ~balls.wicket_kind.isin(NOT_BOWLER_WICKETS),
    )
    per = (b.groupby(["bowler_id", "date"])
           .agg(conceded=("conceded", "sum"), legal=("legal", "sum"), wkts=("wkt", "sum"))
           .reset_index().rename(columns={"bowler_id": "player_id"}))
    per = post_match_ewm(per, "player_id", ["conceded", "legal", "wkts"])
    legal = per.legal.where(per.legal > 0)
    per["bowl_form_econ"] = per.conceded / legal
    per["bowl_form_wkt"] = per.wkts / legal
    return per[["player_id", "date", "bowl_form_econ", "bowl_form_wkt"]]
```

```python
# src/features/players.py
"""Player attributes known at the toss: style, usual batting position, and how
much each bowler bowls in the powerplay. Histories are post-date states read
point-in-time with `as_of`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.config import PARSED, PLAYER_META_RDA
from src.features.form import post_match_ewm

_SPIN_WORDS = ("offbreak", "legbreak", "orthodox", "wrist")


def bowler_type(style) -> str:
    """pace/spin x arm from Cricinfo's free-text style. First listed style wins."""
    if not isinstance(style, str):
        return "unknown"
    s = style.split(",")[0].strip().lower().replace("-", " ")
    if "left" in s:
        arm = "left"
    elif "right" in s or s.startswith("legbreak"):
        arm = "right"
    else:
        return "unknown"
    if any(w in s for w in _SPIN_WORDS) or s.endswith("slow"):
        return f"spin_{arm}"
    if "medium" in s or "fast" in s:
        return f"pace_{arm}"
    return "unknown"


def batting_hand(style) -> str:
    if not isinstance(style, str) or "bat" not in style.lower():
        return "unknown"
    s = style.lower()
    return "left" if s.startswith("left") else "right" if s.startswith("right") else "unknown"


def player_styles() -> pd.DataFrame:
    import pyreadr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # pyreadr warns casting the dob column
        meta = next(iter(pyreadr.read_r(str(PLAYER_META_RDA)).values()))
    meta = meta.dropna(subset=["cricsheet_id"]).drop_duplicates("cricsheet_id")
    return pd.DataFrame({
        "player_id": meta.cricsheet_id.astype(str).to_numpy(),
        "bowler_type": [bowler_type(s) for s in meta.bowling_style],
        "batting_hand": [batting_hand(s) for s in meta.batting_style],
    })


def annotate(balls: pd.DataFrame, styles: pd.DataFrame) -> pd.DataFrame:
    s = styles.set_index("player_id")
    return balls.assign(
        bowler_type=balls.bowler_id.map(s.bowler_type).fillna("unknown"),
        batting_hand=balls.batter_id.map(s.batting_hand).fillna("unknown"),
    )


def batting_positions(balls: pd.DataFrame) -> pd.DataFrame:
    """Order of first appearance (striker before non-striker on the same ball)."""
    seq = np.arange(len(balls), dtype=float)
    cols = ["match_id", "innings_num", "date"]
    a = balls[cols].assign(player_id=balls.batter_id.to_numpy(), seq=seq)
    b = balls[cols].assign(player_id=balls.non_striker_id.to_numpy(), seq=seq + 0.5)
    first = (pd.concat([a, b]).sort_values("seq", kind="stable")
             .drop_duplicates(["match_id", "innings_num", "player_id"]))
    first["bat_pos"] = first.groupby(["match_id", "innings_num"]).cumcount() + 1
    return first[["match_id", "innings_num", "date", "player_id", "bat_pos"]].reset_index(drop=True)


def batting_order_history(balls: pd.DataFrame) -> pd.DataFrame:
    per = batting_positions(balls).groupby(["player_id", "date"], as_index=False).bat_pos.mean()
    return post_match_ewm(per, "player_id", ["bat_pos"])


def load_xi(balls: pd.DataFrame) -> pd.DataFrame:
    xi = pd.read_parquet(PARSED / "playing_xi.parquet")
    dates = balls[["match_id", "date"]].drop_duplicates("match_id")
    return xi.merge(dates, on="match_id")[["match_id", "date", "team", "player_id", "position"]]


def powerplay_history(balls: pd.DataFrame, xi: pd.DataFrame) -> pd.DataFrame:
    """EWM of overs bowled in overs 1-6 per appearance; appearances with none count 0."""
    pp = (balls[balls.over < 6].groupby(["match_id", "bowler_id"]).over.nunique()
          .rename("pp_overs").reset_index().rename(columns={"bowler_id": "player_id"}))
    app = xi[["match_id", "player_id", "date"]].merge(
        pp, on=["match_id", "player_id"], how="left").fillna({"pp_overs": 0})
    per = app.groupby(["player_id", "date"], as_index=False).pp_overs.mean()
    return post_match_ewm(per, "player_id", ["pp_overs"])
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_features.py -q`
Expected: PASS, 24 tests.

- [ ] **Step 5: Smoke-test on real data**

Run:
```bash
python -c "
from src.features.outcomes import load_balls
from src.features.players import player_styles, annotate
b = annotate(load_balls(), player_styles())
print(b.bowler_type.value_counts(normalize=True).round(3).to_dict())
print(b.batting_hand.value_counts(normalize=True).round(3).to_dict())
"
```
Expected: `unknown` under 5% of balls for both (measured coverage: 96% bowler style, 97% batting hand).

- [ ] **Step 6: Commit**

```bash
git add src/features tests/test_features.py
git commit -m "feat: point-in-time lookup, form, player styles, batting order, powerplay share"
```

---

### Task 3: Shrinkage — counts, k fit, levels

**Files:**
- Create: `src/skill/__init__.py`, `src/skill/counts.py`, `src/skill/shrink.py`
- Test: `tests/test_shrink.py`

**Interfaces:**
- Consumes: `CLASSES`, `log_loss` (Task 1); `as_of` (Task 2).
- Produces:
  - `totals(balls, group_cols: list[str]) -> DataFrame[group_cols + CLASSES]`
  - `cumulative_counts(balls, group_cols) -> DataFrame[group_cols + date + CLASSES]` (post-date running totals)
  - `counts_before(cum, query, group_cols) -> np.ndarray (n, 11)`
  - `ConcentrationAtBoundError`, `fit_k(counts, parent) -> float`, `shrink(counts, parent, k) -> np.ndarray`
  - `Level(name, group_cols: tuple[str, ...], parent: str | None)`, `LEVELS: tuple[Level, ...]`
  - `SkillModel(league: np.ndarray, k: dict[str, float], cum: dict[str, pd.DataFrame], levels: tuple[Level, ...])`
  - `fit(balls, cutoff: str, levels=LEVELS) -> SkillModel`
  - `estimates(skill, query) -> dict[str, np.ndarray]` (each `(len(query), 11)`); query needs `date, batter_id, bowler_id, bowler_type, batting_hand, venue_key`
  - `MIN_MEDIAN_BALLS = 30`; `level_report(skill, train, val) -> DataFrame[level, n_groups, median_balls, k, log_loss, parent_log_loss, gain, passes]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shrink.py
import numpy as np
import pandas as pd
import pytest

from src.features.outcomes import CLASSES
from src.skill.counts import cumulative_counts
from src.skill.shrink import (
    LEVELS, ConcentrationAtBoundError, Level, SkillModel, estimates, fit, fit_k, shrink)


def test_shrink_is_the_posterior_mean():
    out = shrink(np.array([[8.0, 2.0]]), np.array([[0.5, 0.5]]), k=10.0)
    np.testing.assert_allclose(out, [[0.65, 0.35]])


def test_a_group_with_no_data_returns_the_parent():
    parent = np.array([[0.3, 0.7]])
    np.testing.assert_allclose(shrink(np.zeros((1, 2)), parent, k=50.0), parent)


def test_rows_sum_to_one():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(20, 11)).astype(float)
    np.testing.assert_allclose(shrink(counts, np.full((20, 11), 1 / 11), 30.0).sum(1), 1.0)


def test_fit_recovers_a_known_concentration():
    rng = np.random.default_rng(42)
    true_k, parent = 80.0, np.array([0.4, 0.35, 0.25])
    counts = np.array([rng.multinomial(120, rng.dirichlet(true_k * parent))
                       for _ in range(600)], dtype=float)
    k_hat = fit_k(counts, np.tile(parent, (600, 1)))
    assert 0.5 * true_k < k_hat < 2.0 * true_k


def test_bound_pinned_fit_raises():
    """Groups with no scatter around the parent push k to its upper bound."""
    counts = np.tile(np.array([50.0, 50.0]), (500, 1))
    with pytest.raises(ConcentrationAtBoundError):
        fit_k(counts, np.full((500, 2), 0.5))


def test_estimates_use_only_earlier_dates():
    balls = pd.DataFrame({
        "batter_id": ["b1"] * 20,
        "date": pd.to_datetime(["2020-01-01"] * 10 + ["2020-02-01"] * 10),
        "outcome": pd.Categorical(["4"] * 10 + ["6"] * 10, categories=CLASSES)})
    lv = Level("batter", ("batter_id",), None)
    skill = SkillModel(league=np.full(11, 1 / 11), k={"batter": 1.0},
                       cum={"batter": cumulative_counts(balls, ["batter_id"])}, levels=(lv,))
    q = pd.DataFrame({"batter_id": ["b1"] * 3,
                      "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])})
    e = estimates(skill, q)["batter"]
    i4, i6 = CLASSES.index("4"), CLASSES.index("6")
    np.testing.assert_allclose(e[0], 1 / 11)
    assert e[1, i4] == pytest.approx((10 + 1 / 11) / 11)
    assert e[1, i6] == pytest.approx((1 / 11) / 11)
    assert e[2, i6] == pytest.approx((10 + 1 / 11) / 21)


def _synthetic(seed=0, n_batters=300, balls_each=300):
    """Batters scatter around the league (k=60); batter x type around the batter (k=200)."""
    rng = np.random.default_rng(seed)
    league = rng.dirichlet(np.full(11, 20.0))
    rows = []
    for b in range(n_batters):
        p_b = rng.dirichlet(60 * league)
        for t in ("pace_right", "spin_left"):
            p_bt = rng.dirichlet(200 * p_b)
            draws = rng.choice(11, size=balls_each // 2, p=p_bt)
            rows.append(pd.DataFrame({"batter_id": f"b{b}", "bowler_type": t, "o": draws}))
    d = pd.concat(rows, ignore_index=True)
    d["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 300, len(d)), "D")
    d["outcome"] = pd.Categorical(np.array(CLASSES)[d.o], categories=CLASSES)
    return d


def test_fit_runs_the_chain_and_orders_k_sensibly():
    levels = (Level("batter", ("batter_id",), None),
              Level("batter_x_bowlertype", ("batter_id", "bowler_type"), "batter"))
    skill = fit(_synthetic(), cutoff="2021-01-01", levels=levels)
    assert skill.k["batter"] < skill.k["batter_x_bowlertype"], \
        "the crossed level scatters less, so it must shrink harder"


def test_levels_are_the_spec_hierarchy():
    by = {lv.name: lv for lv in LEVELS}
    assert set(by) == {"batter", "bowler", "batter_x_bowlertype", "bowler_x_battinghand",
                       "batter_x_venue", "bowler_x_venue"}
    assert by["batter_x_venue"].parent == "batter"
    assert by["bowler_x_venue"].parent == "bowler"
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_shrink.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.skill'`.

- [ ] **Step 3: Implement**

`src/skill/__init__.py`: empty file.

```python
# src/skill/counts.py
"""Outcome counts per group: training totals, and post-date running totals.

Plain integer counts only. The Dirichlet-Multinomial fit reads a group's
total as its quantity of evidence; weighting balls breaks that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import as_of
from src.features.outcomes import CLASSES


def _wide(g) -> pd.DataFrame:
    w = g.size().unstack("outcome", fill_value=0).reindex(columns=CLASSES, fill_value=0)
    w.columns = list(CLASSES)
    return w


def totals(balls: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return _wide(balls.groupby(group_cols + ["outcome"], observed=True)).reset_index()


def cumulative_counts(balls: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    w = _wide(balls.groupby(group_cols + ["date", "outcome"], observed=True))
    return w.groupby(level=group_cols).cumsum().reset_index()


def counts_before(cum: pd.DataFrame, query: pd.DataFrame, group_cols: list[str]) -> np.ndarray:
    return as_of(cum, query, group_cols, CLASSES).fillna(0).to_numpy(float)
```

```python
# src/skill/shrink.py
"""Empirical-Bayes shrinkage of 11-class outcome rates.

    estimate = (counts + k * parent) / (n + k)

One k per level, fitted by maximising the Dirichlet-Multinomial marginal
likelihood of the training-slice counts given the parent. Levels:

    league -> batter -> batter x bowler_type
    league -> bowler -> bowler x batting_hand
    batter x venue (parent batter), bowler x venue (parent bowler)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.features.outcomes import CLASSES, log_loss
from src.skill.counts import counts_before, cumulative_counts, totals

LOG_K_BOUNDS = (np.log(0.1), np.log(1e6))
_BOUND_TOL = 1e-2
MIN_MEDIAN_BALLS = 30


class ConcentrationAtBoundError(RuntimeError):
    """k landed on its search bound: the fit failed, it did not 'shrink hard'."""


def _clean(parent: np.ndarray) -> np.ndarray:
    p = np.clip(parent, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


def dm_loglik(log_k: float, counts: np.ndarray, parent: np.ndarray) -> float:
    k = np.exp(log_k)
    a = k * parent
    n = counts.sum(axis=1)
    return float(np.sum(gammaln(k) - gammaln(n + k)
                        + (gammaln(counts + a) - gammaln(a)).sum(axis=1)))


def fit_k(counts: np.ndarray, parent: np.ndarray) -> float:
    parent = _clean(parent)
    res = minimize_scalar(lambda lk: -dm_loglik(lk, counts, parent),
                          bounds=LOG_K_BOUNDS, method="bounded", options={"xatol": 1e-4})
    if min(res.x - LOG_K_BOUNDS[0], LOG_K_BOUNDS[1] - res.x) < _BOUND_TOL:
        raise ConcentrationAtBoundError(f"k={np.exp(res.x):.4g} is at its search bound")
    return float(np.exp(res.x))


def shrink(counts: np.ndarray, parent: np.ndarray, k: float) -> np.ndarray:
    return (counts + k * _clean(parent)) / (counts.sum(axis=1, keepdims=True) + k)


@dataclass(frozen=True)
class Level:
    name: str
    group_cols: tuple[str, ...]
    parent: str | None          # None = league vector


LEVELS = (
    Level("batter", ("batter_id",), None),
    Level("bowler", ("bowler_id",), None),
    Level("batter_x_bowlertype", ("batter_id", "bowler_type"), "batter"),
    Level("bowler_x_battinghand", ("bowler_id", "batting_hand"), "bowler"),
    Level("batter_x_venue", ("batter_id", "venue_key"), "batter"),
    Level("bowler_x_venue", ("bowler_id", "venue_key"), "bowler"),
)


@dataclass
class SkillModel:
    league: np.ndarray                 # (11,), training slice
    k: dict[str, float]                # fitted on the training slice
    cum: dict[str, pd.DataFrame]       # running counts over ALL balls; read strictly before
    levels: tuple[Level, ...]


def fit(balls: pd.DataFrame, cutoff: str, levels: tuple[Level, ...] = LEVELS) -> SkillModel:
    """k from balls before `cutoff`; running counts from all balls (as_of keeps them honest)."""
    train = balls[balls.date < cutoff]
    league = train.outcome.value_counts().reindex(CLASSES, fill_value=0).to_numpy(float)
    league = league / league.sum()
    by_name = {lv.name: lv for lv in levels}
    at_cutoff: dict[str, pd.DataFrame] = {}
    k: dict[str, float] = {}
    for lv in levels:
        cols = list(lv.group_cols)
        tot = totals(train, cols)
        counts = tot[CLASSES].to_numpy(float)
        if lv.parent is None:
            parent = np.tile(league, (len(tot), 1))
        else:
            pcols = list(by_name[lv.parent].group_cols)
            parent = tot[pcols].merge(at_cutoff[lv.parent], on=pcols, how="left")[CLASSES]
            parent = parent.to_numpy(float)
        k[lv.name] = fit_k(counts, parent)
        at_cutoff[lv.name] = pd.concat(
            [tot[cols].reset_index(drop=True),
             pd.DataFrame(shrink(counts, parent, k[lv.name]), columns=CLASSES)], axis=1)
    cum = {lv.name: cumulative_counts(balls, list(lv.group_cols)) for lv in levels}
    return SkillModel(league, k, cum, tuple(levels))


def estimates(skill: SkillModel, query: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for lv in skill.levels:
        counts = counts_before(skill.cum[lv.name], query, list(lv.group_cols))
        parent = (np.tile(skill.league, (len(query), 1)) if lv.parent is None
                  else out[lv.parent])
        out[lv.name] = shrink(counts, parent, skill.k[lv.name])
    return out


def level_report(skill: SkillModel, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """The density gate: median group size >= 30 and held-out log loss beats the parent."""
    est = estimates(skill, val)
    y = val.outcome.cat.codes.to_numpy()
    league_ll = log_loss(np.tile(skill.league, (len(val), 1)), y)
    rows = []
    for lv in skill.levels:
        n = totals(train, list(lv.group_cols))[CLASSES].sum(axis=1)
        parent_ll = league_ll if lv.parent is None else log_loss(est[lv.parent], y)
        ll = log_loss(est[lv.name], y)
        rows.append({"level": lv.name, "n_groups": len(n), "median_balls": float(n.median()),
                     "k": skill.k[lv.name], "log_loss": ll, "parent_log_loss": parent_ll,
                     "gain": parent_ll - ll})
    out = pd.DataFrame(rows)
    out["passes"] = (out.median_balls >= MIN_MEDIAN_BALLS) & (out.gain > 0)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_shrink.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/skill tests/test_shrink.py
git commit -m "feat: Dirichlet-Multinomial shrinkage over 11 classes, six levels, point-in-time"
```

---

### Task 4: The per-ball feature frame

**Files:**
- Create: `src/model/__init__.py`, `src/model/frame.py`
- Test: `tests/test_frame.py`

**Interfaces:**
- Consumes: `CLASSES`, `NOT_LEGAL`, `is_wicket`, `label_outcomes` (Task 1); `as_of` (Task 2); `SkillModel`, `estimates`, `LEVELS` (Task 3).
- Produces:
  - `SITUATION = ["over", "ball_in_over", "wickets", "runs", "batter_balls", "free_hit"]`
  - `FORM = ["bat_form_sr", "bat_form_out", "bowl_form_econ", "bowl_form_wkt"]`
  - `skill_columns(levels) -> list[str]` (`"{level}__{class}"` per level, then `"hint__{class}"`)
  - `static_columns(levels) = skill_columns + FORM`; `feature_columns(levels) = static_columns + SITUATION` (87 columns with the six levels)
  - `skill_features(skill, query) -> DataFrame` (float32, index = query.index)
  - `form_features(query, bat_form, bowl_form) -> DataFrame` (float32)
  - `situation(first_innings_balls) -> DataFrame` (state BEFORE each delivery)
  - `build_frame(balls, skill, bat_form, bowl_form) -> DataFrame` (first-innings rows: feature columns + `y` (int8 class code) + `date` + `match_id`)
  - `importance_groups(levels) -> dict[str, list[str]]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frame.py
import numpy as np
import pandas as pd
import pytest

from src.features.outcomes import CLASSES, label_outcomes
from src.model.frame import (
    FORM, SITUATION, feature_columns, importance_groups, situation, skill_features)
from src.skill.shrink import LEVELS, Level, SkillModel
from src.skill.counts import cumulative_counts


def _innings(outcomes, batters):
    """One first innings; each outcome is (wide, noball, runs_batter, runs_total, wicket_kind)."""
    rows = []
    for (wide, nb, rb, rt, wk), bat in zip(outcomes, batters):
        rows.append({"match_id": "m", "batter_id": bat, "wide": wide, "noball": nb,
                     "bye": np.nan, "legbye": np.nan, "runs_batter": rb, "runs_total": rt,
                     "wicket_kind": wk})
    d = pd.DataFrame(rows)
    d["outcome"] = label_outcomes(d)
    return d


def test_situation_is_the_state_before_each_delivery():
    n = np.nan
    d = _innings([(n, n, 1, 1, None),        # legal single
                  (1.0, n, 0, 1, None),      # wide
                  (n, n, 4, 4, None),        # legal four
                  (n, n, 0, 0, "bowled"),    # wicket
                  (n, 1.0, 0, 1, None),      # no-ball
                  (n, n, 0, 0, None),        # free hit
                  (n, n, 0, 0, None)],
                 ["A", "B", "B", "B", "C", "C", "C"])
    s = situation(d)
    assert s.runs.tolist() == [0, 1, 2, 6, 6, 7, 7]
    assert s.wickets.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert s.ball_in_over.tolist() == [0, 1, 1, 2, 3, 3, 4], "wide and no-ball are not legal"
    assert s.free_hit.tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert s.batter_balls.tolist() == [0, 0, 0, 1, 0, 1, 2], "a wide is not a ball faced"


def test_over_rolls_after_six_legal_balls():
    n = np.nan
    d = _innings([(n, n, 0, 0, None)] * 8, ["A"] * 8)
    s = situation(d)
    assert s.over.tolist() == [0] * 6 + [1, 1]
    assert s.ball_in_over.tolist() == [0, 1, 2, 3, 4, 5, 0, 1]


def test_feature_columns_count_with_six_levels():
    assert len(feature_columns(LEVELS)) == 6 * 11 + 11 + len(FORM) + len(SITUATION)


def test_hint_rows_sum_to_one():
    levels = (Level("batter", ("batter_id",), None), Level("bowler", ("bowler_id",), None))
    balls = pd.DataFrame({
        "batter_id": ["b"] * 5, "bowler_id": ["w"] * 5,
        "date": pd.to_datetime(["2020-01-01"] * 5),
        "outcome": pd.Categorical(["4", "0", "1", "6", "0"], categories=CLASSES)})
    skill = SkillModel(np.full(11, 1 / 11), {"batter": 5.0, "bowler": 5.0},
                       {lv.name: cumulative_counts(balls, list(lv.group_cols)) for lv in levels},
                       levels)
    q = pd.DataFrame({"batter_id": ["b"], "bowler_id": ["w"],
                      "date": pd.to_datetime(["2020-02-01"])})
    f = skill_features(skill, q)
    hint = f[[f"hint__{c}" for c in CLASSES]].to_numpy()
    assert hint.sum() == pytest.approx(1.0, abs=1e-5)


def test_importance_groups_cover_every_feature_once_plus_all_skill():
    g = importance_groups(LEVELS)
    singles = [c for name, cols in g.items() if name != "all_skill" for c in cols]
    assert sorted(singles) == sorted(feature_columns(LEVELS))
    assert len(g["all_skill"]) == 6 * 11 + 11
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_frame.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model'`.

- [ ] **Step 3: Implement**

`src/model/__init__.py`: empty file.

```python
# src/model/frame.py
"""One row per first-innings delivery: skill, form and the situation before the ball.

The simulator builds exactly these columns, in exactly this order:
static columns (skill, hint, form) then SITUATION.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.asof import as_of
from src.features.outcomes import CLASSES, NOT_LEGAL, is_wicket
from src.skill.shrink import Level, SkillModel, estimates

SITUATION = ["over", "ball_in_over", "wickets", "runs", "batter_balls", "free_hit"]
FORM = ["bat_form_sr", "bat_form_out", "bowl_form_econ", "bowl_form_wkt"]


def skill_columns(levels: tuple[Level, ...]) -> list[str]:
    return ([f"{lv.name}__{c}" for lv in levels for c in CLASSES]
            + [f"hint__{c}" for c in CLASSES])


def static_columns(levels: tuple[Level, ...]) -> list[str]:
    return skill_columns(levels) + FORM


def feature_columns(levels: tuple[Level, ...]) -> list[str]:
    return static_columns(levels) + SITUATION


def skill_features(skill: SkillModel, query: pd.DataFrame) -> pd.DataFrame:
    est = estimates(skill, query)
    bat = est.get("batter_x_bowlertype", est.get("batter"))
    bowl = est.get("bowler_x_battinghand", est.get("bowler"))
    hint = bat * bowl / skill.league            # batter rate x bowler rate / league rate
    hint = hint / hint.sum(axis=1, keepdims=True)
    blocks = [est[lv.name] for lv in skill.levels] + [hint]
    return pd.DataFrame(np.hstack(blocks).astype(np.float32), index=query.index,
                        columns=skill_columns(skill.levels))


def form_features(query: pd.DataFrame, bat_form: pd.DataFrame,
                  bowl_form: pd.DataFrame) -> pd.DataFrame:
    qb = query[["batter_id", "date"]].rename(columns={"batter_id": "player_id"})
    qw = query[["bowler_id", "date"]].rename(columns={"bowler_id": "player_id"})
    a = as_of(bat_form, qb, ["player_id"], ["bat_form_sr", "bat_form_out"])
    b = as_of(bowl_form, qw, ["player_id"], ["bowl_form_econ", "bowl_form_wkt"])
    return pd.concat([a, b], axis=1).astype(np.float32)


def situation(b: pd.DataFrame) -> pd.DataFrame:
    """State before each delivery. `b` is first-innings balls in delivery order."""
    m = b.match_id
    legal = (~b.outcome.isin(NOT_LEGAL)).astype(int)
    wkt = is_wicket(b).astype(int)
    faced = (b.outcome != "wide").astype(int)
    legal_before = legal.groupby(m).cumsum() - legal
    return pd.DataFrame({
        "over": legal_before // 6,
        "ball_in_over": legal_before % 6,
        "wickets": wkt.groupby(m).cumsum() - wkt,
        "runs": b.runs_total.astype(int).groupby(m).cumsum() - b.runs_total.astype(int),
        "batter_balls": faced.groupby([m, b.batter_id]).cumsum() - faced,
        "free_hit": b.noball.notna().groupby(m).shift(1, fill_value=False).astype(int),
    }, index=b.index).astype(np.float32)


def build_frame(balls: pd.DataFrame, skill: SkillModel, bat_form: pd.DataFrame,
                bowl_form: pd.DataFrame) -> pd.DataFrame:
    """`balls` must be annotated (bowler_type, batting_hand)."""
    b = balls[balls.innings_num == 1]
    frame = pd.concat([skill_features(skill, b), form_features(b, bat_form, bowl_form),
                       situation(b)], axis=1)[feature_columns(skill.levels)]
    frame["y"] = b.outcome.cat.codes.astype(np.int8)
    frame["date"] = b.date
    frame["match_id"] = b.match_id
    return frame.reset_index(drop=True)


def importance_groups(levels: tuple[Level, ...]) -> dict[str, list[str]]:
    g = {lv.name: [f"{lv.name}__{c}" for c in CLASSES] for lv in levels}
    g["hint"] = [f"hint__{c}" for c in CLASSES]
    g.update({c: [c] for c in FORM + SITUATION})
    g["all_skill"] = skill_columns(levels)
    return g
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_frame.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/model tests/test_frame.py
git commit -m "feat: per-ball feature frame (skill, hint, form, situation)"
```

---

### Task 5: The ball model and permutation importance

**Files:**
- Create: `src/model/ballmodel.py`
- Test: `tests/test_ballmodel.py`

**Interfaces:**
- Consumes: `log_loss` (Task 1).
- Produces: `MODELS = ("logistic", "lightgbm", "xgboost")`; `BallModel(name, cols, est)` with `.predict(X: DataFrame) -> np.ndarray (n, 11)`; `fit(name, X, y, dates, cols) -> BallModel`; `permutation_importance(model, X, y, groups, n=200_000, seed=0) -> DataFrame[group, ll_increase]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ballmodel.py
import numpy as np
import pandas as pd
import pytest

from src.model.ballmodel import MODELS, fit, permutation_importance


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = np.where(X.signal > 0, 1, 0)
    y[:11] = np.arange(11)                # every class present
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="h"))
    return X, y, dates


@pytest.mark.parametrize("name", MODELS)
def test_fit_returns_eleven_column_probabilities(name):
    X, y, dates = _data()
    m = fit(name, X, y, dates, ["signal", "noise"])
    p = m.predict(X.head(50))
    assert p.shape == (50, 11)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_model_uses_only_its_own_columns():
    X, y, dates = _data()
    m = fit("logistic", X, y, dates, ["signal"])
    assert m.predict(X.assign(extra=1.0)).shape == (len(X), 11)


def test_permutation_importance_ranks_signal_above_noise():
    X, y, dates = _data()
    m = fit("lightgbm", X, y, dates, ["signal", "noise"])
    imp = permutation_importance(m, X, y, {"signal": ["signal"], "noise": ["noise"]}).set_index("group")
    assert imp.loc["signal", "ll_increase"] > 0.1
    assert abs(imp.loc["noise", "ll_increase"]) < imp.loc["signal", "ll_increase"] / 10
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_ballmodel.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.ballmodel'`.

- [ ] **Step 3: Implement**

```python
# src/model/ballmodel.py
"""Per-ball 11-class models: multinomial logistic (the floor), LightGBM, XGBoost.

The boosted models early-stop on the most recent 10% of their own training
rows, so no evaluation year is touched while fitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.outcomes import log_loss

MODELS = ("logistic", "lightgbm", "xgboost")
ES_FRACTION = 0.1
N_CLASSES = 11


@dataclass
class BallModel:
    name: str
    cols: list[str]
    est: object

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est.predict_proba(X[self.cols])


def _recent_split(dates: pd.Series):
    order = np.argsort(dates.to_numpy(), kind="stable")
    cut = int(len(order) * (1 - ES_FRACTION))
    return order[:cut], order[cut:]


def fit(name: str, X: pd.DataFrame, y: np.ndarray, dates: pd.Series,
        cols: list[str]) -> BallModel:
    X = X[cols].reset_index(drop=True)
    y = np.asarray(y)
    if name == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=500))
        est.fit(X, y)
    elif name == "lightgbm":
        import lightgbm as lgb
        a, b = _recent_split(dates)
        est = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=0.05,
                                 num_leaves=63, min_child_samples=500, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        est.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name == "xgboost":
        import xgboost as xgb
        a, b = _recent_split(dates)
        est = xgb.XGBClassifier(objective="multi:softprob", n_estimators=3000,
                                learning_rate=0.05, max_depth=6, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist",
                                early_stopping_rounds=50, eval_metric="mlogloss")
        est.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")
    if len(est.classes_) != N_CLASSES:
        raise ValueError(f"training rows held {len(est.classes_)} classes, need {N_CLASSES}")
    return BallModel(name, list(cols), est)


def permutation_importance(model: BallModel, X: pd.DataFrame, y: np.ndarray,
                           groups: dict[str, list[str]], n: int = 200_000,
                           seed: int = 0) -> pd.DataFrame:
    """Log-loss increase when a group of columns is shuffled together."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = np.asarray(y)[idx]
    base = log_loss(model.predict(Xs), ys)
    rows = []
    for g, cols in groups.items():
        Xp = Xs.copy()
        Xp[cols] = Xs[cols].to_numpy()[rng.permutation(len(Xs))]
        rows.append({"group": g, "ll_increase": log_loss(model.predict(Xp), ys) - base})
    return pd.DataFrame(rows).sort_values("ll_increase", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ballmodel.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/model/ballmodel.py tests/test_ballmodel.py
git commit -m "feat: per-ball models (logistic, LightGBM, XGBoost) and permutation importance"
```

---

### Task 6: The Monte Carlo innings

**Files:**
- Create: `src/sim/__init__.py`, `src/sim/simulate.py`
- Test: `tests/test_simulate.py`

**Interfaces:**
- Consumes: `CLASSES`, `outcome_table` output format (Task 1).
- Produces:
  - `sample_rows(p, rng) -> np.ndarray` (one index per row; rows need not be normalised)
  - `bowler_plan(weights, n_sims, n_overs, rng, max_overs=4) -> np.ndarray (n_sims, n_overs)`
  - `RunTable` and `run_table(table: DataFrame) -> RunTable`
  - `MatchSetup(batters: list[str], bowlers: list[str], bowler_weights: np.ndarray, static: np.ndarray (n_bat, n_bowl, n_static))`
  - `simulate(setup, predict, table: RunTable, n_sims=10_000, legal_balls=36, seed=0) -> np.ndarray` of totals. `predict(X: ndarray) -> ndarray (n, 11)`; X columns are `static` then `over, ball_in_over, wickets, runs, batter_balls, free_hit`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulate.py
import numpy as np
import pandas as pd
import pytest

from src.features.outcomes import CLASSES
from src.sim.simulate import MatchSetup, bowler_plan, run_table, simulate

N_STATIC = 1


def _table():
    rows = [(c, int(c), 0) for c in ["0", "1", "2", "3", "4", "6"]]
    rows += [("5plus", 5, 0), ("wicket", 0, 1), ("wide", 1, 0), ("noball", 1, 0), ("bye", 1, 0)]
    t = pd.DataFrame(rows, columns=["outcome", "runs", "wicket"])
    t["n"] = 1
    t["p"] = 1.0
    return run_table(t)


def _setup():
    # static[b, w, 0] = batting position, so tests can see who is on strike
    static = np.repeat(np.arange(11, dtype=float)[:, None, None], 11, axis=1)
    return MatchSetup([f"b{i}" for i in range(11)], [f"w{i}" for i in range(11)],
                      np.ones(11), static)


def _always(cls):
    i = CLASSES.index(cls)
    def predict(X):
        p = np.zeros((len(X), 11)); p[:, i] = 1.0
        return p
    return predict


def test_always_singles_scores_36():
    assert (simulate(_setup(), _always("1"), _table(), n_sims=50) == 36).all()


def test_always_four_scores_144():
    assert (simulate(_setup(), _always("4"), _table(), n_sims=50) == 144).all()


def test_all_out_stops_the_innings():
    assert (simulate(_setup(), _always("wicket"), _table(), n_sims=50) == 0).all()


def test_wides_alone_never_finish():
    with pytest.raises(RuntimeError):
        simulate(_setup(), _always("wide"), _table(), n_sims=5)


def test_wides_do_not_use_up_legal_balls():
    calls = {"n": 0}
    def predict(X):
        cls = "wide" if calls["n"] % 2 == 0 else "1"
        calls["n"] += 1
        return _always(cls)(X)
    # 36 legal singles plus 36 wides of 1 run each
    assert (simulate(_setup(), predict, _table(), n_sims=20) == 72).all()


def test_noball_makes_the_next_delivery_a_free_hit():
    seen = []
    def predict(X):
        seen.append(X[0, N_STATIC + 5])
        return _always("noball" if len(seen) == 1 else "0")(X)
    simulate(_setup(), predict, _table(), n_sims=3)
    assert seen[:3] == [0, 1, 0]


def test_strike_rotates_on_singles_and_at_end_of_over():
    seen = []
    def predict(X):
        seen.append(X[0, 0])
        return _always("0")(X)
    simulate(_setup(), predict, _table(), n_sims=1)
    assert seen[:6] == [0] * 6 and seen[6:12] == [1] * 6, "dots: openers swap only at the over"
    seen.clear()
    def singles(X):
        seen.append(X[0, 0])
        return _always("1")(X)
    simulate(_setup(), singles, _table(), n_sims=1)
    assert seen[:7] == [0, 1, 0, 1, 0, 1, 1], "single on the 6th ball: same batter keeps strike"


def test_new_batter_comes_in_at_the_fall_of_a_wicket():
    seen = []
    def predict(X):
        seen.append(X[0, 0])
        return _always("wicket" if len(seen) == 1 else "0")(X)
    simulate(_setup(), predict, _table(), n_sims=1)
    assert seen[1] == 2


def test_bowler_plan_respects_the_rules():
    # 6 bowlers, not 5: with exactly 5 x 4 = 20 overs a random order can corner itself
    plan = bowler_plan(np.ones(6), n_sims=500, n_overs=20, rng=np.random.default_rng(0))
    assert (plan[:, 1:] != plan[:, :-1]).all(), "no bowler bowls consecutive overs"
    counts = np.stack([(plan == b).sum(axis=1) for b in range(6)], axis=1)
    assert (counts <= 4).all()


def test_bowler_plan_raises_when_the_rules_cannot_be_met():
    with pytest.raises(RuntimeError):
        bowler_plan(np.ones(2), n_sims=1, n_overs=10, rng=np.random.default_rng(0))


def test_bowler_plan_follows_the_weights():
    plan = bowler_plan(np.array([3.0, 1.0, 1e-6]), n_sims=4000, n_overs=1,
                       rng=np.random.default_rng(0))
    share = (plan[:, 0] == 0).mean()
    assert 0.72 < share < 0.78
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_simulate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sim'`.

- [ ] **Step 3: Implement**

`src/sim/__init__.py`: empty file.

```python
# src/sim/simulate.py
"""The Monte Carlo innings, vectorised across simulations.

Each step asks the ball model for 11 probabilities for every live simulation,
draws a class, then draws what that delivery added (runs, wicket) from the
historical table for that class.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.outcomes import CLASSES

WIDE, NOBALL = CLASSES.index("wide"), CLASSES.index("noball")
MAX_DELIVERIES = 400


def sample_rows(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = np.cumsum(p, axis=1)
    u = rng.random((len(p), 1)) * c[:, -1:]
    return np.minimum((c < u).sum(axis=1), p.shape[1] - 1)


def bowler_plan(weights, n_sims: int, n_overs: int, rng: np.random.Generator,
                max_overs: int = 4) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if (weights > 0).sum() < 2:
        raise ValueError("need at least two bowlers with positive weight")
    plan = np.empty((n_sims, n_overs), dtype=int)
    used = np.zeros((n_sims, len(weights)), dtype=int)
    rows = np.arange(n_sims)
    for o in range(n_overs):
        w = np.broadcast_to(weights, used.shape).copy()
        w[used >= max_overs] = 0
        if o:
            w[rows, plan[:, o - 1]] = 0
        if (w.sum(axis=1) == 0).any():
            raise RuntimeError(f"no bowler can legally bowl over {o + 1}")
        pick = sample_rows(w, rng)
        plan[:, o] = pick
        used[rows, pick] += 1
    return plan


@dataclass
class RunTable:
    runs: list[np.ndarray]
    wicket: list[np.ndarray]
    p: list[np.ndarray]


def run_table(table: pd.DataFrame) -> RunTable:
    runs, wkt, p = [], [], []
    for c in CLASSES:
        t = table[table.outcome == c]
        if t.empty:
            raise ValueError(f"no history for class {c!r}")
        runs.append(t.runs.to_numpy(int))
        wkt.append(t.wicket.to_numpy(int))
        p.append(t.p.to_numpy(float) / t.p.sum())
    return RunTable(runs, wkt, p)


@dataclass
class MatchSetup:
    batters: list[str]               # batting order
    bowlers: list[str]
    bowler_weights: np.ndarray
    static: np.ndarray               # (n_bat, n_bowl, n_static)


def simulate(setup: MatchSetup, predict, table: RunTable, n_sims: int = 10_000,
             legal_balls: int = 36, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_bat = len(setup.batters)
    max_wkts = min(10, n_bat - 1)
    plan = bowler_plan(setup.bowler_weights, n_sims, -(-legal_balls // 6), rng)
    runs = np.zeros(n_sims, dtype=int)
    wkts = np.zeros(n_sims, dtype=int)
    legal = np.zeros(n_sims, dtype=int)
    striker = np.zeros(n_sims, dtype=int)
    non_striker = np.ones(n_sims, dtype=int)
    next_in = np.full(n_sims, 2)
    faced = np.zeros((n_sims, n_bat), dtype=int)
    free_hit = np.zeros(n_sims, dtype=int)
    for _ in range(MAX_DELIVERIES):
        act = np.flatnonzero((legal < legal_balls) & (wkts < max_wkts))
        if act.size == 0:
            return runs
        over = legal[act] // 6
        bat = striker[act]
        X = np.column_stack([setup.static[bat, plan[act, over]], over, legal[act] % 6,
                             wkts[act], runs[act], faced[act, bat], free_hit[act]])
        cls = sample_rows(predict(X), rng)
        r = np.zeros(act.size, dtype=int)
        w = np.zeros(act.size, dtype=int)
        for c in np.unique(cls):
            m = cls == c
            j = rng.choice(len(table.p[c]), size=int(m.sum()), p=table.p[c])
            r[m] = table.runs[c][j]
            w[m] = table.wicket[c][j]
        is_legal = (cls != WIDE) & (cls != NOBALL)
        runs[act] += r
        faced[act, bat] += cls != WIDE
        free_hit[act] = cls == NOBALL
        legal[act] += is_legal
        wkts[act] += w
        # Runs actually run decide the strike; the 1-run penalty on a wide or
        # no-ball is not run.
        swap = ((r - ~is_legal) % 2 == 1) & (w == 0)
        # ponytail: the striker is always the one out and the new batter takes
        # strike; non-striker run-outs (~3% of wickets) are not modelled.
        out = act[w == 1]
        striker[out] = next_in[out]
        next_in[out] += 1
        end_over = is_legal & (legal[act] % 6 == 0)
        flip = act[swap ^ end_over]
        striker[flip], non_striker[flip] = non_striker[flip], striker[flip].copy()
    raise RuntimeError(f"innings unfinished after {MAX_DELIVERIES} deliveries")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_simulate.py -q`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sim tests/test_simulate.py
git commit -m "feat: vectorised Monte Carlo innings with extras, free hits and bowler rules"
```

---

### Task 7: Scoring, calibration and the betting rule

**Files:**
- Create: `src/sim/calibrate.py`, `src/bet/__init__.py`, `src/bet/backtest.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces:
  - `crps(samples, y) -> float`, `pit(samples, y) -> float`, `coverage(sims, ys, lo=0.1, hi=0.9) -> float`
  - `Calibration(a, b, s)`; `apply(cal, samples) -> np.ndarray`; `fit_calibration(sims, ys) -> Calibration`
  - `summary(sims, ys) -> dict` with keys `n, mae, crps, bias, coverage, ks_p`
  - `COMMISSION = 0.05`, `THRESHOLD = 1 / 1.95`; `decide(p_over, p_under) -> "over" | "under" | None`; `pnl(side, line, actual) -> float`; `backtest(sims, lines, actuals) -> dict` with keys `n_matches, n_bets, hit_rate, pnl, roi`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring.py
import numpy as np
import pytest

from src.bet.backtest import THRESHOLD, backtest, decide, pnl
from src.sim.calibrate import Calibration, apply, coverage, crps, fit_calibration, pit


def test_crps_of_a_point_mass_is_absolute_error():
    assert crps(np.full(1000, 60.0), 50.0) == pytest.approx(10.0)


def test_crps_prefers_a_well_spread_forecast_to_a_confident_wrong_one():
    rng = np.random.default_rng(0)
    assert crps(rng.normal(50, 12, 5000), 50.0) < crps(np.full(5000, 80.0), 50.0)


def test_pit_counts_ties_as_half():
    assert pit(np.array([1, 2, 2, 3]), 2) == pytest.approx(0.5)


def test_coverage_of_p10_p90():
    s = [np.arange(100.0)] * 4
    assert coverage(s, [50, 5, 95, 50]) == pytest.approx(0.5)


def test_calibration_removes_a_constant_bias():
    rng = np.random.default_rng(1)
    truth = rng.normal(50, 10, 300)
    sims = [rng.normal(t - 15, 10, 2000) for t in truth]   # every forecast 15 runs low
    cal = fit_calibration(sims, truth)
    fixed = [apply(cal, s) for s in sims]
    assert abs(np.mean([np.median(f) for f in fixed]) - truth.mean()) < 1.0


def test_apply_is_identity_for_the_neutral_calibration():
    s = np.array([40.0, 50.0, 70.0])
    np.testing.assert_allclose(apply(Calibration(0.0, 1.0, 1.0), s), s)


def test_threshold_breaks_even_after_commission():
    assert THRESHOLD == pytest.approx(1 / 1.95)
    assert THRESHOLD * 0.95 - (1 - THRESHOLD) == pytest.approx(0.0)


def test_decide():
    assert decide(0.55, 0.45) == "over"
    assert decide(0.45, 0.55) == "under"
    assert decide(0.51, 0.49) is None


def test_pnl_win_lose_push():
    assert pnl("over", 50.5, 60) == pytest.approx(0.95)
    assert pnl("over", 50.5, 40) == -1.0
    assert pnl("under", 50.0, 50) == 0.0
    assert pnl(None, 50.5, 60) == 0.0


def test_backtest_totals():
    sims = [np.full(100, 70.0), np.full(100, 30.0), np.full(100, 50.0)]
    # over wins; under loses; under loses
    out = backtest(sims, np.array([50.5, 50.5, 50.5]), np.array([60, 60, 60]))
    assert out["n_bets"] == 3
    assert out["pnl"] == pytest.approx(0.95 - 1.0 - 1.0)
    assert out["hit_rate"] == pytest.approx(1 / 3)
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.bet'`.

- [ ] **Step 3: Implement**

```python
# src/sim/calibrate.py
"""Scoring a simulated distribution, and the one correction fitted on 2024.

corrected = a + b * median + s * (samples - median)
a, b by least squares of actual on simulated median; s chosen so 80% of
actuals land inside the corrected p10-p90.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kstest


def crps(samples: np.ndarray, y: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n ** 2)


def pit(samples: np.ndarray, y: float) -> float:
    s = np.asarray(samples)
    return float(np.mean(s < y) + 0.5 * np.mean(s == y))


def coverage(sims, ys, lo: float = 0.1, hi: float = 0.9) -> float:
    inside = [np.quantile(s, lo) <= y <= np.quantile(s, hi) for s, y in zip(sims, ys)]
    return float(np.mean(inside))


@dataclass
class Calibration:
    a: float
    b: float
    s: float


def apply(cal: Calibration, samples: np.ndarray) -> np.ndarray:
    m = np.median(samples)
    return cal.a + cal.b * m + cal.s * (np.asarray(samples, dtype=float) - m)


def fit_calibration(sims, ys) -> Calibration:
    ys = np.asarray(ys, dtype=float)
    med = np.array([np.median(s) for s in sims])
    b, a = np.polyfit(med, ys, 1)
    grid = np.linspace(0.5, 2.5, 201)
    cov = np.array([coverage([apply(Calibration(a, b, s), x) for x in sims], ys) for s in grid])
    return Calibration(float(a), float(b), float(grid[np.argmin(np.abs(cov - 0.8))]))


def summary(sims, ys) -> dict:
    ys = np.asarray(ys, dtype=float)
    med = np.array([np.median(s) for s in sims])
    pits = np.array([pit(s, y) for s, y in zip(sims, ys)])
    return {
        "n": int(len(ys)),
        "mae": float(np.mean(np.abs(med - ys))),
        "crps": float(np.mean([crps(s, y) for s, y in zip(sims, ys)])),
        "bias": float(np.mean([np.mean(s) for s in sims]) - ys.mean()),
        "coverage": coverage(sims, ys),
        "ks_p": float(kstest(pits, "uniform").pvalue),
    }
```

`src/bet/__init__.py`: empty file.

```python
# src/bet/backtest.py
"""Bet the 6-over line when the simulated probability beats evens after commission.

Line markets pay about even money. With commission c on winnings, a bet at
probability p breaks even when p * (1 - c) = 1 - p, i.e. p = 1 / (2 - c).
"""
from __future__ import annotations

import numpy as np

COMMISSION = 0.05
THRESHOLD = 1 / (2 - COMMISSION)


def decide(p_over: float, p_under: float):
    if p_over > THRESHOLD:
        return "over"
    if p_under > THRESHOLD:
        return "under"
    return None


def pnl(side, line: float, actual: float) -> float:
    if side is None or actual == line:
        return 0.0
    won = actual > line if side == "over" else actual < line
    return (1 - COMMISSION) if won else -1.0


def backtest(sims, lines, actuals) -> dict:
    results = []
    for s, line, y in zip(sims, lines, actuals):
        side = decide(float(np.mean(s > line)), float(np.mean(s < line)))
        if side is not None:
            results.append(pnl(side, line, y))
    r = np.array(results)
    return {
        "n_matches": int(len(lines)),
        "n_bets": int(len(r)),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "pnl": float(r.sum()),
        "roi": float(r.mean()) if len(r) else float("nan"),
    }
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_scoring.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add src/sim/calibrate.py src/bet tests/test_scoring.py
git commit -m "feat: CRPS/PIT/coverage scoring, output calibration, betting rule"
```

---

### Task 8: Match setup and the development run

**Files:**
- Create: `src/sim/setup.py`, `src/run_6over.py`
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `Context(skill, styles, bat_form, bowl_form, bat_pos, pp, xi)`
  - `check_xi(rows, xi) -> None` (raises listing match ids without both XIs)
  - `match_setup(row, ctx) -> MatchSetup` (`row` has `match_id, date, batting_team, bowling_team, venue_key`)
  - `simulate_matches(rows, ctx, model, table: DataFrame, n_sims, legal_balls=36) -> list[np.ndarray]`
  - `python -m src.run_6over dev` writes `data/reports/6over_dev.json` with keys `levels, models, model, kept_columns, importance, calibration, dev_raw, dev_calibrated`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup.py
import pandas as pd
import pytest

from src.sim.setup import check_xi


def test_check_xi_names_matches_missing_a_team():
    rows = pd.DataFrame({"match_id": ["1", "2"], "batting_team": ["A", "C"],
                         "bowling_team": ["B", "D"]})
    xi = pd.DataFrame({"match_id": ["1", "1", "2"], "team": ["A", "B", "C"]})
    with pytest.raises(ValueError, match="2"):
        check_xi(rows, xi)


def test_check_xi_passes_when_complete():
    rows = pd.DataFrame({"match_id": ["1"], "batting_team": ["A"], "bowling_team": ["B"]})
    xi = pd.DataFrame({"match_id": ["1", "1"], "team": ["A", "B"]})
    check_xi(rows, xi)
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_setup.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sim.setup'`.

- [ ] **Step 3: Implement `src/sim/setup.py`**

```python
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
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/test_setup.py -q`
Expected: PASS, 2 tests.

- [ ] **Step 5: Write `src/run_6over.py`**

```python
# src/run_6over.py
"""The 6-over pipeline.

    python -m src.run_6over dev     train < 2024; choose model, prune, calibrate on 2024
    python -m src.run_6over final   train < 2025; score the 2025+ holdout ONCE

`final` refuses to run twice unless given --force, so the holdout is looked at once.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.bet.backtest import backtest
from src.config import DEV_END, REPORTS_DIR, TRAIN_END, ensure_dirs
from src.features.form import batter_form, bowler_form
from src.features.outcomes import load_balls, log_loss, outcome_table, venue_key
from src.features.players import (
    annotate, batting_order_history, load_xi, player_styles, powerplay_history)
from src.market.baseline import comparison_bar, comparison_set
from src.model.ballmodel import MODELS, fit as fit_model, permutation_importance
from src.model.frame import build_frame, feature_columns, importance_groups
from src.sim.calibrate import Calibration, apply, fit_calibration, summary
from src.sim.setup import Context, check_xi, simulate_matches
from src.skill.shrink import fit as fit_skill, level_report

SIX = "1_INNING_6_OVR_LINE"
DEV_N_SIMS = 2_000      # model choice; median MC error ~0.3 runs per match, averaged over 435
FINAL_N_SIMS = 10_000
DEV_REPORT = REPORTS_DIR / "6over_dev.json"
FINAL_REPORT = REPORTS_DIR / "6over_final.json"


def market_rows(split: str, start: str | None = None) -> pd.DataFrame:
    r = comparison_set(split)
    r = r[r.market_type == SIX].copy()
    r["date"] = pd.to_datetime(r["date"])
    if start:
        r = r[r.date >= start]
    r["venue_key"] = venue_key(r.venue, r.city)
    return r.sort_values("date", kind="stable").reset_index(drop=True)


def prepare(cutoff: str):
    styles = player_styles()
    balls = annotate(load_balls(), styles)
    xi = load_xi(balls)
    ctx = Context(skill=fit_skill(balls, cutoff), styles=styles,
                  bat_form=batter_form(balls), bowl_form=bowler_form(balls),
                  bat_pos=batting_order_history(balls), pp=powerplay_history(balls, xi), xi=xi)
    frame = build_frame(balls, ctx.skill, ctx.bat_form, ctx.bowl_form)
    table = outcome_table(balls[balls.date < cutoff])
    return balls, ctx, frame, table


def dev() -> dict:
    balls, ctx, frame, table = prepare(DEV_END)
    levels = level_report(ctx.skill, balls[balls.date < DEV_END],
                          balls[(balls.date >= DEV_END) & (balls.date < TRAIN_END)])
    print(levels.to_string(index=False), flush=True)
    if not levels.passes.all():
        raise SystemExit("a shrinkage level failed the density gate; see the table above")

    cols = feature_columns(ctx.skill.levels)
    tr = (frame.date < DEV_END).to_numpy()
    va = ((frame.date >= DEV_END) & (frame.date < TRAIN_END)).to_numpy()
    X, y, dates = frame[cols], frame.y.to_numpy(), frame.date
    rows = market_rows("train", start=DEV_END)
    check_xi(rows, ctx.xi)
    ys = rows.runs_at_6ov.to_numpy(float)

    results, fitted = {}, {}
    for name in MODELS:
        m = fit_model(name, X.loc[tr], y[tr], dates[tr], cols)
        sims = simulate_matches(rows, ctx, m, table, DEV_N_SIMS)
        results[name] = {"val_log_loss": log_loss(m.predict(X.loc[va]), y[va]), **summary(sims, ys)}
        fitted[name] = (m, sims)
        print(name, results[name], flush=True)
    best = min(results, key=lambda n: (results[n]["mae"], results[n]["crps"]))
    m, sims = fitted[best]

    groups = importance_groups(ctx.skill.levels)
    imp = permutation_importance(m, X.loc[va], y[va], groups)
    print(imp.to_string(index=False), flush=True)
    drop = {c for g in imp[(imp.ll_increase <= 0) & (imp.group != "all_skill")].group
            for c in groups[g]}
    kept = [c for c in cols if c not in drop]
    if drop:
        pruned = fit_model(best, X.loc[tr], y[tr], dates[tr], kept)
        if log_loss(pruned.predict(X.loc[va]), y[va]) <= results[best]["val_log_loss"] + 1e-4:
            m = pruned
            sims = simulate_matches(rows, ctx, m, table, DEV_N_SIMS)
        else:
            kept = cols

    cal = fit_calibration(sims, ys)
    report = {
        "levels": levels.to_dict(orient="records"),
        "k": ctx.skill.k,
        "models": results,
        "model": best,
        "kept_columns": kept,
        "importance": imp.to_dict(orient="records"),
        "calibration": cal.__dict__,
        "dev_raw": summary(sims, ys),
        # in-sample for the calibration: a sanity check, not a result
        "dev_calibrated": summary([apply(cal, s) for s in sims], ys),
    }
    ensure_dirs()
    DEV_REPORT.write_text(json.dumps(report, indent=2, default=float))
    return report


def final(force: bool = False) -> dict:
    if FINAL_REPORT.exists() and not force:
        raise SystemExit("the holdout was already scored; --force reruns it (say so in any write-up)")
    dev_report = json.loads(DEV_REPORT.read_text())
    balls, ctx, frame, table = prepare(TRAIN_END)
    tr = (frame.date < TRAIN_END).to_numpy()
    m = fit_model(dev_report["model"], frame.loc[tr], frame.y.to_numpy()[tr],
                  frame.date[tr], dev_report["kept_columns"])
    rows = market_rows("holdout")
    check_xi(rows, ctx.xi)
    ys = rows.runs_at_6ov.to_numpy(float)
    raw = simulate_matches(rows, ctx, m, table, FINAL_N_SIMS)
    sims = [apply(Calibration(**dev_report["calibration"]), s) for s in raw]
    bar = comparison_bar("holdout").set_index("market_type").loc[SIX]
    report = {
        "model": dev_report["model"],
        "raw": summary(raw, ys),
        "calibrated": summary(sims, ys),
        "market_mae": float(bar.market_mae),
        "detectable_gain": float(bar.detectable_gain),
        "backtest": backtest(sims, rows.pre_match_line.to_numpy(float), ys),
    }
    report["beats_bar"] = report["calibrated"]["mae"] < report["market_mae"] - report["detectable_gain"]
    ensure_dirs()
    FINAL_REPORT.write_text(json.dumps(report, indent=2, default=float))
    return report


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "dev":
        print(json.dumps(dev(), indent=2, default=float))
    elif step == "final":
        print(json.dumps(final(force="--force" in sys.argv), indent=2, default=float))
    else:
        raise SystemExit("usage: python -m src.run_6over dev|final [--force]")
```

- [ ] **Step 6: Smoke-run one match before the full run**

Run:
```bash
python -c "
from src.run_6over import prepare, market_rows
from src.config import DEV_END
from src.model.ballmodel import fit
from src.model.frame import feature_columns
from src.sim.setup import simulate_matches, check_xi
balls, ctx, frame, table = prepare(DEV_END)
cols = feature_columns(ctx.skill.levels)
tr = frame.date < DEV_END
m = fit('logistic', frame.loc[tr].head(200_000), frame.y[tr].head(200_000).to_numpy(), frame.date[tr].head(200_000), cols)
rows = market_rows('train', start=DEV_END).head(3); check_xi(rows, ctx.xi)
for s, y in zip(simulate_matches(rows, ctx, m, table, 2000), rows.runs_at_6ov):
    print(round(float(s.mean()), 1), round(float(s.std()), 1), 'actual', y)
"
```
Expected: three lines, simulated means roughly 35-65 and standard deviations roughly 8-20. If `fit_skill` raises `ConcentrationAtBoundError`, stop: report which level and its `k`, and remove that level from `LEVELS` only after telling the user.

- [ ] **Step 7: Run the development pipeline**

Run: `python -m src.run_6over dev`
Expected: the level table (all `passes` True), one line per model, the importance table, then the JSON report. This takes hours: three model fits on ~1M rows and 3–4 rounds of 435 simulated matches. If the density gate fails, stop and report the table to the user before changing `LEVELS`.

Record in the commit message: each model's `val_log_loss`, `mae`, `crps`; the chosen model; dropped feature groups; the `all_skill` importance (if it is ≤ 0, say so plainly — the shrinkage layer is not earning its place); the calibration `a, b, s`; and `dev_raw` bias and coverage.

- [ ] **Step 8: Commit**

```bash
git add src/sim/setup.py src/run_6over.py tests/test_setup.py
git commit -m "feat: dev run - model bake-off, feature pruning, calibration on 2024"
```

---

### Task 9: The holdout run (once)

**Files:**
- No code changes. Produces `data/reports/6over_final.json`.

- [ ] **Step 1: Confirm with the user before running**

The holdout is looked at once. Ask the user for an explicit go-ahead, quoting the dev report's chosen model and 2024 MAE.

- [ ] **Step 2: Run**

Run: `python -m src.run_6over final`
Expected: JSON with `raw`, `calibrated`, `market_mae` (10.48), `detectable_gain` (0.848), `backtest`, `beats_bar`.

- [ ] **Step 3: Report to the user**

Lead with one line: calibrated MAE vs 10.48, and whether it beats 10.48 − 0.85 = 9.63. Then bias, coverage, KS p-value, and the backtest's bets, hit rate and profit. If `beats_bar` is false, say so plainly.

- [ ] **Step 4: Commit the result note**

Add the numbers to `README.md` under a new "## Results" heading (one short paragraph), then:

```bash
git add README.md
git commit -m "docs: 6-over holdout result"
```

---

## Self-review notes

**Spec coverage.** §2 scope → Task 1 `load_balls`, Task 4 first-innings frame, Task 8 XI handling (impact player: the XI table as named; batting order sorted by history). §3 classes, precedence, run tables, free hit, penalty → Tasks 1, 4, 6. §4 shrinkage, levels, k fit, point-in-time, integer counts, bound error, density gate → Tasks 2, 3, 8 Step 7. §5 inputs, hint, bake-off, importance and the shrinkage check → Tasks 4, 5, 8. §6 simulator (order, bowler weights and rules, 36 balls or all out, 10,000 draws) → Tasks 6, 8, 9. §7 calibration → Task 7, fitted in Task 8, applied in Task 9. §8 splits → `DEV_END`/`TRAIN_END`, Tasks 8–9. §9 success and backtest → Tasks 7, 9. §11 tests → every task.

**Deliberate simplifications, marked in code.** The striker is always the batter dismissed (`ponytail:` in `simulate.py`). Free hit is "previous delivery was a no-ball" in both training and simulation, so the two agree. Dev bake-off uses 2,000 draws per match; the holdout uses 10,000. Players with no powerplay history get a 1e-6 bowling weight rather than exactly zero, so the bowler rules can always be met. Isotonic recalibration (spec §7) is not built: it is only needed if the dev `ks_p` shows the PIT is still clearly not flat after the linear correction.

**Known cost.** The `k` values are fitted on the training slice, which training rows then use: five scalars of in-sample information, not per-player leakage. The holdout is unaffected.
