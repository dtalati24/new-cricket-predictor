# Match context, faster runs, walk-forward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add match-level context features (toss, scoring trend, competition, venue/team/opposition powerplay form, weather), make simulation runs 3–5× faster, and replace the single holdout with a walk-forward test that can run on GitHub Actions overnight.

**Architecture:** A new per-match context table (one row per match, for the side batting first) is joined onto every first-innings ball in the training frame and broadcast into the simulator's static feature tensor, so the simulator's row layout changes only by extra static columns. Speed comes from deduplicating identical prediction rows, simulating matches in a process pool, freeing the training frame before simulation, and caching prepared data per cutoff. A walk-forward driver runs one test year per invocation, so each year can be its own GitHub Actions job.

**Tech Stack:** Python 3.14, pandas 3, numpy 2, scikit-learn, LightGBM 4.7, XGBoost 3.4, requests (Open-Meteo), concurrent.futures, threadpoolctl, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-06-match-context-and-speed-design.md`

## Global Constraints

- Point in time: every history feature for a match on date D uses only dates strictly before D, through `src.features.asof.as_of`. Toss, start time and weather at the venue on the day are treated as known at the toss (weather uses historical observations as a stand-in for the forecast).
- Speed changes must not change results: deduplicated prediction and parallel simulation are bit-identical to the old serial path for the same seeds (`seed = row index` per match).
- Context features are one value per match, for the innings-1 batting team; the same table feeds the training frame and the simulator.
- Powerplay runs = runs in the first 36 legal balls of an innings; an innings counts only if it reached 36 legal balls or lost 10 wickets inside them.
- Halflives: team and venue EWMs 5, competition EWM 20 (matches/innings, not days). League window 90 days.
- Weather variables, exactly: `temperature_2m, relative_humidity_2m, dew_point_2m, cloud_cover, wind_speed_10m, precipitation`; plus `start_hour_local`, `is_night` (`start_hour_local >= 17`). Start-hour value when a start time exists, else the mean over local hours 14–20 of the match date.
- Open-Meteo: at most one request per second, requests ≤ 14 days and ≤ 10 variables (each counts as one call on the free tier), cached and resumable.
- Walk-forward test years 2023, 2024, 2025, 2026; validation year = Y−1; 2,000 sims on validation, 10,000 on test.
- Betfair-derived data is licensed: never commit it; it reaches GitHub only as a private release asset.
- Run every command from the repo root `C:\Users\dtala\new-cricket-predictor`. Tests that need built data skip when it is missing. Keep test output pristine.

## File map

```
src/sim/simulate.py          dedupe identical prediction rows                       (Task 1)
src/sim/setup.py             simulate_setups (process pool), simulate_matches        (Task 1, 3)
src/cache.py                 cache key + load_or_build for prepared data            (Task 2)
src/features/outcomes.py     mens_t20_matches, competition_key, leaner load_balls   (Task 2, 3)
src/skill/shrink.py          + competition level                                    (Task 3)
src/features/form.py         post_match_ewm(..., halflife=)                         (Task 4)
src/features/context.py      toss, season, powerplay histories, match_context      (Task 3, 4, 5)
src/features/weather.py      venues, geocoding, download, start times, match weather (Task 5)
src/features/venue_overrides.csv  hand fixes for venue coordinates                 (Task 5)
src/model/frame.py           CONTEXT in static columns; build_frame(context)       (Task 3)
src/run_6over.py             prepare() builds the context table                    (Task 3, 5)
src/walkforward.py           one fold per call, pooled summary                      (Task 6)
src/pack_data.py             data bundle for GitHub                                 (Task 7)
.github/workflows/walkforward.yml                                                  (Task 7)
```

---

### Task 1: Faster simulation — deduplicated predictions and a process pool

**Files:**
- Modify: `src/sim/simulate.py` (the predict call inside `simulate`)
- Modify: `src/sim/setup.py` (`simulate_matches`; add `simulate_setups`, worker helpers)
- Test: `tests/test_speed.py`

**Interfaces:**
- Consumes: `MatchSetup`, `RunTable`, `run_table`, `simulate(setup, predict, table, n_sims, legal_balls, seed)` from `src/sim/simulate.py`; `BallModel` (`.name`, `.est`, `.predict(DataFrame)`).
- Produces:
  - `src.sim.simulate.DEDUPE: bool = True` (module flag, tests flip it).
  - `simulate_setups(setups: list[MatchSetup], model, table: RunTable, cols: list[str], n_sims: int, legal_balls: int = 36, workers: int = 1) -> list[np.ndarray]` — same order as `setups`, match i seeded with `i`.
  - `simulate_matches(rows, ctx, model, table: pd.DataFrame, n_sims, legal_balls=36, workers=1)` — builds setups serially, then calls `simulate_setups`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_speed.py
import numpy as np
import pandas as pd
import pytest

import src.sim.simulate as simmod
from src.features.outcomes import CLASSES
from src.sim.setup import simulate_setups
from src.sim.simulate import MatchSetup, run_table

COLS = ["s0", "over", "ball_in_over", "wickets", "runs", "batter_balls", "free_hit"]


def _table():
    rows = [(c, int(c), 0) for c in ["0", "1", "2", "3", "4", "6"]]
    rows += [("5plus", 5, 0), ("wicket", 0, 1), ("wide", 1, 0), ("noball", 1, 0), ("bye", 1, 0)]
    t = pd.DataFrame(rows, columns=["outcome", "runs", "wicket"])
    t["n"] = 1
    t["p"] = 1.0
    return run_table(t)


def _setup(shift):
    static = np.repeat(np.arange(11, dtype=float)[:, None, None], 11, axis=1) + shift
    return MatchSetup([f"b{i}" for i in range(11)], [f"w{i}" for i in range(11)],
                      np.ones(11), static)


class FakeModel:
    """Deterministic, row-dependent probabilities; picklable for the process pool."""
    name = "fake"
    est = None

    def __init__(self):
        self.rows_seen = 0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.rows_seen += len(X)
        a = X.to_numpy(float)
        z = np.sin(a @ np.arange(1, a.shape[1] + 1)[:, None] * np.arange(1, 12)[None, :])
        p = np.exp(z)
        p[:, CLASSES.index("wide")] *= 0.1
        return p / p.sum(axis=1, keepdims=True)


def test_dedupe_gives_identical_results_with_fewer_predictions(monkeypatch):
    setups, table = [_setup(0.0)], _table()
    m1 = FakeModel()
    monkeypatch.setattr(simmod, "DEDUPE", True)
    a = simulate_setups(setups, m1, table, COLS, n_sims=500)
    m2 = FakeModel()
    monkeypatch.setattr(simmod, "DEDUPE", False)
    b = simulate_setups(setups, m2, table, COLS, n_sims=500)
    np.testing.assert_array_equal(a[0], b[0])
    assert m1.rows_seen < m2.rows_seen, "identical early states are predicted once"


def test_parallel_equals_serial():
    setups, table = [_setup(s) for s in (0.0, 0.5, 1.0)], _table()
    serial = simulate_setups(setups, FakeModel(), table, COLS, n_sims=300, workers=1)
    parallel = simulate_setups(setups, FakeModel(), table, COLS, n_sims=300, workers=2)
    for s, p in zip(serial, parallel):
        np.testing.assert_array_equal(s, p)
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_speed.py -q`
Expected: FAIL — `ImportError: cannot import name 'simulate_setups'`.

- [ ] **Step 3: Implement dedupe in `src/sim/simulate.py`**

Add next to `MAX_DELIVERIES`:

```python
# Identical rows (same striker, bowler and situation) are predicted once. The
# probabilities are unchanged, so results are bit-identical; early overs, where
# most simulations share a state, get much cheaper.
DEDUPE = True
```

Replace the line `cls = sample_rows(predict(X), rng)` with:

```python
        if DEDUPE:
            uniq, inv = np.unique(X, axis=0, return_inverse=True)
            probs = predict(uniq)[inv.reshape(-1)]
        else:
            probs = predict(X)
        cls = sample_rows(probs, rng)
```

- [ ] **Step 4: Implement the pool in `src/sim/setup.py`**

Replace `simulate_matches` with:

```python
_WORKER: dict = {}


def _single_thread(model) -> None:
    """One model per process: stop each process's model from using every core."""
    if getattr(model, "name", "") in ("lightgbm", "xgboost"):
        model.est.set_params(n_jobs=1)


def _init_worker(model, table, cols, n_sims, legal_balls) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _single_thread(model)
    _WORKER.update(model=model, table=table, cols=cols, n_sims=n_sims, legal_balls=legal_balls)


def _run_one(item):
    i, setup = item
    w = _WORKER
    predict = lambda X: w["model"].predict(pd.DataFrame(X, columns=w["cols"]))  # noqa: E731
    return simulate(setup, predict, w["table"], w["n_sims"], w["legal_balls"], seed=i)


def simulate_setups(setups, model, table, cols, n_sims: int, legal_balls: int = 36,
                    workers: int = 1) -> list[np.ndarray]:
    """Simulate each setup; match i uses seed i, so any worker count gives the same output."""
    out: list[np.ndarray] = []
    if workers <= 1:
        predict = lambda X: model.predict(pd.DataFrame(X, columns=cols))  # noqa: E731
        items = (simulate(s, predict, table, n_sims, legal_balls, seed=i)
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
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_speed.py tests/test_simulate.py tests/test_setup.py -q`
Expected: PASS. Then the full suite: `python -m pytest -q` (133 + 2 passing).

- [ ] **Step 6: Commit**

```bash
git add src/sim/simulate.py src/sim/setup.py tests/test_speed.py
git commit -m "perf: dedupe identical prediction rows; simulate matches in a process pool"
```

---

### Task 2: Prepared-data cache and a leaner ball loader

**Files:**
- Create: `src/cache.py`
- Modify: `src/features/outcomes.py` (`load_balls` reads only men's T20 deliveries; add `mens_t20_matches`)
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces:
  - `mens_t20_matches() -> pd.DataFrame` — `matches.parquet` filtered to `match_type in {T20, IT20}`, `gender == "male"`, `balls_per_over == 6`.
  - `cache_key(cutoff: str, columns: list[str], files: list[Path]) -> str` — sha256 hex over cutoff, column names, and each file's path, size and content hash for `.py`/`.csv` files or mtime for others.
  - `load_or_build(cutoff: str, columns: list[str], files: list[Path], build, cache_dir: Path = CACHE_DIR)` — returns the pickled object if `{cutoff}-{key[:12]}.pkl` exists, else calls `build()`, pickles it (protocol 5) and returns it.
  - `CACHE_DIR = MODEL / "cache"`; `input_files() -> list[Path]` — `src/features/*.py`, `src/skill/*.py`, `src/model/frame.py`, `data/parsed/*.parquet`, `data/ticks/match_odds_markets.parquet`, `PLAYER_META_RDA`, plus `data/raw/weather/hourly.parquet`, `data/raw/weather/venues.csv` and `src/features/venue_overrides.csv` when they exist.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache.py
from src.cache import cache_key, load_or_build


def test_key_changes_with_columns_and_file_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1")
    k1 = cache_key("2024-01-01", ["a", "b"], [f])
    assert cache_key("2024-01-01", ["a", "b", "c"], [f]) != k1
    f.write_text("x = 2")
    assert cache_key("2024-01-01", ["a", "b"], [f]) != k1
    assert cache_key("2025-01-01", ["a", "b"], [f]) != cache_key("2024-01-01", ["a", "b"], [f])


def test_load_or_build_builds_once(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return {"v": 42}

    f = tmp_path / "a.py"
    f.write_text("x")
    a = load_or_build("2024-01-01", ["c"], [f], build, cache_dir=tmp_path / "cache")
    b = load_or_build("2024-01-01", ["c"], [f], build, cache_dir=tmp_path / "cache")
    assert a == b == {"v": 42}
    assert len(calls) == 1
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.cache'`.

- [ ] **Step 3: Implement `src/cache.py`**

```python
# src/cache.py
"""Cache prepared data per training cutoff, keyed by everything that shapes it.

The key covers the cutoff, the feature column list, and every input file:
source files by content, data files by size and modification time. Change any
of them and the cache rebuilds; delete data/model/cache to force a rebuild.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from src.config import MODEL, PARSED, PLAYER_META_RDA, ROOT, TICKS

CACHE_DIR = MODEL / "cache"


def input_files() -> list[Path]:
    files = sorted((ROOT / "src" / "features").glob("*.py")) + sorted(
        (ROOT / "src" / "skill").glob("*.py")) + [ROOT / "src" / "model" / "frame.py"]
    files += sorted(PARSED.glob("*.parquet")) + [TICKS / "match_odds_markets.parquet", PLAYER_META_RDA]
    for p in (ROOT / "data" / "raw" / "weather" / "hourly.parquet",
              ROOT / "data" / "raw" / "weather" / "venues.csv",
              ROOT / "src" / "features" / "venue_overrides.csv"):
        if p.exists():
            files.append(p)
    return files


def cache_key(cutoff: str, columns: list[str], files: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(cutoff.encode())
    h.update("\x1f".join(columns).encode())
    for f in files:
        f = Path(f)
        h.update(str(f).encode())
        if f.suffix in (".py", ".csv"):
            h.update(hashlib.sha256(f.read_bytes()).digest())
        else:
            st = f.stat()
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def load_or_build(cutoff: str, columns: list[str], files: list[Path], build,
                  cache_dir: Path = CACHE_DIR):
    path = Path(cache_dir) / f"{cutoff}-{cache_key(cutoff, columns, files)[:12]}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    tmp.replace(path)
    return obj
```

- [ ] **Step 4: Leaner `load_balls` in `src/features/outcomes.py`**

Add above `load_balls` and use it inside:

```python
def mens_t20_matches() -> pd.DataFrame:
    m = pd.read_parquet(PARSED / "matches.parquet")
    return m[m.match_type.isin(["T20", "IT20"]) & (m.gender == "male") & (m.balls_per_over == 6)]
```

In `load_balls`, replace the first two lines (reading and filtering matches) with `m = mens_t20_matches()`, and replace
`d = pd.read_parquet(PARSED / "deliveries.parquet")` + `d = d[d.match_id.isin(set(m.match_id))]` with

```python
    d = pd.read_parquet(PARSED / "deliveries.parquet",
                        filters=[("match_id", "in", m.match_id.tolist())])
```

(reads ~2.4M rows instead of 11.4M; same result).

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_cache.py tests/test_outcomes.py -q` (the real-data `load_balls` test must still pass with the same row-count bounds), then `python -m pytest -q`.

- [ ] **Step 6: Commit**

```bash
git add src/cache.py src/features/outcomes.py tests/test_cache.py
git commit -m "perf: cache prepared data per cutoff; load only men's T20 deliveries"
```

---

### Task 3: Toss, season and competition — the context table wired end to end

**Files:**
- Create: `src/features/context.py`
- Modify: `src/features/outcomes.py` (`competition_key`; `load_balls` adds `event_name`, `competition_key`)
- Modify: `src/skill/shrink.py` (`LEVELS` + competition)
- Modify: `src/model/frame.py` (`static_columns` includes `CONTEXT`; `build_frame(..., context)`)
- Modify: `src/sim/setup.py` (`Context.context`; `match_setup` broadcasts the row)
- Modify: `src/run_6over.py` (`prepare` builds context; `market_rows` adds `competition_key`)
- Test: `tests/test_context.py`; update `tests/test_frame.py`, `tests/test_shrink.py`

**Interfaces:**
- Consumes: `mens_t20_matches()` (Task 2).
- Produces:
  - `competition_key(event_name: pd.Series) -> pd.Series` (renames below; null → `"unknown"`).
  - `Level("competition", ("competition_key",), None)` in `LEVELS`, after `venue`.
  - `src.features.context`: `TOSS_SEASON = ["bat_won_toss", "chose_to_bat", "season"]`, `CONTEXT` (list; this task: `TOSS_SEASON`), `first_innings(balls) -> DataFrame` (index `match_id`; columns `date, batting_team, bowling_team, venue_key, competition_key`), `toss_and_season(first, matches) -> DataFrame`, `match_context(balls, matches) -> DataFrame` (index `match_id`, columns `CONTEXT`, float32).
  - `static_columns(levels) = skill_columns(levels) + FORM + CONTEXT`.
  - `build_frame(balls, skill, bat_form, bowl_form, context)`.
  - `Context(skill, styles, bat_form, bowl_form, bat_pos, pp, xi, context)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_context.py
import numpy as np
import pandas as pd

from src.features.context import CONTEXT, first_innings, toss_and_season
from src.features.outcomes import competition_key


def _balls():
    return pd.DataFrame({
        "match_id": ["m1", "m1", "m2"], "innings_num": [1, 2, 1],
        "date": pd.to_datetime(["2024-05-02", "2024-05-02", "2024-12-31"]),
        "batting_team": ["A", "B", "C"], "bowling_team": ["B", "A", "D"],
        "venue_key": ["V", "V", "W"], "competition_key": ["X", "X", "Y"]})


def test_first_innings_is_one_row_per_match_for_the_side_batting_first():
    f = first_innings(_balls())
    assert list(f.index) == ["m1", "m2"]
    assert f.loc["m1", "batting_team"] == "A"


def test_toss_flags_separate_chose_to_bat_from_put_in():
    matches = pd.DataFrame({"match_id": ["m1", "m2"], "toss_winner": ["A", "D"],
                            "toss_decision": ["bat", "field"]})
    t = toss_and_season(first_innings(_balls()), matches)
    assert t.loc["m1", "bat_won_toss"] == 1.0 and t.loc["m1", "chose_to_bat"] == 1.0
    assert t.loc["m2", "bat_won_toss"] == 0.0 and t.loc["m2", "chose_to_bat"] == 0.0


def test_season_is_year_plus_fraction():
    t = toss_and_season(first_innings(_balls()),
                        pd.DataFrame({"match_id": ["m1", "m2"], "toss_winner": ["A", "C"],
                                      "toss_decision": ["bat", "bat"]}))
    assert 2024.0 < t.loc["m1", "season"] < 2024.5
    assert 2024.99 < t.loc["m2", "season"] < 2025.0


def test_competition_key_merges_sponsor_renames():
    k = competition_key(pd.Series(["NatWest T20 Blast", "Vitality Blast", "Vitality Blast Men",
                                   "Indian Premier League", None]))
    assert k.tolist() == ["T20 Blast", "T20 Blast", "T20 Blast", "Indian Premier League", "unknown"]


def test_context_starts_with_toss_and_season():
    assert CONTEXT[:3] == ["bat_won_toss", "chose_to_bat", "season"]
```

Update `tests/test_frame.py`: replace every hard-coded level count. The feature-count test becomes
`assert len(feature_columns(LEVELS)) == len(LEVELS) * 11 + 11 + len(FORM) + len(CONTEXT) + len(SITUATION)` (import `CONTEXT` from `src.features.context`), and the `all_skill` length becomes `len(LEVELS) * 11 + 11`. `test_build_frame_on_two_matches` passes a context table: `pd.DataFrame(0.0, index=["m1", "m2"], columns=CONTEXT)` and asserts the frame's columns equal `feature_columns(levels) + ["y", "date", "match_id"]` (unchanged assertion).
Update `tests/test_shrink.py::test_levels_are_the_spec_hierarchy`: the set includes `"competition"`, `by["competition"].parent is None`, `by["competition"].group_cols == ("competition_key",)`.

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_context.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.features.context'`.

- [ ] **Step 3: Implement**

In `src/features/outcomes.py`:

```python
# Sponsor renames of the same competition. Checked against the 40 most frequent
# event names (2026-10-06); extend when a new rename appears.
COMPETITION_RENAMES = {
    "NatWest T20 Blast": "T20 Blast",
    "Vitality Blast": "T20 Blast",
    "Vitality Blast Men": "T20 Blast",
}


def competition_key(event_name: pd.Series) -> pd.Series:
    return event_name.replace(COMPETITION_RENAMES).fillna("unknown").astype(str)
```

In `load_balls`, merge `event_name` with the other match columns
(`d.merge(m[["match_id", "date", "venue", "city", "event_name"]], on="match_id")`) and add `d["competition_key"] = competition_key(d["event_name"])` next to `venue_key`.

Before writing the rename map, list the 40 most frequent `event_name` values among `mens_t20_matches()` and add any other sponsor rename of one competition you find to `COMPETITION_RENAMES`; record the list and any additions in your report.

In `src/skill/shrink.py` `LEVELS`, add after the `venue` level:
`Level("competition", ("competition_key",), None),`

Create `src/features/context.py`:

```python
# src/features/context.py
"""Match-level context known at the toss, one row per match for the side batting first.

Joined onto every first-innings ball in the training frame and broadcast into the
simulator's static features, so training and simulation see identical values.
History-based columns are read strictly before the match date with `as_of`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOSS_SEASON = ["bat_won_toss", "chose_to_bat", "season"]
CONTEXT = TOSS_SEASON


def first_innings(balls: pd.DataFrame) -> pd.DataFrame:
    b = balls[balls.innings_num == 1]
    cols = ["date", "batting_team", "bowling_team", "venue_key", "competition_key"]
    return b.drop_duplicates("match_id").set_index("match_id")[cols]


def toss_and_season(first: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.set_index("match_id").reindex(first.index)
    d = pd.to_datetime(first.date)
    return pd.DataFrame({
        "bat_won_toss": (m.toss_winner == first.batting_team).astype(float),
        "chose_to_bat": (m.toss_decision == "bat").astype(float),
        "season": d.dt.year + (d.dt.dayofyear - 1) / 366,
    }, index=first.index)


def match_context(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    first = first_innings(balls)
    out = toss_and_season(first, matches)
    return out[CONTEXT].astype(np.float32)
```

In `src/model/frame.py`: `from src.features.context import CONTEXT`; `static_columns` returns `skill_columns(levels) + FORM + CONTEXT`; `build_frame` gains a `context: pd.DataFrame` parameter (index `match_id`) and adds

```python
    ctx = pd.DataFrame(context.reindex(b.match_id.to_numpy())[CONTEXT].to_numpy(np.float32),
                       index=b.index, columns=CONTEXT)
```

to the `pd.concat` list (between form and situation); the final column selection `[feature_columns(skill.levels)]` stays.

In `src/sim/setup.py`: `Context` gains a last field `context: pd.DataFrame`; import `CONTEXT`; in `match_setup`, after building `pairs`, add

```python
    crow = ctx.context.loc[row.match_id, CONTEXT].to_numpy(np.float32)
    cblock = pd.DataFrame(np.tile(crow, (len(pairs), 1)), index=pairs.index, columns=CONTEXT)
```

and include `cblock` in the `pd.concat([...], axis=1)` that builds `static`, before selecting `static_columns(ctx.skill.levels)`.

In `src/run_6over.py`: `market_rows` adds `r["competition_key"] = competition_key(r.event_name)`; `prepare` builds `context = match_context(balls, mens_t20_matches())`, passes it to `build_frame(..., context)` and sets `Context(..., context=context)`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_context.py tests/test_frame.py tests/test_shrink.py -q`, then `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/features/context.py src/features/outcomes.py src/skill/shrink.py src/model/frame.py src/sim/setup.py src/run_6over.py tests/test_context.py tests/test_frame.py tests/test_shrink.py
git commit -m "feat: match context table (toss, season) and a competition shrinkage level"
```

---

### Task 4: Recent powerplay history — league, competition, venue, team, opposition

**Files:**
- Modify: `src/features/form.py` (`post_match_ewm` gets a `halflife` parameter)
- Modify: `src/features/context.py`
- Test: `tests/test_context.py` (append)

**Interfaces:**
- Consumes: `as_of`, `post_match_ewm`, `NOT_LEGAL`, `is_wicket`, `first_innings`.
- Produces:
  - `post_match_ewm(per, key, cols, halflife=HALFLIFE)` (default unchanged).
  - `HISTORY = ["league_pp_recent", "comp_pp_recent", "venue_pp_recent", "venue_pp_n", "team_bat_pp", "opp_bat_pp", "opp_bowl_pp_conceded"]`; `CONTEXT = TOSS_SEASON + HISTORY`.
  - `innings_powerplay(balls) -> DataFrame[match_id, innings_num, date, batting_team, bowling_team, venue_key, competition_key, pp_runs]` (valid innings only).
  - `history_features(balls, first) -> DataFrame` (index `first.index`, columns `HISTORY`).

- [ ] **Step 1: Write the failing tests (append to `tests/test_context.py`)**

```python
from src.features.context import HISTORY, history_features, innings_powerplay
from src.features.form import HALFLIFE, post_match_ewm
from src.features.outcomes import label_outcomes


def _innings_balls(match_id, innings, date, bat, bowl, venue, comp, runs_per_ball,
                   n_legal=36, wides=0, wkts=0):
    rows = []
    for i in range(wides):
        rows.append({"wide": 1.0, "runs_batter": 0, "runs_total": 1, "wicket_kind": None})
    for i in range(n_legal):
        wk = "bowled" if i < wkts else None
        rows.append({"wide": np.nan, "runs_batter": runs_per_ball, "runs_total": runs_per_ball,
                     "wicket_kind": wk})
    d = pd.DataFrame(rows)
    for c in ("noball", "bye", "legbye"):
        d[c] = np.nan
    d["outcome"] = label_outcomes(d)
    return d.assign(match_id=match_id, innings_num=innings, date=pd.Timestamp(date),
                    batting_team=bat, bowling_team=bowl, venue_key=venue, competition_key=comp)


def test_powerplay_counts_36_legal_balls_and_drops_rain_cut_innings():
    full = _innings_balls("m1", 1, "2024-01-01", "A", "B", "V", "X", 1, n_legal=40, wides=2)
    cut = _innings_balls("m2", 1, "2024-01-02", "A", "B", "V", "X", 1, n_legal=20)
    out_early = _innings_balls("m3", 1, "2024-01-03", "A", "B", "V", "X", 1, n_legal=20, wkts=10)
    pp = innings_powerplay(pd.concat([full, cut, out_early], ignore_index=True))
    by = pp.set_index("match_id").pp_runs
    assert by["m1"] == 36 + 2, "36 legal singles plus the 2 wides bowled inside them"
    assert "m2" not in by.index, "rain-cut: fewer than 36 legal balls and not all out"
    assert by["m3"] == 20, "bowled out inside the powerplay counts"


def test_post_match_ewm_honours_halflife():
    per = pd.DataFrame({"k": ["p", "p"], "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                        "v": [2.0, 0.0]})
    w = 0.5 ** (1 / 5)
    out = post_match_ewm(per, "k", ["v"], halflife=5)
    assert out.v.iloc[1] == pytest.approx(2 * w / (1 + w))
    assert post_match_ewm(per, "k", ["v"]).v.iloc[1] == pytest.approx(
        2 * 0.5 ** (1 / HALFLIFE) / (1 + 0.5 ** (1 / HALFLIFE)))


def test_history_is_point_in_time_and_keyed_correctly():
    balls = pd.concat([
        _innings_balls("m1", 1, "2024-01-01", "A", "B", "V", "X", 1),   # A bats: 36
        _innings_balls("m1", 2, "2024-01-01", "B", "A", "V", "X", 2),   # B bats: 72
        _innings_balls("m2", 1, "2024-01-01", "C", "D", "W", "X", 3),   # same day, other venue
        _innings_balls("m3", 1, "2024-02-01", "A", "B", "V", "X", 1),   # later match
    ], ignore_index=True)
    h = history_features(balls, first_innings(balls))
    assert h.loc["m1"].isna()[["team_bat_pp", "venue_pp_recent", "league_pp_recent"]].all(), \
        "nothing before the first date"
    assert h.loc["m1", "venue_pp_n"] == 0
    assert np.isnan(h.loc["m2", "comp_pp_recent"]), "same-day matches never see each other"
    m3 = h.loc["m3"]
    assert m3.team_bat_pp == 36, "A's own batting powerplay"
    assert m3.opp_bat_pp == 72, "B's batting powerplay"
    assert m3.opp_bowl_pp_conceded == 36, "B's bowlers conceded 36 to A in m1"
    assert m3.venue_pp_recent == pytest.approx((36 + 72) / 2), "both innings at V on one date, averaged"
    assert m3.venue_pp_n == 2
    assert m3.comp_pp_recent == pytest.approx((36 + 108) / 2), "first innings only, both Jan 1 matches"
    assert m3.league_pp_recent == pytest.approx((36 + 108) / 2)
```

(`m2`'s first innings is 36 balls × 3 = 108.) Add `import pytest` to the file's imports, and `first_innings` to the existing `from src.features.context import ...` line.

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_context.py -q`
Expected: FAIL — `ImportError: cannot import name 'HISTORY'`.

- [ ] **Step 3: Implement**

`src/features/form.py`: change the signature to `def post_match_ewm(per, key, cols, halflife: float = HALFLIFE)` and use `halflife=halflife` in the `.ewm(...)` call.

`src/features/context.py` — add imports `from src.features.asof import as_of`, `from src.features.form import post_match_ewm`, `from src.features.outcomes import NOT_LEGAL, is_wicket`, then:

```python
HISTORY = ["league_pp_recent", "comp_pp_recent", "venue_pp_recent", "venue_pp_n",
           "team_bat_pp", "opp_bat_pp", "opp_bowl_pp_conceded"]
CONTEXT = TOSS_SEASON + HISTORY

PP_LEGAL_BALLS = 36
LEAGUE_WINDOW_DAYS = 90
HALFLIFE_TEAM = 5
HALFLIFE_VENUE = 5
HALFLIFE_COMP = 20
_INNINGS_COLS = ["date", "batting_team", "bowling_team", "venue_key", "competition_key"]


def innings_powerplay(balls: pd.DataFrame) -> pd.DataFrame:
    """Runs in the first 36 legal balls; rain-cut innings (short and not all out) dropped."""
    key = [balls.match_id, balls.innings_num]
    legal = (~balls.outcome.isin(NOT_LEGAL)).astype(int)
    in_pp = (legal.groupby(key).cumsum() - legal) < PP_LEGAL_BALLS
    b = balls.loc[in_pp, ["match_id", "innings_num", *_INNINGS_COLS, "runs_total"]].assign(
        legal=legal[in_pp].to_numpy(), wkt=is_wicket(balls)[in_pp].astype(int).to_numpy())
    agg = {c: (c, "first") for c in _INNINGS_COLS}
    g = b.groupby(["match_id", "innings_num"], as_index=False).agg(
        **agg, pp_runs=("runs_total", "sum"), legal=("legal", "sum"), wkts=("wkt", "sum"))
    keep = (g.legal == PP_LEGAL_BALLS) | (g.wkts >= 10)
    return g.loc[keep, ["match_id", "innings_num", *_INNINGS_COLS, "pp_runs"]].reset_index(drop=True)


def _ewm_table(pp: pd.DataFrame, key: str, halflife: float, name: str) -> pd.DataFrame:
    per = pp.groupby([key, "date"], as_index=False).pp_runs.mean().rename(columns={key: "key"})
    return post_match_ewm(per, "key", ["pp_runs"], halflife=halflife).rename(columns={"pp_runs": name})


def _read(table: pd.DataFrame, keys: pd.Series, dates: pd.Series, col: str) -> pd.Series:
    q = pd.DataFrame({"key": keys.to_numpy(), "date": dates.to_numpy()}, index=keys.index)
    return as_of(table, q, ["key"], [col])[col]


def history_features(balls: pd.DataFrame, first: pd.DataFrame) -> pd.DataFrame:
    pp = innings_powerplay(balls)
    f1 = pp[pp.innings_num == 1]
    out = pd.DataFrame(index=first.index)

    daily = f1.groupby("date").pp_runs.agg(["sum", "count"]).sort_index().cumsum().reset_index()
    daily["key"] = "all"
    all_key = pd.Series("all", index=first.index)
    now = as_of(daily, pd.DataFrame({"key": all_key, "date": first.date}), ["key"], ["sum", "count"]).fillna(0)
    then = as_of(daily, pd.DataFrame({"key": all_key,
                                      "date": first.date - pd.Timedelta(days=LEAGUE_WINDOW_DAYS)}),
                 ["key"], ["sum", "count"]).fillna(0)
    n = now["count"] - then["count"]
    out["league_pp_recent"] = (now["sum"] - then["sum"]) / n.where(n > 0)

    out["comp_pp_recent"] = _read(_ewm_table(f1, "competition_key", HALFLIFE_COMP, "v"),
                                  first.competition_key, first.date, "v")
    out["venue_pp_recent"] = _read(_ewm_table(pp, "venue_key", HALFLIFE_VENUE, "v"),
                                   first.venue_key, first.date, "v")
    cnt = pp.groupby(["venue_key", "date"]).size().rename("n").reset_index().rename(
        columns={"venue_key": "key"})
    cnt["v"] = cnt.groupby("key").n.cumsum()
    out["venue_pp_n"] = _read(cnt, first.venue_key, first.date, "v").fillna(0)

    bat = _ewm_table(pp, "batting_team", HALFLIFE_TEAM, "v")
    out["team_bat_pp"] = _read(bat, first.batting_team, first.date, "v")
    out["opp_bat_pp"] = _read(bat, first.bowling_team, first.date, "v")
    out["opp_bowl_pp_conceded"] = _read(_ewm_table(pp, "bowling_team", HALFLIFE_TEAM, "v"),
                                        first.bowling_team, first.date, "v")
    return out[HISTORY]
```

In `match_context`, join the history: `out = out.join(history_features(balls, first))` before `return out[CONTEXT].astype(np.float32)`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_context.py tests/test_features.py tests/test_frame.py -q`, then `python -m pytest -q`.

- [ ] **Step 5: Real-data check (paste output in the report)**

```bash
python -c "
from src.features.outcomes import load_balls, mens_t20_matches
from src.features.context import match_context
c = match_context(load_balls(), mens_t20_matches())
print(c.shape); print(c.isna().mean().round(3).to_dict()); print(c.describe().round(2).T)
"
```
Expected: one row per men's T20 match (~10,500); toss/season never NaN; history columns NaN mostly for the earliest matches only; `league_pp_recent` around 40–55.

- [ ] **Step 6: Commit**

```bash
git add src/features/form.py src/features/context.py tests/test_context.py
git commit -m "feat: recent powerplay history - league, competition, venue, team, opposition"
```

---

### Task 5: Weather and start time

**Files:**
- Create: `src/features/weather.py`, `src/features/venue_overrides.csv` (header only: `venue_key,latitude,longitude,timezone,country`)
- Modify: `src/features/context.py` (`WEATHER` in `CONTEXT`; `match_context(..., weather=None)`)
- Modify: `src/run_6over.py` (`prepare` loads match weather)
- Test: `tests/test_weather.py`

**Interfaces:**
- Produces:
  - `VARS`, `WEATHER = VARS + ["start_hour_local", "is_night"]`, `CONTEXT = TOSS_SEASON + HISTORY + WEATHER`.
  - `venue_cities(matches) -> DataFrame` (index `venue_key`; `city`, `n_matches`).
  - `geocode(name, get=requests.get) -> dict | None` (`latitude, longitude, timezone, country`).
  - `build_venues(matches, get=requests.get, pause=PAUSE_S) -> DataFrame` (writes `VENUES_CSV`).
  - `date_windows(dates, max_days=14) -> list[tuple[date, date]]`.
  - `fetch_window(lat, lon, start, end, get=requests.get) -> DataFrame[time, utc_offset_seconds, *VARS]`.
  - `download_weather(venues, matches, get=requests.get, pause=PAUSE_S) -> DataFrame` (writes `HOURLY`).
  - `start_times() -> pd.Series` (index `match_id`, tz-aware UTC).
  - `match_weather(mv, hourly, starts) -> DataFrame` (index `match_id`, columns `WEATHER`); `mv` has `match_id, date, venue_key`.
  - `load_match_weather(balls) -> DataFrame` (raises `FileNotFoundError` if `HOURLY` is missing).
  - `match_context(balls, matches, weather=None)`: when `weather is None`, the `WEATHER` columns are NaN.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_weather.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features import weather as W


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)


def test_geocode_parses_the_top_result():
    get = lambda url, params, timeout: FakeResponse({"results": [  # noqa: E731
        {"latitude": 19.0, "longitude": 72.8, "timezone": "Asia/Kolkata", "country": "India"}]})
    assert W.geocode("Mumbai", get=get) == {"latitude": 19.0, "longitude": 72.8,
                                             "timezone": "Asia/Kolkata", "country": "India"}
    assert W.geocode("Nowhere", get=lambda u, params, timeout: FakeResponse({})) is None


def test_date_windows_never_exceed_14_days():
    d = [dt.date(2024, 1, 1), dt.date(2024, 1, 5), dt.date(2024, 1, 14), dt.date(2024, 1, 15),
         dt.date(2024, 3, 1)]
    w = W.date_windows(d)
    assert w == [(dt.date(2024, 1, 1), dt.date(2024, 1, 14)),
                 (dt.date(2024, 1, 15), dt.date(2024, 1, 15)),
                 (dt.date(2024, 3, 1), dt.date(2024, 3, 1))]


def _hourly():
    times = pd.date_range("2024-05-02 00:00", periods=24, freq="h")
    h = pd.DataFrame({"venue_key": "V", "time": times, "utc_offset_seconds": 19800})
    for i, v in enumerate(W.VARS):
        h[v] = np.arange(24, dtype=float) + 100 * i
    return h


def test_match_weather_uses_the_start_hour_when_known_else_the_day_window():
    mv = pd.DataFrame({"match_id": ["a", "b"], "date": pd.to_datetime(["2024-05-02"] * 2),
                       "venue_key": ["V", "V"]})
    starts = pd.Series(pd.to_datetime(["2024-05-02 14:00"], utc=True), index=["a"])  # 19:30 local
    out = W.match_weather(mv, _hourly(), starts)
    assert out.loc["a", "temperature_2m"] == 19.0, "local 19:30 floors to the 19:00 row"
    assert out.loc["a", "start_hour_local"] == pytest.approx(19.5)
    assert out.loc["a", "is_night"] == 1.0
    assert out.loc["b", "temperature_2m"] == pytest.approx(np.mean(range(14, 21)))
    assert np.isnan(out.loc["b", "start_hour_local"]) and np.isnan(out.loc["b", "is_night"])


def test_venue_without_coordinates_gets_nan_weather():
    mv = pd.DataFrame({"match_id": ["c"], "date": pd.to_datetime(["2024-05-02"]), "venue_key": ["Z"]})
    out = W.match_weather(mv, _hourly(), pd.Series(dtype="datetime64[ns, UTC]"))
    assert out.loc["c"].isna().all()


def test_download_skips_cached_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(W, "HOURLY", tmp_path / "hourly.parquet")
    calls = []

    def get(url, params, timeout):
        calls.append(params["start_date"])
        n = 24
        return FakeResponse({"utc_offset_seconds": 0, "hourly": {
            "time": [f"{params['start_date']}T{h:02d}:00" for h in range(n)],
            **{v: [1.0] * n for v in W.VARS}}})

    venues = pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}, index=pd.Index(["V"], name="venue_key"))
    matches = pd.DataFrame({"venue_key": ["V"], "date": pd.to_datetime(["2024-05-02"])})
    W.download_weather(venues, matches, get=get, pause=0)
    W.download_weather(venues, matches, get=get, pause=0)
    assert calls == ["2024-05-02"], "second run reuses the cache"
    assert pd.read_parquet(tmp_path / "hourly.parquet").venue_key.unique().tolist() == ["V"]
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_weather.py -q`
Expected: FAIL — `ImportError: cannot import name 'weather'`.

- [ ] **Step 3: Implement `src/features/weather.py`**

```python
# src/features/weather.py
"""Venue weather from Open-Meteo and match start times from Betfair.

    python -m src.features.weather      geocode venues, then download (cached, resumable)

Weather values are historical observations, standing in for the forecast a
bettor has at the toss. Requests stay inside Open-Meteo's free tier: <= 14 days
and <= 10 variables each (one call apiece), at most one per second.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests

from src.config import ROOT, TICKS
from src.features.outcomes import mens_t20_matches, venue_key

WEATHER_DIR = ROOT / "data" / "raw" / "weather"
CACHE = WEATHER_DIR / "cache"
VENUES_CSV = WEATHER_DIR / "venues.csv"
HOURLY = WEATHER_DIR / "hourly.parquet"
OVERRIDES = ROOT / "src" / "features" / "venue_overrides.csv"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
        "wind_speed_10m", "precipitation"]
WEATHER = VARS + ["start_hour_local", "is_night"]
NIGHT_HOUR = 17
DAY_HOURS = (14, 20)
PAUSE_S = 1.0
MAX_RETRIES = 5


def venue_cities(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.assign(venue_key=venue_key(matches.venue, matches.city))
    n = m.groupby("venue_key").size().rename("n_matches")
    city = m.dropna(subset=["city"]).groupby("venue_key").city.agg(lambda s: s.value_counts().index[0])
    out = pd.DataFrame(n).join(city.rename("city"))
    fallback = pd.Series(out.index.str.split("|").str[0], index=out.index)
    out["city"] = out.city.fillna(fallback)
    return out.sort_values("n_matches", ascending=False)


def geocode(name: str, get=requests.get) -> dict | None:
    r = get(GEOCODE_URL, params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=30)
    r.raise_for_status()
    res = r.json().get("results") or []
    if not res:
        return None
    top = res[0]
    return {"latitude": top["latitude"], "longitude": top["longitude"],
            "timezone": top.get("timezone"), "country": top.get("country")}


def build_venues(matches: pd.DataFrame, get=requests.get, pause: float = PAUSE_S) -> pd.DataFrame:
    vc = venue_cities(matches)
    over = (pd.read_csv(OVERRIDES).set_index("venue_key") if OVERRIDES.exists()
            else pd.DataFrame(columns=["latitude", "longitude", "timezone", "country"]))
    old = pd.read_csv(VENUES_CSV).set_index("venue_key") if VENUES_CSV.exists() else None
    rows = []
    for vk, r in vc.iterrows():
        rec = {"venue_key": vk, "city": r.city, "n_matches": int(r.n_matches)}
        if vk in over.index:
            rec.update(over.loc[vk].to_dict(), source="override")
        elif old is not None and vk in old.index and pd.notna(old.loc[vk, "latitude"]) \
                and old.loc[vk, "source"] != "override":
            rec.update(old.loc[vk, ["latitude", "longitude", "timezone", "country", "source"]].to_dict())
        else:
            g, src = geocode(r.city, get), "geocode:city"
            name = vk.split("|")[0]
            if g is None and name != r.city:
                g, src = geocode(name, get), "geocode:venue"
            rec.update(g or {}, source=src if g else "unresolved")
            time.sleep(pause)
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("venue_key")
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(VENUES_CSV)
    return out


def date_windows(dates, max_days: int = 14):
    ds = sorted(set(pd.to_datetime(pd.Series(list(dates))).dt.date))
    out, start, end = [], None, None
    for d in ds:
        if start is None:
            start = end = d
        elif (d - start).days < max_days:
            end = d
        else:
            out.append((start, end))
            start = end = d
    if start is not None:
        out.append((start, end))
    return out


def fetch_window(lat, lon, start, end, get=requests.get) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(end),
              "hourly": ",".join(VARS), "timezone": "auto"}
    for attempt in range(1, MAX_RETRIES + 1):
        r = get(ARCHIVE_URL, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(60 * attempt)
            continue
        r.raise_for_status()
        p = r.json()
        h = pd.DataFrame(p["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h["utc_offset_seconds"] = int(p.get("utc_offset_seconds", 0))
        return h[["time", "utc_offset_seconds", *VARS]]
    raise RuntimeError(f"Open-Meteo kept failing for {lat},{lon} {start}..{end}")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def download_weather(venues: pd.DataFrame, matches: pd.DataFrame, get=requests.get,
                     pause: float = PAUSE_S) -> pd.DataFrame:
    """`venues` indexed by venue_key with latitude/longitude; `matches` has venue_key, date."""
    CACHE.mkdir(parents=True, exist_ok=True)
    have = venues.dropna(subset=["latitude", "longitude"])
    for vk, v in have.iterrows():
        for start, end in date_windows(matches.loc[matches.venue_key == vk, "date"]):
            path = CACHE / f"{_slug(vk)}_{start}_{end}.parquet"
            if path.exists():
                continue
            h = fetch_window(v.latitude, v.longitude, start, end, get).assign(venue_key=vk)
            h.to_parquet(path)
            time.sleep(pause)
    parts = [pd.read_parquet(p) for p in sorted(CACHE.glob("*.parquet"))]
    hourly = (pd.concat(parts, ignore_index=True).drop_duplicates(["venue_key", "time"])
              if parts else pd.DataFrame(columns=["venue_key", "time", "utc_offset_seconds", *VARS]))
    hourly.to_parquet(HOURLY)
    return hourly


def start_times() -> pd.Series:
    mo = pd.read_parquet(TICKS / "match_odds_markets.parquet", columns=["match_id", "market_time"])
    mo["market_time"] = pd.to_datetime(mo.market_time, utc=True)
    mo = mo.dropna(subset=["match_id", "market_time"])
    mo = mo[mo.market_time.dt.year < 2030]          # 2099 placeholder start times
    return mo.groupby("match_id").market_time.min()


def match_weather(mv: pd.DataFrame, hourly: pd.DataFrame, starts: pd.Series) -> pd.DataFrame:
    m = mv[["match_id", "date", "venue_key"]].copy().reset_index(drop=True)
    m["day"] = pd.to_datetime(m.date).dt.normalize()
    h = hourly.assign(time=pd.to_datetime(hourly.time))
    h = h.assign(day=h.time.dt.normalize(), hour=h.time.dt.hour)
    off = h.groupby(["venue_key", "day"]).utc_offset_seconds.first().rename("off").reset_index()
    m = m.merge(off, on=["venue_key", "day"], how="left")
    start = pd.Series(starts.reindex(m.match_id).to_numpy(), index=m.index)
    start = pd.to_datetime(start, utc=True).dt.tz_localize(None)
    local = start + pd.to_timedelta(m.off, unit="s")
    m["start_hour_local"] = local.dt.hour + local.dt.minute / 60
    m["is_night"] = (m.start_hour_local >= NIGHT_HOUR).astype(float).where(m.start_hour_local.notna())
    m["time"] = local.dt.floor("h")
    at = m[["venue_key", "time"]].merge(h[["venue_key", "time", *VARS]],
                                        on=["venue_key", "time"], how="left")[VARS]
    win = h[h.hour.between(*DAY_HOURS)].groupby(["venue_key", "day"])[VARS].mean().reset_index()
    day = m[["venue_key", "day"]].merge(win, on=["venue_key", "day"], how="left")[VARS]
    have = at.notna().all(axis=1).to_numpy()
    vals = at.where(np.repeat(have[:, None], len(VARS), axis=1), day)
    out = pd.concat([vals, m[["start_hour_local", "is_night"]]], axis=1)
    out.index = m.match_id
    return out[WEATHER].astype(float)


def load_match_weather(balls: pd.DataFrame) -> pd.DataFrame:
    if not HOURLY.exists():
        raise FileNotFoundError(f"{HOURLY} missing: run `python -m src.features.weather` first")
    mv = balls[["match_id", "date", "venue_key"]].drop_duplicates("match_id")
    return match_weather(mv, pd.read_parquet(HOURLY), start_times())


if __name__ == "__main__":
    matches = mens_t20_matches()
    venues = build_venues(matches)
    unresolved = venues[venues.latitude.isna()]
    print(f"{len(venues)} venues, {len(unresolved)} unresolved:", unresolved.index.tolist()[:40])
    print(venues.head(60)[["city", "country", "latitude", "longitude", "source", "n_matches"]]
          .to_string())
    mv = matches.assign(venue_key=venue_key(matches.venue, matches.city))[["venue_key", "date"]]
    hourly = download_weather(venues, mv)
    print(f"hourly rows: {len(hourly):,}")
```

Create `src/features/venue_overrides.csv` with the single header line `venue_key,latitude,longitude,timezone,country`.

- [ ] **Step 4: Wire weather into the context**

`src/features/context.py`: `from src.features.weather import WEATHER`; `CONTEXT = TOSS_SEASON + HISTORY + WEATHER`; `match_context(balls, matches, weather: pd.DataFrame | None = None)`: after the history join,

```python
    if weather is None:
        out[WEATHER] = np.nan
    else:
        out = out.join(weather.reindex(out.index)[WEATHER])
```

`src/run_6over.py` `prepare`: `context = match_context(balls, mens_t20_matches(), load_match_weather(balls))` (import `load_match_weather` from `src.features.weather`).

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_weather.py tests/test_context.py tests/test_frame.py -q`, then `python -m pytest -q`. Do NOT run `python -m src.features.weather` (network); the controller runs it and reviews the venue list.

- [ ] **Step 6: Commit**

```bash
git add src/features/weather.py src/features/venue_overrides.csv src/features/context.py src/run_6over.py tests/test_weather.py
git commit -m "feat: venue weather from Open-Meteo and start time from Betfair"
```

- [ ] **Step 7 (controller): download and check**

Run `python -m src.features.weather` (about 1–2 hours, resumable). Read the printed top-60 venues: every country must be plausible for the ground (e.g. `Hamilton` should be New Zealand for Seddon Park, `Providence` should be Guyana). Put corrections in `src/features/venue_overrides.csv`, delete `data/raw/weather/venues.csv` rows you corrected, rerun (cached windows are reused, but corrected venues refetch — delete their cache files `data/raw/weather/cache/{slug}_*.parquet`). Record the share of men's T20 matches with non-NaN `temperature_2m` and commit the overrides file.

---

### Task 6: Walk-forward driver

**Files:**
- Create: `src/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `prepare(cutoff)` (returns `balls, ctx, frame, table`), `market_rows`-style filtering, `fit` and `MODELS` from ballmodel, `simulate_matches(..., workers=)`, `fit_calibration`, `apply`, `summary`, `per_match`, `choose_model`, `backtest`, `load_or_build`, `input_files`, `feature_columns`, `LEVELS`, `venue_key`, `competition_key`, `comparison_set`.
- Produces:
  - `YEARS = (2023, 2024, 2025, 2026)`, `OUT = REPORTS_DIR / "walkforward"`.
  - `year_rows(year: int) -> DataFrame` — usable men's 6-over market rows dated in `year`, sorted by date, with `venue_key`, `competition_key`.
  - `prepare_cached(cutoff) -> (ctx, frame, table)`.
  - `run_fold(year, workers=4, n_val=2000, n_test=10000, smoke=False, out_dir=OUT) -> dict` — writes `{year}.json` and `{year}_{model}_matches.parquet`.
  - `summarize(out_dir=OUT, years=YEARS) -> dict` — writes `summary.json`.
  - CLI `python -m src.walkforward --year Y [--workers N] [--n-sims-val N] [--n-sims-test N] [--smoke]` and `python -m src.walkforward --summary`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_walkforward.py
import json

import numpy as np
import pandas as pd
import pytest

from src.config import MARKET
from src.walkforward import summarize


def _write_year(d, year, chosen, n=40, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.normal(50, 12, n)
    df = pd.DataFrame({"match_id": [f"{year}-{i}" for i in range(n)], "line": 50.5,
                       "actual": actual, "cal_median": actual + rng.normal(0, 9, n),
                       "side": ["over"] * (n // 2) + [None] * (n - n // 2),
                       "pnl": [0.95] * (n // 4) + [-1.0] * (n // 4) + [0.0] * (n - n // 2)})
    df.to_parquet(d / f"{year}_{chosen}_matches.parquet")
    (d / f"{year}.json").write_text(json.dumps({"year": year, "chosen": chosen}))


def test_summarize_pools_chosen_models_and_splits_clean_years(tmp_path):
    _write_year(tmp_path, 2023, "logistic", seed=1)
    _write_year(tmp_path, 2024, "lightgbm", seed=2)
    _write_year(tmp_path, 2025, "logistic", seed=3)
    s = summarize(out_dir=tmp_path, years=(2023, 2024, 2025, 2026))
    assert set(s["per_year"]) == {"2023", "2024", "2025"}
    assert s["pooled"]["n_matches"] == 120
    assert s["pooled_2023_2024"]["n_matches"] == 80
    p = s["pooled"]
    assert p["n_bets"] == 60 and p["roi"] == pytest.approx((0.95 * 30 - 30) / 60)
    assert p["roi_se"] > 0
    assert (tmp_path / "summary.json").exists()


@pytest.mark.skipif(not (MARKET / "overline_comparison.parquet").exists(), reason="market tables not built")
def test_year_rows_match_the_known_2024_count():
    from src.walkforward import year_rows
    r = year_rows(2024)
    assert len(r) == 435
    assert (r.date.dt.year == 2024).all()
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_walkforward.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.walkforward'`.

- [ ] **Step 3: Implement `src/walkforward.py`**

```python
# src/walkforward.py
"""Walk-forward test: for each test year Y, choose and calibrate on Y-1, score Y.

    python -m src.walkforward --year 2024 [--workers 4] [--n-sims-val 2000] [--n-sims-test 10000] [--smoke]
    python -m src.walkforward --summary

One year per call, so each year can be its own GitHub Actions job. 2025-2026
were seen before the context features were designed; the summary reports the
pooled result with and without them (2023-2024 is the cleaner evidence).
"""
from __future__ import annotations

import gc
import json
import sys

import numpy as np
import pandas as pd

from src.bet.backtest import backtest
from src.cache import input_files, load_or_build
from src.config import REPORTS_DIR
from src.features.outcomes import competition_key, venue_key
from src.market.baseline import comparison_set
from src.model.ballmodel import MODELS, fit as fit_model
from src.model.frame import feature_columns
from src.run_6over import SIX, choose_model, per_match, prepare
from src.sim.calibrate import apply, fit_calibration, summary
from src.sim.setup import simulate_matches
from src.skill.shrink import LEVELS

YEARS = (2023, 2024, 2025, 2026)
OUT = REPORTS_DIR / "walkforward"
SMOKE_TRAIN_ROWS = 200_000
SMOKE_MATCHES = 5


def year_rows(year: int) -> pd.DataFrame:
    r = comparison_set("all")
    r = r[r.market_type == SIX].copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r[r.date.dt.year == year]
    r["venue_key"] = venue_key(r.venue, r.city)
    r["competition_key"] = competition_key(r.event_name)
    return r.sort_values("date", kind="stable").reset_index(drop=True)


def prepare_cached(cutoff: str):
    def build():
        balls, ctx, frame, table = prepare(cutoff)
        return ctx, frame, table
    return load_or_build(cutoff, feature_columns(LEVELS), input_files(), build)


def _fit_all(frame, cutoff, models, cols, smoke):
    tr = frame.date < cutoff
    train = frame.loc[tr]
    if smoke:
        train = train.tail(SMOKE_TRAIN_ROWS)
    return {n: fit_model(n, train, train.y.to_numpy(), train.date, cols) for n in models}


def run_fold(year: int, workers: int = 4, n_val: int = 2000, n_test: int = 10000,
             smoke: bool = False, out_dir=OUT) -> dict:
    out_dir = out_dir / "smoke" if smoke else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    models = ("logistic",) if smoke else MODELS
    val_cut, test_cut = f"{year - 1}-01-01", f"{year}-01-01"
    val_rows, test_rows = year_rows(year - 1), year_rows(year)
    if smoke:
        val_rows, test_rows = val_rows.head(SMOKE_MATCHES), test_rows.head(SMOKE_MATCHES)

    # Phase 1: fit on < Y-1, simulate Y-1, calibrate each model, choose one.
    ctx, frame, table = prepare_cached(val_cut)
    cols = feature_columns(ctx.skill.levels)
    fitted = _fit_all(frame, val_cut, models, cols, smoke)
    del frame
    gc.collect()
    ys = val_rows.runs_at_6ov.to_numpy(float)
    val, cals = {}, {}
    for n, m in fitted.items():
        sims = simulate_matches(val_rows, ctx, m, table, n_val, workers=workers)
        val[n] = {"raw": summary(sims, ys)}
        try:
            cals[n] = fit_calibration(sims, ys)
            val[n]["calibrated"] = summary([apply(cals[n], s) for s in sims], ys)
        except ValueError as err:
            val[n]["calibrated"] = None
            val[n]["calibration_error"] = str(err)
        print(year, "val", n, val[n], flush=True)
    if not cals:
        raise SystemExit(f"{year}: no model could be calibrated on {year - 1}")
    chosen = choose_model(val)
    del ctx, fitted
    gc.collect()

    # Phase 2: refit on < Y, simulate Y, apply each model's Y-1 calibration.
    ctx, frame, table = prepare_cached(test_cut)
    fitted = _fit_all(frame, test_cut, [n for n in models if n in cals], cols, smoke)
    del frame
    gc.collect()
    ys = test_rows.runs_at_6ov.to_numpy(float)
    lines = test_rows.pre_match_line.to_numpy(float)
    report = {"year": year, "chosen": chosen, "smoke": smoke, "n_val_sims": n_val,
              "n_test_sims": n_test, "n_val_matches": len(val_rows),
              "n_test_matches": len(test_rows),
              "market_mae": float(np.mean(np.abs(lines - ys))), "val": val, "test": {}}
    for n in [chosen] + [k for k in fitted if k != chosen]:
        raw = simulate_matches(test_rows, ctx, fitted[n], table, n_test, workers=workers)
        sims = [apply(cals[n], s) for s in raw]
        per_match(test_rows, raw, sims).to_parquet(out_dir / f"{year}_{n}_matches.parquet")
        report["test"][n] = {"role": "chosen" if n == chosen else "comparison",
                             "calibration": cals[n].__dict__, "raw": summary(raw, ys),
                             "calibrated": summary(sims, ys), "backtest": backtest(sims, lines, ys)}
        (out_dir / f"{year}.json").write_text(json.dumps(report, indent=2, default=float))
        print(year, "test", n, report["test"][n]["calibrated"], flush=True)
    return report


def _stats(df: pd.DataFrame) -> dict:
    bets = df[df.side.notna()]
    n = len(bets)
    return {
        "n_matches": int(len(df)),
        "mae_model": float(np.mean(np.abs(df.cal_median - df.actual))),
        "mae_market": float(np.mean(np.abs(df.line - df.actual))),
        "n_bets": int(n),
        "hit_rate": float((bets.pnl > 0).mean()) if n else float("nan"),
        "roi": float(bets.pnl.mean()) if n else float("nan"),
        "roi_se": float(bets.pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
    }


def summarize(out_dir=OUT, years=YEARS) -> dict:
    frames = {}
    for y in years:
        meta = out_dir / f"{y}.json"
        if not meta.exists():
            continue
        chosen = json.loads(meta.read_text())["chosen"]
        path = out_dir / f"{y}_{chosen}_matches.parquet"
        if path.exists():
            frames[y] = pd.read_parquet(path)
    if not frames:
        raise SystemExit(f"no fold results in {out_dir}")
    clean = [frames[y] for y in frames if y in (2023, 2024)]
    out = {"per_year": {str(y): _stats(f) for y, f in frames.items()},
           "pooled": _stats(pd.concat(frames.values(), ignore_index=True)),
           "pooled_2023_2024": _stats(pd.concat(clean, ignore_index=True)) if clean else None}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def _arg(argv, name, default, cast=int):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--summary" in argv:
        print(json.dumps(summarize(), indent=2, default=float))
    elif "--year" in argv:
        rep = run_fold(_arg(argv, "--year", None), workers=_arg(argv, "--workers", 4),
                       n_val=_arg(argv, "--n-sims-val", 2000),
                       n_test=_arg(argv, "--n-sims-test", 10000), smoke="--smoke" in argv)
        print(json.dumps({k: rep[k] for k in ("year", "chosen", "market_mae")}, default=float))
    else:
        raise SystemExit("usage: python -m src.walkforward --year Y [--workers N] "
                         "[--n-sims-val N] [--n-sims-test N] [--smoke] | --summary")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_walkforward.py -q`, then `python -m pytest -q`.

- [ ] **Step 5: Smoke run (paste output and wall-clock time in the report)**

Run in the background (it builds two cached prepares, ~30–40 min the first time):
`python -m src.walkforward --year 2024 --smoke --workers 2 --n-sims-val 200 --n-sims-test 200`
Expected: `data/reports/walkforward/smoke/2024.json` and `2024_logistic_matches.parquet` exist; the JSON shows 5 val and 5 test matches; no errors. Run it a second time and confirm it starts simulating within a couple of minutes (cache hit).

- [ ] **Step 6: Commit**

```bash
git add src/walkforward.py tests/test_walkforward.py
git commit -m "feat: walk-forward test - one fold per call, pooled summary"
```

---

### Task 7: Data bundle and the GitHub Actions workflow

**Files:**
- Create: `src/pack_data.py`, `.github/workflows/walkforward.yml`
- Modify: `.gitignore` (add `data_bundle.tar.gz`), `requirements.txt` (`lightgbm>=4.7`), `README.md` (a "Walk-forward on GitHub" section)
- Test: `tests/test_pack_data.py`

**Interfaces:**
- Produces: `BUNDLE_PATTERNS: list[str]`, `bundle_files(root=ROOT) -> list[Path]` (raises `FileNotFoundError` naming any pattern with no match), `pack(out=ROOT / "data_bundle.tar.gz") -> Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pack_data.py
import pytest

from src.pack_data import BUNDLE_PATTERNS, bundle_files


def test_missing_pattern_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/parsed"):
        bundle_files(root=tmp_path)


def test_bundle_covers_what_the_pipeline_reads():
    joined = " ".join(BUNDLE_PATTERNS)
    for needed in ("data/parsed", "data/market", "data/ticks", "player_meta.rda",
                   "hourly.parquet", "venues.csv"):
        assert needed in joined
```

- [ ] **Step 2: Run to see it fail**

Run: `python -m pytest tests/test_pack_data.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pack_data'`.

- [ ] **Step 3: Implement `src/pack_data.py`**

```python
# src/pack_data.py
"""Pack the ~80 MB of processed tables the pipeline reads into one tarball.

    python -m src.pack_data      writes data_bundle.tar.gz at the repo root

Betfair-derived tables are licensed: the bundle goes to GitHub only as a
release asset on this private repository, never into git history.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

from src.config import ROOT

BUNDLE_PATTERNS = [
    "data/parsed/*.parquet",
    "data/market/*.parquet",
    "data/ticks/*.parquet",
    "data/raw/cricsheet/player_meta.rda",
    "data/raw/weather/hourly.parquet",
    "data/raw/weather/venues.csv",
]


def bundle_files(root: Path = ROOT) -> list[Path]:
    files = []
    for pat in BUNDLE_PATTERNS:
        hits = sorted(Path(root).glob(pat))
        if not hits:
            raise FileNotFoundError(f"no files match {pat} under {root}")
        files += hits
    return files


def pack(out: Path = ROOT / "data_bundle.tar.gz") -> Path:
    with tarfile.open(out, "w:gz") as tar:
        for f in bundle_files():
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    pack()
```

- [ ] **Step 4: Write `.github/workflows/walkforward.yml`**

```yaml
name: walk-forward

on:
  workflow_dispatch:
    inputs:
      years:
        description: "Test years, comma-separated"
        default: "2023,2024,2025,2026"
        required: true
      n_sims_test:
        description: "Simulations per test match"
        default: "10000"
        required: true
      data_tag:
        description: "Release tag holding data_bundle.tar.gz"
        default: "data-v1"
        required: true

permissions:
  contents: read

jobs:
  years:
    runs-on: ubuntu-latest
    outputs:
      list: ${{ steps.y.outputs.list }}
    steps:
      - id: y
        run: echo "list=[${{ inputs.years }}]" >> "$GITHUB_OUTPUT"

  fold:
    needs: years
    runs-on: ubuntu-latest
    timeout-minutes: 355
    strategy:
      fail-fast: false
      matrix:
        year: ${{ fromJSON(needs.years.outputs.list) }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Fetch data bundle (private release asset)
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release download "${{ inputs.data_tag }}" --pattern data_bundle.tar.gz
          tar -xzf data_bundle.tar.gz
      - name: Run fold
        run: python -u -m src.walkforward --year ${{ matrix.year }} --workers 2 --n-sims-test ${{ inputs.n_sims_test }}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: walkforward-${{ matrix.year }}
          path: data/reports/walkforward/

  summary:
    needs: fold
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
          cache: pip
      - run: pip install -r requirements.txt
      - uses: actions/download-artifact@v4
        with:
          pattern: walkforward-*
          merge-multiple: true
          path: data/reports/walkforward/
      - run: python -m src.walkforward --summary
      - uses: actions/upload-artifact@v4
        with:
          name: walkforward-summary
          path: data/reports/walkforward/summary.json
```

- [ ] **Step 5: `.gitignore`, requirements, README**

- `.gitignore`: add a line `data_bundle.tar.gz`.
- `requirements.txt`: change the LightGBM line to `lightgbm>=4.7` (the code uses `eval_X`/`eval_y`).
- `README.md`: add a section "## Walk-forward on GitHub" after the PIT section, explaining: `python -m src.pack_data` builds the bundle; it is uploaded once as release `data-v1` (`gh release create data-v1 data_bundle.tar.gz --notes "Processed tables (licensed; private)"`); the run is started from the Actions tab → "walk-forward" → Run workflow; each year is one job (~4–5 h on GitHub's 2-core runners), results are downloadable artifacts, and `walkforward-summary` holds the pooled numbers. Note the free private-repo quota (2,000 minutes a month) means about one full run a month.

- [ ] **Step 6: Run the tests, build and verify the bundle**

Run `python -m pytest -q`. Then `python -m src.pack_data` and verify it in a clean checkout:

```bash
git worktree add ../ncp-bundle-check HEAD
tar -xzf data_bundle.tar.gz -C ../ncp-bundle-check
cd ../ncp-bundle-check && python -c "from src.walkforward import year_rows; print(len(year_rows(2024)))" && cd -
git worktree remove --force ../ncp-bundle-check
```
Expected: `435`. Paste the bundle size and this output in the report. Do NOT upload or push; the controller does that.

- [ ] **Step 7: Commit**

```bash
git add src/pack_data.py .github/workflows/walkforward.yml .gitignore requirements.txt README.md tests/test_pack_data.py
git commit -m "feat: data bundle and GitHub Actions walk-forward workflow"
```

---

### Task 8 (controller): Publish and launch

- [ ] Merge the branch into `main` (the workflow must be on the default branch to appear in the Actions tab) and push.
- [ ] `gh release create data-v1 data_bundle.tar.gz --repo dtalati24/new-cricket-predictor --notes "Processed tables (licensed; private)"`.
- [ ] Start one short check run: years `2024`, `n_sims_test` `200`. Confirm the job installs, fetches the bundle, completes and uploads its artifact; read its log for memory or time problems. If `pip install` fails on Python 3.14, change both `python-version` lines to `"3.13"`, commit, push and retry.
- [ ] Tell the user the full run is ready to start from the Actions tab (default inputs), and what it costs in free minutes.

---

## Self-review notes

**Spec coverage.** §1.1 toss → Task 3. §1.2 season, league window, competition level, competition EWM → Tasks 3–4. §1.3 venue/team/opposition → Task 4. §1.4 start time, weather, venue geocoding, overrides, download → Task 5 (overrides live at `src/features/venue_overrides.csv` instead of under `data/`, because `data/` is gitignored; same content, versioned as the spec intends). §2 speed: free memory (Task 6 drops the frame before simulating; Task 2 reads only men's T20 deliveries), parallel matches and dedupe (Task 1), cache (Task 2). §3 walk-forward → Task 6, with the honesty split in `summarize`. §4 testing → each task's tests. §5 order → task order.

**Interface consistency.** `CONTEXT` grows in Tasks 3 → 4 → 5; tests compute counts from `len(CONTEXT)` and `len(LEVELS)`, so they stay valid. `simulate_matches(..., workers=)` (Task 1) is what Task 6 calls. `prepare` keeps its 4-tuple; `prepare_cached` drops `balls`. `choose_model`, `per_match`, `SIX` are imported from `src.run_6over`.

**Known limits.** GitHub's private-repo runners have 2 cores and ~7 GB RAM; if a fold runs out of memory, the fallback is a rented machine. Weather is observed, not forecast, so it slightly flatters what a bettor knew at the toss.
