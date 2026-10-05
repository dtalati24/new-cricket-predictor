"""Paths shared across the build.

Raw inputs (~5.7 GB of Cricsheet JSON and the Betfair BASIC archive) and every
derived table live under `data/` and are gitignored -- they are reproducible by
re-running the pipeline. `data/.gitkeep` holds the directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- raw inputs
RAW_BETFAIR = ROOT / "data" / "raw" / "betfair" / "BASIC"
RAW_CRICSHEET = ROOT / "data" / "raw" / "cricsheet"

# ------------------------------------------------------------------- outputs
PARSED = ROOT / "data" / "parsed"      # Cricsheet -> normalised parquet
TICKS = ROOT / "data" / "ticks"        # Betfair price paths and definitions
MARKET = ROOT / "data" / "market"      # settled labels, line comparisons
SKILL = ROOT / "data" / "skill"        # shrunk hierarchy estimates
MODEL = ROOT / "data" / "model"        # trained models, calibration fits
REPORTS = ROOT / "data" / "reports"    # measured numbers, gate evidence
FEATURES = ROOT / "data" / "features"  # form, player style, assembled feature frames

MATCH_ODDS_TICKS = TICKS / "match_odds_ticks.parquet"
MATCH_ODDS_MARKETS = TICKS / "match_odds_markets.parquet"
MATCH_ODDS_DEFS = TICKS / "match_odds_defs.parquet"
# Live tables this repo builds (the pipeline inputs ticks.py draws from).
LIVE_MARKET_MANIFEST = TICKS / "market_manifest.parquet"
LIVE_EVENT_MATCH = MARKET / "event_match.parquet"
LIVE_MATCHES = PARSED / "matches.parquet"

# ------------------------------------------------ aliases for ported Build 2
# Modules ported from archive/2026-09-build/src/ import these names unchanged,
# so they are aliased onto the constants above rather than redefined.
PARSED_DIR = PARSED
MARKET_DIR = MARKET
REPORTS_DIR = REPORTS
# Not data/betfair (Build 2's location): this repo already carries the Betfair
# tick tables, populated, at data/ticks. One copy of that data, not two.
BETFAIR_DIR = TICKS
BETFAIR_RAW_DIR = RAW_BETFAIR
FEATURES_DIR = FEATURES      # features/form.py, features/player_style.py
MODELS_DIR = MODEL           # model/ trained artefacts, calibration fits
CRICSHEET_RAW_DIR = RAW_CRICSHEET
CRICSHEET_ZIP = CRICSHEET_RAW_DIR / "all_json.zip"
CRICSHEET_MATCHES_DIR = CRICSHEET_RAW_DIR / "matches"
CRICSHEET_PEOPLE_CSV = CRICSHEET_RAW_DIR / "people.csv"
CRICSHEET_NAMES_CSV = CRICSHEET_RAW_DIR / "names.csv"
PLAYER_META_RDA = CRICSHEET_RAW_DIR / "player_meta.rda"  # skill/player_style

_DERIVED = (PARSED_DIR, BETFAIR_DIR, MARKET_DIR, SKILL, MODEL, REPORTS_DIR, FEATURES)


def ensure_dirs() -> None:
    """Create every derived-output directory. Never touches raw inputs."""
    for d in _DERIVED:
        d.mkdir(parents=True, exist_ok=True)


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

# The unconditional-mean MAE for T20 first-innings totals, measured over the
# full Cricsheet corpus (n=14,002, mean 149.1, sd 41.8). Any model or market
# must beat this to be worth anything.
UNCONDITIONAL_T20_FIRST_INNINGS_MAE = 33.4

# Betfair market types this project targets. Over-lines are the PRIMARY target
# (single runner "Total Runs", ltp is the traded line IN RUNS, not odds).
OVER_LINE_MARKET_TYPES = (
    "1_INNING_6_OVR_LINE",
    "1_INNING_10_OVR_LINE",
    "1_INNING_15_OVR_LINE",
    "1_INNING_20_OVR_LINE",
)
# Ladder of binary "X Runs or more" rungs; hc = -X. Secondary / distribution check.
# NOTE: only ~16% of markets TYPED 1ST_INNINGS_RUNS are actually this ladder
# (bettingType == "ASIAN_HANDICAP_SINGLE_LINE"). Always route on bettingType,
# never on market_type.
LADDER_MARKET_TYPES = ("1ST_INNINGS_RUNS",)

# DROPPED, and it must stay dropped. The other ~84% of 1ST_INNINGS_RUNS markets
# are bettingType == "LINE" with a single "Total Runs" runner, and a single
# event carries SEVERAL of them with wildly different lines (verified: 41.5,
# 12.5 and 140.5 on one T20I at the same marketTime). Betfair's Basic tier
# strips `marketName`, so there is no field that says which is the innings
# total and which is a powerplay or single-over line. Comparing a model's
# innings-total forecast against an unidentified line would manufacture pure
# fake edge, so these are excluded rather than guessed at.
AMBIGUOUS_LINE_MARKET_TYPES = ("1ST_INNINGS_RUNS",)
# Sharp, liquid benchmark used to validate the price-extraction method itself.
BENCHMARK_MARKET_TYPES = ("MATCH_ODDS",)

TARGET_MARKET_TYPES = OVER_LINE_MARKET_TYPES + LADDER_MARKET_TYPES

# The single runner id Betfair uses for the "Total Runs" runner on LINE markets.
TOTAL_RUNS_RUNNER_ID = 15316

# --- over-line data-quality gates (measured at Gate 4, see reports/) --------
# Catastrophic line errors were traced to markets with a single pre-match
# trade at an absurd value (999.5 and 2.5 both appeared as "15 over" lines on
# real T20s). Both filters below use ONLY pre-match information, so they are
# legitimate at bet time.
#
# Caveat that must travel with any filtered result: n_ticks_pre correlates
# +0.17 with the actual score, so a tick-filtered sample skews toward
# higher-scoring fixtures. It is a valid like-for-like model-vs-market
# comparison set, but it is NOT a representative sample of all fixtures.
MIN_PRE_MATCH_TICKS = 2

# Cricket-plausible line range per checkpoint. Uses no outcome information --
# we simply would not bet into a line outside these bounds.
OVER_LINE_PLAUSIBLE_BAND = {
    "1_INNING_6_OVR_LINE": (15.0, 110.0),
    "1_INNING_10_OVR_LINE": (25.0, 160.0),
    "1_INNING_15_OVR_LINE": (40.0, 220.0),
    "1_INNING_20_OVR_LINE": (50.0, 300.0),
}

# EXCLUDED from modelling targets. Measured at Gate 4: after both filters the
# 15-over line has corr 0.222 and R^2 of -0.576 against its own settlement --
# genuinely worse than predicting a constant. Only 21.0% of these markets ever
# trade pre-match and 56% of those carry a single tick. The checkpoint mapping
# was verified correct (its line still predicts runs_at_15ov better than any
# other checkpoint), so this is a market that barely functions pre-match, not
# a pipeline bug. Kept in the tables and flagged, never silently dropped.
UNRELIABLE_OVER_LINE_TYPES = ("1_INNING_15_OVR_LINE",)

# Only men's T20 is modelled. The measured bar pools men's and women's, whose
# first-innings scores differ by ~19% (6-over mean 49.1 vs 40.0) and whose time
# trends DIVERGE: men's 6-over market MAE rose 8.36 -> 10.56 across 2022-2026
# while women's stayed flat. A model scored against the pooled bar could look
# good by fitting one population. Restricting to men's costs sample -- the 2025+
# holdout falls from 1,012 rows to 764, which raises the 6-over detectable gain
# from 0.72 to 0.85 -- and that trade was made deliberately. See docs/gates.md.
MODELLING_GENDER = "male"

MODELLING_OVER_LINE_TYPES = tuple(
    m for m in OVER_LINE_MARKET_TYPES if m not in UNRELIABLE_OVER_LINE_TYPES
)

# How many balls each over-line market covers, keyed by market type. Expressed
# in OVERS; multiply by the match's own balls_per_over rather than assuming 6.
OVER_LINE_OVERS = {
    "1_INNING_6_OVR_LINE": 6,
    "1_INNING_10_OVR_LINE": 10,
    "1_INNING_15_OVR_LINE": 15,
    "1_INNING_20_OVR_LINE": 20,
}
