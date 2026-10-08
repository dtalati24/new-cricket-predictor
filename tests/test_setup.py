from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from src.features.context import CONTEXT
from src.features.outcomes import CLASSES
from src.model.frame import skill_columns, static_columns
from src.sim.setup import Context, check_xi, match_setup
from src.skill.counts import cumulative_counts
from src.skill.shrink import LEVELS, SkillModel


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


def test_match_setup_builds_every_level_and_broadcasts_context():
    """Full LEVELS: a group column missing from `pairs` (e.g. competition_key) raises here."""
    hist = pd.DataFrame({
        "batter_id": ["a1", "a1", "a2", "a2"], "bowler_id": ["b1", "b2", "b1", "b2"],
        "bowler_type": ["pace", "spin", "pace", "spin"], "batting_hand": ["right", "right", "left", "left"],
        "venue_key": "V", "competition_key": "X", "phase": "pp", "date": pd.Timestamp("2024-01-01"),
        "outcome": pd.Categorical(["4", "0", "1", "wicket"], categories=CLASSES)})
    skill = SkillModel(np.full(11, 1 / 11), {lv.name: 5.0 for lv in LEVELS},
                       {lv.name: cumulative_counts(hist, list(lv.group_cols)) for lv in LEVELS},
                       LEVELS, phase_league={"pp": np.full(11, 1 / 11)})
    day = pd.Timestamp("2024-01-01")
    styles = pd.DataFrame({"player_id": ["a1", "a2", "b1", "b2"],
                           "bowler_type": ["none", "none", "pace", "spin"],
                           "batting_hand": ["right", "left", "right", "right"]})
    form = lambda cols: pd.DataFrame({"player_id": ["a1", "a2", "b1", "b2"], "date": day,  # noqa: E731
                                      **{c: 1.0 for c in cols}})
    xi = pd.DataFrame({"match_id": "m", "date": pd.Timestamp("2024-02-01"),
                       "team": ["A", "A", "B", "B"], "player_id": ["a1", "a2", "b1", "b2"],
                       "position": [1, 2, 1, 2]})
    context = pd.DataFrame([[1.0, 2024.5, *range(2, len(CONTEXT))]], index=["m"], columns=CONTEXT).astype(np.float32)
    ctx = Context(skill=skill, styles=styles,
                  bat_form=form(["bat_form_sr", "bat_form_out", "bat_pp_sr", "bat_pp_out"]),
                  bowl_form=form(["bowl_form_econ", "bowl_form_wkt", "bowl_pp_econ", "bowl_pp_wkt"]),
                  bat_pos=form(["bat_pos"]), pp=form(["pp_overs"]), xi=xi, context=context)
    Row = namedtuple("Row", "match_id date batting_team bowling_team venue_key competition_key")
    s = match_setup(Row("m", "2024-02-01", "A", "B", "V", "X"), ctx)
    cols = static_columns(LEVELS)
    assert s.static.shape == (2, 2, len(cols))
    ci = [cols.index(c) for c in CONTEXT]
    assert (s.static[:, :, ci] == context.loc["m", CONTEXT].to_numpy(np.float32)).all()
    assert not np.isnan(s.static[:, :, :len(skill_columns(LEVELS))]).any()
    fi = [cols.index(c) for c in ("bat_pp_sr", "bowl_pp_econ")]
    assert (s.static[:, :, fi] == 1.0).all(), "powerplay form reaches the simulator"
