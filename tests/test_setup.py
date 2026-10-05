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
