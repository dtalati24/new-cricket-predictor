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


def test_recency_weights_halve_every_halflife():
    from src.model.ballmodel import recency_weights
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
    np.testing.assert_allclose(recency_weights(dates, 1.0), [0.25, 0.5, 1.0], atol=0.002)
    assert recency_weights(dates, None) is None


@pytest.mark.parametrize("name", MODELS)
def test_short_halflife_follows_the_recent_regime(name):
    X, y, dates = _data()
    recent = np.arange(len(y)) >= 0.7 * len(y)
    y = np.where(recent, 1 - (X.signal > 0), y)       # relationship flips for the last 30%
    y[:11] = np.arange(11)
    q = X.tail(300)
    flat = fit(name, X, y, dates, ["signal", "noise"]).predict(q)
    weighted = fit(name, X, y, dates, ["signal", "noise"], halflife=0.005).predict(q)
    new = (1 - (q.signal > 0)).to_numpy()
    hit = lambda p: p[np.arange(len(q)), new].mean()
    assert hit(weighted) > hit(flat) + 0.1
