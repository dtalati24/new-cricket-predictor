"""Per-ball 11-class models: multinomial logistic (the floor), LightGBM, XGBoost.

The boosted models early-stop on the most recent 10% of their own training
rows, so no evaluation year is touched while fitting. An optional recency
halflife (years) down-weights older training balls.
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


def recency_weights(dates: pd.Series, halflife: float | None) -> np.ndarray | None:
    """Weight 1 for the latest training ball, halving every `halflife` years before it."""
    if halflife is None:
        return None
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    age = (d.max() - d) / np.timedelta64(1, "D") / 365.25
    return 0.5 ** (age / halflife)


def fit(name: str, X: pd.DataFrame, y: np.ndarray, dates: pd.Series,
        cols: list[str], halflife: float | None = None) -> BallModel:
    X = X[cols].reset_index(drop=True)
    y = np.asarray(y)
    w = recency_weights(dates, halflife)
    sub = lambda idx: None if w is None else w[idx] / w[idx].mean()   # mean 1: regularisation keeps its scale
    if name == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=500))
        est.fit(X, y, logisticregression__sample_weight=sub(slice(None)))
    elif name == "lightgbm":
        import lightgbm as lgb
        a, b = _recent_split(dates)
        est = lgb.LGBMClassifier(objective="multiclass", n_estimators=3000, learning_rate=0.05,
                                 num_leaves=63, min_child_samples=500, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        est.fit(X.iloc[a], y[a], sample_weight=sub(a),
                eval_X=X.iloc[b], eval_y=y[b],
                callbacks=[lgb.early_stopping(50, verbose=False)])
    elif name == "xgboost":
        import xgboost as xgb
        a, b = _recent_split(dates)
        est = xgb.XGBClassifier(objective="multi:softprob", n_estimators=3000,
                                learning_rate=0.05, max_depth=6, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist",
                                early_stopping_rounds=50, eval_metric="mlogloss")
        est.fit(X.iloc[a], y[a], sample_weight=sub(a),
                eval_set=[(X.iloc[b], y[b])], verbose=False)
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
