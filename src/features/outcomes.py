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


def mens_t20_matches() -> pd.DataFrame:
    m = pd.read_parquet(PARSED / "matches.parquet")
    return m[m.match_type.isin(["T20", "IT20"]) & (m.gender == "male") & (m.balls_per_over == 6)]


def load_balls() -> pd.DataFrame:
    """Men's T20/IT20, 6-ball overs, innings 1-2, in delivery order, labelled."""
    m = mens_t20_matches()
    inn = pd.read_parquet(PARSED / "innings.parquet")
    inn = inn[inn.innings_num.isin([1, 2]) & ~inn.is_super_over]
    d = pd.read_parquet(PARSED / "deliveries.parquet",
                        filters=[("match_id", "in", m.match_id.tolist())])
    d = d.merge(inn[["match_id", "innings_num", "batting_team", "bowling_team"]],
                on=["match_id", "innings_num"])
    d = d.merge(m[["match_id", "date", "venue", "city"]], on="match_id")
    d["date"] = pd.to_datetime(d["date"])
    d["venue_key"] = venue_key(d["venue"], d["city"])
    d["outcome"] = label_outcomes(d)
    d = d.sort_values(["date", "match_id", "innings_num", "over", "ball_seq"], kind="stable")
    return d.reset_index(drop=True)
