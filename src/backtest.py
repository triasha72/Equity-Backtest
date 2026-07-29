"""Walk-forward backtest and portfolio construction.

The evaluation protocol here is deliberately different from k-fold. Observations
in a financial panel are not exchangeable across time: the ordering *is* the
problem. Shuffling, or fitting on any month whose label had not yet been
realised, leaks the future into the training set. This module uses an expanding
window instead.

At each formation month t:
  * training pairs are (features at s, realised return over s+1) for all s+1 <= t
  * the model predicts the cross-section of t+1 returns from features at t
  * nothing observed after t enters the fit
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["mom_12_1", "reversal", "vol_12m", "liquidity"]


@dataclass(frozen=True)
class Config:
    decile: float = 0.10          # fraction of the cross-section in each leg
    cost_bps: float = 10.0        # charged on every dollar traded
    min_train_months: int = 36
    min_names: int = 50
    ridge_alpha: float = 1.0
    features: tuple[str, ...] = tuple(FEATURES)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["features"] = ",".join(self.features)
        return d


def _make_model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _weights(pred: pd.Series, decile: float) -> pd.Series:
    """Dollar-neutral decile spread: +0.5 gross long, -0.5 gross short."""
    n = max(int(round(len(pred) * decile)), 1)
    ranked = pred.sort_values(ascending=False)
    longs, shorts = ranked.index[:n], ranked.index[-n:]
    w = pd.Series(0.0, index=pred.index)
    w[longs] = 0.5 / n
    w[shorts] = -0.5 / n
    return w


def run(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a per-month frame: gross return, turnover, cost, net return."""
    feats = list(cfg.features)
    dates = panel.index.get_level_values("date").unique().sort_values()

    rows, prev_w = [], pd.Series(dtype=float)

    for i, t in enumerate(dates):
        if i < cfg.min_train_months:
            continue

        # Labels are only known for formation months strictly before t.
        train = panel[panel.index.get_level_values("date") < dates[i - 1]]
        train = train.dropna(subset=["target"])
        cur = panel.xs(t, level="date")

        if len(train) < cfg.min_names * 2 or len(cur) < cfg.min_names:
            continue
        if cur["target"].isna().all():
            continue

        model = _make_model(cfg.ridge_alpha)
        model.fit(train[feats].values, train["target"].values)
        pred = pd.Series(model.predict(cur[feats].values), index=cur.index)

        w = _weights(pred, cfg.decile)
        realised = cur["target"].reindex(w.index).fillna(0.0)
        gross = float((w * realised).sum())

        aligned_prev = prev_w.reindex(w.index).fillna(0.0)
        turnover = float((w - aligned_prev).abs().sum())
        cost = turnover * cfg.cost_bps / 1e4

        rows.append(
            {"date": t, "gross": gross, "turnover": turnover,
             "cost": cost, "net": gross - cost, "n_names": len(cur)}
        )
        prev_w = w

    return pd.DataFrame(rows).set_index("date")


def benchmark(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight return of the same universe, for comparison."""
    return panel.groupby(level="date")["target"].mean().dropna()
