"""Performance statistics.

Reported gross *and* net. A signal that only works gross of costs is not a
signal, it is a description of the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def _sharpe(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float("nan") if sd == 0 else float(r.mean() / sd * np.sqrt(MONTHS))


def _tstat(r: pd.Series) -> float:
    sd = r.std(ddof=1)
    return float("nan") if sd == 0 else float(r.mean() / (sd / np.sqrt(len(r))))


def _max_drawdown(r: pd.Series) -> float:
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def summarise(bt: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    g, n = bt["gross"], bt["net"]
    out = {
        "months": len(bt),
        "start": str(bt.index.min().date()),
        "end": str(bt.index.max().date()),
        "sharpe_gross": round(_sharpe(g), 3),
        "sharpe_net": round(_sharpe(n), 3),
        "ann_return_net": round(float(n.mean() * MONTHS), 4),
        "ann_vol_net": round(float(n.std(ddof=1) * np.sqrt(MONTHS)), 4),
        "tstat_net": round(_tstat(n), 3),
        "max_drawdown_net": round(_max_drawdown(n), 4),
        "hit_rate_net": round(float((n > 0).mean()), 3),
        "avg_monthly_turnover": round(float(bt["turnover"].mean()), 3),
        "avg_monthly_cost_bps": round(float(bt["cost"].mean() * 1e4), 2),
    }
    if bench is not None and len(bench):
        b = bench.reindex(bt.index).dropna()
        out["benchmark_sharpe_ew"] = round(_sharpe(b), 3) if len(b) > 1 else None
    return out


def report(stats: dict) -> str:
    width = max(len(k) for k in stats)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in stats.items())
