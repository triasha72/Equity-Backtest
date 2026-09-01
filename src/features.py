"""Cross-sectional feature construction.

Every feature at month t is computed from information available at or before the
close of month t. Nothing downstream of t enters the feature set.

Design note on size: true market capitalisation would require historical shares
outstanding. Applying *current* shares outstanding retroactively is lookahead
bias, so this study uses log average daily dollar volume as a size/liquidity
proxy instead and says so rather than quietly introducing the bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK = 11  # months, after skipping the most recent month
VOL_WINDOW_DAYS = 252
DOLLAR_VOL_WINDOW_DAYS = 63


def _month_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()


def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a long panel indexed by (date, ticker) with features and target.

    Columns
    -------
    mom_12_1 : cumulative return months t-11..t-1 (most recent month skipped)
    reversal : return in month t
    vol_12m  : annualised std of daily returns, trailing 252 days
    liquidity: log mean daily dollar volume, trailing 63 days
    target   : return in month t+1  (the label; never used as an input)
    """
    close = prices["close"].sort_index()
    volume = prices["volume"].sort_index()

    daily_ret = close.pct_change()
    dollar_vol = close * volume

    m_close = _month_end(close)
    m_ret = m_close.pct_change()

    # 12-1 momentum: skip the most recent month, compound the 11 before it.
    gross = (1.0 + m_ret).shift(1)
    mom = gross.rolling(MOMENTUM_LOOKBACK).apply(np.prod, raw=True) - 1.0

    reversal = m_ret

    vol = _month_end(daily_ret.rolling(VOL_WINDOW_DAYS).std()) * np.sqrt(252)

    liq = _month_end(dollar_vol.rolling(DOLLAR_VOL_WINDOW_DAYS).mean())
    liq = np.log(liq.where(liq > 0))

    target = m_ret.shift(-1)

    frames = {
        "mom_12_1": mom,
        "reversal": reversal,
        "vol_12m": vol,
        "liquidity": liq,
        "target": target,
    }
    panel = pd.concat(
        {k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1
    )
    panel.index.names = ["date", "ticker"]
    return panel.dropna(subset=["mom_12_1", "reversal", "vol_12m", "liquidity"])


def cross_sectional_zscore(
    panel: pd.DataFrame, cols: list[str], clip: float | None = 3.0
) -> pd.DataFrame:
    """Standardise each feature *within* each date.

    Pooling across dates would compare 2008 volatility against 2021 volatility.
    Standardising within the cross-section is what makes the ranks comparable.

    ``clip`` winsorises the standardised scores at +/- that many standard
    deviations. Winsorising deliberately breaks exact zero-centring, which is why
    it is a separate, explicit argument rather than folded silently into the
    standardisation: the tests check centring with ``clip=None`` and check the
    bound separately.
    """
    out = panel.copy()
    g = out.groupby(level="date")
    for c in cols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std")
        z = (out[c] - mu) / sd.replace(0.0, np.nan)
        out[c] = z if clip is None else z.clip(-clip, clip)
    return out
