"""Price data loading.

SURVIVORSHIP BIAS WARNING
-------------------------
The default universe file is a list of *currently listed* tickers. Companies that
were delisted, acquired, or went bankrupt during the sample are absent, which
biases results optimistically: the sample conditions on survival.

This is a known, unfixed limitation of the free-data version of this study. A
survivorship-bias-free run requires a point-in-time database with delisting
returns (CRSP via WRDS). See README section "Limitations".
"""

from __future__ import annotations

import os

import pandas as pd


def load_universe(path: str) -> list[str]:
    """Read one ticker per line; blank lines and '#' comments ignored."""
    with open(path) as fh:
        tickers = [ln.strip().upper() for ln in fh]
    return [t for t in tickers if t and not t.startswith("#")]


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache: str | None = "data/prices.parquet",
    force: bool = False,
) -> pd.DataFrame:
    """Daily adjusted close and volume.

    Returns a DataFrame with a MultiIndex column (field, ticker) where field is
    one of {'close', 'volume'}. Cached to parquet so repeated runs do not refetch.
    """
    if cache and os.path.exists(cache) and not force:
        return pd.read_parquet(cache)

    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )
    close = raw["Close"]
    volume = raw["Volume"]
    out = pd.concat({"close": close, "volume": volume}, axis=1)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        out.to_parquet(cache)
    return out
