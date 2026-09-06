"""Licensed CRSP monthly-data adapter with delisting-return handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"PERMNO", "date", "RET", "DLRET", "PRC", "VOL", "SHROUT"}


def load_crsp_monthly(path: str | Path) -> pd.DataFrame:
    """Load a WRDS export without replacing missing delisting information."""
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"CRSP export is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = (
        pd.to_datetime(frame["date"], errors="raise").dt.to_period("M").dt.to_timestamp("M")
    )
    frame["ticker"] = frame["PERMNO"].astype("Int64").astype(str)
    for column in ("RET", "DLRET", "PRC", "VOL", "SHROUT"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("CRSP export contains duplicate PERMNO-month rows")
    regular = frame["RET"]
    delisting = frame["DLRET"]
    frame["adjusted_return"] = np.where(
        regular.notna() | delisting.notna(),
        (1.0 + regular.fillna(0.0)) * (1.0 + delisting.fillna(0.0)) - 1.0,
        np.nan,
    )
    frame["market_cap"] = frame["PRC"].abs() * frame["SHROUT"] * 1_000
    frame["dollar_volume"] = frame["PRC"].abs() * frame["VOL"]
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_crsp_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Build monthly features using only information available by formation time."""
    rows = []
    for ticker, security in frame.groupby("ticker", sort=True):
        security = security.sort_values("date").copy()
        lagged = security["adjusted_return"].shift(1)
        security["mom_12_1"] = (1.0 + lagged).rolling(11, min_periods=11).apply(np.prod) - 1.0
        security["reversal"] = security["adjusted_return"]
        security["vol_12m"] = lagged.rolling(12, min_periods=12).std()
        security["liquidity"] = np.log1p(
            security["dollar_volume"].shift(1).rolling(12, min_periods=12).mean()
        )
        security["target"] = security["adjusted_return"].shift(-1)
        security["ticker"] = ticker
        rows.append(security)
    panel = pd.concat(rows, ignore_index=True)
    columns = ["date", "ticker", "mom_12_1", "reversal", "vol_12m", "liquidity", "target"]
    return panel[columns].dropna().set_index(["date", "ticker"]).sort_index()
