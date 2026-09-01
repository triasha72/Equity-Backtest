"""Point-in-time universe controls for survivorship-aware experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"ticker", "start_date", "end_date"}


def load_membership(path: str | Path) -> pd.DataFrame:
    """Load ticker membership intervals without filling unknown history."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"membership file is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="raise")
    frame["end_date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    if frame["ticker"].eq("").any():
        raise ValueError("membership file contains an empty ticker")
    if (frame["end_date"].notna() & (frame["end_date"] < frame["start_date"])).any():
        raise ValueError("membership end_date precedes start_date")
    return frame.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def apply_point_in_time_membership(
    panel: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    """Retain a stock-month only when its membership interval was active."""
    if panel.index.names != ["date", "ticker"]:
        raise ValueError("panel must use a (date, ticker) index")
    observations = panel.reset_index()
    observations["ticker"] = observations["ticker"].astype(str).str.upper()
    joined = observations.merge(
        membership, on="ticker", how="inner", validate="many_to_many"
    )
    active = (joined["date"] >= joined["start_date"]) & (
        joined["end_date"].isna() | (joined["date"] <= joined["end_date"])
    )
    filtered = joined.loc[active, observations.columns]
    if filtered.duplicated(["date", "ticker"]).any():
        raise ValueError(
            "overlapping membership intervals create duplicate stock-months"
        )
    return filtered.set_index(["date", "ticker"]).sort_index()
