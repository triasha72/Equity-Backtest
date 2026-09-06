import numpy as np
import pandas as pd

from src.crsp import build_crsp_panel, load_crsp_monthly


def test_crsp_adapter_combines_regular_and_delisting_returns(tmp_path):
    path = tmp_path / "crsp.csv"
    pd.DataFrame([{
        "PERMNO": 10001, "date": "2020-01-31", "RET": -0.2, "DLRET": -0.5,
        "PRC": -10, "VOL": 1000, "SHROUT": 200,
    }]).to_csv(path, index=False)
    frame = load_crsp_monthly(path)
    assert frame.loc[0, "adjusted_return"] == -0.6
    assert frame.loc[0, "market_cap"] == 2_000_000


def test_crsp_panel_uses_next_month_return_as_target():
    dates = pd.date_range("2018-01-31", periods=20, freq="ME")
    frame = pd.DataFrame({
        "date": dates, "ticker": "10001", "adjusted_return": np.arange(20) / 100,
        "dollar_volume": 1_000_000.0,
    })
    panel = build_crsp_panel(frame)
    first_date = panel.index.get_level_values("date")[0]
    source_index = dates.get_loc(first_date)
    assert panel.iloc[0]["target"] == frame.iloc[source_index + 1]["adjusted_return"]
