import numpy as np
import pandas as pd
import pytest
from src.backtest import Config, run
from src.features import cross_sectional_zscore


def panel():
    rows = []
    for i, date in enumerate(pd.date_range("2020-01-31", periods=6, freq="ME")):
        for name, x in [("A", 1), ("B", -1), ("C", 0.2)]:
            if i == 4 and name == "A":
                continue
            rows.append(
                {
                    "date": date,
                    "ticker": name,
                    "mom_12_1": x,
                    "reversal": x,
                    "vol_12m": x,
                    "liquidity": x,
                    "log_daily_dollar_volume": np.log(1e9),
                    "target": x * 0.01,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_departing_position_is_charged():
    result = run(panel(), Config(min_train_months=2, min_names=2))
    assert result.loc["2020-05-31", "exit_turnover"] == pytest.approx(0.5)
    assert result.loc["2020-05-31", "turnover"] >= 1


def test_missing_held_return_is_not_zero_imputed():
    data = panel()
    data.loc[(pd.Timestamp("2020-04-30"), "A"), "target"] = np.nan
    with pytest.raises(ValueError, match="Missing realized"):
        run(data, Config(min_train_months=2, min_names=2))


def test_capacity_units_survive_standardization():
    data = panel()
    transformed = cross_sectional_zscore(data, ["liquidity"])
    pd.testing.assert_series_equal(
        data.log_daily_dollar_volume, transformed.log_daily_dollar_volume
    )
    with pytest.raises(ValueError, match="raw log_daily"):
        run(data.drop(columns="log_daily_dollar_volume"), Config())
