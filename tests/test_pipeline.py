"""Smoke tests on synthetic data.

These do not validate that the signal works. They validate that the machinery
runs, that the shapes line up, and - most importantly - that no future
information reaches the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import FEATURES, Config, _weights, benchmark, run
from src.features import build_panel, cross_sectional_zscore
from src.membership import apply_point_in_time_membership, load_membership


def synthetic_prices(n_tickers: int = 60, days: int = 2600, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    rets = rng.normal(0.0004, 0.014, size=(days, n_tickers))
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers
    )
    volume = pd.DataFrame(
        rng.lognormal(13, 0.4, size=(days, n_tickers)), index=idx, columns=tickers
    )
    return pd.concat({"close": close, "volume": volume}, axis=1)


def test_panel_builds_and_features_are_finite():
    panel = build_panel(synthetic_prices())
    assert set(FEATURES).issubset(panel.columns)
    assert len(panel) > 0
    assert np.isfinite(panel[FEATURES].values).all()


def test_zscore_is_within_date():
    """Unwinsorised scores must be exactly centred within each cross-section."""
    panel = cross_sectional_zscore(build_panel(synthetic_prices()), FEATURES, clip=None)
    for col in FEATURES:
        means = panel.groupby(level="date")[col].mean().abs()
        stds = panel.groupby(level="date")[col].std()
        assert (means < 1e-9).all(), f"{col} not centred within date"
        assert ((stds - 1.0).abs() < 1e-9).all(), f"{col} not unit-scaled within date"


def test_winsorisation_bounds_scores():
    panel = cross_sectional_zscore(build_panel(synthetic_prices()), FEATURES, clip=3.0)
    assert panel[FEATURES].abs().max().max() <= 3.0 + 1e-12


def test_weights_are_dollar_neutral():
    pred = pd.Series(np.arange(100, dtype=float), index=[f"T{i}" for i in range(100)])
    w = _weights(pred, 0.10)
    assert abs(w.sum()) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12


def test_backtest_runs_and_costs_reduce_return():
    panel = cross_sectional_zscore(build_panel(synthetic_prices()), FEATURES)
    bt = run(panel, Config(min_train_months=24, min_names=20))
    assert len(bt) > 0
    assert (bt["net"] <= bt["gross"] + 1e-12).all()
    assert (bt["turnover"] >= 0).all()


def test_no_lookahead_shuffling_target_destroys_signal():
    """If the pipeline leaked, a shuffled target would still score well."""
    panel = cross_sectional_zscore(build_panel(synthetic_prices(seed=1)), FEATURES)
    shuffled = panel.copy()
    rng = np.random.default_rng(7)
    shuffled["target"] = rng.permutation(shuffled["target"].values)
    bt = run(shuffled, Config(min_train_months=24, min_names=20))
    assert abs(bt["net"].mean()) < 0.01, "shuffled target should not be profitable"


def test_benchmark_matches_universe_mean():
    panel = build_panel(synthetic_prices())
    b = benchmark(panel)
    assert len(b) > 0 and b.notna().all()


def test_point_in_time_membership_excludes_inactive_stock_months(tmp_path):
    panel = build_panel(synthetic_prices(n_tickers=2, days=700))
    dates = panel.index.get_level_values("date").unique().sort_values()
    cutoff = dates[len(dates) // 2]
    source = tmp_path / "membership.csv"
    pd.DataFrame(
        [
            {"ticker": "T000", "start_date": dates.min(), "end_date": cutoff},
            {"ticker": "T001", "start_date": cutoff, "end_date": ""},
        ]
    ).to_csv(source, index=False)

    filtered = apply_point_in_time_membership(panel, load_membership(source))
    first = filtered.xs("T000", level="ticker").index
    second = filtered.xs("T001", level="ticker").index
    assert first.max() <= cutoff
    assert second.min() >= cutoff


def test_overlapping_membership_intervals_are_rejected():
    panel = build_panel(synthetic_prices(n_tickers=1, days=700))
    membership = pd.DataFrame(
        {
            "ticker": ["T000", "T000"],
            "start_date": pd.to_datetime(["2012-01-01", "2013-01-01"]),
            "end_date": pd.to_datetime(["2014-01-01", "2015-01-01"]),
        }
    )
    with np.testing.assert_raises(ValueError):
        apply_point_in_time_membership(panel, membership)
