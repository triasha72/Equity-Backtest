#!/usr/bin/env python3
"""Entry point.

Every run appends one row to results/variants_log.csv. That log is the honest
denominator for any t-statistic quoted in the README: if twenty specifications
were tried, the twentieth's significance must be read against twenty attempts,
not one. See Harvey, Liu & Zhu (2016).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from src.backtest import FEATURES, Config, benchmark, run
from src.data import fetch_prices, load_universe
from src.evaluate import report, summarise
from src.features import build_panel, cross_sectional_zscore
from src.membership import apply_point_in_time_membership, load_membership

LOG = "results/variants_log.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-sectional equity signal backtest")
    p.add_argument("--universe", default="data/universe.txt")
    p.add_argument(
        "--membership",
        help="optional point-in-time CSV with ticker,start_date,end_date columns",
    )
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--features", default=",".join(FEATURES))
    p.add_argument("--decile", type=float, default=0.10)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--note", default="", help="what this variant was testing")
    p.add_argument("--force-refetch", action="store_true")
    a = p.parse_args()

    tickers = load_universe(a.universe)
    print(f"universe: {len(tickers)} tickers  {a.start} -> {a.end}")

    prices = fetch_prices(tickers, a.start, a.end, force=a.force_refetch)
    panel = build_panel(prices)
    if a.membership:
        panel = apply_point_in_time_membership(panel, load_membership(a.membership))
        print(f"point-in-time membership: {a.membership}")
    feats = [f.strip() for f in a.features.split(",") if f.strip()]
    panel = cross_sectional_zscore(panel, feats)
    print(f"panel: {len(panel):,} stock-months, {len(feats)} features")

    cfg = Config(
        decile=a.decile,
        cost_bps=a.cost_bps,
        ridge_alpha=a.ridge_alpha,
        features=tuple(feats),
    )
    bt = run(panel, cfg)
    if bt.empty:
        raise SystemExit("no periods produced - check the sample window")

    stats = summarise(bt, benchmark(panel))
    print("\n" + report(stats))

    os.makedirs("results", exist_ok=True)
    bt.to_csv("results/monthly_returns.csv")
    with open("results/summary.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    row = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": a.note,
        **cfg.as_dict(),
        **stats,
    }
    pd.DataFrame([row]).to_csv(
        LOG, mode="a", header=not os.path.exists(LOG), index=False
    )
    with open(LOG) as log_file:
        variant_number = sum(1 for _ in log_file) - 1
    print(f"\nlogged to {LOG} (variant #{variant_number})")


if __name__ == "__main__":
    main()
