#!/usr/bin/env python3
"""Render the figures the README needs.

Reads results/monthly_returns.csv (written by run.py) and writes three PNGs to
results/. Run after run.py.

The gross-vs-net pair is the important one: the gap between the two curves is
the cost of turnover, and showing it is the point of the whole exercise.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = "results/monthly_returns.csv"
OUT = "results"
INK, MUTED, WARN = "#1b3a5c", "#9aa5b1", "#b5442f"


def _load() -> pd.DataFrame:
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found - run `python run.py` first")
    df = pd.read_csv(SRC, parse_dates=["date"]).set_index("date")
    if df.empty:
        raise SystemExit(f"{SRC} is empty")
    return df


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def equity_curve(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot((1 + df["gross"]).cumprod(), color=MUTED, lw=1.6,
            label="Gross of costs", linestyle="--")
    ax.plot((1 + df["net"]).cumprod(), color=INK, lw=1.9,
            label="Net of costs")
    ax.axhline(1.0, color="black", lw=0.7, alpha=0.4)
    _style(ax, "Cumulative return, gross vs net of transaction costs",
           "Growth of 1.0")
    ax.legend(frameon=False, fontsize=8.5)
    path = f"{OUT}/equity_curve.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def drawdown(df: pd.DataFrame) -> str:
    curve = (1 + df["net"]).cumprod()
    dd = curve / curve.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.fill_between(dd.index, dd.values, 0, color=WARN, alpha=0.30)
    ax.plot(dd, color=WARN, lw=1.2)
    _style(ax, "Drawdown, net of costs", "Peak-to-trough")
    path = f"{OUT}/drawdown.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def turnover(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(df["turnover"], color=INK, lw=1.0, alpha=0.75)
    ax.axhline(df["turnover"].mean(), color=WARN, lw=1.1, linestyle="--",
               label=f"mean {df['turnover'].mean():.2f}")
    _style(ax, "Monthly turnover (total absolute weight change)", "Turnover")
    ax.legend(frameon=False, fontsize=8.5)
    path = f"{OUT}/turnover.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main() -> None:
    df = _load()
    for fn in (equity_curve, drawdown, turnover):
        print("wrote", fn(df))


if __name__ == "__main__":
    main()
