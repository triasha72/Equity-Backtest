# What would make this backtest more credible

[Read the project story and measured result](README.md)

The current experiment answers a limited question: a ridge-ranked momentum
strategy did not produce a statistically convincing net return on a fixed list
of large US stocks from 2014 through 2024. It does not establish that momentum
does or does not work in a survivorship-free market universe.

## Now implemented

- Expanding-window training with no shuffled future observations.
- Real adjusted prices and volume, cached with the experiment.
- Turnover-based cost deductions and a complete variant log.
- A point-in-time membership adapter that accepts explicit listing intervals and
  rejects overlapping records.
- Synthetic tests for mechanics and leakage. These tests are not return evidence.

## The next real-data run

A credible follow-up needs a licensed point-in-time security database with
delisting returns, such as CRSP through WRDS. The membership adapter is ready,
but no CRSP result is claimed because those records are not available in this
repository.

That run should also include the 2008–2009 momentum crash, use a validation
period for model and portfolio choices, and reserve a later test period for one
final evaluation. Costs should vary with liquidity, and capacity should be
reported alongside return.
