# Cross-Sectional Equity Signal: An Honest Backtest

![tests](https://github.com/triasha72/Equity-Backtest/actions/workflows/tests.yml/badge.svg)

A monthly, dollar-neutral, cross-sectional equity strategy built to a strict
walk-forward protocol — and reported gross **and** net of transaction costs.

The point of this repository is not to present a profitable strategy. It is to
run a well-known effect through an evaluation protocol strict enough that the
result, whatever it turns out to be, can be believed.

---

## Hypothesis

Cross-sectional momentum (Jegadeesh & Titman, 1993) predicts the relative
ordering of next-month equity returns. Stated in the form actually tested:

> At the end of each month, a dollar-neutral portfolio long the top decile and
> short the bottom decile of predicted returns earns a positive average return
> over the following month, **net of transaction costs**.

The null is that any gross spread is consumed by turnover.

---

## Results

<!-- Fill in after running on real data. Do not publish this section with
     placeholder numbers - state the result you actually measured. -->

| Metric | Gross | Net |
| --- | --- | --- |
| Annualised Sharpe | _TBD_ | _TBD_ |
| Annualised return | — | _TBD_ |
| t-statistic | — | _TBD_ |
| Max drawdown | — | _TBD_ |
| Avg monthly turnover | _TBD_ | |
| Equal-weight universe Sharpe | _TBD_ | |

Sample: _TBD_ to _TBD_, _N_ months, _M_ names.

![Gross vs net of costs](results/equity_curve.png)

![Drawdown](results/drawdown.png)

![Monthly turnover](results/turnover.png)

**Verdict:** _TBD — state plainly whether the signal survives costs. If it does
not, say so here, in the first sentence._

**Significance:** _TBD variants were tested (see `results/variants_log.csv`)._
Harvey, Liu & Zhu (2016) argue that a new factor should clear a t-statistic of
3.0 rather than 2.0, precisely because the literature has tested so many
candidates. That hurdle is adopted here and stated before the result, not after.

---

## Method

**Universe.** A fixed list of large-cap US equities (`data/universe.txt`),
monthly rebalancing.

**Features**, each computed at month *t* from information available at or before
the close of month *t*:

| Feature | Definition |
| --- | --- |
| `mom_12_1` | Cumulative return over months *t-11* … *t-1*, skipping the most recent month |
| `reversal` | Return during month *t* |
| `vol_12m` | Annualised std of daily returns, trailing 252 days |
| `liquidity` | Log mean daily dollar volume, trailing 63 days |

**Target.** Return over month *t+1*.

**Model.** Ridge regression on cross-sectionally standardised features. A linear
model is deliberate: with four features and a monthly panel, the binding
constraint is signal, not model capacity.

**Portfolio.** Rank predictions, long the top decile, short the bottom decile,
equal-weighted within each leg, dollar-neutral, gross exposure of 1.0.

**Costs.** 10 bps charged on every dollar traded, applied to realised turnover
computed month over month.

---

## What was controlled for

**Expanding-window walk-forward, not k-fold.** Observations in a financial panel
are not exchangeable across time — the ordering *is* the problem. At formation
month *t*, the training set contains only pairs whose labels had already been
realised by *t*. Nothing observed after *t* enters the fit. Shuffled k-fold on a
time series is the most common way a backtest fools its author, and it is not
used here.

**Cross-sectional standardisation, not pooled.** Features are standardised
within each date. Pooling would compare 2008 volatility against 2021 volatility
and let the level of the market leak into the ranks.

**No post-formation data as a feature.** Every input is causally prior to the
predicted return. This is a structural control rather than a procedural one: the
model cannot see downstream information because it is never given any.

**No retroactive shares outstanding.** True market capitalisation would need
historical share counts; applying today's share count to 2010 is lookahead. Log
dollar volume is used as a size/liquidity proxy instead, and the substitution is
stated rather than quietly made.

**A shuffled-target test.** `tests/test_pipeline.py` permutes the labels and
asserts the pipeline then earns nothing. If a leak existed, a shuffled target
would still score.

---

## Known limitations

**Survivorship bias — present and unfixed.** The universe file lists *currently
traded* tickers. Companies delisted, acquired, or bankrupted during the sample
are absent, so the sample conditions on survival and results are optimistic. A
clean run needs point-in-time constituents with delisting returns (CRSP via
WRDS). This is the largest single caveat on any number in this repository.

**Costs are a flat assumption.** 10 bps per dollar traded is a reasonable
stand-in, not a market-impact model. Real costs vary with size, spread, and
capacity.

**No capacity analysis.** Equal-weighted decile portfolios over large caps say
nothing about how much capital the signal absorbs.

**Monthly frequency only.** Nothing here speaks to intraday or weekly horizons.

---

## Multiple testing

Every invocation of `run.py` appends a row to `results/variants_log.csv` with the
full configuration and the resulting statistics. That file is the denominator: a
t-statistic from the twentieth specification tried must be read against twenty
attempts. The log is committed deliberately, including the variants that failed.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/ -q                    # 7 tests, synthetic data, no network needed

python run.py --note "baseline: 4 features, 10bps, decile spread"
python make_plots.py

python run.py --features mom_12_1 --note "momentum alone"
python run.py --cost-bps 20 --note "cost sensitivity: 20bps"
python run.py --cost-bps 0  --note "zero-cost upper bound"
python run.py --decile 0.20 --note "quintile spread"
```

Outputs land in `results/`: per-month returns, a summary JSON, and the appended
variants log.

---

## References

Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling
Losers: Implications for Stock Market Efficiency. *The Journal of Finance*,
48(1), 65–91.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the Cross-Section of Expected
Returns. *The Review of Financial Studies*, 29(1), 5–68.
