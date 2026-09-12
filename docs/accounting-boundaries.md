# Accounting corrections and evidence limits

Capacity constraints now consume raw log daily dollar volume, separate from the
cross-sectional standardized liquidity predictor. Previously recorded capacity
results must be rerun; they are not evidence for the corrected implementation.
Exiting names contribute liquidation turnover at their last mark. Missing held
forward returns now stop the run rather than silently becoming zero. Non-held
missing outcomes are not treated as trades. Membership dates are validated.

`python run.py --help` lists the new `--crsp` monthly input route. CRSP security
histories with missing months are rejected before rolling features or forward
labels are computed. Delisting returns use the existing adapter. Membership files
must use matching PERMNO identifiers. Monthly dollar volume divided by 21 is an
approximation of daily capacity, not a calibrated execution model.

No licensed survivorship-complete panel has been run here. Last-mark liquidation,
pre-return target-weight accounting, fixed participation caps and incomplete market
impact remain research simplifications. Results do not establish investable alpha.
