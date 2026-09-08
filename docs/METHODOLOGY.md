# A1-1 methodology and risk notes

## Missing values

`description` is categorical and repeated within `stock_code`, so the product-group
mode preserves product identity better than a global mode. Dropping rows would
discard valid transaction amounts; mean imputation is not defined for product
names. If a group has no observed name, the explicit sentinel `UNKNOWN PRODUCT`
is used. `customer_id` is never imputed because it is an identifier.

This policy is not neutral. Mode imputation reduces within-group variation and
can make between-product differences look stronger than they are. Production
use should retain an imputation flag, compare before/after distributions, and
prefer a governed product master when one exists.

Text missingness is therefore imputed explicitly. Image arrays follow a stricter
fail-fast rule: empty, unequal-length, NaN, or infinite arrays raise an error;
invented zero pixels would bias `image_mean` and `image_std`.

## Outliers

IQR uses the middle 50% and does not require a normal distribution, making it
more robust than mean/standard-deviation rules for right-skewed purchase values.
Its limitation is that legitimate high-value orders in a skewed B2B distribution
may be flagged. This project applies the rule only to positive purchases,
retains returns, preserves the raw column, and clips rather than deletes rows.

## RFM thresholds

Quartiles are a transparent exploratory baseline when no contractual or campaign
thresholds exist. Four bins provide enough observations per bin and map cleanly
to four operational labels. Three bins merge more customers; five create smaller
groups and more boundary churn. Before deployment, compare quartiles with fixed
30/90/180-day Recency, minimum order counts, margin, and LTV thresholds.

## Vectorization and scale

`to_numpy()` and `np.stack()` create homogeneous contiguous arrays that avoid
Python per-element dispatch and allow NumPy's compiled loops and SIMD-friendly
operations. The trade-off is peak memory: a full dense image matrix must fit in
RAM. At larger scale, store arrays in binary columns, process batches, retain
`uint8`/`float32`, use memory maps, and parallelize only after profiling.

## Predictive extension

An auditable next target is `churn_90d`: no purchase in the 90 days after a
cutoff. Candidate pre-cutoff features are RFM, average order value, return rate,
unique products, active months, country, recent-window counts and values, and an
imputation flag. Future orders, future amounts, and post-cutoff segments are
excluded as leakage. Use chronological splits and prioritize Recall, PR-AUC,
calibration, and campaign value over Accuracy alone.

