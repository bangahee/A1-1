# A1-1: start-to-finish build map

This is the learning path behind the finished repository. Run every command
from the project root.

## Step 1 - Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Checkpoint: `python -c "import numpy, pandas, matplotlib, seaborn"` exits
without error.

## Step 2 - Acquire and document data

The project uses UCI Online Retail under CC BY 4.0. Rebuild the local 25,000-row
sample with:

```bash
python scripts/prepare_data.py
```

Checkpoint: `data/online_retail_sample.csv` has 25,000 rows and 14 columns.
Read `data/DATASET.md` before using the data.

## Step 3 - Understand the schema

Open the notebook and run the data-loading cells. Use `head()`, `info()`, and
`describe()` to inspect values, types, dates, and missingness. Do not treat an
identifier such as `customer_id` as a regular measurable number.

Checkpoint: explain why a missing customer ID must not be mean-imputed.

## Step 4 - Build the reusable class

Study `src/pipeline.py` in this order:

1. `load_data()` validates and parses the CSV.
2. `handle_missing_values()` uses product-group evidence.
3. `engineer_features()` applies vectorized numeric/text/image operations.
4. `detect_outliers()` implements the IQR formula.
5. `calculate_rfm()` aggregates customers and assigns segments.

Checkpoint: all four tests pass with `python -m unittest -v`.

## Step 5 - Handle missing values and outliers

Descriptions are filled from the most common description for the same product.
Missing customer IDs remain missing and are excluded only from RFM. Positive
purchase outliers are clipped at `Q3 + 1.5*IQR`; negative returns are retained
in the raw analysis.

Checkpoint: the quality report shows 65 missing descriptions before treatment,
0 after, and 0 remaining positive-purchase outliers after clipping.

## Step 6 - Compute and interpret RFM

Recency counts days since last purchase, Frequency counts distinct invoices,
and Monetary sums valid IQR-adjusted purchase values. Scores use quartiles.

Checkpoint: the output contains exactly VIP, Loyal, New, and Churned, and you
can explain why lower Recency is better.

## Step 7 - Produce EDA and insights

```bash
python -m scripts.run_analysis
```

Inspect the histogram, box plot, bar chart, heatmap, scatter plot, and time-series
line chart. Each business proposal must include evidence, action, expected
effect, and the extra data needed to validate the claim.

Checkpoint: seven PNGs, RFM customer results, segment summaries, group
statistics, a quality report, and three evidence-based insights exist.

## Step 8 - Build the submitted notebook

```bash
python scripts/build_notebook.py
```

Checkpoint: every code cell in `notebooks/analysis_report.ipynb` has executed
without an exception and its written conclusions agree with the CSV outputs.

## Step 9 - Final verification

```bash
python -m unittest -v
python -m scripts.run_analysis
python scripts/verify_project.py
```

Checkpoint: all checks report success. Then commit and push the repository to
obtain the GitHub URL requested by the mission.

