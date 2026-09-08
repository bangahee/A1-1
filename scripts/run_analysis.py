"""Run the complete A1-1 EDA and RFM workflow and save all artifacts."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.pipeline import DataAnalyzer


SEGMENT_ORDER = ["VIP", "Loyal", "New", "Churned"]
PALETTE = {
    "VIP": "#6C5CE7",
    "Loyal": "#00B894",
    "New": "#0984E3",
    "Churned": "#D63031",
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    frame: pd.DataFrame,
    rfm: pd.DataFrame,
    segment_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    purchases = frame.loc[frame["amount"].gt(0), "amount"]
    upper_99 = purchases.quantile(0.99)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(purchases[purchases.le(upper_99)], bins=45, color="#0984E3", ax=ax)
    ax.set(title="Purchase amount distribution (up to 99th percentile)", xlabel="Amount (GBP)")
    save_figure(fig, figures_dir / "01_amount_histogram.png")

    positive = frame.loc[frame["amount"].gt(0), ["amount", "amount_clean"]].rename(
        columns={"amount": "Before", "amount_clean": "After IQR clipping"}
    )
    plot_values = positive.melt(var_name="Treatment", value_name="Amount")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=plot_values, x="Treatment", y="Amount", color="#74B9FF", ax=ax)
    ax.set_yscale("log")
    ax.set(title="Outliers before and after treatment", ylabel="Amount (GBP, log scale)")
    save_figure(fig, figures_dir / "02_outlier_boxplot.png")

    counts = rfm["Segment"].value_counts().reindex(SEGMENT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=ax)
    ax.set(title="Customers by RFM segment", xlabel="Segment", ylabel="Customers")
    for patch, count in zip(ax.patches, counts.values):
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f"{count:,}", ha="center", va="bottom")
    save_figure(fig, figures_dir / "03_segment_bar.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    corr = rfm[["Recency", "Frequency", "Monetary", "RFM_score"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("RFM correlation heatmap")
    save_figure(fig, figures_dir / "04_rfm_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=rfm,
        x="Frequency",
        y="Monetary",
        hue="Segment",
        hue_order=SEGMENT_ORDER,
        palette=PALETTE,
        alpha=0.68,
        s=42,
        ax=ax,
    )
    ax.set(xscale="log", yscale="log", title="Frequency vs monetary value by segment")
    save_figure(fig, figures_dir / "05_rfm_scatter.png")

    monthly = (
        frame.loc[frame["amount"].gt(0)]
        .set_index("order_date")
        .resample("ME")["amount_clean"]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o", color="#6C5CE7", ax=ax)
    ax.set(title="Monthly sales trend", xlabel="Month", ylabel="IQR-adjusted sales (GBP)")
    save_figure(fig, figures_dir / "06_monthly_sales_line.png")

    table = segment_summary.reset_index().copy()
    table["customer_share"] = table["customer_share"].map(lambda x: f"{x:.1%}")
    table["revenue_share"] = table["revenue_share"].map(lambda x: f"{x:.1%}")
    table["mean_recency"] = table["mean_recency"].map(lambda x: f"{x:.1f}")
    table["mean_monetary"] = table["mean_monetary"].map(lambda x: f"£{x:,.0f}")
    table = table[["Segment", "customers", "customer_share", "mean_recency", "median_frequency", "mean_monetary", "revenue_share"]]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    rendered = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="center", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.6)
    ax.set_title("RFM segment evidence table", pad=12, fontweight="bold")
    save_figure(fig, figures_dir / "07_segment_summary_table.png")


def build_insights(summary: pd.DataFrame) -> dict[str, dict[str, str]]:
    vip = summary.loc["VIP"]
    churned = summary.loc["Churned"]
    new = summary.loc["New"]
    return {
        "vip_retention": {
            "evidence": f"VIPs are {vip.customer_share:.1%} of customers and generate {vip.revenue_share:.1%} of adjusted revenue; mean frequency is {vip.mean_frequency:.1f} orders.",
            "action": "Offer tiered early access and service benefits to VIP customers.",
            "expected_effect": "Protect revenue concentration and improve repeat purchase rate.",
            "validation_data": "Campaign/control repeat-purchase rate, gross margin, benefit cost, and 90-day churn.",
        },
        "churn_reactivation": {
            "evidence": f"Churned customers are {churned.customer_share:.1%} of customers with mean recency {churned.mean_recency:.1f} days.",
            "action": "Run a randomized reactivation test with a time-limited incentive.",
            "expected_effect": "Recover dormant customers without discounting the entire base.",
            "validation_data": "Holdout reactivation, incremental revenue, contribution margin, unsubscribe rate, and churn-reason survey.",
        },
        "new_customer_onboarding": {
            "evidence": f"The New segment contains {int(new.customers):,} customers ({new.customer_share:.1%}) and median frequency {new.median_frequency:.0f}.",
            "action": "Trigger product education and a second-order recommendation within 14 days.",
            "expected_effect": "Increase conversion from first purchase to the second order.",
            "validation_data": "14/30-day second-order rate, recommendation click-through, returns, and cohort retention.",
        },
    }


def run_analysis(
    data_path: str | Path = "data/online_retail_sample.csv",
    output_dir: str | Path = "outputs",
    figures_dir: str | Path = "figures",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyzer = DataAnalyzer(data_path, outlier_threshold=1.5)
    frame = analyzer.load_data()
    overview = analyzer.overview()

    info_buffer = io.StringIO()
    frame.info(buf=info_buffer)
    (output_dir / "data_info.txt").write_text(info_buffer.getvalue(), encoding="utf-8")
    # Array objects are represented by image_mean/std later; comparing thousands
    # of NumPy arrays in object-level describe is both noisy and very slow.
    describable = frame.drop(columns=["product_image"])
    describable.describe(include="all").transpose().to_csv(
        output_dir / "descriptive_statistics.csv"
    )

    missing_report = analyzer.handle_missing_values(strategy="group_mode", group_col="stock_code")
    frame = analyzer.engineer_features()
    outlier_report = analyzer.treat_outliers(
        "amount", method="clip", output_col="amount_clean", positive_only=True
    )
    rfm = analyzer.calculate_rfm(amount_col="amount_clean")
    summary = analyzer.segment_summary()

    rfm.to_csv(output_dir / "rfm_customers.csv")
    summary.to_csv(output_dir / "segment_summary.csv")
    group_stats = rfm.groupby("Segment", observed=True)[["Recency", "Frequency", "Monetary"]].agg(
        ["count", "mean", "median", "std"]
    )
    group_stats.to_csv(output_dir / "segment_group_statistics.csv")
    create_figures(frame, rfm, summary, figures_dir)

    quality = {
        "overview": overview,
        "missing_values": missing_report,
        "outliers": outlier_report,
        "rfm_reference_date": (
            analyzer.rfm_transactions(amount_col="amount_clean")["order_date"].max().normalize()
            + pd.Timedelta(days=1)
        ),
        "valid_rfm_transactions": len(analyzer.rfm_transactions(amount_col="amount_clean")),
        "rfm_customers": len(rfm),
    }
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    insights = build_insights(summary)
    (output_dir / "insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "analyzer": analyzer,
        "frame": frame,
        "overview": overview,
        "missing_report": missing_report,
        "outlier_report": outlier_report,
        "rfm": rfm,
        "segment_summary": summary,
        "group_statistics": group_stats,
        "quality": quality,
        "insights": insights,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/online_retail_sample.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    result = run_analysis(args.data, args.output_dir, args.figures_dir)
    print(result["segment_summary"].round(3).to_string())
    print(f"\nSaved {len(list(Path(args.figures_dir).glob('*.png')))} figures")


if __name__ == "__main__":
    main()
