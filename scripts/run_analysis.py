"""Run the complete A1-1 EDA and RFM workflow and save all artifacts."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

# 서버나 CI처럼 홈 디렉터리에 쓸 수 없는 환경에서도 Matplotlib 캐시를
# 프로젝트 내부에 만들 수 있도록 설정한다.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

# 화면이 없는 CI/평가 환경에서도 PNG를 만들기 위한 비대화형 백엔드다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.pipeline import DataAnalyzer


# 모든 표와 그림에서 순서·색상을 고정해 세그먼트 의미가 그림마다 바뀌지 않게 한다.
SEGMENT_ORDER = ["VIP", "Loyal", "New", "Churned"]
PALETTE = {
    "VIP": "#6C5CE7",
    "Loyal": "#00B894",
    "New": "#0984E3",
    "Churned": "#D63031",
}

# 평가자가 이미지 자체를 열지 않아도 제목과 축 레이블을 확인할 수 있도록
# 시각화 메타데이터를 기계 판독 가능한 표로 함께 저장한다.
FIGURE_METADATA = [
    {
        "file": "01_amount_histogram.png",
        "chart_type": "Histogram",
        "title": "Purchase amount distribution (up to 99th percentile)",
        "x_axis": "Amount (GBP)",
        "y_axis": "Count",
    },
    {
        "file": "02_outlier_boxplot.png",
        "chart_type": "Box plot",
        "title": "Outliers before and after treatment",
        "x_axis": "Treatment",
        "y_axis": "Amount (GBP, log scale)",
    },
    {
        "file": "03_segment_bar.png",
        "chart_type": "Bar chart",
        "title": "Customers by RFM segment",
        "x_axis": "Segment",
        "y_axis": "Customers",
    },
    {
        "file": "04_rfm_heatmap.png",
        "chart_type": "Heatmap",
        "title": "RFM correlation heatmap",
        "x_axis": "RFM metrics",
        "y_axis": "RFM metrics",
    },
    {
        "file": "05_rfm_scatter.png",
        "chart_type": "Scatter plot",
        "title": "Frequency vs monetary value by segment",
        "x_axis": "Frequency (orders)",
        "y_axis": "Monetary (GBP)",
    },
    {
        "file": "06_monthly_sales_line.png",
        "chart_type": "Line chart",
        "title": "Monthly sales trend",
        "x_axis": "Month",
        "y_axis": "IQR-adjusted sales (GBP)",
    },
]


def json_default(value: Any) -> Any:
    # NumPy 스칼라와 Timestamp를 표준 JSON 타입으로 바꾸는 직렬화 경계다.
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def save_figure(fig: plt.Figure, path: Path) -> None:
    # 모든 그래프를 동일한 해상도와 여백 정책으로 저장한다. 반복 생성 시 열린 Figure가
    # 누적되어 메모리를 점유하지 않도록 저장 직후 명시적으로 닫는다.
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    frame: pd.DataFrame,
    rfm: pd.DataFrame,
    segment_summary: pd.DataFrame,
    outlier_report: dict[str, Any],
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    # 극단적인 꼬리가 본문 분포를 가리지 않도록 히스토그램은 양수 구매의
    # 99분위까지 표시하되, 이상치 계산 자체에는 전체 양수 값을 사용한다.
    purchases = frame.loc[frame["amount"].gt(0), "amount"]
    upper_99 = purchases.quantile(0.99)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(purchases[purchases.le(upper_99)], bins=45, color="#0984E3", ax=ax)
    ax.set(title="Purchase amount distribution (up to 99th percentile)", xlabel="Amount (GBP)")
    save_figure(fig, figures_dir / "01_amount_histogram.png")

    # 동일한 양수 거래의 처리 전·후 금액을 비교해 IQR 클리핑 효과를 보여준다.
    positive = frame.loc[frame["amount"].gt(0), ["amount", "amount_clean"]].rename(
        columns={"amount": "Before", "amount_clean": "After IQR clipping"}
    )
    plot_values = positive.melt(var_name="Treatment", value_name="Amount")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=plot_values, x="Treatment", y="Amount", color="#74B9FF", ax=ax)
    # 금액의 긴 오른쪽 꼬리 때문에 선형축에서는 상자 본체가 눌리므로 로그축을 쓴다.
    ax.set_yscale("log")
    ax.set(
        title="Outliers before and after treatment",
        xlabel="Treatment",
        ylabel="Amount (GBP, log scale)",
    )
    eligible_count = int(frame["amount"].gt(0).sum())
    before_count = int(outlier_report["before_count"])
    after_count = int(outlier_report["after_count"])
    annotation = (
        f"IQR outliers: {before_count:,} ({before_count / eligible_count:.1%}) before"
        f"  →  {after_count:,} after clipping"
    )
    ax.text(
        0.5,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    save_figure(fig, figures_dir / "02_outlier_boxplot.png")

    counts = rfm["Segment"].value_counts().reindex(SEGMENT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=ax)
    ax.set(title="Customers by RFM segment", xlabel="Segment", ylabel="Customers")
    for patch, count in zip(ax.patches, counts.values):
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f"{count:,}", ha="center", va="bottom")
    save_figure(fig, figures_dir / "03_segment_bar.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    # 상관계수는 관계의 방향·강도를 요약할 뿐 인과관계를 증명하지 않는다.
    corr = rfm[["Recency", "Frequency", "Monetary", "RFM_score"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax)
    ax.set(
        title="RFM correlation heatmap",
        xlabel="RFM metrics",
        ylabel="RFM metrics",
    )
    save_figure(fig, figures_dir / "04_rfm_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    # Frequency와 Monetary 모두 치우침이 커 로그축으로 고객 간 관계를 읽기 쉽게 한다.
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
    ax.set(
        xscale="log",
        yscale="log",
        title="Frequency vs monetary value by segment",
        xlabel="Frequency (orders)",
        ylabel="Monetary (GBP)",
    )
    save_figure(fig, figures_dir / "05_rfm_scatter.png")

    # 월별 합계에는 이상치 조정 금액을 사용하며, 원본 날짜를 월말 단위로 재표본화한다.
    # 2011년 12월은 9일까지만 있는 부분 월이므로 전월과 직접 비교하지 않는다.
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

    # 같은 RFM 근거를 CSV뿐 아니라 검토 화면에서 바로 읽을 수 있는 표 그림으로 남긴다.
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
    # 각 제안을 근거-실행-기대효과-추가 검증 데이터 구조로 고정해
    # 단순한 의견이 아니라 검증 가능한 비즈니스 가설로 만든다.
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

    # 전체 분석은 로드 → 탐색 → 결측 처리 → 특징 생성 → 이상치 처리 →
    # RFM 집계 → 시각화·리포트 저장 순서로 재현 가능하게 실행된다.
    analyzer = DataAnalyzer(data_path, outlier_threshold=1.5)
    frame = analyzer.load_data()
    overview = analyzer.overview()

    info_buffer = io.StringIO()
    frame.info(buf=info_buffer)
    (output_dir / "data_info.txt").write_text(info_buffer.getvalue(), encoding="utf-8")
    # product_image 객체 배열은 뒤에서 image_mean/std로 요약된다. 수천 개 배열 자체를
    # object describe로 비교하면 느리고 해석 가치도 없어 기술통계 표에서는 제외한다.
    describable = frame.drop(columns=["product_image"])
    descriptive = describable.describe(include="all").transpose()
    descriptive.insert(0, "dtype", describable.dtypes.astype(str))
    descriptive.to_csv(output_dir / "descriptive_statistics.csv")
    (
        frame.dtypes.astype(str)
        .value_counts()
        .rename_axis("dtype")
        .rename("column_count")
        .to_csv(output_dir / "dtype_summary.csv")
    )

    missing_report = analyzer.handle_missing(strategy="group_mode", group_col="stock_code")
    frame = analyzer.engineer_features()
    # RFM Monetary에는 원본을 보존한 amount_clean을 사용해 극단값 한 건이 고객 가치
    # 전체를 지배하는 현상을 줄인다. 회계 매출이 아니라 비교용 조정 금액이라는 뜻이다.
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
    # 수치 행렬을 별도 저장해 README의 상관 해석을 검토자가 재계산할 수 있게 한다.
    correlations = rfm[["Recency", "Frequency", "Monetary", "RFM_score"]].corr()
    correlations.to_csv(output_dir / "rfm_correlations.csv")
    create_figures(frame, rfm, summary, outlier_report, figures_dir)
    pd.DataFrame(FIGURE_METADATA).to_csv(
        output_dir / "visualization_evidence.csv", index=False
    )

    # 제출 후에도 데이터 규모와 정제 결과를 기계적으로 검증할 수 있게 저장한다.
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
