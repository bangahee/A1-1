"""거래 분석과 한국어 그림·통계·프로젝트 설명을 같은 결과에서 생성한다."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from src.pipeline import DataAnalyzer
from scripts.reporting import (SEGMENT_LABELS, METRIC_LABELS, imputation_example,
                               scaling_estimates, korean_summary, write_readme)

SEGMENT_ORDER = list(SEGMENT_LABELS)
PALETTE = dict(zip(SEGMENT_ORDER, ["#6C5CE7", "#00B894", "#0984E3", "#D63031"]))
FIGURE_METADATA = [
    {"file": "01_amount_histogram.png", "chart_type": "히스토그램", "title": "양수 구매금액 분포(99백분위 이하)", "x_axis": "구매금액(GBP)", "y_axis": "거래 수"},
    {"file": "02_outlier_boxplot.png", "chart_type": "박스플롯", "title": "IQR 처리 전후 구매금액 비교", "x_axis": "처리 단계", "y_axis": "구매금액(GBP, 로그 척도)"},
    {"file": "03_segment_bar.png", "chart_type": "막대그래프", "title": "RFM 고객군별 고객 수", "x_axis": "고객군", "y_axis": "고객 수"},
    {"file": "04_rfm_heatmap.png", "chart_type": "히트맵", "title": "RFM 지표의 상관관계", "x_axis": "RFM 지표", "y_axis": "RFM 지표"},
    {"file": "05_rfm_scatter.png", "chart_type": "산점도", "title": "고객군별 구매 빈도와 조정 구매금액", "x_axis": "구매 빈도(주문 수, 로그 척도)", "y_axis": "조정 구매금액(GBP, 로그 척도)"},
    {"file": "06_monthly_sales_line.png", "chart_type": "라인차트", "title": "월별 조정 구매금액 추이", "x_axis": "거래 월", "y_axis": "조정 구매금액 합계(GBP)"},
]


def json_default(value: Any) -> Any:
    """NumPy 수치와 날짜를 JSON으로 기록한다."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"JSON으로 변환할 수 없는 자료형: {type(value).__name__}")


def configure_plotting() -> str:
    """설치된 한국어 글꼴을 선택하고 통일된 그림 스타일을 적용한다."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"] if name in available), None)
    if font is None:
        raise RuntimeError("한국어 글꼴이 필요합니다. 나눔고딕 또는 Noto Sans CJK KR을 설치하세요")
    sns.set_theme(style="whitegrid", context="notebook", font=font)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    return font


def save_figure(fig: plt.Figure, path: Path) -> None:
    """동일 해상도로 저장하고 그림 객체의 메모리를 해제한다."""
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_axes(ax: plt.Axes, index: int) -> None:
    """그림 제목과 축 이름을 기록용 메타데이터와 동일하게 설정한다."""
    meta = FIGURE_METADATA[index]
    ax.set(title=meta["title"], xlabel=meta["x_axis"], ylabel=meta["y_axis"])


def create_figures(frame: pd.DataFrame, rfm: pd.DataFrame, summary: pd.DataFrame,
                   outlier_report: dict[str, Any], figures_dir: Path) -> str:
    """현재 DataFrame으로 여섯 분석 그림과 고객군 표·배열 예시를 만든다."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    font = configure_plotting()
    purchases = frame.loc[frame.amount.gt(0), "amount"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(purchases[purchases.le(purchases.quantile(.99))], bins=45, color="#0984E3", ax=ax)
    label_axes(ax, 0)
    save_figure(fig, figures_dir / FIGURE_METADATA[0]["file"])

    positive = frame.loc[frame.amount.gt(0), ["amount", "amount_clean"]].rename(columns={"amount": "처리 전", "amount_clean": "IQR 클리핑 후"})
    melted = positive.melt(var_name="처리 단계", value_name="구매금액")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=melted, x="처리 단계", y="구매금액", color="#74B9FF", ax=ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    label_axes(ax, 1)
    before, after = outlier_report["before_count"], outlier_report["after_count"]
    ax.text(.5, .97, f"경계 밖 거래: {before:,}건({before/len(positive):.1%}) → {after:,}건",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "alpha": .9})
    save_figure(fig, figures_dir / FIGURE_METADATA[1]["file"])

    counts = rfm.Segment.value_counts().reindex(SEGMENT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [SEGMENT_LABELS[s] for s in counts.index]
    bars = ax.bar(labels, counts.values, color=[PALETTE[s] for s in counts.index])
    label_axes(ax, 2)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=3)
    ax.margins(y=.12)
    save_figure(fig, figures_dir / FIGURE_METADATA[2]["file"])

    corr = rfm[list(METRIC_LABELS)].corr().rename(index=METRIC_LABELS, columns=METRIC_LABELS)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"label": "Pearson 상관계수"}, ax=ax)
    label_axes(ax, 3)
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, figures_dir / FIGURE_METADATA[3]["file"])

    plot_rfm = rfm.assign(고객군=rfm.Segment.map(SEGMENT_LABELS))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_rfm, x="Frequency", y="Monetary", hue="고객군",
                    hue_order=list(SEGMENT_LABELS.values()),
                    palette={SEGMENT_LABELS[k]: v for k, v in PALETTE.items()}, alpha=.65, s=40, ax=ax)
    ax.set(xscale="log", yscale="log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    label_axes(ax, 4)
    save_figure(fig, figures_dir / FIGURE_METADATA[4]["file"])

    monthly = frame.loc[frame.amount.gt(0)].set_index("order_date").resample(pd.offsets.MonthEnd())["amount_clean"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    month_labels = monthly.index.strftime("%Y-%m")
    ax.plot(month_labels, monthly.values, marker="o", color="#6C5CE7")
    ax.tick_params(axis="x", rotation=40)
    ax.annotate("부분 월(12월 9일까지)", xy=(len(monthly)-1, monthly.iloc[-1]),
                xytext=(-135, 35), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    label_axes(ax, 5)
    save_figure(fig, figures_dir / FIGURE_METADATA[5]["file"])

    table = korean_summary(summary)
    fig, ax = plt.subplots(figsize=(14, 3.4))
    ax.axis("off")
    rendered = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="center", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(10)
    rendered.scale(1, 1.8)
    ax.set_title("고객군별 규모와 조정 구매금액", pad=14)
    save_figure(fig, figures_dir / "07_segment_summary_table.png")

    image = frame.product_image.iloc[0]
    height = int(frame.image_height.iloc[0]) if "image_height" in frame else 1
    width = int(frame.image_width.iloc[0]) if "image_width" in frame else len(image)
    if height * width != len(image):
        height, width = 1, len(image)
    fig, ax = plt.subplots(figsize=(7, 5))
    rendered_image = ax.imshow(image.reshape(height, width), cmap="gray", vmin=0, vmax=255)
    fig.colorbar(rendered_image, ax=ax, label="교육용 배열 값")
    ax.set(title=f"교육용 배열 예시: 평균 {image.mean():.2f}, 표준편차 {image.std():.2f}", xlabel="열 위치", ylabel="행 위치")
    save_figure(fig, figures_dir / "08_educational_array.png")
    return font


def sensitivity_analysis(analyzer: DataAnalyzer, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """점수 수·최근성 임계값·금액 처리 변경에 따른 분류 이동을 계산한다."""
    rows, comparisons = [], []
    scenarios = [("기본 4구간·조정 금액", 4, None, "amount_clean"),
                 ("3구간·조정 금액", 3, None, "amount_clean"),
                 ("5구간·조정 금액", 5, None, "amount_clean"),
                 ("최근성 30일", 4, 30, "amount_clean"),
                 ("최근성 90일", 4, 90, "amount_clean"),
                 ("최근성 180일", 4, 180, "amount_clean"),
                 ("4구간·원본 금액", 4, None, "amount")]
    for label, bins, days, amount in scenarios:
        current = analyzer.calculate_rfm(amount_col=amount, score_bins=bins, recent_days=days)
        movement = current.Segment.ne(baseline.Segment.reindex(current.index)).mean()
        for segment, group in current.groupby("Segment"):
            rows.append({"시나리오": label, "점수 구간 수": bins, "최근성 기준(일)": days,
                         "금액 열": amount, "고객군": SEGMENT_LABELS[segment], "고객 수": len(group),
                         "고객 비중": len(group)/len(current), "금액 비중": group.Monetary.sum()/current.Monetary.sum(),
                         "기본 대비 분류 이동률": movement})
        if amount == "amount":
            comparisons = current[["Monetary", "Segment"]].rename(columns={"Monetary": "원본 구매금액", "Segment": "원본 기준 고객군"}).join(
                baseline[["Monetary", "Segment"]].rename(columns={"Monetary": "조정 구매금액", "Segment": "조정 기준 고객군"}))
    # 시나리오 분석 후 기본 결과를 복원해 이후 요약과 기준일이 섞이지 않게 한다.
    analyzer.rfm = baseline
    return pd.DataFrame(rows), comparisons


def build_insights(summary: pd.DataFrame) -> dict[str, dict[str, str]]:
    """관측 근거와 실행 가설, 실패 조건을 구분해 사업 제안을 만든다."""
    vip, dormant, recent = (summary.loc[s] for s in ["VIP", "Churned", "New"])
    return {
        "vip_retention": {
            "title": "핵심 우수 고객 유지",
            "evidence": f"핵심 우수 고객은 {int(vip.customers):,}명({vip.customer_share:.1%})이며 조정 금액의 {vip.revenue_share:.1%}를 만든다. 평균 주문 수는 {vip.mean_frequency:.2f}회다.",
            "action": "핵심 우수 고객에게 등급별 선공개·서비스 혜택을 제공한다. 첫 2주에 혜택 비용·마진 기준을 정하고 4주 캠페인 후 90일 반복구매율을 대조군과 비교한다.",
            "expected_effect": "기대 효과는 반복 구매와 주요 매출 기반의 유지다. 금액 기여도가 커 우선 검토하되 고객별 마진과 혜택 비용에 따라 우선순위를 조정한다.",
            "validation_data": "대조군의 반복구매율·90일 미구매율, 고객별 마진과 혜택 비용이 필요하다. 반복구매율이 개선되지 않거나 증분 공헌이익이 0 이하라면 제안의 효과는 지지되지 않는다.",
        },
        "dormant_reactivation": {
            "title": "장기 미구매 고객 재방문",
            "evidence": f"장기 미구매 고객은 {int(dormant.customers):,}명({dormant.customer_share:.1%})이며 평균 최근성은 {dormant.mean_recency:.1f}일이다. 상대 점수 기준의 미구매 집단으로 확정 이탈자는 아니다.",
            "action": "장기 미구매 고객을 대상으로 2주 준비 후 4주간 기간 제한 인센티브를 무작위 대조 실험한다. 증분 매출·공헌이익·수신거부율을 추적한다.",
            "expected_effect": "기대 효과는 전체 고객 할인 없이 일부 고객의 재방문을 유도하는 것이다. 할인비와 자연 재구매를 캠페인 효과로 오인할 위험을 함께 관리한다.",
            "validation_data": "무작위 대조군의 자연 재구매, 이탈 사유, 할인 비용과 마진 데이터가 필요하다. 재방문율이나 증분 공헌이익이 대조군보다 개선되지 않으면 가설을 지지하지 않는다.",
        },
        "recent_repeat_purchase": {
            "title": "최근 저빈도 고객의 추가 구매",
            "evidence": f"최근 저빈도 고객은 {int(recent.customers):,}명({recent.customer_share:.1%})이고 표본 내 주문 빈도 중앙값은 {recent.median_frequency:g}회다. 신규 가입이나 생애 첫 구매가 확인된 집단은 아니다.",
            "action": "최근 저빈도 고객에게 마지막 관측 구매 후 상품 사용 안내와 연관상품을 제안하고 14/30일 추가 구매율을 대조군과 비교한다. 첫 구매 온보딩은 전체 구매 이력에서 첫 구매가 확인된 대상에만 적용한다.",
            "expected_effect": "기대 효과는 최근 구매 고객의 재방문과 추가 구매 전환이다. 추천의 관련성과 반품 증가를 함께 살핀다.",
            "validation_data": "전체 구매 이력·추천 노출·클릭·반품 데이터가 필요하다. 추가 구매율이 대조군보다 높지 않거나 반품·혜택 비용이 성과를 상쇄하면 효과는 지지되지 않는다.",
        },
    }


def run_analysis(data_path: str | Path = ROOT / "data/online_retail_sample.csv",
                 output_dir: str | Path = ROOT / "outputs",
                 figures_dir: str | Path = ROOT / "figures", *, update_readme: bool = False) -> dict[str, Any]:
    """현재 입력에서 모든 분석 결과를 재계산한다. 기본적으로 README는 변경하지 않는다."""
    output_dir, figures_dir = Path(output_dir), Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = DataAnalyzer(data_path)
    frame = analyzer.load_data()
    raw_frame = frame.copy()
    overview = analyzer.overview()
    buffer = io.StringIO()
    raw_frame.info(buf=buffer)
    info_text = "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n"
    (output_dir / "data_info.txt").write_text(info_text, encoding="utf-8")
    describable = raw_frame.drop(columns="product_image")
    descriptive = describable.describe(include="all").T
    descriptive.insert(0, "dtype", describable.dtypes.astype(str))
    descriptive.to_csv(output_dir / "descriptive_statistics.csv")
    raw_frame.dtypes.astype(str).value_counts().rename_axis("dtype").rename("column_count").to_csv(output_dir / "dtype_summary.csv")
    raw_stats = raw_frame.select_dtypes(include="number").describe().T
    missing = analyzer.handle_missing_values()
    frame = analyzer.engineer_features()
    outliers = analyzer.treat_outliers("amount", positive_only=True, output_col="amount_clean")
    rfm = analyzer.calculate_rfm(amount_col="amount_clean")
    summary = analyzer.segment_summary()
    group_stats = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].agg(["count", "mean", "median", "std", lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
    group_stats.columns = pd.MultiIndex.from_tuples([(a, "Q1" if b == "<lambda_0>" else "Q3" if b == "<lambda_1>" else b) for a,b in group_stats.columns])
    correlations = rfm[list(METRIC_LABELS)].corr()
    sensitivity, amounts = sensitivity_analysis(analyzer, rfm)
    ranges = []
    for metric, score in [("Recency", "R_score"), ("Frequency", "F_score"), ("Monetary", "M_score")]:
        part = rfm.groupby(score)[metric].agg(["count", "min", "max"]).reset_index()
        part.columns = ["점수", "고객 수", "최솟값", "최댓값"]
        part.insert(0, "지표", metric)
        ranges.append(part)
    score_ranges = pd.concat(ranges, ignore_index=True)
    demo_values, demo_stats = imputation_example()
    scaling = scaling_estimates()
    for name, table, index in [
        ("rfm_customers", rfm, True), ("segment_summary", summary, True),
        ("segment_group_statistics", group_stats, True), ("rfm_correlations", correlations, True),
        ("rfm_sensitivity", sensitivity, False), ("monetary_comparison", amounts, True),
        ("rfm_score_ranges", score_ranges, False), ("imputation_example_values", demo_values, False),
        ("imputation_example_statistics", demo_stats, False), ("scaling_estimates", scaling, False),
    ]:
        table.to_csv(output_dir / f"{name}.csv", index=index)
    font = create_figures(frame, rfm, summary, outliers, figures_dir)
    pd.DataFrame(FIGURE_METADATA).to_csv(output_dir / "visualization_evidence.csv", index=False)
    captions = "# 그림 설명\n\n"
    for meta in FIGURE_METADATA:
        captions += f"- `{meta['file']}`: **{meta['title']}**. 가로축: {meta['x_axis']}, 세로축: {meta['y_axis']}.\n"
    captions += "\n히스토그램은 양수 금액의 99백분위 이하만 표시한다. 박스플롯과 산점도의 로그 척도는 긴 꼬리를 읽기 위한 선택이다. 상관행렬은 인과를 뜻하지 않는다. 월별 그림의 마지막 12월은 부분 월이다.\n\n07은 고객군 요약 표, 08은 실제 사진이 아닌 교육용 배열 예시다. 모든 그림은 현재 분석 실행에서 생성한다.\n"
    (figures_dir / "CAPTIONS.md").write_text(captions, encoding="utf-8")
    quality = {"overview": overview, "missing_values": missing, "outliers": outliers,
               "rfm_reference_date": analyzer.rfm_reference_date,
               "valid_rfm_transactions": len(analyzer.rfm_transactions(amount_col="amount_clean")),
               "rfm_customers": len(rfm), "segment_count": rfm.Segment.nunique(),
               "score_policy": "평균 순위 백분위, 동점 보존, 4구간", "korean_font": font}
    insights = build_insights(summary)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "matplotlib": matplotlib.__version__, "seaborn": sns.__version__, "korean_font": font}
    for name, value in [("data_quality_report", quality), ("insights", insights), ("environment", environment)]:
        (output_dir / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    result = {"analyzer": analyzer, "frame": frame, "raw_frame": raw_frame, "overview": overview,
              "raw_descriptive_statistics": raw_stats, "missing_report": missing, "outlier_report": outliers,
              "rfm": rfm, "segment_summary": summary, "group_statistics": group_stats,
              "correlations": correlations, "sensitivity": sensitivity, "score_ranges": score_ranges,
              "imputation_values": demo_values, "imputation_statistics": demo_stats, "scaling": scaling,
              "quality": quality, "insights": insights}
    if update_readme:
        write_readme(result, ROOT)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="거래 분석과 한국어 산출물 생성")
    parser.add_argument("--data", default=str(ROOT / "data/online_retail_sample.csv"), help="입력 CSV")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"), help="통계 저장 위치")
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"), help="그림 저장 위치")
    args = parser.parse_args()
    result = run_analysis(args.data, args.output_dir, args.figures_dir, update_readme=True)
    print(korean_summary(result["segment_summary"]).to_string(index=False))
    print("\n분석·한국어 그림·README 생성 완료")


if __name__ == "__main__":
    main()
