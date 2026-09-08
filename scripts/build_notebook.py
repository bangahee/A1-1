"""Build and execute the submitted A1-1 analysis notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            """
# A1-1 쇼핑몰 고객 분석과 RFM 세분화

이 노트북은 공개 거래 데이터를 불러와 결측치와 이상치를 처리하고, 숫자·텍스트·이미지 배열 특징을 만든 뒤 RFM으로 고객을 네 그룹으로 분류한다. 모든 결과는 `src/pipeline.py`의 재사용 가능한 `DataAnalyzer`를 통해 계산한다.
"""
        ),
        markdown(
            """
## 1. 데이터 출처와 분석 설계

- 출처: UCI Machine Learning Repository, **Online Retail**
- DOI: <https://doi.org/10.24432/C5BW33>
- 라이선스: CC BY 4.0
- 원본: 541,909행, 8개 원본 필드, 2010-12-01~2011-12-09
- 분석 표본: seed 42로 재현 가능하게 추출한 25,000행, 14개 열

`product_image`는 실제 상품 사진이라고 주장하지 않는다. `stock_code`에서 결정론적으로 만든 8×8 교육용 배열이며, CSV 문자열을 `np.fromstring`으로 읽어 배열 평균과 표준편차를 반복문 없이 계산한다. RFM 기준일은 마지막 유효 구매일 다음 날이다.
"""
        ),
        code(
            """
from pathlib import Path
from IPython.display import Image, display
import json
import numpy as np
import pandas as pd

from src.pipeline import DataAnalyzer

pd.set_option("display.max_columns", 30)
DATA_PATH = Path("data/online_retail_sample.csv")
FIGURES = Path("figures")
OUTPUTS = Path("outputs")
"""
        ),
        markdown("## 2. 데이터 로드와 기본 탐색"),
        code(
            """
analyzer = DataAnalyzer(DATA_PATH, outlier_threshold=1.5)
df = analyzer.load_data()
print(f"shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
display(df.drop(columns="product_image").head())
df.info()
"""
        ),
        code(
            """
overview = analyzer.overview()
display(pd.Series(overview["missing_by_column"], name="missing_count").to_frame())
display(df.select_dtypes(include="number").describe().T)
"""
        ),
        markdown(
            """
### 탐색 해석

수치형, 문자열형, 날짜형, NumPy 배열형이 함께 존재한다. 실제 표본에서 `description` 65건과 `customer_id` 6,300건의 결측을 확인했다. 고객 ID는 식별자이므로 평균이나 임의 값으로 채우면 서로 다른 고객을 합치는 오류가 생긴다. 따라서 설명은 상품 그룹의 최빈값으로 보완하고, 고객 ID 결측은 보존했다가 RFM 집계에서만 제외한다.
"""
        ),
        markdown("## 3. 결측치 처리와 멀티모달 특징 공학"),
        code(
            """
missing_report = analyzer.handle_missing_values(strategy="group_mode", group_col="stock_code")
display(pd.DataFrame(missing_report))

df = analyzer.engineer_features()
display(df[["quantity", "unit_price", "amount", "description", "word_count", "image_mean", "image_std"]].head())
"""
        ),
        markdown(
            """
- 숫자: `amount = quantity × unit_price`를 NumPy 벡터 연산으로 계산했다.
- 텍스트: 상품 설명을 공백 단위로 나눈 `word_count`를 생성했다.
- 이미지 배열: 25,000×64 행렬로 쌓아 행 방향 `mean`과 `std`를 한 번에 계산했다.

이 방식은 Python `for` 반복으로 개별 픽셀을 계산하는 것보다 간결하고, 실제 계산을 최적화된 NumPy 루틴에 위임한다.
"""
        ),
        markdown(
            r"""
## 4. IQR 이상치 탐지와 처리

\[
IQR = Q_3 - Q_1, \quad
Lower = Q_1 - 1.5IQR, \quad
Upper = Q_3 + 1.5IQR
\]

반품과 취소는 음수라는 비즈니스 의미가 있으므로 별도로 보존하고, 양수 구매 금액의 극단값만 상한·하한으로 클리핑했다. 원본 `amount`도 남겨 처리 전후를 검증할 수 있게 했다.
"""
        ),
        code(
            """
outliers_before = analyzer.detect_outliers("amount", positive_only=True)
outlier_report = analyzer.treat_outliers(
    "amount", method="clip", output_col="amount_clean", positive_only=True
)
display(pd.Series(outlier_report))
print(f"IQR positive-purchase outliers: {len(outliers_before):,}")
display(Image(filename=str(FIGURES / "02_outlier_boxplot.png")))
"""
        ),
        markdown("## 5. 여섯 종류 이상의 시각화"),
        code(
            """
for filename in [
    "01_amount_histogram.png",
    "03_segment_bar.png",
    "04_rfm_heatmap.png",
    "05_rfm_scatter.png",
    "06_monthly_sales_line.png",
]:
    display(Image(filename=str(FIGURES / filename)))
"""
        ),
        markdown(
            """
히스토그램은 구매금액의 강한 오른쪽 꼬리를 보여준다. 상관 히트맵에서 Frequency와 Monetary의 상관은 약 0.79로 높지만, 상관관계가 구매 횟수 증가의 인과 효과를 증명하지는 않는다. 월별 추세는 2011년 가을의 상승과 11월 고점을 보여주며, 마지막 12월은 9일까지만 포함된 불완전 월이므로 전월과 직접 비교하면 안 된다.
"""
        ),
        markdown(
            """
## 6. RFM 고객 세분화

- Recency: 기준일에서 마지막 구매일까지의 일수(작을수록 좋음)
- Frequency: 서로 다른 주문서 수(클수록 좋음)
- Monetary: IQR 조정된 유효 구매 금액 합계(클수록 좋음)

각 지표를 사분위 점수 1~4로 바꾸고 `VIP`, `Loyal`, `New`, `Churned`로 분류한다. 기준일은 데이터의 마지막 유효 구매일 다음 날인 2011-12-10이다.
"""
        ),
        code(
            """
rfm = analyzer.calculate_rfm(amount_col="amount_clean")
segment_summary = analyzer.segment_summary()
display(rfm.head(10))
display(segment_summary.style.format({
    "mean_recency": "{:.1f}",
    "mean_frequency": "{:.1f}",
    "mean_monetary": "£{:,.0f}",
    "total_monetary": "£{:,.0f}",
    "customer_share": "{:.1%}",
    "revenue_share": "{:.1%}",
}))
"""
        ),
        code(
            """
group_statistics = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].agg(
    ["count", "mean", "median", "std"]
)
display(group_statistics)
display(Image(filename=str(FIGURES / "07_segment_summary_table.png")))
"""
        ),
        markdown(
            """
## 7. 데이터 근거 기반 비즈니스 제안

1. **VIP 유지** — 근거: VIP는 고객의 27.8%지만 조정 매출의 63.4%를 만든다. 실행: 등급별 선공개와 서비스 혜택을 제공한다. 기대효과: 핵심 매출과 반복구매율 방어. 검증: 캠페인/대조군 반복구매율, 이익률, 혜택 비용, 90일 이탈률이 필요하다.
2. **Churned 재활성화** — 근거: 고객의 50.0%가 Churned이며 평균 Recency는 179.5일이다. 실행: 기간 제한 인센티브를 무작위 대조 실험으로 운영한다. 기대효과: 전체 할인 없이 휴면 고객 회수. 검증: 홀드아웃 대비 증분 매출, 공헌이익, 수신거부율, 이탈 사유 설문이 필요하다.
3. **New의 두 번째 구매 유도** — 근거: New는 531명(16.3%)이고 구매빈도 중앙값은 1회다. 실행: 첫 구매 후 14일 내 사용 가이드와 연관상품 추천을 보낸다. 기대효과: 1회 구매자의 2회차 전환. 검증: 14/30일 2회차 구매율, 클릭률, 반품률, 코호트 유지율이 필요하다.

이 제안들은 관찰 데이터에서 나온 가설이다. 캠페인의 인과 효과를 주장하려면 대조군과 비용·마진 데이터가 추가로 필요하다.
"""
        ),
        markdown(
            """
## 8. 한계와 재현성

- 25,000행 표본이므로 전체 541,909행의 정확한 모집단 추정치가 아니다.
- 거래 고객 ID 결측은 RFM에서 제외되어 비식별 구매를 대표하지 못한다.
- `product_image`는 배열 연습용 파생 데이터이며 시각적 상품 특성을 뜻하지 않는다.
- 반품/취소는 RFM 구매 집계에서 제외했지만 별도의 반품 행동 분석이 필요하다.
- 2011년 12월은 부분 월이다.

모든 표본 추출, 특징 생성, 경계값, 기준일은 코드에 명시되어 있다. `python -m scripts.run_analysis`로 산출물을 다시 만들 수 있다.
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    return notebook


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "notebooks" / "analysis_report.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute()
    nbf.write(notebook, output)
    print(f"Built and executed {output}")


if __name__ == "__main__":
    main()
