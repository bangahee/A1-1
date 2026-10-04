"""현재 분석 결과에서 한국어 표와 설명을 생성한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline import DataAnalyzer


SEGMENT_LABELS = {
    "VIP": "핵심 우수 고객",
    "Loyal": "반복 구매 고객",
    "New": "최근 저빈도 고객",
    "Churned": "장기 미구매 고객",
}
METRIC_LABELS = {
    "Recency": "최근성(일)", "Frequency": "구매 빈도(주문 수)",
    "Monetary": "조정 구매금액(GBP)", "RFM_score": "RFM 종합점수",
}


def markdown_table(frame: pd.DataFrame) -> str:
    """추가 표 라이브러리 없이 작은 DataFrame을 마크다운 표로 바꾼다."""
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def korean_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """고객군 표의 표시 이름과 단위를 한국어로 바꾼다."""
    return pd.DataFrame({
        "고객군": summary.index.map(SEGMENT_LABELS),
        "고객 수": summary.customers.map(lambda x: f"{x:,}"),
        "고객 비중": summary.customer_share.map(lambda x: f"{x:.1%}"),
        "평균 최근성": summary.mean_recency.map(lambda x: f"{x:.1f}일"),
        "빈도 중앙값": summary.median_frequency.map(lambda x: f"{x:g}회"),
        "평균 조정 구매금액": summary.mean_monetary.map(lambda x: f"£{x:,.2f}"),
        "조정 금액 비중": summary.revenue_share.map(lambda x: f"{x:.1%}"),
    }).reset_index(drop=True)


def imputation_example() -> tuple[pd.DataFrame, pd.DataFrame]:
    """실제 거래와 분리한 교육용 예제로 전역·그룹 대치의 영향을 비교한다."""
    original = pd.DataFrame({"stock_code": ["A"] * 4 + ["B"] * 4,
                             "unit_price": [1., 3., 101., np.nan, 10., 12., 14., np.nan],
                             "order_date": pd.to_datetime(["2024-01-01"] * 8)})
    variants = {"관측값만": original.copy()}
    global_fill = original.copy()
    global_fill["unit_price"] = global_fill.unit_price.fillna(global_fill.unit_price.mean())
    variants["전역 평균"] = global_fill
    for strategy, label in [("group_mean", "그룹 평균"), ("group_median", "그룹 중앙값")]:
        analyzer = DataAnalyzer("교육용_예제.csv", min_rows=1)
        analyzer.df = original.copy()
        analyzer.handle_missing_values(strategy, columns=["unit_price"])
        variants[label] = analyzer.df
    values = pd.DataFrame({"상품 그룹": original.stock_code})
    stats = []
    for label, frame in variants.items():
        values[label] = frame.unit_price
        for group, rows in frame.groupby("stock_code"):
            stats.append({"대치 방법": label, "상품 그룹": group,
                          "관측 수": int(rows.unit_price.count()),
                          "평균": float(rows.unit_price.mean()),
                          "표본 분산": float(rows.unit_price.var())})
    return values, pd.DataFrame(stats)


def scaling_estimates() -> pd.DataFrame:
    """고객당 비압축 이미지 한 장을 가정한 배열 용량을 계산한다."""
    rows = []
    for label, channels, itemsize in [("흑백 uint8", 1, 1), ("RGB uint8", 3, 1), ("RGB float32", 3, 4)]:
        size = 1_000_000 * 1024 * 1024 * channels * itemsize
        rows.append({"배열 형식": label, "이미지 수": 1_000_000, "용량(TB)": size / 10**12,
                     "용량(TiB)": size / 1024**4})
    return pd.DataFrame(rows)


DESIGN_NOTES = """`DataAnalyzer`는 DataFrame 상태를 관리하고 각 단계의 책임을 분리한다.
`load_data()`는 입력 형식·크기·날짜·배열을 확인하고, `handle_missing_values()`는 결측 정책을 적용한다.
`engineer_features()`는 수치·텍스트·배열 특징을 만들고, `detect_outliers()`는 경계 밖 행을 찾는다.
`treat_outliers()`는 처리 정책을 적용하며, `calculate_rfm()`은 고객 단위 집계와 점수를 만든다.
`segment_summary()`는 고객군의 규모와 금액 기여도를 요약한다.
이 분리는 결측 정책만 교체하거나 집계만 다시 실행하고, 단계별 입력·출력과 상태 변화를 따로 검증할 수 있게 한다.

`outlier_threshold`는 분포에 맞는 탐지 강도를, `strategy`·`group_col`·`columns`는 데이터에 맞는 대치를,
`reference_date`는 같은 시점의 재현 가능한 최근성을 지정한다. `score_bins`와 `recent_days`는 상대 점수와
절대 구매주기 기준을 비교한다. 다른 파일·정책을 적용할 때 함수 내부 상수를 고칠 필요가 없다.
`handle_missing()`은 기존 호출과의 호환을 위한 이름이며 공개 API인 `handle_missing_values()`와 같은 구현을 사용한다."""

VECTOR_NOTES = """금액은 `quantity.to_numpy() * unit_price.to_numpy()`로 계산한다.
배열 특징은 `np.stack()`으로 동종 자료형의 2차원 행렬을 만든 뒤 `mean(axis=1)`과 `std(axis=1)`로
이미지별 값을 한 번에 구한다. 단어 수는 Pandas 문자열 연산으로 만든다.
CSV 문자열을 배열로 복원하는 입출력 단계에는 행별 처리가 남아 있지만, 평균·표준편차 계산에 Python 행별 for문은 사용하지 않는다.

동종 자료형의 연속 메모리는 포인터 추적을 줄이고 CPU 캐시의 지역성을 높인다.
NumPy의 컴파일된 내부 루프는 Python 원소별 호출과 동적 타입 검사 비용을 줄이며,
지원되는 연산·자료형·CPU에서는 SIMD로 여러 값을 함께 처리할 수 있다.
모든 연산이 SIMD를 사용하거나 작은 배열에서도 항상 빠른 것은 아니다. 전체 행렬을 쌓는 메모리 비용도 고려해야 한다.
현재 0~255 배열은 float32로 충분하며 float64보다 핵심 행렬 메모리가 절반이다."""

IMAGE_NOTES = """`product_image`는 `stock_code`를 해시해 만든 8×8 교육용 배열이다.
실제 상품 사진이나 UCI가 제공한 이미지가 아니며 색상·형태·품질을 뜻하지 않는다.
숫자·범주·날짜는 원본 거래에서 확보하고 배열은 파싱·벡터 통계의 동작을 살펴보는 데 사용한다.
실제 상품 이미지 분석으로 확장하려면 공개 라이선스 이미지와 정확한 상품 대응 관계가 필요하다.
관련 없는 사진을 임의로 상품에 연결해서는 안 된다. 빈 배열·길이가 다른 배열·비유한 값은 오류로 드러낸다."""

SCALING_NOTES = """고객당 1024×1024 비압축 이미지 한 장을 가정하면 흑백 uint8는 약 1.05 TB,
RGB uint8는 약 3.15 TB, RGB float32는 약 12.58 TB다. 압축 파일 용량과 다르며
CSV 문자열·파싱 객체·임시 복사본은 추가 메모리를 쓴다. 실제 이미지 수는 고객 수가 아니라 고유 상품 수와 연결해 계산해야 한다.

전체 `np.stack()`의 RAM 요구량은 분명한 실행 불가 지점이다. 디스크·네트워크 읽기 또는 디코딩이
먼저 병목이 되는지는 프로파일링으로 확인한다. 고유 상품 이미지의 평균·표준편차를 한 번만 계산해 캐시하고,
이미지 또는 작은 배치마다 처리한 뒤 원본 배열을 해제한다. 배열 대신 작은 피처 표를 거래에 연결한다.
원본 uint8를 유지하고 필요시 `arr[::n, ::n]`으로 다운샘플링하며, CSV 픽셀 문자열 대신 NumPy 바이너리 등으로 저장한다.
필요시 메모리 매핑을 사용하고 CPU 병목이 확인된 뒤에만 병렬화한다. 프로세스마다 전체 데이터를 복사하면 RAM 문제가 커진다."""

ML_NOTES = """후속 예측 타깃은 과거 기준일 이후 90일간 구매가 없으면 `churn_90d=1`, 구매가 있으면 0으로 정의할 수 있다.
2011-12-10 이후의 거래는 현재 데이터에 없으므로 이 시점의 미래 레이블은 만들 수 없다.
마지막 관측일까지 이후 90일이 완전히 포함되는 과거 기준일을 선택하고, 그 이전에 활동한 고객을 대상으로 한다.
관측 기간 부족을 미구매로 오인하는 우측 검열을 관리하고, 거래 표본에서 빠진 구매도 이탈 오분류를 만들 수 있어 전체 이력이 필요하다.

현재 사용 가능한 피처는 최근성·구매 빈도·원본/조정 구매금액과 상품 설명의 단어 수 등이다.
거래 수준 피처를 고객 수준으로 집계하는 규칙을 정하고, 반품률·고유 상품 수·활동 개월 수·최근 30/60/90일 활동은 추가로 구현할 후보로 구분한다.
교육용 이미지 배열 통계는 시각적 의미나 예측력을 가정하지 않는다. 미래 주문·금액과 미래 캠페인 결과를 입력에서 제외한다.
기준일에 이미 알려진 RFM 세그먼트 자체는 미래 정보가 아니지만, 원 지표로부터 결정된 중복 피처이고 기준 변경에 민감해 기본 입력에서는 제외한다.
시간순 학습·검증·테스트 분할을 사용하고 정확도 외 PR-AUC·재현율·정밀도·확률 보정과 캠페인 비용을 반영한 기대가치를 확인한다.
확장 설계이며 현재 저장소에서 머신러닝 모델을 학습한 결과는 아니다."""


def analytical_notes(result: dict[str, Any]) -> dict[str, str]:
    """최신 수치와 고정 방법론을 합쳐 README와 노트북이 공유할 설명을 만든다."""
    stats = result["raw_descriptive_statistics"]
    lines = []
    for col, label in [("quantity", "수량"), ("unit_price", "단가(GBP)"), ("amount", "거래금액(GBP)")]:
        s = stats.loc[col]
        lines.append(f"- **{label}**: 평균 {s['mean']:.2f}, 중앙값 {s['50%']:.2f}, 표준편차 {s['std']:.2f}, Q1 {s['25%']:.2f}, Q3 {s['75%']:.2f}. "
                     "평균·중앙값 차이와 큰 변동성을 함께 확인하고 일반적인 거래는 중앙값과 사분위 범위로 설명한다.")
    b = result["outlier_report"]["bounds"]
    count = result["outlier_report"]["before_count"]
    eligible = int(result["frame"].amount.gt(0).sum())
    corr = result["correlations"]
    valid = result["analyzer"].rfm_transactions(amount_col="amount_clean")
    raw, adjusted = valid.amount.sum(), valid.amount_clean.sum()
    return {
        "design": DESIGN_NOTES, "vector": VECTOR_NOTES, "image": IMAGE_NOTES,
        "statistics": "\n".join(lines) + "\n\n반품·취소가 포함된 거래 표본의 기술통계다. 원본과 조정 금액을 구분해서 사용한다.",
        "missing": f"상품 설명 결측 {result['missing_report']['before']['description']:,}건은 같은 상품 코드의 최빈값으로 보완해 "
                   f"{result['missing_report']['after']['description']:,}건이 되었다. `description_imputed`에 대치 위치를 남긴다. "
                   f"고객 ID 결측 {result['overview']['missing_by_column']['customer_id']:,}건은 식별자를 조작하지 않고 RFM에서 제외한다.\n\n"
                   "상품명에는 평균이 정의되지 않아 최빈값이 적합하다. 수치형은 그룹 평균 또는 중앙값을 인자로 선택한다. "
                   "평균은 극단값에 민감하며 중앙값은 상대적으로 강건하다. 그룹 전체가 결측이면 전체 통계로 보완하거나 결측을 유지한다. "
                   "대치는 관측값만으로 계산한 분산을 줄이고 기존 그룹 차이를 실제보다 뚜렷하게 보이게 할 수 있다. "
                   "별도 교육용 예제에서 관측 수·그룹 평균·표본 분산을 비교하며 실제 거래에 인위적 결측을 넣지 않는다.",
        "iqr": f"IQR = Q3 − Q1이며 경계는 Q1 − 1.5×IQR, Q3 + 1.5×IQR이다. 양수 거래의 "
               f"Q1={b['q1']:.3f}, Q3={b['q3']:.3f}, IQR={b['iqr']:.3f}, 상한={b['upper']:.3f} GBP다. "
               f"경계 밖 {count:,}건({count/eligible:.1%})을 클리핑했고 같은 경계 밖 값은 처리 후 0건이다.\n\n"
               "IQR은 정규분포를 가정하지 않고 중앙 50%를 이용해 극단값 영향에 비교적 강건하다. "
               "그러나 비대칭의 긴 꼬리나 여러 봉우리가 있는 분포에서는 정상 고액 주문도 과다 표시할 수 있고 전역 경계는 상품·국가별 차이를 무시한다. "
               "이상치는 곧 오류라는 뜻이 아니다. 음수 반품은 보존하고 원본 amount와 조정 amount_clean을 분리한다. "
               f"유효 RFM 구매의 원본 합계는 £{raw:,.2f}, 조정 합계는 £{adjusted:,.2f}로 {1-adjusted/raw:.1%} 감소한다. "
               "조정 금액의 매출 비중을 회계 매출이나 실현 이익으로 해석하지 않는다.",
        "correlations": f"- **구매 빈도–조정 구매금액**: Pearson 상관계수 {corr.loc['Frequency','Monetary']:.3f}. "
                        "더 자주 주문한 고객이 큰 누적 금액을 보이는 경향으로, 반복 구매 고객의 경제적 기여를 함께 살펴볼 근거다. 누적 합계와 빈도의 구조적 연결도 있어 인과 효과는 아니다.\n"
                        f"- **최근성–구매 빈도**: Pearson 상관계수 {corr.loc['Recency','Frequency']:.3f}. "
                        "마지막 구매 후 시간이 긴 고객은 주문 빈도가 낮은 경향이지만 관계가 강하지 않으므로 최근성만으로 구매 가치를 판단하지 않는다. "
                        "반복 구매 유지와 장기 미구매 회수를 구분해 실험할 시사점이 있다.\n\n"
                        f"최근성과 종합점수의 상관은 {corr.loc['Recency','RFM_score']:.3f}이며 최근성 점수가 산식에 포함된 영향을 받는다.",
        "rfm": f"기준일은 {result['quality']['rfm_reference_date']:%Y-%m-%d}이다. 기준일과 마지막 구매일의 현지 날짜 차이가 최근성이다. "
               "빈도는 고유 주문번호 수이고 금액은 유효 양수 구매의 조정 금액 합계다.\n\n"
               "각 지표의 평균 순위 백분위를 네 점수 구간으로 나누며 최근성은 방향을 반대로 적용한다. 동일 값은 동일 점수를 받고 "
               "고객 ID나 정렬 순서에 영향을 받지 않는다. 고객 한 명 또는 상수 지표는 중간 점수로 처리한다. 동점 때문에 각 구간이 정확히 25%가 되거나 모든 점수가 존재하지 않을 수 있다.\n\n"
               "기본 네 구간에서 R≥3을 최근 구매로 보고, F≥3·M≥3이면 핵심 우수 고객이다. 최근 구매이면서 F≤2이면 최근 저빈도 고객, "
               "R≤2이면 장기 미구매 고객, 나머지는 반복 구매 고객이다. New·Churned는 내부 호환 식별자이며 신규 가입·생애 첫 구매·확정 이탈을 의미하지 않는다. "
               "거래 표본에서 관측한 행동이므로 고객의 전체 이력과 다를 수 있다.",
        "sensitivity": "점수 구간 3·4·5개, 최근성 30·90·180일, 원본·조정 금액 시나리오를 비교한다. "
                       "홀수 구간은 중앙 점수를 상위 쪽에 포함하므로 단순히 구간 수만 바뀌는 비교는 아니다. "
                       "후보 기준은 정답이 아닌 가정이며 고객 수·금액 비중·기본 분류 대비 이동률로 운영 안정성을 확인한다. "
                       "조정 금액은 고객 순위와 매출 집중도를 바꿀 수 있어 두 기준을 함께 본다.",
        "scaling": SCALING_NOTES, "ml": ML_NOTES,
        "limitations": "거래 행 표본은 일부 주문 품목과 과거 구매를 빠뜨려 빈도·금액을 낮게 측정하거나 최근성을 늘릴 수 있다. "
                        "고객 단위로 표본을 뽑고 관측 기간 내 전체 거래를 유지하는 방식이 고객 행동 분석에 더 적합하다. "
                        "익명 거래는 고객 단위 결과에서 제외된다. 반품·취소는 구매 집계와 분리했으며 순매출·반품 행동은 별도 분석이 필요하다. "
                        "2011년 12월은 9일까지인 부분 월로 전월과 직접 비교하지 않는다. 교육용 배열을 실제 상품 이미지로 해석할 수 없다.",
    }


def write_readme(result: dict[str, Any], root: Path) -> None:
    """현재 결과를 사용해 프로젝트 설명 문서를 한국어로 저장한다."""
    notes = analytical_notes(result)
    insights = result["insights"]
    proposals = []
    for item in insights.values():
        proposals.append(f"### {item['title']}\n\n**(근거)** {item['evidence']}\n\n"
                         f"**(실행)** {item['action']} {item['expected_effect']}\n\n**(검증)** {item['validation_data']}")
    overview = result['overview']
    sections = [
        f"# 쇼핑몰 거래 분석과 RFM 고객 세분화\n\n{overview['rows']:,}건의 거래를 전처리하고 수치·텍스트·교육용 배열 특징을 만든 뒤 "
        f"{len(result['rfm']):,}명의 고객을 네 집단으로 나누는 데이터 분석 프로젝트다. 분석과 문서의 수치는 같은 실행 결과에서 생성한다.",
        "## 분석 결과\n\n" + markdown_table(korean_summary(result['segment_summary'])) + "\n\n![고객군별 고객 수](figures/03_segment_bar.png)",
        "## 데이터와 출처\n\n- 출처: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail)\n"
        "- 인용: Chen, D. (2015). Online Retail. [DOI](https://doi.org/10.24432/C5BW33)\n"
        "- 라이선스: CC BY 4.0\n- 원본: 541,909행, 8개 원본 필드\n"
        f"- 분석 표본: seed 42, {overview['rows']:,}행, {overview['columns']}열\n"
        f"- 기간: {overview['date_min'][:10]}~{overview['date_max'][:10]}\n\n"
        + notes['image'] + "\n\n[데이터 설명](data/DATASET.md) · [배열 설계](docs/IMAGE_ARRAY_DECISION.md)",
        "## 파이프라인 설계\n\n" + notes['design'],
        "## 결측 처리\n\n" + notes['missing'],
        "## 수치·텍스트·배열 특징\n\n" + notes['vector'] + "\n\n![교육용 배열 예시](figures/08_educational_array.png)",
        "## 기술통계\n\n" + notes['statistics'],
        "## IQR 탐지와 처리\n\n" + notes['iqr'] + "\n\n![이상치 처리 전후](figures/02_outlier_boxplot.png)",
        "## 상관관계\n\n" + notes['correlations'],
        "## RFM 기준과 고객군\n\n" + notes['rfm'] + "\n\n" + notes['sensitivity'] +
        "\n\n[점수별 지표 범위](outputs/rfm_score_ranges.csv) · [기준 민감도](outputs/rfm_sensitivity.csv) · [고객군 기술통계](outputs/segment_group_statistics.csv)",
        "## 시각화\n\n각 그림은 같은 분석 실행에서 생성하며 제목·축·범례를 한국어로 표시한다.\n\n"
        "![구매금액 분포](figures/01_amount_histogram.png)\n\n![상관행렬](figures/04_rfm_heatmap.png)\n\n"
        "![빈도와 금액](figures/05_rfm_scatter.png)\n\n![월별 조정 구매액](figures/06_monthly_sales_line.png)\n\n"
        "[전체 그림 설명](figures/CAPTIONS.md)",
        "## 비즈니스 제안\n\n" + "\n\n".join(proposals),
        "## 대규모 데이터로 확장하기\n\n" + notes['scaling'] + "\n\n" + markdown_table(result['scaling'].round(3)),
        "## 머신러닝으로 확장하기\n\n" + notes['ml'],
        "## 한계\n\n" + notes['limitations'],
        "## 실행 방법\n\n검증 환경은 Python 3.14.5다. 이전 Python 버전 지원은 검증하지 않았다. "
        "정확한 패키지 버전은 `outputs/environment.json`에 기록한다.\n\n"
        "```bash\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements-dev.txt\n"
        "python -m unittest -v\npython -m scripts.run_analysis\npython scripts/build_notebook.py\npython scripts/verify_project.py\n```\n\n"
        "Windows PowerShell의 활성화 명령은 `.venv\\Scripts\\Activate.ps1`이다. "
        "기본 분석에만 필요한 라이브러리는 `requirements.txt`의 NumPy·Pandas·Matplotlib·Seaborn이다. "
        "`requirements-dev.txt`는 이를 포함하고 노트북 실행·원본 Excel 처리를 위한 도구를 추가한다. "
        "노트북 생성기는 실행 중인 Python으로 새 커널을 시작하고 현재 통계와 그림을 다시 생성한다. "
        "한국어 표시에는 AppleGothic·맑은 고딕·나눔고딕·Noto Sans CJK KR 중 설치된 글꼴을 사용한다.\n\n"
        "원본 표본 재생성은 `python scripts/prepare_data.py` 또는 "
        '`python scripts/prepare_data.py --raw-xlsx "/path/to/Online Retail.xlsx"`로 실행한다.',
        "## 프로젝트 구조\n\n```text\nsrc/pipeline.py                  데이터 분석 클래스\n"
        "scripts/run_analysis.py          분석과 산출물 생성\nscripts/reporting.py             결과에 따른 한국어 설명\n"
        "scripts/build_notebook.py        노트북 생성과 새 커널 실행\nscripts/verify_project.py        산출물 일관성 검증\n"
        "notebooks/analysis_report.ipynb   실행된 분석 보고서\ndata/                           표본과 출처\n"
        "outputs/                        통계·민감도·실행 기록\nfigures/                        한국어 시각화\n"
        "tests/                          핵심 계산 회귀 검증\n```\n\n[분석 흐름](docs/IMPLEMENTATION_GUIDE.md) · [방법과 한계](docs/METHODOLOGY.md)",
    ]
    (root / "README.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    headings = {
        "design": "메서드 책임과 재사용", "missing": "그룹별 결측 대치",
        "vector": "배열 연산과 성능", "iqr": "IQR 선택과 한계", "rfm": "고객 점수와 의미",
        "sensitivity": "기준 변경의 영향", "scaling": "대규모 이미지 처리",
        "ml": "미래 구매 예측", "limitations": "관측 범위의 한계",
    }
    methodology = ["# 분석 방법과 한계"] + [f"## {heading}\n\n{notes[key]}" for key, heading in headings.items()]
    methodology.append("## 수치형 대치 예제\n\n실제 거래와 분리한 예제다. 관측 수가 늘어나는 대치 전후 표본 분산 비교이며 참 분산이나 대치의 정확성을 입증하지 않는다.\n\n"
                       + markdown_table(result["imputation_statistics"].round(3)))
    comparison = result["sensitivity"].groupby("시나리오", sort=False).agg({"고객 수": "sum", "기본 대비 분류 이동률": "first"}).reset_index().round(3)
    methodology.append("## 민감도 결과\n\n분류 이동률은 기본 4구간·조정 금액과 고객별 분류가 달라진 비율이다.\n\n" + markdown_table(comparison))
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/METHODOLOGY.md").write_text("\n\n".join(methodology) + "\n", encoding="utf-8")
