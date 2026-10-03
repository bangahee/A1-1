"""Build and execute the submitted A1-1 analysis notebook.

The builder is the notebook's single source of truth: markdown, code cells, and
execution evidence are regenerated together instead of being edited separately.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time

import nbformat as nbf
from nbclient import NotebookClient


def markdown(text: str):
    # 들여쓰기된 여러 줄 문자열의 바깥 공백을 제거해 마크다운 셀을 만든다.
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    # 생성 스크립트와 제출 노트북의 코드를 한 곳에서 관리하기 위한 헬퍼다.
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    # 설명 셀과 실행 셀을 분석 순서대로 조립한다. 노트북을 수동 편집하지 않고
    # 이 함수에서 재생성하면 코드와 실행 증거의 불일치를 줄일 수 있다.
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

# 경로를 상수로 모아 모든 셀이 같은 데이터와 산출물 위치를 사용하게 한다.
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
# 로드 단계에서 최소 행·열 수, 필수 열, 날짜와 이미지 배열 형식을 검증한다.
df = analyzer.load_data()
print(f"shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
display(df.drop(columns="product_image").head())
df.info()
"""
        ),
        code(
            """
overview = analyzer.overview()
# 타입별 열 수와 수치형 기술 통계를 함께 확인해 데이터 구조를 증명한다.
display(pd.Series(overview["missing_by_column"], name="missing_count").to_frame())
display(
    df.dtypes.astype(str)
    .value_counts()
    .rename_axis("dtype")
    .rename("column_count")
    .to_frame()
)
display(df.select_dtypes(include="number").describe().T)
"""
        ),
        markdown(
            """
### 주요 수치형 변수의 기술 통계 해석

- `quantity`는 평균 **9.51**, 중앙값 **3**, 표준편차 **43.22**, Q1 **1**, Q3 **10**이다. 평균이 중앙값보다 크고 표준편차가 큰 점은 소수의 대량 주문과 음수 반품 때문에 분포가 비대칭임을 보여준다. 따라서 일반적인 주문 수량은 평균보다 중앙값과 사분위 범위로 설명하는 편이 안전하다.
- `unit_price`는 평균 **£5.11**, 중앙값 **£2.08**, 표준편차 **£120.07**, Q1 **£1.25**, Q3 **£4.13**이다. 중앙 50%는 비교적 좁지만 표준편차가 매우 크므로 일부 고가 품목이나 조정 거래가 전체 변동성을 크게 높인다.
- `amount`는 평균 **£17.49**, 중앙값 **£9.75**, 표준편차 **£131.27**, Q1 **£3.38**, Q3 **£17.40**이다. 평균이 중앙값의 약 1.8배이고 표준편차도 크므로 오른쪽 꼬리가 긴 거래금액 분포임을 확인할 수 있다. 이 결과는 평균·표준편차만으로 이상치를 정하기보다 IQR을 사용하고 처리 전후를 함께 비교해야 한다는 근거가 된다.

위 수치는 표본의 기술 통계이며, 반품과 취소를 포함한 원본 거래 기준이다. 고객 세분화용 Monetary에서는 뒤에서 유효 양수 구매와 IQR 조정 금액을 별도로 사용한다.
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
missing_report = analyzer.handle_missing(strategy="group_mode", group_col="stock_code")
display(pd.DataFrame(missing_report))

# 금액, 단어 수, 이미지 평균·표준편차를 반복문 없이 벡터화해 생성한다.
df = analyzer.engineer_features()
display(df[["quantity", "unit_price", "amount", "description", "word_count", "image_mean", "image_std"]].head())
"""
        ),
        markdown(
            """
- 숫자: `amount = quantity × unit_price`를 NumPy 벡터 연산으로 계산했다.
- 텍스트: 상품 설명을 공백 단위로 나눈 `word_count`를 생성했다.
- 이미지 배열: 25,000×64 행렬로 쌓아 행 방향 `mean`과 `std`를 한 번에 계산했다.

텍스트 결측은 그룹 대치 후에도 값이 없으면 `UNKNOWN PRODUCT`로 표시하고, 이미지 배열은 길이가 다르거나 비어 있거나 NaN/무한대를 포함하면 통계를 만들지 않고 명시적으로 실패시킨다. 이 정책은 품질 문제를 0으로 조용히 대치해 숨기지 않는다.

`to_numpy()`와 `np.stack()`은 동종 값을 연속적인 배열로 배치하여 Python 객체 순회 비용을 줄이고, NumPy의 컴파일된 루프와 SIMD 친화적 연산을 사용하기 위한 선택이다. 다만 전체 이미지 행렬을 한 번에 적재하므로 원본 전체 규모에서는 배치 처리나 메모리 매핑이 필요하다.
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

IQR은 평균·표준편차처럼 정규분포를 가정하지 않고 중앙 50%에 기반하므로 오른쪽 꼬리가 긴 거래금액에 비교적 견고하다. 하지만 비대칭 분포의 정상적인 고액 주문도 이상치로 과다 표시할 수 있다. 반품과 취소는 음수라는 비즈니스 의미가 있으므로 별도로 보존하고, 양수 구매 금액의 극단값만 상한·하한으로 클리핑했다. 원본 `amount`도 남겨 처리 전후를 검증할 수 있게 했다.
"""
        ),
        code(
            """
outliers_before = analyzer.detect_outliers("amount", positive_only=True)
# 원본 amount는 보존하고 양수 구매의 IQR 극단값만 amount_clean에 클리핑한다.
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
# 제목과 x/y축 레이블 증거를 표로 먼저 제시한 뒤 실제 PNG를 표시한다.
visualization_evidence = pd.read_csv(OUTPUTS / "visualization_evidence.csv")
display(visualization_evidence)

# 서로 다른 분석 목적을 가진 정적 그래프를 순서대로 노트북에 표시한다.
for filename in [
    "02_outlier_boxplot.png",
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

각 지표를 사분위 점수 1~4로 바꾸고 `VIP`, `Loyal`, `New`, `Churned`로 분류한다. 사분위는 사전 비즈니스 임계값이 없는 탐색 단계에서 각 점수 구간에 관측치를 충분히 확보하면서 네 운영 그룹과 연결하기 쉬워 선택했다. 3분위는 경계가 거칠어 그룹이 커지고, 5분위는 경계 근처 고객 이동과 작은 그룹을 늘릴 수 있다. 운영 전에는 30/90/180일 같은 실제 캠페인·구매주기 임계값과 비교해야 한다. 기준일은 데이터의 마지막 유효 구매일 다음 날인 2011-12-10이다.
"""
        ),
        code(
            """
# 마지막 유효 구매일 다음 날을 기준으로 RFM을 계산하고 네 세그먼트로 분류한다.
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
# 세그먼트별 count/mean/median/std를 계산해 해석의 수치 근거를 남긴다.
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

1. **VIP 유지** — **(근거)** VIP는 고객의 27.8%지만 조정 매출의 63.4%를 만든다. **(실행)** 등급별 선공개와 서비스 혜택을 제공한다. 기대효과는 핵심 매출과 반복구매율 방어다. **(검증)** 대조군보다 90일 반복구매율·이탈률이 개선되지 않거나 증분 공헌이익이 0 이하면 가설을 지지하지 않는다.
2. **Churned 재활성화** — **(근거)** 고객의 50.0%가 Churned이며 평균 Recency는 179.5일이다. **(실행)** 기간 제한 인센티브를 무작위 대조 실험으로 운영한다. 기대효과는 전체 할인 없이 휴면 고객을 회수하는 것이다. **(검증)** 홀드아웃 대비 재활성화율·증분 공헌이익이 개선되지 않으면 가설을 지지하지 않으며, 이탈 사유 데이터가 추가로 필요하다.
3. **New의 두 번째 구매 유도** — **(근거)** New는 531명(16.3%)이고 구매빈도 중앙값은 1회다. **(실행)** 첫 구매 후 14일 내 사용 가이드와 연관상품 추천을 보낸다. 기대효과는 1회 구매자의 2회차 전환이다. **(검증)** 14/30일 2회차 구매율이 대조군보다 높지 않거나 반품 증가가 효과를 상쇄하면 가설을 지지하지 않는다.

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
        markdown(
            """
## 9. 다음 단계: 예측 모델 확장

권장 타깃은 기준일 이후 90일 동안 구매가 없으면 `churn_90d=1`인 이진 레이블이다. 기준일 이전 관측창에서 `Recency`, `Frequency`, `Monetary`, 평균 주문금액, 반품률, 구매 상품 수, 활동 개월 수, 국가, 최근 30/60/90일 구매 횟수와 금액을 피처로 사용할 수 있다. 미래 주문, 사후 RFM 세그먼트, 타깃 기간의 구매금액은 데이터 누수이므로 제외한다. 시간순 train/validation/test 분할과 Recall·PR-AUC·캘리브레이션을 함께 평가한다.
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    # 고정 버전 문자열 대신 실제 빌드 인터프리터를 기록한다. 재현성 판단에는 아래
    # 실행 리포트의 Python 버전과 노트북 SHA-256을 함께 사용한다.
    notebook.metadata.language_info = {
        "name": "python",
        "version": platform.python_version(),
    }
    return notebook


def main() -> None:
    # 모든 경로를 스크립트 위치에서 계산하므로 어느 디렉터리에서 호출해도 같은
    # 제출 파일을 대상으로 한다.
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "notebooks" / "analysis_report.ipynb"
    report_path = project_root / "outputs" / "notebook_execution_report.json"
    log_path = project_root / "outputs" / "notebook_execution.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    # nbclient가 프로젝트 루트를 작업 경로로 사용해 모든 코드 셀을 실제 실행한다.
    # 600초 제한은 무한 대기를 막되 25,000행 분석에는 충분한 여유를 둔다.
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
    )
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    try:
        # 실행이 끝난 뒤에만 notebook을 덮어써 실패한 부분 실행본을 제출하지 않는다.
        client.execute()
        nbf.write(notebook, output)
    except Exception as exc:
        # 실행 실패도 구조화된 JSON과 로그로 남겨 원인을 재현할 수 있게 한다.
        failed_report = {
            "status": "failed",
            "started_at": started_at.isoformat(),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "executor": "nbclient.NotebookClient",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        report_path.write_text(
            json.dumps(failed_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_path.write_text(
            "NOTEBOOK EXECUTION FAILED\n"
            + json.dumps(failed_report, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [
        item
        for cell in code_cells
        for item in cell.get("outputs", [])
        if item.output_type == "error"
    ]
    # 예외가 없어도 셀 출력에 error가 남았는지 별도로 세고, 실행된 노트북의 해시를
    # 저장해 이후 코드가 수동 변경되지 않았음을 verify_project.py에서 검증한다.
    notebook_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    finished_at = datetime.now().astimezone()
    report = {
        "status": "success" if not error_outputs else "failed",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "executor": "nbclient.NotebookClient",
        "kernel": "python3",
        "python_version": platform.python_version(),
        "notebook": str(output.relative_to(project_root)),
        "sha256": notebook_hash,
        "total_cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(cell.execution_count is not None for cell in code_cells),
        "error_outputs": len(error_outputs),
        "execution_counts": [cell.execution_count for cell in code_cells],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_lines = [
        "NOTEBOOK EXECUTION SUCCESS",
        f"executor={report['executor']}",
        f"started_at={report['started_at']}",
        f"finished_at={report['finished_at']}",
        f"duration_seconds={report['duration_seconds']}",
        f"python_version={report['python_version']}",
        f"code_cells={report['code_cells']}",
        f"executed_code_cells={report['executed_code_cells']}",
        f"error_outputs={report['error_outputs']}",
        f"execution_counts={report['execution_counts']}",
        f"sha256={report['sha256']}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Built and executed {output}")
    print(f"Execution evidence: {report_path} and {log_path}")


if __name__ == "__main__":
    main()
