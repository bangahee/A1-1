"""한국어 분석 노트북을 생성하고 현재 Python의 새 커널로 전체 실행한다."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager


def markdown(text: str):
    """설명 셀을 만든다."""
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    """코드 셀을 만든다."""
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    """저장된 그림을 전제로 하지 않고 현재 입력에서 분석을 재실행하는 보고서다."""
    cells = [
        markdown("# 쇼핑몰 거래 분석과 RFM 고객 세분화\n\n거래 전처리부터 배열 특징, 통계, 고객군과 실행 가설까지 같은 분석 결과에서 살펴본다."),
        markdown("## 1. 환경과 분석 실행\n\nUCI Online Retail의 거래 표본을 사용한다. CSV의 수치·범주·날짜를 복원하고 교육용 배열을 처리한다. 분석 실행은 통계와 그림을 새로 만든다."),
        code('''from pathlib import Path
import sys
from IPython.display import Image, Markdown, display
import numpy as np
import pandas as pd

# 저장소 루트 또는 notebooks 하위에서 실행해도 같은 파일을 찾는다.
start = Path.cwd().resolve()
ROOT = next((p for p in [start, *start.parents] if (p / "src/pipeline.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("저장소 안에서 노트북을 실행하세요")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_analysis import run_analysis, FIGURE_METADATA
from scripts.reporting import analytical_notes, korean_summary, SEGMENT_LABELS, write_readme

result = run_analysis(ROOT / "data/online_retail_sample.csv", ROOT / "outputs", ROOT / "figures")
notes = analytical_notes(result)
raw = result["raw_frame"]
df = result["frame"]
rfm = result["rfm"]
write_readme(result, ROOT)
pd.set_option("display.max_columns", 30)
print("입력부터 통계·그림까지 현재 데이터로 생성했습니다.")'''),
        markdown("## 2. 기본 탐색과 기술통계"),
        code('''print(f"원본 표본: {raw.shape[0]:,}행 × {raw.shape[1]}열")
display(raw.drop(columns="product_image").head())
raw.info()
display(raw.select_dtypes(include="number").describe().T)
display(raw.dtypes.astype(str).value_counts().rename("열 수").to_frame())
display(pd.Series(result["overview"]["missing_by_column"], name="결측 수").to_frame())
display(Markdown(notes["statistics"]))
display(Markdown(notes["missing"]))'''),
        markdown("## 3. 파이프라인 책임과 인자"),
        code('''display(Markdown(notes["design"]))'''),
        markdown("## 4. 결측 처리와 그룹 통계 비교\n\n아래 수치형 대치 예제는 실제 거래와 분리한 교육용 데이터다."),
        code('''display(pd.DataFrame(result["missing_report"]).rename(columns={"before": "처리 전", "after": "처리 후"}))
display(result["imputation_values"])
display(result["imputation_statistics"].round(3))
print("그룹 전체가 결측일 때는 전체 통계 대치 또는 결측 유지 정책을 선택합니다.")'''),
        markdown("## 5. 수치·텍스트·배열 특징 공학"),
        code('''display(df[["amount", "word_count", "image_mean", "image_std", "description_imputed"]].head())
display(Markdown(notes["vector"]))
display(Markdown(notes["image"]))
display(Image(filename=str(ROOT / "figures/08_educational_array.png")))'''),
        markdown("## 6. IQR 이상치와 분포 변화"),
        code('''display(pd.Series(result["outlier_report"]).to_frame("처리 결과"))
display(Markdown(notes["iqr"]))
display(Image(filename=str(ROOT / "figures/02_outlier_boxplot.png")))'''),
        markdown("## 7. RFM 정의와 고객군"),
        code('''display(Markdown(notes["rfm"]))
display(rfm.head(10))
display(korean_summary(result["segment_summary"]))
display(result["group_statistics"].rename(index=SEGMENT_LABELS).round(3))
display(result["score_ranges"])
display(Image(filename=str(ROOT / "figures/07_segment_summary_table.png")))'''),
        markdown("## 8. 상관분석과 여섯 종류의 시각화"),
        code('''# 현재 커널의 고객별 지표로 상관계수를 직접 다시 계산한다.
correlations = rfm[["Recency", "Frequency", "Monetary", "RFM_score"]].corr()
display(correlations)
display(Markdown(notes["correlations"]))
display(pd.DataFrame(FIGURE_METADATA).rename(columns={"file": "파일", "chart_type": "종류", "title": "제목", "x_axis": "가로축", "y_axis": "세로축"}))
for meta in FIGURE_METADATA:
    display(Image(filename=str(ROOT / "figures" / meta["file"])))
print("월별 그림의 마지막 12월은 부분 월이므로 전월과 직접 비교하지 않습니다.")'''),
        markdown("## 9. 점수 기준과 금액 처리의 민감도"),
        code('''display(Markdown(notes["sensitivity"]))
display(result["sensitivity"].round(3))'''),
        markdown("## 10. 비즈니스 제안"),
        code('''for insight in result["insights"].values():
    display(Markdown(f"### {insight['title']}\\n\\n**근거**: {insight['evidence']}\\n\\n**실행과 기대 효과**: {insight['action']} {insight['expected_effect']}\\n\\n**검증**: {insight['validation_data']}"))'''),
        markdown("## 11. 대규모 처리와 머신러닝 확장"),
        code('''display(result["scaling"].round(3))
display(Markdown(notes["scaling"]))
display(Markdown(notes["ml"]))
display(Markdown(notes["limitations"]))'''),
    ]
    # 실행 결과와 함께 읽을 수 있도록 주요 의사결정 근거는 마크다운 셀 자체에도 남긴다.
    from scripts.reporting import DESIGN_NOTES, VECTOR_NOTES, IMAGE_NOTES, SCALING_NOTES, ML_NOTES
    cells.insert(6, markdown(DESIGN_NOTES))
    for heading, body in [("배열 연산의 선택 이유", VECTOR_NOTES), ("이미지 배열의 적용 범위", IMAGE_NOTES),
                          ("대규모 처리의 가정", SCALING_NOTES), ("미래 타깃의 관측 조건", ML_NOTES)]:
        cells.append(markdown(f"### {heading}\n\n{body}"))
    cells.append(markdown("### 결측 정책과 점수 선택\n\n상품명은 그룹 최빈값, 수치형은 그룹 평균·중앙값으로 다룬다. 평균은 극단값에 민감하고 대치는 그룹 내부 분산을 줄일 수 있다. 고객 식별자는 대치하지 않는다. RFM은 사전 구매주기 기준이 없는 탐색에서 평균 순위 백분위 네 구간을 기본으로 사용하며, 동점을 보존하고 실제 운영 임계값과 민감도를 비교한다."))
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": platform.python_version()}
    return notebook


def execute_notebook(notebook: nbf.NotebookNode, working_dir: Path) -> nbf.NotebookNode:
    """호출 Python과 같은 환경의 새 커널에서 전체 셀을 실행한다."""
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(working_dir)}})
    try:
        return client.execute()
    finally:
        # 외부에서 전달한 커널도 성공·실패와 관계없이 명시적으로 종료한다.
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    notebook = build_notebook()
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    output = root / "notebooks/analysis_report.ipynb"
    report_path = root / "outputs/notebook_execution_report.json"
    log_path = root / "outputs/notebook_execution.log"
    try:
        # notebooks 디렉터리에서 시작해 경로 탐색까지 함께 검증한다.
        execute_notebook(notebook, root / "notebooks")
        nbf.write(notebook, output)
    except Exception as exc:
        report = {"status": "failed", "started_at": started_at.isoformat(), "error_type": type(exc).__name__, "error": str(exc)}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text("노트북 전체 실행 실패\n" + str(exc), encoding="utf-8")
        raise
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [o for cell in code_cells for o in cell.outputs if o.output_type == "error"]
    report = {"status": "success", "started_at": started_at.isoformat(),
              "finished_at": datetime.now().astimezone().isoformat(), "duration_seconds": round(time.perf_counter()-started, 3),
              "executor": "nbclient.NotebookClient", "kernel": "python3", "python_version": platform.python_version(),
              "python_executable": sys.executable, "working_directory": "notebooks", "notebook": "notebooks/analysis_report.ipynb",
              "total_cells": len(notebook.cells), "code_cells": len(code_cells),
              "executed_code_cells": sum(c.execution_count is not None for c in code_cells), "error_outputs": len(errors),
              "execution_counts": [c.execution_count for c in code_cells], "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text("노트북 전체 실행 성공\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"새 커널에서 코드 셀 {len(code_cells)}개 실행, 오류 {len(errors)}개")


if __name__ == "__main__":
    main()
