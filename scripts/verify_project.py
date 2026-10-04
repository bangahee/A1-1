"""원본 입력·저장 결과·한국어 문서·노트북 실행 기록의 일관성을 확인한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.pipeline import DataAnalyzer
from scripts.reporting import SEGMENT_LABELS, analytical_notes, korean_summary, markdown_table


def png_dimensions(path: Path) -> tuple[int, int]:
    """디코더 없이 PNG 헤더에서 크기를 확인한다."""
    with path.open("rb") as handle:
        header = handle.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"PNG 형식이 아닙니다: {path}"
    return struct.unpack(">II", header[16:24])


def main() -> None:
    """현재 입력으로 핵심 결과를 재계산하고 저장된 결과와 비교한다."""
    required = ["src/pipeline.py", "notebooks/analysis_report.ipynb", "README.md", "data/DATASET.md",
                "outputs/environment.json", "outputs/rfm_sensitivity.csv", "outputs/rfm_score_ranges.csv",
                "outputs/monetary_comparison.csv", "outputs/imputation_example_statistics.csv",
                "outputs/notebook_execution_report.json", "figures/CAPTIONS.md", "docs/METHODOLOGY.md",
                "docs/IMPLEMENTATION_EVIDENCE.md"]
    assert all((ROOT / p).is_file() for p in required), "필수 파일이 없습니다"
    data = pd.read_csv(ROOT / "data/online_retail_sample.csv")
    assert data.shape[0] >= 1000 and data.shape[1] >= 8, "입력 규모가 부족합니다"
    analyzer = DataAnalyzer(ROOT / "data/online_retail_sample.csv")
    typed = analyzer.load_data()
    assert pd.api.types.is_datetime64_any_dtype(typed.order_date), "날짜 변환이 필요합니다"
    assert typed.select_dtypes(include="number").shape[1] > 0
    assert any(pd.api.types.is_string_dtype(dtype) for dtype in typed.dtypes)
    assert isinstance(typed.product_image.iloc[0], np.ndarray)
    before = analyzer.overview()
    missing = analyzer.handle_missing_values()
    frame = analyzer.engineer_features()
    np.testing.assert_allclose(frame.image_mean, np.stack(frame.product_image).mean(axis=1))
    np.testing.assert_allclose(frame.image_std, np.stack(frame.product_image).std(axis=1))
    assert frame.description_imputed.sum() == before["missing_by_column"]["description"]
    outliers = analyzer.treat_outliers("amount", positive_only=True, output_col="amount_clean")
    current = analyzer.calculate_rfm(amount_col="amount_clean")
    assert current.Segment.nunique() >= 4, "네 고객군이 필요합니다"
    assert set(current.Segment).issubset(SEGMENT_LABELS)
    saved = pd.read_csv(ROOT / "outputs/rfm_customers.csv", index_col=0)
    pd.testing.assert_frame_equal(current, saved, check_dtype=False, check_index_type=False, atol=1e-9)
    transactions = analyzer.rfm_transactions(amount_col="amount_clean")
    calendar_days = (analyzer.rfm_reference_date - transactions.groupby("customer_id").order_date.max().dt.normalize()).dt.days
    np.testing.assert_array_equal(current.Recency, calendar_days.reindex(current.index))
    for metric, score in [("Recency", "R_score"), ("Frequency", "F_score"), ("Monetary", "M_score")]:
        assert current.groupby(metric)[score].nunique().eq(1).all(), "동점 점수가 일치하지 않습니다"
    summary = analyzer.segment_summary()
    pd.testing.assert_frame_equal(summary, pd.read_csv(ROOT / "outputs/segment_summary.csv", index_col=0), check_dtype=False, atol=1e-9)
    correlations = current[["Recency", "Frequency", "Monetary", "RFM_score"]].corr()
    pd.testing.assert_frame_equal(correlations, pd.read_csv(ROOT / "outputs/rfm_correlations.csv", index_col=0), atol=1e-9)
    quality = json.loads((ROOT / "outputs/data_quality_report.json").read_text())
    assert quality["overview"] == before
    assert quality["missing_values"] == missing
    assert quality["outliers"] == outliers
    assert quality["rfm_customers"] == len(current)
    assert quality["rfm_reference_date"] == analyzer.rfm_reference_date.isoformat()
    dtypes = pd.read_csv(ROOT / "outputs/dtype_summary.csv")
    assert dtypes.column_count.sum() == data.shape[1]
    descriptive = pd.read_csv(ROOT / "outputs/descriptive_statistics.csv")
    assert {"dtype", "mean", "std", "25%", "50%", "75%"}.issubset(descriptive.columns)
    visual = pd.read_csv(ROOT / "outputs/visualization_evidence.csv")
    assert set(visual.chart_type) >= {"히스토그램", "박스플롯", "막대그래프", "히트맵", "산점도", "라인차트"}
    for row in visual.itertuples():
        assert all(str(getattr(row, col)).strip() for col in ["title", "x_axis", "y_axis"])
        assert all(re.search("[가-힣]", str(getattr(row, col))) for col in ["title", "x_axis", "y_axis"])
        w, h = png_dimensions(ROOT / "figures" / row.file)
        assert w >= 900 and h >= 500, "그림 해상도가 부족합니다"
    notebook = nbformat.read(ROOT / "notebooks/analysis_report.ipynb", as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert cells and all(c.execution_count is not None for c in cells)
    errors = [o for c in cells for o in c.outputs if o.output_type == "error"]
    assert not errors, "노트북 오류 출력이 있습니다"
    execution = json.loads((ROOT / "outputs/notebook_execution_report.json").read_text())
    assert execution["status"] == "success"
    assert execution["error_outputs"] == 0
    assert execution["executed_code_cells"] == execution["code_cells"] == len(cells)
    assert execution["total_cells"] == len(notebook.cells)
    assert execution["sha256"] == hashlib.sha256((ROOT / "notebooks/analysis_report.ipynb").read_bytes()).hexdigest()
    assert "노트북 전체 실행 성공" in (ROOT / "outputs/notebook_execution.log").read_text()
    markdown_cells = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    assert all(term in markdown_cells for term in ["중앙값", "분산", "SIMD", "90일", "동점"])
    for c in cells:
        for output in c.outputs:
            if "text/markdown" in output.get("data", {}):
                assert "\\n" not in output.data["text/markdown"], "마크다운에 이스케이프 줄바꿈이 남아 있습니다"
    readme = (ROOT / "README.md").read_text()
    assert "평가항목" not in readme and "체크리스트" not in readme and "제출" not in readme
    assert markdown_table(korean_summary(summary)) in readme, "README 수치가 다릅니다"
    for metric1, metric2 in [("Frequency", "Monetary"), ("Recency", "Frequency")]:
        assert f"{correlations.loc[metric1,metric2]:.3f}" in readme
    assert all(readme.count(label) >= 3 for label in ["**(근거)**", "**(실행)**", "**(검증)**"])
    insights = json.loads((ROOT / "outputs/insights.json").read_text())
    assert len(insights) >= 3
    assert all({"evidence", "action", "expected_effect", "validation_data"}.issubset(x) for x in insights.values())
    sensitivity = pd.read_csv(ROOT / "outputs/rfm_sensitivity.csv")
    assert sensitivity.groupby("시나리오")["고객 수"].sum().eq(len(current)).all()
    assert sensitivity["기본 대비 분류 이동률"].between(0, 1).all()
    requirements = (ROOT / "requirements.txt").read_text().lower()
    packages = {re.match(r"^[a-z0-9_.-]+", line).group() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")}
    assert packages == {"numpy", "pandas", "matplotlib", "seaborn"}
    print("분석 결과와 실행 기록의 일관성 검증 통과")
    print(f"입력 {len(data):,}행·{data.shape[1]}열 / 고객 {len(current):,}명·{current.Segment.nunique()}개 고객군")
    print(f"한국어 분석 그림 {len(visual)}종 / 새 커널 코드 셀 {len(cells)}개 / 오류 0개")


if __name__ == "__main__":
    main()
