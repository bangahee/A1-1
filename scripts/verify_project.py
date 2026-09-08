"""Verify A1-1 deliverables, outputs, and notebook execution state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import DataAnalyzer


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"Not a PNG: {path}")
    return struct.unpack(">II", signature[16:24])


def main() -> None:
    required = [
        ROOT / "src/pipeline.py",
        ROOT / "notebooks/analysis_report.ipynb",
        ROOT / "README.md",
        ROOT / "data/online_retail_sample.csv",
        ROOT / "data/source_evidence.png",
        ROOT / "outputs/rfm_customers.csv",
        ROOT / "outputs/segment_summary.csv",
        ROOT / "outputs/data_quality_report.json",
        ROOT / "outputs/dtype_summary.csv",
        ROOT / "outputs/rfm_correlations.csv",
        ROOT / "outputs/notebook_execution_report.json",
        ROOT / "outputs/notebook_execution.log",
        ROOT / "figures/CAPTIONS.md",
        ROOT / "docs/METHODOLOGY.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"Missing required files: {missing}"

    data = pd.read_csv(ROOT / "data/online_retail_sample.csv", low_memory=False)
    assert data.shape[0] >= 1_000, data.shape
    assert data.shape[1] >= 8, data.shape
    assert {"customer_id", "order_date", "amount", "product_image"}.issubset(data.columns)
    pd.to_datetime(data["order_date"], errors="raise")
    assert callable(getattr(DataAnalyzer, "handle_missing", None))

    rfm = pd.read_csv(ROOT / "outputs/rfm_customers.csv")
    assert set(rfm["Segment"]) == {"VIP", "Loyal", "New", "Churned"}
    assert rfm[["Recency", "Frequency", "Monetary"]].notna().all().all()

    descriptive = pd.read_csv(ROOT / "outputs/descriptive_statistics.csv")
    assert "dtype" in descriptive.columns
    dtype_summary = pd.read_csv(ROOT / "outputs/dtype_summary.csv")
    assert dtype_summary["column_count"].sum() == data.shape[1]

    correlations = pd.read_csv(ROOT / "outputs/rfm_correlations.csv", index_col=0)
    assert {"Recency", "Frequency", "Monetary", "RFM_score"}.issubset(correlations.columns)

    figures = sorted((ROOT / "figures").glob("*.png"))
    assert len(figures) >= 6, f"Expected 6+ PNG figures, got {len(figures)}"
    for figure in figures:
        width, height = png_dimensions(figure)
        assert width >= 900 and height >= 500, (figure.name, width, height)

    notebook = nbformat.read(ROOT / "notebooks/analysis_report.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors, f"Notebook contains error outputs: {errors}"

    execution = json.loads(
        (ROOT / "outputs/notebook_execution_report.json").read_text(encoding="utf-8")
    )
    assert execution["status"] == "success"
    assert execution["error_outputs"] == 0
    assert execution["executed_code_cells"] == execution["code_cells"] == len(code_cells)
    assert execution["sha256"] == hashlib.sha256(
        (ROOT / "notebooks/analysis_report.ipynb").read_bytes()
    ).hexdigest()
    assert "NOTEBOOK EXECUTION SUCCESS" in (
        ROOT / "outputs/notebook_execution.log"
    ).read_text(encoding="utf-8")

    quality = json.loads((ROOT / "outputs/data_quality_report.json").read_text(encoding="utf-8"))
    assert quality["overview"]["rows"] == 25_000
    assert quality["missing_values"]["after"]["description"] == 0
    assert quality["outliers"]["after_count"] == 0

    print("A1-1 verification passed")
    print(f"- data: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"- RFM: {len(rfm):,} customers, 4 segments")
    print(f"- figures: {len(figures)} valid PNG files")
    print(f"- notebook: {len(code_cells)} executed code cells, 0 errors")
    print("- evidence: dtype, correlations, captions, and nbclient execution log")


if __name__ == "__main__":
    main()
