"""Reusable data-analysis pipeline for customer RFM segmentation.

Only NumPy and Pandas are used for data processing.  The image-like feature is
stored in CSV as a flattened numeric array and parsed directly with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "order_date",
    "unit_price",
    "customer_id",
    "country",
    "product_image",
}


@dataclass(frozen=True)
class OutlierBounds:
    """IQR limits for one numeric column."""

    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float


class DataAnalyzer:
    """Load, validate, clean, enrich, and segment retail transactions."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        outlier_threshold: float = 1.5,
        min_rows: int = 1_000,
        min_columns: int = 8,
    ) -> None:
        if outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")
        self.data_path = Path(data_path)
        self.outlier_threshold = float(outlier_threshold)
        self.min_rows = int(min_rows)
        self.min_columns = int(min_columns)
        self.df: pd.DataFrame | None = None
        self.rfm: pd.DataFrame | None = None

    @staticmethod
    def parse_image_array(value: Any) -> np.ndarray:
        """Parse a finite flattened CSV image using compact float32 storage.

        Missing or empty inputs become empty arrays and are rejected together in
        :meth:`engineer_features`; silently inventing pixels would bias the image
        statistics.  float32 halves working memory versus float64 while retaining
        far more precision than the source 0-255 grayscale-like values require.
        """

        if isinstance(value, np.ndarray):
            return value.astype(np.float32, copy=False)
        if pd.isna(value):
            return np.array([], dtype=np.float32)
        cleaned = str(value).strip().strip("[]").replace(",", " ")
        return np.fromstring(cleaned, sep=" ", dtype=np.float32)

    def _require_loaded(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call load_data() before this operation")
        return self.df

    def load_data(self) -> pd.DataFrame:
        """Load CSV data, parse dates/images, and enforce the mission schema."""

        if not self.data_path.exists():
            raise FileNotFoundError(self.data_path)
        frame = pd.read_csv(self.data_path, low_memory=False)
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        if len(frame) < self.min_rows:
            raise ValueError(f"Expected at least {self.min_rows:,} rows, got {len(frame):,}")
        if frame.shape[1] < self.min_columns:
            raise ValueError(
                f"Expected at least {self.min_columns} columns, got {frame.shape[1]}"
            )

        frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
        frame["product_image"] = frame["product_image"].map(self.parse_image_array)
        self.df = frame
        return self.df

    def overview(self) -> dict[str, Any]:
        """Return compact, serializable exploration facts."""

        frame = self._require_loaded()
        return {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "duplicate_rows": int(frame.duplicated(subset=["row_id"]).sum())
            if "row_id" in frame
            else int(frame.duplicated().sum()),
            "date_min": frame["order_date"].min().isoformat(),
            "date_max": frame["order_date"].max().isoformat(),
            "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        }

    def handle_missing(
        self,
        strategy: str = "group_mode",
        group_col: str = "stock_code",
    ) -> dict[str, dict[str, int]]:
        """Impute descriptions by product group and retain missing customer IDs.

        Customer IDs are identifiers, so inventing them would corrupt RFM.  They
        remain missing here and are excluded only in the RFM transaction view.
        """

        frame = self._require_loaded()
        before = frame.isna().sum().astype(int).to_dict()
        if strategy == "group_mode":
            if group_col not in frame:
                raise KeyError(group_col)
            group_value = frame.groupby(group_col, dropna=False)["description"].transform(
                lambda values: values.mode().iat[0] if not values.mode().empty else np.nan
            )
            frame["description"] = frame["description"].fillna(group_value)
            frame["description"] = frame["description"].fillna("UNKNOWN PRODUCT")
        elif strategy == "drop":
            frame.dropna(subset=["description", "order_date"], inplace=True)
        else:
            raise ValueError("strategy must be 'group_mode' or 'drop'")

        frame.dropna(subset=["order_date"], inplace=True)
        after = frame.isna().sum().astype(int).to_dict()
        return {"before": before, "after": after}

    def handle_missing_values(
        self,
        strategy: str = "group_mode",
        group_col: str = "stock_code",
    ) -> dict[str, dict[str, int]]:
        """Backward-compatible alias for the public :meth:`handle_missing` API."""

        return self.handle_missing(strategy=strategy, group_col=group_col)

    def engineer_features(self) -> pd.DataFrame:
        """Create numeric, text, and image statistics with vector operations."""

        frame = self._require_loaded()
        frame["amount"] = frame["quantity"].to_numpy() * frame["unit_price"].to_numpy()
        frame["word_count"] = (
            frame["description"].fillna("").astype(str).str.split().str.len().astype("int64")
        )

        lengths = frame["product_image"].map(len)
        if lengths.nunique() != 1 or lengths.iat[0] == 0:
            raise ValueError("All product_image arrays must have one non-zero length")
        image_matrix = np.stack(frame["product_image"].to_numpy()).astype(
            np.float32, copy=False
        )
        if not np.isfinite(image_matrix).all():
            raise ValueError("product_image arrays must contain only finite values")
        frame["image_mean"] = image_matrix.mean(axis=1)
        frame["image_std"] = image_matrix.std(axis=1)
        return frame

    def outlier_bounds(
        self,
        column: str,
        threshold: float | None = None,
        *,
        positive_only: bool = False,
    ) -> OutlierBounds:
        """Compute IQR limits directly from Q1 and Q3."""

        frame = self._require_loaded()
        factor = self.outlier_threshold if threshold is None else float(threshold)
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if positive_only:
            values = values[values > 0]
        if values.empty:
            raise ValueError(f"No values available for {column}")
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        return OutlierBounds(q1, q3, iqr, q1 - factor * iqr, q3 + factor * iqr)

    def detect_outliers(
        self,
        column: str,
        threshold: float | None = None,
        *,
        positive_only: bool = False,
    ) -> pd.DataFrame:
        """Return rows outside direct IQR limits."""

        frame = self._require_loaded()
        bounds = self.outlier_bounds(column, threshold, positive_only=positive_only)
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.lt(bounds.lower) | values.gt(bounds.upper)
        if positive_only:
            mask &= values.gt(0)
        return frame.loc[mask].copy()

    def treat_outliers(
        self,
        column: str,
        *,
        method: str = "clip",
        output_col: str | None = None,
        positive_only: bool = False,
    ) -> dict[str, Any]:
        """Clip or remove IQR outliers and report the before/after change."""

        frame = self._require_loaded()
        bounds = self.outlier_bounds(column, positive_only=positive_only)
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.lt(bounds.lower) | values.gt(bounds.upper)
        if positive_only:
            mask &= values.gt(0)
        before_count = int(mask.sum())

        if method == "clip":
            target = output_col or f"{column}_clean"
            frame[target] = values
            eligible = values.gt(0) if positive_only else values.notna()
            frame.loc[eligible, target] = values.loc[eligible].clip(bounds.lower, bounds.upper)
            after_mask = frame[target].lt(bounds.lower) | frame[target].gt(bounds.upper)
            if positive_only:
                after_mask &= frame[target].gt(0)
            after_count = int(after_mask.sum())
        elif method == "remove":
            self.df = frame.loc[~mask].copy()
            target = column
            after_count = 0
        else:
            raise ValueError("method must be 'clip' or 'remove'")

        return {
            "column": column,
            "method": method,
            "output_column": target,
            "before_count": before_count,
            "after_count": after_count,
            "bounds": bounds.__dict__,
        }

    def rfm_transactions(
        self,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Return completed purchases suitable for RFM aggregation."""

        frame = self._require_loaded()
        invoice = frame["invoice_no"].astype(str)
        valid = (
            frame[customer_col].notna()
            & frame[date_col].notna()
            & frame[amount_col].gt(0)
            & frame["quantity"].gt(0)
            & ~invoice.str.startswith("C", na=False)
        )
        return frame.loc[valid].copy()

    @staticmethod
    def _quartile_score(series: pd.Series, *, high_is_good: bool) -> pd.Series:
        labels = [1, 2, 3, 4] if high_is_good else [4, 3, 2, 1]
        ranked = series.rank(method="first")
        return pd.qcut(ranked, q=4, labels=labels).astype("int64")

    def calculate_rfm(
        self,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "amount",
        *,
        reference_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Aggregate RFM, score quartiles, and assign four customer segments."""

        transactions = self.rfm_transactions(customer_col, date_col, amount_col)
        if transactions.empty:
            raise ValueError("No valid transactions remain for RFM")
        analysis_date = (
            pd.Timestamp(reference_date)
            if reference_date is not None
            else transactions[date_col].max().normalize() + pd.Timedelta(days=1)
        )

        rfm = transactions.groupby(customer_col).agg(
            last_purchase=(date_col, "max"),
            Frequency=("invoice_no", "nunique"),
            Monetary=(amount_col, "sum"),
        )
        rfm["Recency"] = (analysis_date - rfm.pop("last_purchase")).dt.days
        rfm = rfm[["Recency", "Frequency", "Monetary"]]
        rfm["R_score"] = self._quartile_score(rfm["Recency"], high_is_good=False)
        rfm["F_score"] = self._quartile_score(rfm["Frequency"], high_is_good=True)
        rfm["M_score"] = self._quartile_score(rfm["Monetary"], high_is_good=True)
        rfm["RFM_score"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

        conditions = [
            rfm["R_score"].ge(3) & rfm["F_score"].ge(3) & rfm["M_score"].ge(3),
            rfm["R_score"].ge(3) & rfm["F_score"].le(2),
            rfm["R_score"].le(2),
        ]
        rfm["Segment"] = np.select(conditions, ["VIP", "New", "Churned"], default="Loyal")
        rfm.index = rfm.index.astype("Int64")
        self.rfm = rfm.sort_values(["RFM_score", "Monetary"], ascending=False)
        return self.rfm

    def segment_summary(self) -> pd.DataFrame:
        """Summarize each segment with customer count and RFM evidence."""

        if self.rfm is None:
            raise RuntimeError("Call calculate_rfm() before segment_summary()")
        summary = self.rfm.groupby("Segment", observed=True).agg(
            customers=("RFM_score", "size"),
            mean_recency=("Recency", "mean"),
            median_frequency=("Frequency", "median"),
            mean_frequency=("Frequency", "mean"),
            mean_monetary=("Monetary", "mean"),
            total_monetary=("Monetary", "sum"),
        )
        summary["customer_share"] = summary["customers"] / summary["customers"].sum()
        summary["revenue_share"] = summary["total_monetary"] / summary["total_monetary"].sum()
        return summary.sort_values("total_monetary", ascending=False)
