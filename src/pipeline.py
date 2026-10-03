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


# 분석 파이프라인이 기대하는 최소 데이터 계약이다. 입력 단계에서 검증해
# 잘못된 스키마가 후속 통계나 RFM 결과를 조용히 왜곡하지 않게 한다.
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

    # 동료 검토 포인트: 경계값을 불변 객체로 묶어 탐지와 처리 단계가 정확히
    # 같은 Q1/Q3/IQR 정의를 공유하고, 계산 후 실수로 변경되지 않게 한다.
    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float


class DataAnalyzer:
    """Load, validate, clean, enrich, and segment retail transactions."""

    # 이 클래스는 DataFrame 상태를 보유하되, 로드·결측·특징·이상치·RFM을
    # 별도 메서드로 나눈다. 각 책임을 독립적으로 테스트하거나 교체하기 위해서다.
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
        # 임계값과 최소 규모를 생성자 인자로 두면 과제 데이터뿐 아니라 규모와
        # 이상치 정책이 다른 환경에서도 같은 클래스를 재사용할 수 있다.
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

        # CSV에는 이미지가 평탄화된 문자열로 저장되므로, 모든 행을 동일한
        # float32 배열로 복원한 뒤 한 번에 벡터 연산할 수 있게 준비한다.
        if isinstance(value, np.ndarray):
            return value.astype(np.float32, copy=False)
        if pd.isna(value):
            return np.array([], dtype=np.float32)
        cleaned = str(value).strip().strip("[]").replace(",", " ")
        return np.fromstring(cleaned, sep=" ", dtype=np.float32)

    def _require_loaded(self) -> pd.DataFrame:
        # 상태 기반 API의 호출 순서를 명시적으로 검사한다. None을 후속 메서드에서
        # 암묵적으로 실패시키는 것보다 사용자가 먼저 load_data()를 호출하게 안내한다.
        if self.df is None:
            raise RuntimeError("Call load_data() before this operation")
        return self.df

    def load_data(self) -> pd.DataFrame:
        """Load CSV data, parse dates/images, and enforce the mission schema."""

        # 파일 존재 여부, 필수 열, 최소 크기를 가장 앞에서 확인하는 fail-fast
        # 방식으로 불완전한 데이터가 분석 단계까지 흘러가는 것을 막는다.
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

        # 변환할 수 없는 날짜는 NaT로 표시해 결측치 처리 단계에서 일관되게 다룬다.
        frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
        frame["product_image"] = frame["product_image"].map(self.parse_image_array)
        self.df = frame
        return self.df

    def overview(self) -> dict[str, Any]:
        """Return compact, serializable exploration facts."""

        frame = self._require_loaded()
        # row_id가 있으면 원본 행의 정체성을 기준으로 중복을 확인한다.
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
            # 같은 상품 코드는 같은 상품명을 공유한다는 도메인 가정을 사용한다.
            # 전역 최빈값보다 상품 정체성을 보존하면서 결측 설명을 대치할 수 있다.
            # 단, 최빈값 반복은 그룹 내부 다양성과 분산을 줄이고 그룹 차이를 실제보다
            # 크게 보이게 할 수 있으므로 운영에서는 대치 플래그와 원본을 함께 보존한다.
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

        # 과제 명세의 정확한 메서드명을 유지하면서 실제 정책 구현은 한 곳에만 둔다.
        return self.handle_missing(strategy=strategy, group_col=group_col)

    def engineer_features(self) -> pd.DataFrame:
        """Create numeric, text, and image statistics with vector operations."""

        frame = self._require_loaded()
        # 숫자 특징은 Python 반복문 대신 NumPy 배열끼리 곱해 벡터화한다.
        frame["amount"] = frame["quantity"].to_numpy() * frame["unit_price"].to_numpy()
        frame["word_count"] = (
            frame["description"].fillna("").astype(str).str.split().str.len().astype("int64")
        )

        lengths = frame["product_image"].map(len)
        if lengths.nunique() != 1 or lengths.iat[0] == 0:
            raise ValueError("All product_image arrays must have one non-zero length")
        # 행별 배열을 하나의 연속 2차원 행렬로 쌓아 이미지별 평균과 표준편차를
        # axis=1 연산 한 번으로 계산한다. Python 행 반복을 피하고 NumPy의 컴파일된
        # 루프와 SIMD 친화적 메모리 접근을 사용한다는 것이 핵심 설계 이유다.
        # 대신 전체 행렬이 RAM에 올라가므로 대규모 데이터에서는 배치 처리가 필요하다.
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
            # 반품·취소는 음수라는 별도 의미가 있으므로 구매금액 이상치 계산에서는
            # 양수 거래만 사용하고 음수 행은 원본 그대로 보존한다.
            values = values[values > 0]
        if values.empty:
            raise ValueError(f"No values available for {column}")
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        # IQR은 정규분포를 가정하지 않아 치우친 금액 분포에 강건하지만, 정상적인
        # 고액 주문도 이상치로 분류할 수 있으므로 경계를 절대 오류 판정으로 보지 않는다.
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
            # 행 삭제는 희소한 고액 고객을 잃을 수 있어 기본 분석에서는 clip을 쓴다.
            # 원본 열을 덮어쓰지 않고 정제 열을 별도로 만들어 처리 전후를 검증한다.
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
        # 고객 식별이 가능하고 실제 구매로 볼 수 있는 행만 RFM 집계에 사용한다.
        # 주문번호가 C로 시작하는 취소 주문과 음수/0 수량·금액은 제외한다.
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
        # 동일 값이 많아 qcut 경계가 겹치는 문제를 피하려고 먼저 안정적인 순위를
        # 만든다. Recency는 낮을수록 좋으므로 점수 방향을 반대로 배정한다.
        # 4분위는 사전 업무 임계값이 없을 때 네 운영 그룹과 연결하기 쉬운 탐색 기준이며,
        # 실제 캠페인에서는 30/90/180일 같은 업무 기준으로 민감도를 다시 확인해야 한다.
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
        # 기준일을 지정하지 않으면 데이터가 관측된 마지막 구매일의 다음 날을
        # 사용하여 모든 고객의 Recency가 0 이상이 되게 한다.
        analysis_date = (
            pd.Timestamp(reference_date)
            if reference_date is not None
            else transactions[date_col].max().normalize() + pd.Timedelta(days=1)
        )

        # 고객별 마지막 구매일, 고유 주문 수, 조정 구매금액 합계를 한 번에 집계한다.
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

        # np.select는 첫 번째 참 조건을 사용하므로 조건 순서 자체가 비즈니스 규칙이다.
        # 최근성·빈도·금액이 모두 높은 고객을 VIP로 먼저 분리한 뒤 신규, 이탈 위험,
        # 나머지 고객을 Loyal로 구분한다.
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
        # 고객 비중과 매출 비중을 함께 제공해 규모와 경제적 중요도를 구분한다.
        summary["customer_share"] = summary["customers"] / summary["customers"].sum()
        summary["revenue_share"] = summary["total_monetary"] / summary["total_monetary"].sum()
        return summary.sort_values("total_monetary", ascending=False)
