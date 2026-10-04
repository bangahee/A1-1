"""NumPy·Pandas로 거래 전처리, 배열 특징 생성, 고객 세분화를 수행한다."""

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
    """한 수치형 변수의 사분위수와 IQR 경계를 보관한다."""

    # 경계값을 불변 객체로 묶어 탐지와 처리 단계가 정확히
    # 같은 Q1/Q3/IQR 정의를 공유하고, 계산 후 실수로 변경되지 않게 한다.
    q1: float
    q3: float
    iqr: float
    lower: float
    upper: float


class DataAnalyzer:
    """거래 데이터 상태를 관리하고 각 분석 단계를 독립적으로 실행한다."""

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
        if not np.isfinite(outlier_threshold) or outlier_threshold <= 0:
            raise ValueError("이상치 임계값은 유한한 양수여야 합니다")
        # 임계값과 최소 규모를 생성자 인자로 두면 과제 데이터뿐 아니라 규모와
        # 이상치 정책이 다른 환경에서도 같은 클래스를 재사용할 수 있다.
        self.data_path = Path(data_path)
        self.outlier_threshold = float(outlier_threshold)
        self.min_rows = int(min_rows)
        self.min_columns = int(min_columns)
        self.df: pd.DataFrame | None = None
        self.rfm: pd.DataFrame | None = None
        self.rfm_reference_date: pd.Timestamp | None = None

    @staticmethod
    def parse_image_array(value: Any) -> np.ndarray:
        """CSV의 평탄화 배열을 float32로 복원한다. 빈 입력은 특징 단계에서 거부한다."""

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
            raise RuntimeError("먼저 load_data()로 데이터를 불러오세요")
        return self.df

    def load_data(self) -> pd.DataFrame:
        """CSV를 읽고 필수 열·규모를 확인한 뒤 날짜와 배열 자료형을 복원한다."""

        # 파일 존재 여부, 필수 열, 최소 크기를 가장 앞에서 확인하는 fail-fast
        # 방식으로 불완전한 데이터가 분석 단계까지 흘러가는 것을 막는다.
        if not self.data_path.exists():
            raise FileNotFoundError(self.data_path)
        frame = pd.read_csv(self.data_path, low_memory=False)
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"필수 열이 없습니다: {sorted(missing_columns)}")
        if len(frame) < self.min_rows:
            raise ValueError(f"최소 {self.min_rows:,}행이 필요합니다. 입력: {len(frame):,}행")
        if frame.shape[1] < self.min_columns:
            raise ValueError(
                f"최소 {self.min_columns}열이 필요합니다. 입력: {frame.shape[1]}열"
            )

        # 변환할 수 없는 날짜는 NaT로 표시해 결측치 처리 단계에서 일관되게 다룬다.
        frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
        frame["product_image"] = frame["product_image"].map(self.parse_image_array)
        self.df = frame
        return self.df

    def overview(self) -> dict[str, Any]:
        """규모·관측 기간·중복 식별자·결측 현황을 요약한다."""

        frame = self._require_loaded()
        # row_id가 있으면 원본 행의 정체성을 기준으로 중복을 확인한다.
        return {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "duplicate_rows": int(frame.duplicated(subset=["row_id"]).sum())
            if "row_id" in frame
            else int(frame.drop(columns=["product_image"], errors="ignore").duplicated().sum()),
            "date_min": frame["order_date"].min().isoformat(),
            "date_max": frame["order_date"].max().isoformat(),
            "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        }

    def handle_missing(
        self,
        strategy: str = "group_mode",
        group_col: str = "stock_code",
        *,
        columns: list[str] | None = None,
        fallback: str = "global",
    ) -> dict[str, dict[str, int]]:
        """설명 최빈값 또는 지정 수치 열의 그룹 평균·중앙값으로 결측을 보완한다.

        그룹 전체가 결측이면 global은 전체 통계로 보완하고 keep은 결측을 유지한다.
        식별자는 대치하지 않는다. *_imputed에는 실제로 대치한 위치를 기록한다.
        """

        frame = self._require_loaded()
        before = frame.isna().sum().astype(int).to_dict()
        if fallback not in {"global", "keep"}:
            raise ValueError("fallback은 'global' 또는 'keep'이어야 합니다")
        if strategy == "group_mode":
            if group_col not in frame:
                raise KeyError(group_col)
            missing = frame["description"].isna()
            # 같은 상품 코드는 같은 상품명을 공유한다는 도메인 가정을 사용한다.
            # 전역 최빈값보다 상품 정체성을 보존하면서 결측 설명을 대치할 수 있다.
            # 단, 최빈값 반복은 그룹 내부 다양성과 분산을 줄이고 그룹 차이를 실제보다
            # 크게 보이게 할 수 있으므로 운영에서는 대치 플래그와 원본을 함께 보존한다.
            group_value = frame.groupby(group_col, dropna=False)["description"].transform(
                lambda values: values.mode().iat[0] if not values.mode().empty else np.nan
            )
            frame["description"] = frame["description"].fillna(group_value)
            frame["description"] = frame["description"].fillna("UNKNOWN PRODUCT")
            frame["description_imputed"] = frame.get(
                "description_imputed", pd.Series(False, index=frame.index)
            ) | missing
        elif strategy in {"group_mean", "group_median"}:
            if group_col not in frame:
                raise KeyError(group_col)
            if not columns:
                raise ValueError("평균·중앙값 대치에는 수치형 columns를 지정해야 합니다")
            statistic = "mean" if strategy == "group_mean" else "median"
            for column in columns:
                if column in {"customer_id", "row_id", "source_row_id", "invoice_no", "stock_code"}:
                    raise ValueError(f"식별자 열은 대치하지 않습니다: {column}")
                if column == group_col or not pd.api.types.is_numeric_dtype(frame[column]):
                    raise ValueError(f"대치 대상은 그룹 열과 다른 수치형 열이어야 합니다: {column}")
                missing = frame[column].isna()
                group_value = frame.groupby(group_col, dropna=False)[column].transform(statistic)
                filled = frame[column].fillna(group_value)
                if fallback == "global":
                    filled = filled.fillna(getattr(frame[column], statistic)())
                flag = f"{column}_imputed"
                frame[flag] = frame.get(flag, pd.Series(False, index=frame.index)) | (missing & filled.notna())
                frame[column] = filled
        elif strategy == "drop":
            frame.dropna(subset=["description", "order_date"], inplace=True)
        else:
            raise ValueError("strategy는 group_mode, group_mean, group_median, drop 중 하나여야 합니다")

        frame.dropna(subset=["order_date"], inplace=True)
        after = frame.isna().sum().astype(int).to_dict()
        return {"before": before, "after": after}

    def handle_missing_values(
        self,
        strategy: str = "group_mode",
        group_col: str = "stock_code",
        *,
        columns: list[str] | None = None,
        fallback: str = "global",
    ) -> dict[str, dict[str, int]]:
        """결측치 정책을 수행하는 공개 API. 기존 handle_missing 호출도 유지한다."""

        # 과제 명세의 정확한 메서드명을 유지하면서 실제 정책 구현은 한 곳에만 둔다.
        return self.handle_missing(strategy=strategy, group_col=group_col, columns=columns, fallback=fallback)

    def engineer_features(self) -> pd.DataFrame:
        """금액·단어 수와 배열의 행별 평균·표준편차를 생성한다."""

        frame = self._require_loaded()
        # 숫자 특징은 Python 반복문 대신 NumPy 배열끼리 곱해 벡터화한다.
        frame["amount"] = frame["quantity"].to_numpy() * frame["unit_price"].to_numpy()
        frame["word_count"] = (
            frame["description"].fillna("").astype(str).str.split().str.len().astype("int64")
        )

        lengths = frame["product_image"].map(len)
        if frame.empty or lengths.nunique() != 1 or lengths.iat[0] == 0:
            raise ValueError("이미지 배열은 비어 있지 않고 모두 같은 길이여야 합니다")
        # 행별 배열을 하나의 연속 2차원 행렬로 쌓아 이미지별 평균과 표준편차를
        # axis=1 연산 한 번으로 계산한다. Python 행 반복을 피하고 NumPy의 컴파일된
        # 루프와 SIMD 친화적 메모리 접근을 사용한다는 것이 핵심 설계 이유다.
        # 대신 전체 행렬이 RAM에 올라가므로 대규모 데이터에서는 배치 처리가 필요하다.
        image_matrix = np.stack(frame["product_image"].to_numpy()).astype(
            np.float32, copy=False
        )
        if not np.isfinite(image_matrix).all():
            raise ValueError("이미지 배열에는 유한한 값만 포함할 수 있습니다")
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
        """Q1·Q3에서 IQR과 상·하한을 직접 계산한다."""

        frame = self._require_loaded()
        factor = self.outlier_threshold if threshold is None else float(threshold)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("이상치 임계값은 유한한 양수여야 합니다")
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if positive_only:
            # 반품·취소는 음수라는 별도 의미가 있으므로 구매금액 이상치 계산에서는
            # 양수 거래만 사용하고 음수 행은 원본 그대로 보존한다.
            values = values[values > 0]
        if values.empty:
            raise ValueError(f"IQR을 계산할 값이 없습니다: {column}")
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
        """직접 계산한 IQR 경계를 벗어난 행을 반환한다."""

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
        """IQR 경계 밖 값을 클리핑하거나 제거하고 전후 건수를 반환한다."""

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
            raise ValueError("method는 'clip' 또는 'remove'여야 합니다")

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
        """식별 가능한 양수 구매 중 취소 주문을 제외한 거래를 반환한다."""

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
    def _quartile_score(series: pd.Series, *, high_is_good: bool, bins: int = 4) -> pd.Series:
        """평균 순위의 백분위를 구간으로 바꾼다. 동점은 같은 점수를 받는다.

        고객 한 명 또는 모두 같은 값은 중간 점수로 처리한다. 동점을 유지하므로
        구간별 고객 수가 같거나 모든 점수가 반드시 존재하는 것은 아니다.
        """
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
            raise ValueError("점수 구간 수는 2 이상의 정수여야 합니다")
        if series.empty or series.isna().any():
            raise ValueError("점수 입력은 비어 있지 않고 결측이 없어야 합니다")
        if series.nunique() == 1:
            return pd.Series((bins + 1) // 2, index=series.index, dtype="int64")
        score = np.ceil(series.rank(method="average", pct=True) * bins).clip(1, bins).astype("int64")
        return score if high_is_good else bins + 1 - score

    def calculate_rfm(
        self,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "amount",
        *,
        reference_date: str | pd.Timestamp | None = None,
        score_bins: int = 4,
        recent_days: int | None = None,
    ) -> pd.DataFrame:
        """날짜 기준 RFM과 동점 보존 점수를 계산해 네 고객군으로 분류한다.

        recent_days를 지정하면 상대 점수 대신 해당 일수 이내를 최근 구매로 본다.
        날짜에 시간대가 있으면 그 데이터의 시간대를 기준으로 날짜를 정규화한다.
        """

        transactions = self.rfm_transactions(customer_col, date_col, amount_col)
        if transactions.empty:
            raise ValueError("RFM에 사용할 유효 구매가 없습니다")
        # 기준일을 지정하지 않으면 데이터가 관측된 마지막 구매일의 다음 날을
        # 사용하여 모든 고객의 Recency가 0 이상이 되게 한다.
        analysis_date = (
            pd.Timestamp(reference_date).normalize()
            if reference_date is not None
            else transactions[date_col].max().normalize() + pd.Timedelta(days=1)
        )
        dates_timezone = transactions[date_col].dt.tz
        if analysis_date.tzinfo is None and dates_timezone is not None:
            analysis_date = analysis_date.tz_localize(dates_timezone)
        elif analysis_date.tzinfo is not None and dates_timezone is None:
            raise ValueError("시간대 없는 거래에는 시간대 없는 기준일을 지정하세요")
        elif dates_timezone is not None:
            analysis_date = analysis_date.tz_convert(dates_timezone).normalize()
        if analysis_date < transactions[date_col].max().normalize():
            raise ValueError("기준일은 마지막 유효 구매일보다 빠를 수 없습니다")
        if recent_days is not None and (not isinstance(recent_days, int) or isinstance(recent_days, bool) or recent_days < 0):
            raise ValueError("recent_days는 0 이상의 정수여야 합니다")
        self.rfm_reference_date = analysis_date

        # 고객별 마지막 구매일, 고유 주문 수, 조정 구매금액 합계를 한 번에 집계한다.
        rfm = transactions.groupby(customer_col).agg(
            last_purchase=(date_col, "max"),
            Frequency=("invoice_no", "nunique"),
            Monetary=(amount_col, "sum"),
        )
        # 시간대의 DST 변화에도 달력 날짜 차이가 유지되도록 현지 날짜로 변환한다.
        last_dates = rfm.pop("last_purchase").dt.normalize()
        if dates_timezone is not None:
            last_dates = last_dates.dt.tz_localize(None)
            calendar_reference = analysis_date.tz_localize(None)
        else:
            calendar_reference = analysis_date
        rfm["Recency"] = (calendar_reference - last_dates).dt.days
        rfm = rfm[["Recency", "Frequency", "Monetary"]]
        rfm["R_score"] = self._quartile_score(rfm["Recency"], high_is_good=False, bins=score_bins)
        rfm["F_score"] = self._quartile_score(rfm["Frequency"], high_is_good=True, bins=score_bins)
        rfm["M_score"] = self._quartile_score(rfm["Monetary"], high_is_good=True, bins=score_bins)
        rfm["RFM_score"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

        # np.select는 첫 번째 참 조건을 사용하므로 조건 순서 자체가 비즈니스 규칙이다.
        # 최근성·빈도·금액이 모두 높은 고객을 VIP로 먼저 분리한 뒤 최근 저빈도, 장기 미구매,
        # 나머지 고객을 Loyal로 구분한다.
        upper_half = score_bins // 2 + 1
        recent = rfm["R_score"].ge(upper_half) if recent_days is None else rfm["Recency"].le(recent_days)
        conditions = [
            recent & rfm["F_score"].ge(upper_half) & rfm["M_score"].ge(upper_half),
            recent & rfm["F_score"].lt(upper_half),
            ~recent,
        ]
        rfm["Segment"] = np.select(conditions, ["VIP", "New", "Churned"], default="Loyal")
        # 숫자 ID로 제한하지 않아 문자열 고객 식별자도 그대로 보존한다.
        if pd.api.types.is_numeric_dtype(rfm.index.dtype):
            rfm.index = rfm.index.astype("Int64")
        self.rfm = rfm.sort_values(["RFM_score", "Monetary"], ascending=False)
        return self.rfm

    def segment_summary(self) -> pd.DataFrame:
        """고객군의 규모·최근성·빈도·금액과 상대 기여도를 요약한다."""

        if self.rfm is None:
            raise RuntimeError("먼저 calculate_rfm()으로 고객을 집계하세요")
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
