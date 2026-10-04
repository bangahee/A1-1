"""원본 거래에서 재현 가능한 표본과 교육용 배열을 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
# 원본 UCI 열 이름을 분석 파이프라인의 데이터 계약과 분리하는 어댑터다.
# 원천 스키마가 달라져도 이 매핑만 바꾸고 DataAnalyzer는 그대로 재사용할 수 있다.
RENAME = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "order_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def acquire_xlsx(raw_xlsx: Path | None, cache_dir: Path) -> Path:
    # 사용자가 원본 경로를 주면 네트워크를 사용하지 않고 해당 파일을 우선한다.
    if raw_xlsx is not None:
        if not raw_xlsx.exists():
            raise FileNotFoundError(raw_xlsx)
        return raw_xlsx

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "online_retail.zip"
    target = cache_dir / "Online Retail.xlsx"
    # 이미 내려받은 압축 파일과 엑셀 파일을 재사용해 반복 실행을 빠르게 한다.
    if not archive.exists():
        print(f"원본 데이터 다운로드: {DATA_URL}")
        urlretrieve(DATA_URL, archive)
    if not target.exists():
        with ZipFile(archive) as bundle:
            bundle.extract("Online Retail.xlsx", cache_dir)
    return target


def deterministic_images(stock_codes: pd.Series, size: int = 64) -> np.ndarray:
    """상품 코드로부터 같은 값이 재생성되는 교육용 배열을 만든다."""

    # 상품 코드의 안정적인 해시를 행별 seed처럼 사용한다. 같은 상품 코드는
    # 실행할 때마다 같은 8x8 교육용 픽셀 배열을 생성하므로 결과가 재현된다.
    # 실제 사진의 색·형태를 표현하지 않으므로 이 값으로 시각적 결론을 내리지 않는다.
    seeds = pd.util.hash_pandas_object(stock_codes.astype(str), index=False).to_numpy(
        dtype=np.uint64
    )
    pixel_position = np.arange(size, dtype=np.uint64)
    coefficients = pixel_position * np.uint64(1_664_525) + np.uint64(1_013_904_223)
    # (상품 seed 열) × (픽셀 위치 행) 브로드캐스팅으로 전체 배열을 한 번에 만든다.
    pixels = (seeds[:, None] * coefficients[None, :] + pixel_position[None, :] * 97) % 256
    return pixels.astype(np.uint8)


def prepare(raw_xlsx: Path, output_csv: Path, sample_size: int, seed: int) -> pd.DataFrame:
    source = pd.read_excel(raw_xlsx)
    if len(source) < sample_size:
        raise ValueError(f"표본 {sample_size:,}행이 원본 {len(source):,}행보다 큽니다")

    # seed가 고정된 표본을 시간순으로 정렬해 동일 입력에서 동일 CSV를 만든다.
    # 거래 행 표본은 고객의 전체 이력을 보존하지 않으므로 분석 범위를 명시한다.
    sample = (
        source.sample(n=sample_size, random_state=seed)
        .rename(columns=RENAME)
        .rename_axis("source_row_id")
        .reset_index()
        .sort_values("order_date")
        .reset_index(drop=True)
    )
    sample.insert(0, "row_id", np.arange(len(sample), dtype=np.int64))
    sample["amount"] = sample["quantity"].to_numpy() * sample["unit_price"].to_numpy()
    # 실제 상품 사진이 아니라 NumPy 배열 처리 능력을 검증하기 위한 파생 데이터다.
    pixels = deterministic_images(sample["stock_code"], size=64)
    # 아래 반복은 CSV에 행별 배열을 문자열로 직렬화하는 I/O 경계다. 평가 핵심인
    # Mean/Std 특징 추출은 pipeline.py에서 np.stack과 axis 연산으로 벡터화한다.
    sample["product_image"] = [
        "[" + " ".join(row.astype(str)) + "]" for row in pixels
    ]
    sample["image_height"] = 8
    sample["image_width"] = 8

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_csv, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return sample


def main() -> None:
    # CLI 인자로 원본 위치·표본 크기·seed를 노출해 코드를 수정하지 않고 재사용한다.
    parser = argparse.ArgumentParser(description="거래 표본과 교육용 배열 생성")
    parser.add_argument("--raw-xlsx", type=Path, help="이미 받은 원본 Excel 경로")
    parser.add_argument("--output", type=Path, default=Path("data/online_retail_sample.csv"), help="표본 저장 경로")
    parser.add_argument("--sample-size", type=int, default=25_000, help="표본 행 수")
    parser.add_argument("--seed", type=int, default=42, help="재현용 난수 시드")
    args = parser.parse_args()

    source = acquire_xlsx(args.raw_xlsx, Path("data/raw"))
    sample = prepare(source, args.output, args.sample_size, args.seed)
    print(f"표본 저장: {len(sample):,}행 × {sample.shape[1]}열, {args.output}")
    print(f"관측 기간: {sample['order_date'].min()} → {sample['order_date'].max()}")
    print(f"고객 ID 결측: {sample['customer_id'].isna().sum():,}건")


if __name__ == "__main__":
    main()
