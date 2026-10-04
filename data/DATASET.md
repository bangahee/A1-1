# 데이터 출처와 구성

- 출처: [UCI Machine Learning Repository, Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
- 제작자: Daqing Chen
- 인용: Chen, D. (2015). Online Retail. [DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
- 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 원본: 541,909행, 원본 거래 필드 8개
- 관측 기간: 2010-12-01~2011-12-09
- 현재 표본: NumPy/Pandas seed 42로 선택한 거래 25,000행, 파생 필드를 포함한 14열

영국의 비점포 소매업체 거래 데이터다. `source_row_id`는 원본 행 번호를 보존한다. `source_evidence.png`는 공식 출처 페이지의 기존 캡처다.

| 열 | 의미 |
|---|---|
| row_id / source_row_id | 표본 내 행 번호 / 원본 행 번호 |
| invoice_no | 주문번호, C로 시작하면 취소 거래 |
| stock_code | 상품 코드 |
| description | 상품 설명 |
| quantity / unit_price | 거래 수량 / 단가(GBP) |
| order_date | 거래 날짜와 시간 |
| customer_id / country | 고객 식별자 / 국가 |
| amount | 수량 × 단가로 만든 원본 거래금액 |
| product_image | 상품 코드에서 생성한 교육용 64개 값의 배열 |
| image_height / image_width | 배열 표시 크기, 각각 8 |

`product_image`는 UCI가 제공한 상품 사진이 아니다. 상품 코드 해시와 픽셀 위치를 NumPy 브로드캐스팅해 만든 결정론적 배열이며 같은 코드는 같은 배열을 갖는다. CSV에서는 문자열로 저장하고 `np.fromstring()`으로 복원한다. 실사진의 색상·형태·품질을 나타내지 않는다. 출처가 확인된 상품 사진 또는 명시적으로 허용된 제공 배열을 쓰는 경우에는 해당 출처와 상품 대응 관계를 별도로 기록해야 한다.

파이프라인은 `description_imputed`, `word_count`, `image_mean`, `image_std`, `amount_clean`을 추가한다. 원본 거래금액은 보존한다. 상품명 결측은 상품 그룹 최빈값으로 채우고 고객 ID는 채우지 않는다. RFM은 고객을 식별할 수 있는 양수 구매 중 취소 주문을 제외한 거래로 계산한다.

거래 행 표본은 전체 고객·주문 이력을 보존하지 않는다. 표본 내 최근성·빈도·금액을 생애 구매 이력으로 해석할 수 없으며, 고객 행동을 더 정확하게 분석하려면 고객 단위 표본과 관측 기간 내 전체 거래가 필요하다. 수치형 대치 예제는 원본 데이터와 별도로 만든 교육용 데이터다.
