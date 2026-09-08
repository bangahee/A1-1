# Figure captions

| File | Title | X axis | Y axis | Caption |
|---|---|---|---|---|
| `01_amount_histogram.png` | Purchase amount distribution (up to 99th percentile) | Amount (GBP) | Count | 양수 구매금액의 99백분위 이하 분포. 긴 오른쪽 꼬리와 소액 주문 집중을 보여준다. |
| `02_outlier_boxplot.png` | Outliers before and after treatment | Treatment | Amount (GBP, log scale) | IQR 클리핑 전후 양수 구매금액 비교. 로그 축과 주석으로 처리 건수·비율을 함께 표시한다. |
| `03_segment_bar.png` | Customers by RFM segment | Segment | Customers | VIP, Loyal, New, Churned별 고객 수와 세그먼트 규모 차이. |
| `04_rfm_heatmap.png` | RFM correlation heatmap | RFM metrics | RFM metrics | Recency, Frequency, Monetary, RFM 점수 사이 Pearson 상관계수. 상관은 인과를 뜻하지 않는다. |
| `05_rfm_scatter.png` | Frequency vs monetary value by segment | Frequency (orders) | Monetary (GBP) | 로그 축에서 구매 빈도와 조정 구매금액의 관계를 고객 세그먼트별로 표시한다. |
| `06_monthly_sales_line.png` | Monthly sales trend | Month | IQR-adjusted sales (GBP) | IQR 조정 양수 구매금액의 월별 합계. 2011년 12월은 9일까지만 있는 부분 월이다. |
| `07_segment_summary_table.png` | RFM segment evidence table | 해당 없음 | 해당 없음 | 세그먼트별 고객 수, 비중, 평균 Recency, 빈도 중앙값, 평균 Monetary, 매출 비중 요약. |

모든 그래프는 `python -m scripts.run_analysis`로 재생성되며, 계산 원자료는 `outputs/`의 CSV와 JSON에 저장된다.
