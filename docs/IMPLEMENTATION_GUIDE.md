# A1-1 구현 가이드: 처음부터 제출까지

이 문서는 A1-1 쇼핑몰 고객 분석과 RFM 세분화 미션을 실제로 구현한 순서를 설명한다. 각 단계에서 무엇을 만들었고, 왜 그렇게 설계했으며, 어떤 결과로 검증했는지를 함께 기록한다.

## 0. 최종 산출물

~~~text
README.md
data/
  DATASET.md
  online_retail_sample.csv
  source_evidence.png
docs/
  IMAGE_ARRAY_DECISION.md
  IMPLEMENTATION_GUIDE.md
  PEER_REVIEW_GUIDE.md
figures/
  01_amount_histogram.png
  02_outlier_boxplot.png
  03_segment_bar.png
  04_rfm_heatmap.png
  05_rfm_scatter.png
  06_monthly_sales_line.png
  07_segment_summary_table.png
  CAPTIONS.md
notebooks/
  analysis_report.ipynb
outputs/
  descriptive_statistics.csv
  dtype_summary.csv
  notebook_execution.log
  notebook_execution_report.json
  rfm_correlations.csv
  rfm_customers.csv
  segment_summary.csv
scripts/
  prepare_data.py
  run_analysis.py
  build_notebook.py
  verify_project.py
src/
  pipeline.py
tests/
  test_pipeline.py
~~~

## 1. 명세를 구현 체크리스트로 변환

PDF 요구사항을 먼저 코드와 산출물에 연결했다.

| 요구사항 | 구현 위치 |
|---|---|
| 1,000행·8열 이상과 3종 이상의 데이터 타입 | 데이터 CSV, DataAnalyzer.load_data() |
| OOP 분석 파이프라인 | src/pipeline.py |
| 필수 공개 메서드 | load_data(), handle_missing(), detect_outliers(), calculate_rfm() |
| NumPy 벡터화 | engineer_features() |
| 텍스트 단어 수 | word_count |
| 이미지 배열 평균·표준편차 | image_mean, image_std |
| 그룹별 결측치 처리 | handle_missing() |
| IQR 직접 구현과 전후 비교 | outlier_bounds(), treat_outliers(), Figure 02 |
| 기술통계와 그룹통계 | outputs 디렉터리의 CSV |
| 6종 이상의 시각화 | figures 디렉터리의 PNG 7개 |
| 4개 이상의 RFM 세그먼트 | VIP, Loyal, New, Churned |
| 데이터 근거 기반 제안 | README와 insights.json |
| 실행 가능한 노트북 | notebooks/analysis_report.ipynb |
| 실행 증거와 재현성 | notebook execution report, verify_project.py |

이 매핑을 먼저 만들면 구현 후 제출 항목을 빠뜨리는 일을 줄일 수 있다.

## 2. 개발 환경 구성

프로젝트 루트에서 가상환경을 만든다. 실제 분석 코드와 단위 테스트만 실행할 때는 과제에서 허용한 분석 라이브러리만 설치한다.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

`requirements.txt`의 역할:

- NumPy: 배열 파싱과 벡터화
- Pandas: CSV, 결측치, 그룹 집계, RFM
- Matplotlib·Seaborn: 정적 시각화

노트북 생성·실행 검증과 원본 Excel 재생성까지 수행하려면 선택적인 개발 환경을 설치한다.

~~~bash
python -m pip install -r requirements-dev.txt
~~~

`requirements-dev.txt`의 역할:

- 첫 줄의 `-r requirements.txt`: 기본 분석 라이브러리 포함
- Jupyter·nbformat·nbclient: 노트북 생성·저장·실행 검증
- openpyxl: 원본 Excel 로드

따라서 전체 재현 시 두 파일을 각각 설치할 필요는 없다. `requirements-dev.txt` 하나를 설치하면 분석 환경과 개발 도구가 함께 준비된다.

OpenCV, Pillow, Scikit-learn, NLTK, 자동 EDA 라이브러리는 분석 코드에 사용하지 않았다.

## 3. 공개 데이터 선정과 출처 증빙

UCI Machine Learning Repository의 Online Retail 데이터를 선택했다.

- URL: <https://archive.ics.uci.edu/dataset/352/online%2Bretail>
- DOI: <https://doi.org/10.24432/C5BW33>
- 라이선스: CC BY 4.0
- 원본 규모: 541,909행
- 거래 기간: 2010-12-01~2011-12-09

고객 ID, 주문일, 수량, 단가, 상품명, 국가가 실제 거래 단위로 제공되어 RFM에 적합하다. 공식 페이지 화면은 data/source_evidence.png에, 출처와 파생 데이터 설명은 data/DATASET.md에 기록했다.

## 4. 재현 가능한 표본 생성

원본 전체 대신 seed 42로 25,000행을 추출했다.

~~~bash
python scripts/prepare_data.py
~~~

원본 파일이 있다면 다음처럼 실행한다.

~~~bash
python scripts/prepare_data.py --raw-xlsx "/path/to/Online Retail.xlsx"
~~~

prepare_data.py의 작업 순서:

1. 열 이름을 snake_case로 바꾼다.
2. random_state 42로 25,000행을 추출한다.
3. amount = quantity × unit_price를 벡터 연산으로 만든다.
4. stock_code에서 결정론적인 8×8 교육용 배열을 만든다.
5. 배열을 CSV에서 복원 가능한 문자열로 저장한다.
6. 행 ID와 이미지 높이·너비를 추가한다.

최종 데이터는 25,000행 × 14열이다. product_image는 실제 상품 사진이 아니라 NumPy 배열 실습용 파생 데이터다.

## 5. DataAnalyzer 클래스 설계

분석 로직을 노트북에 흩어놓지 않고 src/pipeline.py에 캡슐화했다.

~~~python
analyzer = DataAnalyzer(
    "data/online_retail_sample.csv",
    outlier_threshold=1.5,
    min_rows=1_000,
    min_columns=8,
)
~~~

주요 책임:

- load_data(): 파일, 필수 열, 최소 행·열, 날짜와 배열을 검증한다.
- handle_missing(): 상품 그룹별 텍스트 대치를 수행한다.
- engineer_features(): 숫자·텍스트·이미지 특징을 생성한다.
- detect_outliers(): IQR 경계 밖의 행을 반환한다.
- treat_outliers(): clip 또는 remove 방식으로 처리한다.
- calculate_rfm(): RFM, 점수, 세그먼트를 만든다.
- segment_summary(): 세그먼트별 고객·매출 비중을 요약한다.

명세와 일치하도록 handle_missing()을 공개 API로 사용한다. handle_missing_values()는 기존 호출을 깨지 않기 위한 호환 별칭이다.

## 6. 데이터 로드와 탐색

~~~python
analyzer = DataAnalyzer("data/online_retail_sample.csv")
df = analyzer.load_data()
df.head()
df.info()
df.describe()
analyzer.overview()
~~~

확인 결과:

- 데이터 크기: 25,000행 × 14열
- 날짜 범위: 2010-12-01~2011-12-09
- description 결측: 65건
- customer_id 결측: 6,300건
- 중복 row_id: 0건

data_info.txt에는 info() 결과를, descriptive_statistics.csv에는 컬럼별 dtype과 기술통계를, dtype_summary.csv에는 타입별 컬럼 수를 저장했다.

## 7. 결측치 처리

~~~python
missing_report = analyzer.handle_missing(
    strategy="group_mode",
    group_col="stock_code",
)
~~~

처리 정책:

1. 상품 설명은 같은 stock_code에서 가장 자주 나타난 값으로 대치한다.
2. 그룹에도 설명이 없으면 UNKNOWN PRODUCT를 사용한다.
3. customer_id는 식별자이므로 임의로 대치하지 않는다.
4. 고객 ID 결측 거래는 보존하고 RFM에서만 제외한다.
5. 이미지가 비었거나 길이가 다르거나 NaN·무한대를 포함하면 실패시킨다.

description 결측은 65건에서 0건이 되었다. 그룹 최빈값은 정보 손실을 줄이지만 그룹 내부 다양성과 분산을 축소하고 그룹 간 차이를 과장할 수 있으므로 README와 노트북에 이 위험을 함께 기록했다.

## 8. NumPy 멀티모달 특징 생성

~~~python
df = analyzer.engineer_features()
~~~

숫자 특징:

~~~python
frame["amount"] = (
    frame["quantity"].to_numpy()
    * frame["unit_price"].to_numpy()
)
~~~

텍스트 특징:

~~~python
frame["word_count"] = (
    frame["description"].fillna("").astype(str).str.split().str.len()
)
~~~

이미지 배열 특징:

~~~python
image_matrix = np.stack(frame["product_image"].to_numpy())
frame["image_mean"] = image_matrix.mean(axis=1)
frame["image_std"] = image_matrix.std(axis=1)
~~~

to_numpy()와 np.stack()은 동종 데이터를 연속 배열에 배치하여 Python의 원소별 호출 비용을 줄이고 NumPy의 컴파일된 루프와 SIMD 친화적 계산을 사용한다. 이미지 행렬은 float32로 유지해 float64 대비 작업 메모리를 절반으로 줄였다.

## 9. IQR 이상치 탐지와 처리

공식을 직접 구현했다.

~~~text
IQR = Q3 - Q1
Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR
~~~

~~~python
outliers = analyzer.detect_outliers("amount", positive_only=True)
report = analyzer.treat_outliers(
    "amount",
    method="clip",
    output_col="amount_clean",
    positive_only=True,
)
~~~

양수 구매 중 1,925건, 약 7.9%가 경계 밖이었다. 클리핑 후 경계 밖 양수 구매는 0건이다. 음수 반품·취소는 거래 의미가 있으므로 처리 대상에서 제외했다. 원본 amount를 보존하고 amount_clean을 따로 만들어 전후 비교가 가능하다.

IQR은 정규분포를 가정하지 않지만 오른쪽 꼬리가 긴 정상 고액 주문을 과다 탐지할 수 있다. 따라서 amount_clean은 회계 매출이 아니라 세분화 비교용이다.

## 10. 통계와 시각화 생성

~~~bash
python -m scripts.run_analysis
~~~

생성한 시각화:

1. 구매금액 히스토그램
2. IQR 전후 박스플롯
3. 세그먼트 고객 수 막대그래프
4. RFM 상관 히트맵
5. Frequency–Monetary 산점도
6. 월별 조정 매출 시계열
7. 세그먼트 요약 표

박스플롯에는 1,925건(7.9%)에서 0건으로 줄어든 결과를 표시했다. 각 그림의 설명은 figures/CAPTIONS.md에 있다.

대표 상관관계:

- Frequency–Monetary: 약 0.79
- Recency–RFM score: 약 -0.69

상관은 인과를 의미하지 않는다. Recency는 RFM 점수 산식에 직접 포함되므로 두 번째 상관은 독립적인 행동 발견으로 과대 해석하지 않았다.

## 11. RFM 계산과 세그먼트

RFM에는 고객 ID와 주문일이 있고, 수량·금액이 양수이며, 주문번호가 C로 시작하지 않는 거래만 사용했다.

~~~python
rfm = analyzer.calculate_rfm(
    customer_col="customer_id",
    date_col="order_date",
    amount_col="amount_clean",
)
~~~

- Recency: 기준일에서 마지막 구매일까지의 일수
- Frequency: 고객별 고유 주문서 수
- Monetary: 고객별 조정 구매금액 합계
- 기준일: 마지막 유효 구매일 다음 날인 2011-12-10

각 지표를 사분위 점수 1~4로 변환했다. 사전 임계값이 없는 탐색에서 충분한 관측치를 유지하면서 네 운영 그룹과 연결하기 쉬워 4분위를 선택했다.

- VIP: 최근 구매했고 Frequency·Monetary도 상위
- New: 최근 구매했지만 Frequency가 낮음
- Churned: 마지막 구매 후 오래 지남
- Loyal: 위 조건 사이의 반복 구매 고객

3분위는 그룹을 거칠게 합치고 5분위는 작은 그룹과 경계 이동을 늘린다. 운영에서는 30·90·180일 같은 실제 구매주기 임계값도 비교해야 한다.

## 12. RFM 결과

유효 거래 18,270건에서 고객 3,251명을 분석했다.

| 세그먼트 | 고객 수 | 고객 비중 | 평균 Recency | 빈도 중앙값 | 평균 Monetary | 조정 매출 비중 |
|---|---:|---:|---:|---:|---:|---:|
| VIP | 903 | 27.8% | 21.2일 | 4회 | £180 | 63.4% |
| Churned | 1,625 | 50.0% | 179.5일 | 1회 | £45 | 28.8% |
| New | 531 | 16.3% | 31.2일 | 1회 | £29 | 6.0% |
| Loyal | 192 | 5.9% | 27.6일 | 2회 | £25 | 1.9% |

고객별 결과는 rfm_customers.csv, 그룹 결과는 segment_summary.csv에 저장했다.

## 13. 비즈니스 제안

### 1순위: VIP 유지

- 근거: 고객 27.8%가 조정 매출 63.4% 생성
- 실행: 등급별 선공개와 서비스 혜택
- KPI: 매출 유지율, 반복구매율, 90일 이탈률
- 기간: 90일
- 가드레일: 혜택 비용 대비 증분 공헌이익

### 2순위: Churned 재활성화

- 근거: 고객 50.0%, 평균 Recency 179.5일
- 실행: 기간 제한 인센티브의 무작위 대조 실험
- KPI: 재활성화율과 증분 매출
- 기간: 6주
- 가드레일: 할인비, 공헌이익, 수신거부율

### 3순위: New 두 번째 구매

- 근거: 531명, Frequency 중앙값 1회
- 실행: 14일 내 사용 가이드와 연관상품 추천
- KPI: 14일·30일 2회차 구매율
- 기간: 30일
- 가드레일: 클릭률과 반품률

관찰 데이터에서 만든 가설이므로 인과 효과는 대조 실험으로 확인한다.

## 14. 노트북 생성과 실행 증거

~~~bash
python scripts/build_notebook.py
~~~

build_notebook.py는 다음 순서로 동작한다.

1. 21개 셀의 보고서를 구성한다.
2. NotebookClient가 모든 코드 셀을 실행한다.
3. 출력이 포함된 analysis_report.ipynb를 저장한다.
4. 실패 시 오류 리포트를 남기고 예외를 발생시킨다.
5. 성공 시 시간, Python 버전, 셀 번호, 오류 수, SHA-256을 기록한다.

현재 실행 증거:

- 전체 셀: 21개
- 코드 셀: 8개
- 실행된 코드 셀: 8개
- 오류 출력: 0개

증빙은 notebook_execution_report.json과 notebook_execution.log에 있다.

## 15. 단위 테스트

~~~bash
python -m unittest -v
~~~

테스트 범위:

1. 이미지 배열과 벡터 특징
2. 그룹별 결측 대치
3. IQR 탐지와 클리핑
4. RFM과 네 세그먼트
5. 이전 API 호환 별칭

현재 5개 테스트 모두 통과한다.

## 16. 프로젝트 전체 검증

~~~bash
python scripts/verify_project.py
~~~

검증기는 필수 파일, 데이터 규모, 날짜, 공개 API, RFM 세그먼트, dtype, 상관계수, PNG, 노트북 실행 번호, 오류 수, 실행 리포트 SHA-256, 결측·이상치 처리 결과를 확인한다.

~~~text
A1-1 verification passed
- data: 25,000 rows x 14 columns
- RFM: 3,251 customers, 4 segments
- figures: 7 valid PNG files
- notebook: 8 executed code cells, 0 errors
- evidence: dtype, correlations, captions, and nbclient execution log
~~~

## 17. 대규모 데이터 확장

25,000×64 float32 행렬은 약 6.1 MiB지만 원본 541,909행에서는 핵심 행렬만 약 132 MiB다. CSV 문자열과 Python 객체 오버헤드까지 고려하면 실제 메모리는 더 크다.

확장 순서:

1. CSV 배열을 Parquet 또는 NumPy 바이너리로 바꾼다.
2. 청크별로 파싱하고 통계를 계산한다.
3. 원본 uint8과 계산용 float32를 유지한다.
4. 메모리 매핑이나 분산 프레임을 검토한다.
5. 초기 EDA는 월·국가 층화 표본으로 수행한다.
6. CPU 병목이 확인된 경우에만 병렬화한다.

전체 데이터를 프로세스마다 복사하는 병렬화는 메모리 사용량을 늘릴 수 있으므로 프로파일링이 먼저다.

## 18. 머신러닝 확장

후속 타깃 예시는 churn_90d다.

~~~text
기준일 이후 90일 동안 구매 없음 → churn_90d = 1
90일 안에 구매 발생               → churn_90d = 0
~~~

기준일 이전 피처:

- Recency, Frequency, Monetary
- 평균 주문금액과 반품률
- 고유 상품 수와 활동 개월 수
- 국가
- 최근 30·60·90일 주문 수와 금액
- 결측 대치 여부

타깃 기간의 주문·금액, 미래 날짜, 사후 RFM 세그먼트는 누수이므로 제외한다. 시간순으로 train/validation/test를 나누고 Recall, Precision, PR-AUC, ROC-AUC, 캘리브레이션과 캠페인 기대가치를 평가한다.

## 19. 처음부터 다시 실행

준비된 CSV를 사용하는 절차:

~~~bash
source .venv/bin/activate
python -m unittest -v
python -m scripts.run_analysis
python scripts/build_notebook.py
python scripts/verify_project.py
~~~

원본 다운로드부터 실행하는 절차:

~~~bash
source .venv/bin/activate
python scripts/prepare_data.py
python -m unittest -v
python -m scripts.run_analysis
python scripts/build_notebook.py
python scripts/verify_project.py
~~~

모든 단계가 성공하면 README 수치, outputs 표, figures 그래프, 실행된 노트북이 동일한 데이터와 코드에서 다시 생성된다.
