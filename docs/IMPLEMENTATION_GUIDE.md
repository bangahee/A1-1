# 데이터 분석의 실행 흐름

## 환경 준비

검증한 Python 버전과 패키지는 `outputs/environment.json`에 기록한다. 현재 검증 환경은 Python 3.14.5이며 이전 버전에서의 실행은 별도로 확인해야 한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

기본 분석만 실행하려면 `requirements.txt`를 설치한다. 개발용 파일은 기본 분석 라이브러리를 포함하고 노트북·Excel 도구를 추가한다. 한국어 그림을 위해 AppleGothic, 맑은 고딕, 나눔고딕 또는 Noto Sans CJK KR이 필요하다.

## 입력과 클래스

```python
from src.pipeline import DataAnalyzer
analyzer = DataAnalyzer("data/online_retail_sample.csv", outlier_threshold=1.5)
raw = analyzer.load_data()
raw.head()
raw.info()
raw.select_dtypes(include="number").describe()
```

로드 단계에서 크기·필수 열을 확인하고 날짜와 배열을 복원한다. `overview()`는 관측 기간과 결측 현황을 요약한다. 각 메서드는 DataFrame 상태를 변경하거나 결과를 반환하므로 먼저 `load_data()`를 호출한다.

## 결측치와 특징

```python
missing = analyzer.handle_missing_values(strategy="group_mode", group_col="stock_code")
frame = analyzer.engineer_features()
```

상품명은 그룹 최빈값으로 보완하고 `description_imputed`에 대치 위치를 남긴다. 고객 ID는 보완하지 않는다. 금액은 NumPy 곱셈, 단어 수는 Pandas 문자열 연산, 이미지 통계는 `np.stack()`과 행별 평균·표준편차로 계산한다. 교육용 배열은 실제 상품 사진을 의미하지 않는다.

수치 결측이 있는 다른 데이터에서는 다음과 같이 지정한다.

```python
analyzer.handle_missing_values(
    strategy="group_median", group_col="stock_code",
    columns=["unit_price"], fallback="keep",
)
```

`fallback="global"`은 그룹 전체가 결측일 때 전체 통계로 보완한다. `keep`은 결측을 유지한다. 평균·중앙값 대치는 식별자에 적용하지 않으며, 실제 표본에 없는 수치 결측을 만들어 비즈니스 분석에 섞지 않는다.

## IQR과 고객 집계

```python
outliers = analyzer.detect_outliers("amount", positive_only=True)
report = analyzer.treat_outliers("amount", positive_only=True, output_col="amount_clean")
rfm = analyzer.calculate_rfm(amount_col="amount_clean", score_bins=4)
summary = analyzer.segment_summary()
```

IQR 경계는 직접 구현한다. 양수 구매를 클리핑하고 반품과 원본 금액은 보존한다. 최근성은 기준일과 마지막 구매일의 달력 날짜 차이, 빈도는 고유 주문번호 수, 금액은 유효 조정 구매액 합계다. 평균 순위 백분위로 점수를 만들고 동점에는 같은 점수를 부여한다. 기준일을 명시하면 마지막 관측 구매일보다 이르지 않아야 한다. 과거 시점의 분석은 그 시점까지 입력 데이터를 먼저 제한한다.

`score_bins=3/5`, `recent_days=30/90/180`, 원본 금액 등으로 가정을 비교한다. 시나리오 변경 후 기본 결과를 복원해 이후 요약과 혼동하지 않는다.

## 분석과 문서 재생성

```bash
python -m scripts.run_analysis
python scripts/build_notebook.py
python scripts/verify_project.py
```

첫 명령은 통계 CSV·한국어 그림·README를 생성한다. 두 번째 명령은 실행 중인 Python으로 새 Jupyter 커널을 시작해 `notebooks/`를 작업 위치로 전체 셀을 실행한다. 노트북도 원본 CSV에서 분석·그림을 다시 생성하고 현재 결과로 README를 갱신한다.

노트북은 저장소 루트 또는 하위 `notebooks/`에서 실행할 수 있다. 원본 CSV는 필요하지만 기존 `outputs/` CSV나 그림이 미리 있을 필요는 없다. `scripts/reporting.py`가 한국어 설명을 만들고 `scripts/build_notebook.py`가 설명과 실행 셀을 구성한다. 변경은 생성 코드에도 적용해야 다음 생성 시 유지된다.

실행된 셀 수·오류 수·환경·해시는 `outputs/notebook_execution_report.json`에 저장한다. 현재 통계는 `outputs/`에서, 그림 설명은 `figures/CAPTIONS.md`에서 확인한다.

## 회귀 검증

```bash
python -m unittest -v
```

날짜 차이와 DST, 잘못된 기준일, 동점·고객 식별자에 대한 점수 불변성, 작은 고객 집단, 그룹 대치와 표시 열, 이미지 Mean/Std와 잘못된 배열, IQR 처리를 검증한다. 산출물 검증기는 저장된 고객 결과를 다시 계산해 비교하고 노트북 해시를 확인한다. 결과 숫자나 셀 수를 고정한 검증에 의존하지 않는다.
