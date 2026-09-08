# A1-1: 처음부터 끝까지 따라가는 구현 안내서

이 문서는 완성된 저장소를 만들기까지의 학습 경로를 설명한다. 모든 명령어는 프로젝트
루트 디렉터리에서 실행한다.

## 1단계 - 개발 환경 만들기

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

확인 사항: `python -c "import numpy, pandas, matplotlib, seaborn"` 명령이 오류 없이
종료되어야 한다.

## 2단계 - 데이터 확보 및 문서화

이 프로젝트는 CC BY 4.0 라이선스가 적용된 UCI Online Retail 데이터를 사용한다. 다음
명령으로 로컬의 25,000행 표본을 다시 생성한다.

```bash
python scripts/prepare_data.py
```

확인 사항: `data/online_retail_sample.csv`에는 25,000행과 14개 열이 있어야 한다.
데이터를 사용하기 전에 `data/DATASET.md`를 읽는다.

## 3단계 - 스키마 이해하기

노트북을 열고 데이터 로드 셀을 실행한다. `head()`, `info()`, `describe()`를 사용해 값,
자료형, 날짜, 결측치를 확인한다. `customer_id` 같은 식별자를 일반적인 측정 수치로
취급해서는 안 된다.

확인 사항: 고객 ID의 결측치를 평균값으로 대치하면 안 되는 이유를 설명할 수 있어야 한다.

## 4단계 - 재사용 가능한 클래스 구현하기

`src/pipeline.py`를 다음 순서로 살펴본다.

1. `load_data()`는 CSV를 검증하고 파싱한다.
2. `handle_missing()`은 상품 그룹의 정보를 근거로 결측치를 처리한다.
   (`handle_missing_values()`는 호환성을 위한 별칭으로만 유지한다.)
3. `engineer_features()`는 수치·텍스트·이미지 데이터에 벡터화 연산을 적용한다.
4. `detect_outliers()`는 IQR 공식을 구현한다.
5. `calculate_rfm()`은 고객 단위로 집계하고 세그먼트를 할당한다.

확인 사항: `python -m unittest -v`를 실행했을 때 다섯 개 테스트가 모두 통과해야 한다.

## 5단계 - 결측치와 이상치 처리하기

상품 설명의 결측치는 동일한 상품에서 가장 자주 등장한 설명으로 채운다. 고객 ID의
결측치는 그대로 두고 RFM 계산에서만 제외한다. 양수 구매 금액의 이상치는
`Q3 + 1.5*IQR`로 조정하며, 음수인 반품 거래는 원본 분석에 유지한다.

확인 사항: 데이터 품질 보고서에서 처리 전 상품 설명 결측치가 65개, 처리 후에는 0개이고,
상한 조정 후 남은 양수 구매 이상치가 0개인지 확인한다.

## 6단계 - RFM 계산 및 해석하기

Recency는 마지막 구매 이후 경과 일수, Frequency는 고유 주문서 수, Monetary는 IQR로
조정한 유효 구매 금액의 합계를 뜻한다. 점수는 사분위수를 사용한다.

확인 사항: 결과에 VIP, Loyal, New, Churned 세그먼트가 정확히 포함되어야 하며,
Recency 값은 낮을수록 좋은 이유를 설명할 수 있어야 한다.

## 7단계 - EDA와 인사이트 생성하기

```bash
python -m scripts.run_analysis
```

히스토그램, 박스플롯, 막대그래프, 히트맵, 산점도, 시계열 선 그래프를 확인한다. 각 비즈니스
제안에는 근거, 실행 방안, 기대 효과, 주장 검증에 필요한 추가 데이터가 포함되어야 한다.

확인 사항: PNG 파일 7개, 고객별 RFM 결과, 세그먼트 요약, 그룹 통계, 데이터 품질 보고서,
근거 기반 인사이트 3개가 생성되어 있어야 한다.

## 8단계 - 제출용 노트북 만들기

```bash
python scripts/build_notebook.py
```

확인 사항: `notebooks/analysis_report.ipynb`의 모든 코드 셀이 예외 없이 실행되어야 하며,
노트북에 작성한 결론이 CSV 출력 결과와 일치해야 한다.

## 9단계 - 최종 검증하기

```bash
python -m unittest -v
python -m scripts.run_analysis
python scripts/verify_project.py
```

확인 사항: 모든 검사가 성공해야 한다. 그다음 저장소를 커밋하고 푸시하여 미션에서 요구한
GitHub URL을 확보한다.
