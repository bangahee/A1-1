# A1-1 제출 전 평가항목 체크리스트

이 문서는 A1-1 미션 PDF의 필수 요구사항을 현재 저장소의 코드와 산출물에 다시 연결한 최종 점검표다. 자동 검증은 `python scripts/verify_project.py`로 수행한다.

## 1. 최종 결과물

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| 실행 가능한 데이터 분석 파이프라인 모듈 | 완료 | `src/pipeline.py`의 `DataAnalyzer` |
| 분석 리포트 노트북 | 완료 | `notebooks/analysis_report.ipynb` |
| 프로젝트 설명과 실행 방법 | 완료 | `README.md` |
| GitHub 저장소로 제출 가능한 구조 | 완료 | `src/`, `scripts/`, `tests/`, `data/`, `outputs/`, `figures/`, `docs/` |

## 2. 데이터와 기본 탐색

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| 최소 1,000행, 8열 이상 | 완료 | `data/online_retail_sample.csv`: 25,000행 × 14열 |
| 숫자, 문자열, 날짜, 이미지 배열 등 3종 이상 | 완료 | CSV 스키마와 `outputs/dtype_summary.csv` |
| `head()`, `info()`, `describe()` 확인 | 완료 | 실행 완료 노트북 2절 |
| 데이터 출처와 라이선스 명시 | 완료 | `data/DATASET.md`, `data/source_evidence.png`, README의 CC BY 4.0 표기 |
| 개인정보 미포함 확인 | 완료 | 공개 UCI 거래 데이터이며 고객 ID는 비식별 숫자 식별자 |

## 3. 객체 지향 분석 파이프라인

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| 분석 로직을 `DataAnalyzer` 클래스로 구현 | 완료 | `src/pipeline.py` |
| `load_data()` | 완료 | 스키마·크기 검증, 날짜·배열 파싱 |
| `handle_missing_values()` | 완료 | 공개 호환 API이며 `handle_missing()`에 위임 |
| `detect_outliers()` | 완료 | 직접 계산한 IQR 경계 밖 행 반환 |
| `calculate_rfm()` | 완료 | 고객별 R/F/M 집계, 사분위 점수, 네 세그먼트 생성 |
| 주요 옵션을 생성자·메서드 인자로 노출 | 완료 | `outlier_threshold`, `min_rows`, `min_columns`, 결측 전략, 기준일 등 |

## 4. 멀티모달 특징 공학

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| 숫자 특징 | 완료 | `amount = quantity × unit_price` |
| 텍스트 특징 | 완료 | `description`의 `word_count` |
| 이미지 배열 특징 | 완료 | `product_image`의 `image_mean`, `image_std` |
| 이미지 특징을 반복문 없이 벡터화 | 완료 | `np.stack(...).mean(axis=1)`, `std(axis=1)` |
| 이미지 처리 금지 라이브러리 미사용 | 완료 | OpenCV·Pillow 미사용, NumPy로 직접 처리 |

## 5. 결측치와 이상치

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| 결측 현황 파악 | 완료 | `outputs/data_quality_report.json`, 노트북 2·3절 |
| 그룹 통계를 이용한 대치 | 완료 | `stock_code`별 `description` 최빈값 대치 |
| 식별자 결측의 의미 보존 | 완료 | `customer_id`는 임의 대치하지 않고 RFM에서만 제외 |
| IQR 로직 직접 구현 | 완료 | `outlier_bounds()`, `detect_outliers()` |
| 처리 전후 분포 비교 | 완료 | `figures/02_outlier_boxplot.png`: 1,925건 → 0건 |
| 원본 값 보존 | 완료 | 원본 `amount`와 정제 `amount_clean`을 별도 유지 |

## 6. 통계 분석과 해석

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| 평균·중앙값·표준편차·사분위수 계산 | 완료 | `outputs/descriptive_statistics.csv`, 노트북 2절 |
| 주요 수치형 변수의 통계 해석 | 완료 | README 및 노트북의 `quantity`, `unit_price`, `amount` 해석 |
| 최소 두 쌍의 상관계수 계산 | 완료 | `outputs/rfm_correlations.csv` |
| 상관관계의 수치 근거와 시사점 | 완료 | Frequency–Monetary 0.79, Recency–RFM score -0.69 및 인과 해석 주의 |

## 7. 시각화

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| 히스토그램 | 완료 | `figures/01_amount_histogram.png` |
| 박스플롯 | 완료 | `figures/02_outlier_boxplot.png` |
| 막대그래프 | 완료 | `figures/03_segment_bar.png` |
| 히트맵 | 완료 | `figures/04_rfm_heatmap.png` |
| 산점도 | 완료 | `figures/05_rfm_scatter.png` |
| 라인차트 | 완료 | `figures/06_monthly_sales_line.png` |
| 제목과 축 레이블 | 완료 | 생성 코드와 PNG 결과, `figures/CAPTIONS.md` |
| 추가 요약 표 | 완료 | `figures/07_segment_summary_table.png` |

## 8. RFM 고객 세분화

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| Recency | 완료 | 기준일 2011-12-10에서 마지막 구매일까지의 일수 |
| Frequency | 완료 | 고객별 고유 주문번호 수 |
| Monetary | 완료 | 고객별 IQR 조정 유효 구매금액 합계 |
| 최소 네 고객 그룹 | 완료 | VIP, Loyal, New, Churned |
| 각 그룹의 특징 분석 | 완료 | `outputs/segment_summary.csv`, README 핵심 결과 표 |

## 9. README 비즈니스 인사이트

각 제안은 PDF가 요구한 세 요소인 **근거**, **실행**, **검증**을 모두 포함한다.

| 제안 | 근거 | 실행 | 검증 |
|---|---|---|---|
| VIP 유지 | 고객 27.8%, 조정 매출 63.4% | 등급별 선공개·서비스 혜택 | 반복구매율, 마진, 비용, 90일 이탈률 |
| Churned 재활성화 | 고객 50.0%, 평균 Recency 179.5일 | 기간 제한 인센티브 A/B 테스트 | 증분 매출, 공헌이익, 수신거부율 |
| New 2회차 구매 | 531명, 빈도 중앙값 1회 | 14일 내 교육·연관상품 추천 | 14/30일 2회차 구매율, 클릭률, 반품률 |

## 10. 제약 사항

| 평가항목 | 상태 | 저장소 증거 |
|---|---|---|
| OpenCV·Pillow 미사용 | 완료 | 전체 Python import 및 `requirements.txt` 확인 |
| Scikit-learn·NLTK 미사용 | 완료 | 전체 Python import 및 `requirements.txt` 확인 |
| 자동 EDA 도구 미사용 | 완료 | pandas-profiling·sweetviz 미사용 |
| 이미지 배열을 NumPy로 직접 처리 | 완료 | `parse_image_array()`, `engineer_features()` |
| Python 3.8 이상 | 완료 | 코드의 타입 표기는 Python 3.10+ 권장, 현재 검증 환경 Python 3.12 |

## 11. 실행과 품질 검증

| 검사 | 결과 |
|---|---|
| Python 컴파일 검사 | 통과 |
| 단위 테스트 | 5개 통과 |
| 전체 분석 재실행 | 통과, PNG 7개 생성 |
| 노트북 코드 셀 | 8개 전체 실행, 오류 0개 |
| 제출 검증기 | `A1-1 verification passed` |
| 데이터 결과 | 25,000행 × 14열, RFM 고객 3,251명, 세그먼트 4개 |

노트북 실행 시점과 SHA-256은 `outputs/notebook_execution_report.json`과 `outputs/notebook_execution.log`에 기록된다.

## 12. 선택 보너스 과제

다음 항목은 필수 평가항목이 아니며 현재 제출 범위에는 포함하지 않았다.

- 고급 이미지 특징 또는 엣지 특징
- 코호트 분석과 Retention Heatmap
- Plotly 기반 인터랙티브 RFM 시각화

필수 항목은 모두 충족하며, 보너스를 구현하지 않은 것은 제출 차단 요인이 아니다.
