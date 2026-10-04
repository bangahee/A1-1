# 구현과 설명의 연결

## 실행 결과

| 구성 | 코드 또는 실행 결과 | 설명 위치 |
|---|---|---|
| 클래스와 네 공개 메서드 | src/pipeline.py의 DataAnalyzer | README의 파이프라인 설계 |
| 처음부터 끝까지 노트북 실행 | notebook_execution_report.json의 셀 수·오류 수·해시 | 실행 방법 |
| 데이터 규모와 세 가지 실제 유형 | data_quality_report.json, dtype_summary.csv | 데이터 설명·기본 탐색 |
| 배열 Mean/Std와 텍스트 특징 | engineer_features(), 노트북 특징 표 | 벡터 연산·교육용 배열의 범위 |
| 이상치 전후 비교 | outlier_bounds(), 02_outlier_boxplot.png | IQR 공식·선택 이유·한계 |
| 여섯 차트와 한국어 제목·축 | visualization_evidence.csv와 각 PNG | 그림 설명 |
| 기술통계·두 쌍의 상관 | descriptive_statistics.csv, rfm_correlations.csv | 노트북의 수치 해석 |
| 네 고객군·세 제안 | rfm_customers.csv, segment_summary.csv, insights.json | RFM 및 비즈니스 제안 |

## 설계 선택

단계별 책임·입력·출력은 README와 IMPLEMENTATION_GUIDE.md에서 설명한다. 배열 통계는 NumPy의 행별 축 연산으로 계산하고, 임계값·대치 전략·그룹·기준일·점수 구간·최근성 기준을 인자로 전달한다. 주요 선택 이유는 노트북의 마크다운 셀 자체에도 남아 있다.

## 원리와 데이터 근거

IQR의 중앙 50% 기반 탐지, 비대칭·다봉·그룹 차이의 한계, 그룹 평균·중앙값 대치의 분산 변화, 평균 순위 점수의 동점 보존, 연속 메모리·CPU 캐시·SIMD 조건은 METHODOLOGY.md와 노트북에서 확인한다. 점수 구간의 실제 최솟값·최댓값은 rfm_score_ranges.csv, 기준 변경 결과는 rfm_sensitivity.csv에 있다.

## 확장 가정

대규모 배열의 형식별 용량은 scaling_estimates.csv에 계산한다. 배치 처리·캐시·프로파일링의 순서, 마케팅 대상·비용·대조군, 미래 90일 타깃의 관측 가능 기간·우측 검열·누수·현재와 후보 피처는 README와 METHODOLOGY.md에 연결한다. 설명 문서는 구현 의도를 기록하며 개인의 이해·설명 능력을 자동으로 증명하지 않는다.
