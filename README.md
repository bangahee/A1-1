# A1-1 쇼핑몰 고객 분석과 RFM 세분화

25,000건의 쇼핑몰 거래에서 결측치와 이상치를 처리하고, 숫자·텍스트·이미지 배열 특징을 벡터화한 뒤 RFM으로 고객을 `VIP`, `Loyal`, `New`, `Churned` 네 그룹으로 분류한 재현 가능한 데이터 분석 프로젝트입니다.

## 핵심 결과

| 세그먼트 | 고객 수 | 고객 비중 | 평균 Recency | 빈도 중앙값 | 평균 Monetary | 조정 매출 비중 |
|---|---:|---:|---:|---:|---:|---:|
| VIP | 903 | 27.8% | 21.2일 | 4회 | £180 | 63.4% |
| Churned | 1,625 | 50.0% | 179.5일 | 1회 | £45 | 28.8% |
| New | 531 | 16.3% | 31.2일 | 1회 | £29 | 6.0% |
| Loyal | 192 | 5.9% | 27.6일 | 2회 | £25 | 1.9% |

![RFM segment counts](figures/03_segment_bar.png)

분석 기준일은 마지막 유효 구매일 다음 날인 `2011-12-10`입니다. Monetary는 양수 구매금액에 IQR 클리핑을 적용한 뒤 합산했습니다. 따라서 표의 매출은 회계 매출이 아니라 세분화 비교용 **조정 금액**입니다.

## 데이터

- 출처: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online%2Bretail)
- 인용: Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository. <https://doi.org/10.24432/C5BW33>
- 라이선스: CC BY 4.0
- 원본 근거: 541,909행, 8개 원본 필드, 2010-12-01~2011-12-09
- 분석 데이터: seed 42로 추출한 25,000행, 14개 열

![Official UCI dataset page](data/source_evidence.png)

`product_image`는 실제 사진이 아니라 `stock_code`에서 결정론적으로 생성한 8×8 교육용 배열입니다. 실제 이미지 출처를 가장하지 않으면서 CSV 배열 파싱과 NumPy 이미지 통계 연습을 재현하기 위한 파생 필드입니다. 자세한 출처와 스키마는 [data/DATASET.md](data/DATASET.md)에 있습니다.

## 처리 방식

- `description` 결측 65건: 같은 `stock_code` 그룹의 최빈 설명으로 대치, 처리 후 0건
- `customer_id` 결측 6,300건: 식별자를 조작하지 않고 보존, RFM에서만 제외
- 숫자 특징: `amount = quantity * unit_price` NumPy 벡터 연산
- 텍스트 특징: 설명의 `word_count`
- 이미지 배열 특징: 25,000×64 행렬의 행별 `image_mean`, `image_std`
- 이상치: 직접 구현한 `Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`; 양수 구매 1,925건을 클리핑, 처리 후 0건
- RFM: 유효 구매 18,270건, 고객 3,251명

## 비즈니스 제안

1. **VIP 유지** — 근거: 고객 27.8%가 조정 매출 63.4%를 만든다. 실행: 등급별 선공개와 서비스 혜택. 기대효과: 핵심 매출과 반복구매율 방어. 검증에는 캠페인/대조군 반복구매율, 마진, 혜택 비용, 90일 이탈률이 필요하다.
2. **Churned 재활성화** — 근거: 고객 50.0%, 평균 Recency 179.5일. 실행: 기간 제한 인센티브의 무작위 대조 실험. 기대효과: 전체 할인 없이 휴면 고객 회수. 검증에는 증분 매출, 공헌이익, 수신거부율, 이탈 사유가 필요하다.
3. **New 두 번째 구매** — 근거: 531명, 빈도 중앙값 1회. 실행: 첫 구매 후 14일 내 사용 가이드와 연관상품 추천. 기대효과: 2회차 전환율 상승. 검증에는 14/30일 재구매율, 클릭률, 반품률, 코호트 유지율이 필요하다.

위 제안은 관찰 데이터에서 얻은 가설이며 인과관계를 뜻하지 않습니다.

## 실행 방법

Python 3.8 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python -m scripts.run_analysis
python scripts/build_notebook.py
python scripts/verify_project.py
```

원본에서 표본 CSV를 다시 만들려면 `python scripts/prepare_data.py`를 실행합니다. 네트워크 없이 받은 원본이 있다면 `--raw-xlsx "/path/to/Online Retail.xlsx"`를 사용합니다.

## 프로젝트 구조

```text
data/                     데이터, 출처·라이선스, 출처 캡처
docs/STEP_BY_STEP.md      처음부터 끝까지 학습 순서
figures/                  6종 이상 시각화와 요약 표
notebooks/analysis_report.ipynb
outputs/                  RFM, 통계, 품질 보고서, 인사이트
scripts/                  데이터 준비·분석·노트북·검증 실행기
src/pipeline.py           재사용 가능한 DataAnalyzer 클래스
tests/                    IQR, 결측, 벡터 특징, RFM 테스트
```

## 한계

- 원본 전체가 아닌 25,000행 표본이므로 전체 모집단의 정확한 추정치가 아닙니다.
- 고객 ID 결측 거래는 고객 수준 RFM에서 제외됩니다.
- 파생 이미지 배열은 시각적 상품 의미를 갖지 않습니다.
- 반품/취소는 RFM 구매에서 제외했으며 별도 반품 분석이 필요합니다.
- 2011년 12월은 9일까지만 포함된 부분 월입니다.

완전한 실행 흐름은 [단계별 가이드](docs/STEP_BY_STEP.md), 분석 설명과 출력은 [실행 완료 노트북](notebooks/analysis_report.ipynb)에서 확인할 수 있습니다.

