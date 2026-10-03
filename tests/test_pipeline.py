from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.pipeline import DataAnalyzer


class DataAnalyzerTests(unittest.TestCase):
    def setUp(self):
        # 작은 합성 fixture를 매 테스트마다 새로 만들어 테스트 간 DataFrame 변경이
        # 누출되지 않게 한다. min_rows=1은 운영 기본값이 아닌 단위 테스트 전용 설정이다.
        self.temp_dir = TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "sample.csv"
        rows = []
        # 결측 상품명, 양·음수 극단값, 반복 주문을 한 표본에 넣어 핵심 분기를 검증한다.
        for i in range(16):
            rows.append(
                {
                    "row_id": i,
                    "invoice_no": f"I{i // 2}",
                    "stock_code": "A" if i < 8 else "B",
                    "description": None if i == 1 else "RED CUP SET",
                    "quantity": 100 if i == 15 else (-100 if i == 14 else i % 4 + 1),
                    "order_date": f"2024-01-{i + 1:02d}",
                    "unit_price": 2.5,
                    "customer_id": 10 + i % 5,
                    "country": "UK",
                    "product_image": "[0 10 20 30]",
                }
            )
        pd.DataFrame(rows).to_csv(self.path, index=False)
        self.analyzer = DataAnalyzer(self.path, min_rows=1)
        self.analyzer.load_data()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_image_parser_and_vectorized_features(self):
        # 평탄화 이미지 배열과 숫자·텍스트 특징이 행 단위로 올바르게 계산되는지 확인한다.
        self.analyzer.handle_missing()
        enriched = self.analyzer.engineer_features()
        np.testing.assert_allclose(enriched["image_mean"], 15.0)
        self.assertEqual(enriched.loc[0, "word_count"], 3)
        self.assertEqual(enriched.loc[0, "amount"], 2.5)

    def test_groupwise_missing_description(self):
        # 단순히 결측 수만 줄이는 것이 아니라 같은 상품 그룹의 설명으로 채웠는지 본다.
        report = self.analyzer.handle_missing()
        self.assertEqual(report["before"]["description"], 1)
        self.assertEqual(report["after"]["description"], 0)
        self.assertEqual(self.analyzer.df.loc[1, "description"], "RED CUP SET")

    def test_iqr_detection_and_clipping(self):
        # 양수 구매의 극단값은 탐지되고, 원본을 보존한 정제 열에서는 제거되어야 한다.
        self.analyzer.handle_missing()
        self.analyzer.engineer_features()
        outliers = self.analyzer.detect_outliers("amount", positive_only=True)
        self.assertIn(15, outliers.index)
        report = self.analyzer.treat_outliers(
            "amount", output_col="amount_clean", positive_only=True
        )
        self.assertGreater(report["before_count"], 0)
        self.assertEqual(report["after_count"], 0)

    def test_rfm_has_all_scores_and_valid_reference_date(self):
        # 명시적 기준일을 사용해 Recency와 네 세그먼트의 계약을 검증한다.
        self.analyzer.handle_missing()
        self.analyzer.engineer_features()
        rfm = self.analyzer.calculate_rfm(reference_date="2024-02-01")
        expected = {"Recency", "Frequency", "Monetary", "R_score", "F_score", "M_score", "Segment"}
        self.assertTrue(expected.issubset(rfm.columns))
        self.assertTrue(rfm["Recency"].ge(0).all())
        self.assertTrue(rfm["Segment"].isin(["VIP", "Loyal", "New", "Churned"]).all())

    def test_legacy_missing_value_alias_matches_public_api(self):
        # 과제 명세가 요구하는 정확한 공개 메서드명이 실제 정책 구현으로 위임되는지 확인한다.
        report = self.analyzer.handle_missing_values()
        self.assertEqual(report["after"]["description"], 0)


if __name__ == "__main__":
    unittest.main()
