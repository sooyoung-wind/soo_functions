import unittest
import numpy as np
import pandas as pd
from statistic import wd_statistic, me, mae, rmse, d, wd_diff_cal, mean_wd


class TestSooFunctions(unittest.TestCase):

    def setUp(self):
        # 테스트용 예측값과 실제값 데이터
        # self.F_data = np.array([10, 20, 30, 40, 50])
        self.F_data = pd.DataFrame([10, 20, 30, 40, 50])
        # self.A_data = np.array([12, 18, 29, 41, 52])
        self.A_data = pd.DataFrame([12, 18, 29, 41, 52])

    def test_me(self):
        # ME (Mean Error) 테스트
        result = me(self.F_data, self.A_data)
        expected = np.mean(self.F_data - self.A_data)
        self.assertAlmostEqual(result, expected, places=5)

    def test_mae(self):
        # MAE (Mean Absolute Error) 테스트
        result = mae(self.F_data, self.A_data)
        expected = np.mean(np.abs(self.F_data - self.A_data))
        self.assertAlmostEqual(result, expected, places=5)

    def test_rmse(self):
        # RMSE (Root Mean Square Error) 테스트
        result = rmse(self.F_data, self.A_data)
        expected = np.sqrt(np.mean((self.F_data - self.A_data) ** 2))
        self.assertAlmostEqual(result, expected, places=5)

    def test_d(self):
        # IOA (Index of Agreement) 테스트
        result = d(self.F_data, self.A_data)
        obs_mean = np.mean(self.A_data)
        numerator = np.sum((self.F_data - self.A_data) ** 2)
        denominator = np.sum((np.abs(self.F_data - obs_mean) + np.abs(self.A_data - obs_mean)) ** 2)
        expected = 1 - numerator / denominator
        self.assertAlmostEqual(result, expected, places=5)

    def test_wd_statistic(self):
        # wd_statistic 함수 테스트
        result = wd_statistic(self.F_data, self.A_data)
        self.assertIsInstance(result, pd.DataFrame)  # 결과가 DataFrame인지 확인
        self.assertIn("ME", result.columns)  # "ME" 컬럼이 있는지 확인
        self.assertIn("RMSE", result.columns)  # "RMSE" 컬럼이 있는지 확인

    def test_wd_diff_cal(self):
        # wd_diff_cal 테스트
        result = wd_diff_cal(self.F_data, self.A_data)
        self.assertEqual(len(result), len(self.F_data))  # 결과 길이가 입력 데이터와 같은지 확인

    def test_mean_wd(self):
        # mean_wd 함수 테스트
        ws = np.ones(len(self.F_data))
        result_mean = mean_wd(ws, self.F_data, "mean")
        result_mode = mean_wd(ws, self.F_data, "mode")
        result_median = mean_wd(ws, self.F_data, "median")

        # 평균, 최빈값, 중앙값이 모두 계산되는지 확인
        self.assertIsNotNone(result_mean)
        self.assertIsNotNone(result_mode)
        self.assertIsNotNone(result_median)


if __name__ == '__main__':
    unittest.main()
