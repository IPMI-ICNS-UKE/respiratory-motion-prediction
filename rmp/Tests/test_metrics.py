from unittest import TestCase

import numpy as np

from rmp.metrics import calculate_relative_rmse
from rmp.my_utils.logger import LoggerMixin


class CalculateRelativeRmse(TestCase, LoggerMixin):
    def setUp(self) -> None:
        self.t = np.arange(0, 100, 0.1)
        self.a = np.cos(self.t)
        self.pred_horizons = [6, 12, 24, 48]

    def test_perfect_prediction(self):
        for future_steps in self.pred_horizons:
            e_rmse = calculate_relative_rmse(
                y_pred=self.a, y_true=self.a, future_steps=future_steps
            )
            self.assertAlmostEqual(e_rmse, 0.0)

    def test_worst_prediction(self):
        for future_steps in self.pred_horizons:
            e_rmse = calculate_relative_rmse(
                y_pred=-2 + np.zeros((len(self.a),)),
                y_true=self.a,
                future_steps=future_steps,
            )
            self.assertGreater(e_rmse, 1.0)

    def test_delayed_prediction(self):
        for future_steps in self.pred_horizons:
            y_delayed = np.concatenate(
                (np.zeros((future_steps,)), self.a[:-future_steps])
            )
            e_rmse = calculate_relative_rmse(
                y_pred=y_delayed, y_true=self.a, future_steps=future_steps
            )
            self.assertAlmostEqual(e_rmse, 1.0)
