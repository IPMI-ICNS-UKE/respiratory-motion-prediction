from unittest import TestCase

import numpy as np

from rmp.dataloader import RpmSignals


class SlidingWindowWrapper(TestCase):
    def setUp(self) -> None:
        self.time_series = np.arange(0, 25 * 200, 1)

    def test_case_01(self):
        future_points, input_features, output_features = 20, 15, 1
        input_wdws, output_wdws = RpmSignals.sliding_wdw_wrapper(
            time_series=self.time_series,
            future_points=future_points,
            input_features=input_features,
            output_features=output_features,
        )
        self.assertTrue(
            input_wdws.shape[1] == input_features
            and output_wdws.shape[1] == output_features
        )
        self.assertTrue(
            np.allclose(input_wdws[0], np.arange(0, 15, 1))
            and np.allclose(input_wdws[1], np.arange(1, 16, 1))
        )
        self.assertTrue(
            np.allclose(output_wdws[0], 34) and np.allclose(output_wdws[1], 35)
        )
        self.assertTrue(np.allclose(output_wdws[-1], self.time_series.max()))

    def test_case_02(self):
        future_points, input_features, output_features = 20, 15, 1
        input_wdws, output_wdws = RpmSignals.sliding_wdw_wrapper(
            time_series=self.time_series,
            future_points=future_points,
            input_features=input_features,
            output_features=output_features,
        )
        self.assertTrue(
            input_wdws.shape[1] == input_features
            and output_wdws.shape[1] == output_features
        )
        self.assertTrue(
            np.allclose(input_wdws[0], np.arange(0, 15, 1))
            and np.allclose(input_wdws[1], np.arange(1, 16, 1))
        )
        self.assertTrue(
            np.allclose(output_wdws[0], 34) and np.allclose(output_wdws[1], 35)
        )
        self.assertTrue(np.allclose(output_wdws[-1], self.time_series.max()))

    def test_case_03(self):
        future_points, input_features, output_features = 12, 15, 12
        input_wdws, output_wdws = RpmSignals.sliding_wdw_wrapper(
            time_series=self.time_series,
            future_points=future_points,
            input_features=input_features,
            output_features=output_features,
        )
        self.assertTrue(
            input_wdws.shape[1] == input_features
            and output_wdws.shape[1] == output_features
        )
        self.assertTrue(
            np.allclose(input_wdws[0], np.arange(0, 15, 1))
            and np.allclose(input_wdws[1], np.arange(12, 27, 1))
        )
        self.assertTrue(
            np.allclose(output_wdws[0], np.arange(15, 27, 1))
            and np.allclose(output_wdws[1], np.arange(27, 39, 1))
        )
        max_value = (
            int(len(self.time_series) / (input_features + future_points))
            * (input_features + future_points)
            - 1
        )
        self.assertTrue(
            np.allclose(output_wdws[-1].max(), max_value), output_wdws[-1].max()
        )

    def test_case_04(self):
        future_points, input_features, output_features = 20, 15, 22
        with self.assertRaises(ValueError):
            _, _ = RpmSignals.sliding_wdw_wrapper(
                time_series=self.time_series,
                future_points=future_points,
                input_features=input_features,
                output_features=output_features,
            )
