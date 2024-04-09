import logging
from collections import namedtuple
from unittest import TestCase

import numpy as np
import torch

from rmp.dataloader import RpmSignals
from rmp.global_config import DEVICE
from rmp.models import Transformer
from rmp.utils.logger import LoggerMixin

logger = logging.getLogger(__name__)


class TransformerInferenceTester(TestCase, LoggerMixin):
    """forward and inference method of the Transformer should return the same
    output (if model in eval mode (no dropout etc.))"""

    def setUp(self) -> None:
        time = np.arange(0, 30, 0.04)
        amplitude = np.cos(time).astype(np.float32)
        input_features = 128
        future_steps = 12
        dataloader_paras = namedtuple(
            "dataloader_paras", ["future_points", "input_features", "output_features"]
        )
        data_paras = dataloader_paras(
            input_features=input_features, future_points=future_steps, output_features=1
        )

        self.input_wdws, _ = RpmSignals.sliding_wdw_wrapper(
            time_series=amplitude,
            future_points=data_paras.future_points,
            input_features=data_paras.input_features,
            output_features=data_paras.output_features,
        )
        self.input_wdws = torch.from_numpy(self.input_wdws).unsqueeze(0).to(DEVICE)

        self.model = Transformer(
            input_features_dim=input_features,
            embedding_features=64,
            num_layers=8,
            dropout=0,
            n_heads=4,
            future_steps=future_steps,
            pre_training_size=0,
            output_dim=1,
            max_signal_length_s=300,
        ).to(DEVICE)

        torch.manual_seed(0)

    def test_case_01(self):
        self.model.eval()
        with torch.no_grad():
            input_wdws = self.input_wdws[:, :, :]
            self.logger.info(f"{input_wdws.shape=}")
            result_forward = self.model(input_wdws)
            result_inference = self.model.infer(input_wdws)
        diff = (
            result_forward[:, self.model.pre_training_size :, :]
            - result_inference[:, self.model.pre_training_size :, :]
        )
        diff = diff.abs()
        max_, min_, mean_ = diff.max(), diff.min(), diff.mean()
        self.logger.info(f"diff stats: {max_=}, {min_=}, {mean_=}")
        self.assertTrue(max_ < 0.0001)
