"""Loads respiratory signal and performs preprocessing (smoothing, de-noising,
scaling etc.)"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from random import randrange
from typing import Any

import numpy as np
import torch
from resp_db.client import RpmDatabaseClient
from torch import Tensor
from torch.utils.data import Dataset

from rmp.my_utils.logger import LoggerMixin
from rmp.my_utils.time_series_utils import (
    add_white_noise_to_signal,
    fourier_smoothing,
    time_series_scaling,
)


class ModelPhases(Enum):
    """Phase of the model in the pipeline."""

    TRAINING = "train"
    VALIDATION = "val"
    TESTING = "test"


class RpmSignals(Dataset, LoggerMixin):
    def __init__(
        self,
        db_root: Path,
        max_num_curves: int | None = None,
        phase: ModelPhases = ModelPhases.TRAINING,
        signal_length_s: int | None = None,
        scaling_period_s: tuple[float, float] | None = (0, 20),
        white_noise_db: int | None = None,
        future_points: int = 12,
        input_features: int = 1,
        output_features: int = 1,
        min_length_s: int = 0,
        use_smooth_target: bool = True,
        cache_returned_data: bool = False,
    ):
        super().__init__()
        self.mode = phase
        self.db_root = db_root
        self.signal_length_s = signal_length_s
        self.use_smooth_target = use_smooth_target
        self.white_noise_db = white_noise_db
        if self.white_noise_db is None:
            self.logger.warning("Smooth input signals are used.")
        self.sampling_rate = 25
        self.scaling_period_s = scaling_period_s
        self.input_features = input_features  # input features per time step
        self.output_features = output_features  # output features per time step
        if self.output_features != 1:
            raise ValueError("This project supports only a single point prediction")
        self.cache_returned_data = cache_returned_data  # True
        self.future_points = future_points
        client = RpmDatabaseClient(db_filepath=db_root)
        with client:
            query = client.get_signals_of_dl_dataset(
                dl_dataset=self.mode.value, project="short-term-prediction"
            )
        self.query = [
            signal
            for signal in query
            if client.preprocess_signal(signal.df_signal).time.max() > min_length_s
        ]
        self.query = self.query[:max_num_curves]
        self.logger.info(
            f"{self.mode.value.upper()} dataset was successfully loaded. It contains {len(self.query)} signals."
        )

    def __len__(self):
        return len(self.query)

    def __getitem__(
        self, index: int
    ) -> tuple[str, Tensor, Tensor, dict[str, Any]] | tuple[str, Tensor, Tensor]:
        """Preprocessing of raw signals in a sliding window fashion."""
        signal = self.query[index]
        research_number, modality, fraction = (
            signal.research_number,
            signal.modality,
            signal.fraction,
        )
        name = f"{research_number}_{modality}_{fraction}"
        df = RpmDatabaseClient.preprocess_signal(
            df_signal=signal.df_signal,
            sampling_rate=self.sampling_rate,
            only_beam_on=True,
            remove_offset=True,
        )
        df = df[df["time"] < 300]
        time = df.time.values
        time_series_smooth = fourier_smoothing(
            time_series=df.amplitude.values,
            freq_threshold_hz=1,
            sampling_rate=self.sampling_rate,
            return_spectrum=False,
        )
        if self.white_noise_db:
            if self.mode is ModelPhases.TESTING:
                seed = 42
            else:
                seed = None
            time_series_noisy = add_white_noise_to_signal(
                target_snr_db=self.white_noise_db, signal=time_series_smooth, seed=seed
            )
            self.logger.debug(
                f"White noise {self.white_noise_db} dB added to each signal"
            )
        else:
            time_series_noisy = time_series_smooth  # no white noise added

        time_series_noisy, scaler = time_series_scaling(
            time_series=time_series_noisy,
            feature_range=(-1, 1),
            scaler="MinMax",
            scaling_period_s=self.scaling_period_s,
            return_scaler=True,
        )
        time_series_smooth = scaler.transform(time_series_smooth.reshape(-1, 1))[:, 0]
        if self.signal_length_s is not None:
            start_signal_index, end_signal_index = self.select_random_subset(
                time_series_len=len(time_series_noisy),
                signal_length_s=self.signal_length_s,
                future_points=self.future_points,
                samples_per_second=self.sampling_rate,
            )
            time_series_noisy = time_series_noisy[start_signal_index:end_signal_index]
            time_series_smooth = time_series_smooth[start_signal_index:end_signal_index]
            time = time[start_signal_index:end_signal_index]

        if not (time_series_noisy.shape == time_series_smooth.shape == time.shape):
            raise ValueError(
                f"Different shapes... \n "
                f"{time.shape=}  \n "
                f"{time_series_smooth=} \n"
                f"{time_series_noisy=}"
            )
        input_wdws, output_wdws = self.sliding_wdw_wrapper(
            time_series=time_series_noisy,
            future_points=self.future_points,
            input_features=self.input_features,
            output_features=self.output_features,
        )
        if self.use_smooth_target:
            _, output_wdws = self.sliding_wdw_wrapper(
                time_series=time_series_smooth,
                future_points=self.future_points,
                input_features=self.input_features,
                output_features=self.output_features,
            )
        time_series_input = torch.from_numpy(input_wdws.astype(np.float32))
        time_series_target = torch.from_numpy(output_wdws.astype(np.float32))
        assert time_series_input.shape[-1] == self.input_features
        assert (
            time_series_target.shape[-1] == self.output_features
        ), f"{time_series_target.shape=}vs.{self.output_features=}"
        if self.cache_returned_data:
            if self.use_smooth_target:
                target_time_series = time_series_smooth
            else:
                target_time_series = time_series_noisy
            all_raw_data = self.cache_time_series(
                name=name,
                time=time,
                input_time_series=time_series_noisy.astype(np.float32),
                target_time_series=target_time_series.astype(np.float32),
            )
            return name, time_series_input, time_series_target, all_raw_data
        return name, time_series_input, time_series_target

    @staticmethod
    def cache_time_series(name, time, input_time_series, target_time_series):
        return dict(
            name=name,
            time=time,
            input_time_series=input_time_series,
            target_time_series=target_time_series,
        )

    @staticmethod
    def select_random_subset(
        time_series_len: int,
        signal_length_s: int,
        future_points: int,
        samples_per_second: int,
    ) -> tuple[int, int]:
        num_points = samples_per_second * signal_length_s
        if time_series_len < num_points:
            raise ValueError("Time series shorter than desired signal")
        if time_series_len - 1 - num_points - future_points <= 0:
            raise ValueError(
                "Cannot pick a random snippet cause time series is to short"
            )
        start_signal_index = randrange(
            0, time_series_len - 1 - num_points - future_points
        )
        end_signal_index = start_signal_index + num_points
        return start_signal_index, end_signal_index

    @staticmethod
    def sliding_wdw_wrapper(
        time_series: np.array,
        future_points: int,
        input_features: int,
        output_features: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if input_features < output_features or output_features > future_points:
            raise ValueError(
                f"Cannot handle given input parameters in order to produce continuous output. \n"
                f"Parameters: \n"
                f"{future_points=}\n"
                f"{input_features=}\n"
                f"{output_features=}\n"
            )
        input_wdws, output_wdws = RpmSignals.sliding_wdws_vectorized(
            data=time_series,
            wdw_size_i=input_features,
            wdw_size_o=future_points,
            step_size=output_features,
        )
        output_wdws = output_wdws[:, -output_features:]
        return input_wdws, output_wdws

    @staticmethod
    def sliding_wdws_vectorized(
        data: np.ndarray, wdw_size_i: int = 6, wdw_size_o: int = 2, step_size: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Taken from https://github.com/LMUK-RADONC-PHYS-
        RES/lstm_centroid_prediction. Given a sequence of input data, subdivide
        it in input and output windows (vectorized operations). Adapted from:
        https://towardsdatascience.com/fast-and-robust-sliding-window-
        vectorization-with-numpy-3ad950ed62f5.

        :param data: array with input data
        :param wdw_size_i: length of generated window to be used as input
        :param wdw_size_o: length of generated window to be used as output
        :param step_size: number of data points the window rolls at each step
        :return: input and output windows
        """
        start = 0
        stop = len(data)
        idx_windows_i = (
            start
            + np.expand_dims(np.arange(wdw_size_i), 0)
            + np.expand_dims(
                np.arange(stop - wdw_size_i - wdw_size_o + 1, step=step_size), 0
            ).T
        )
        idx_windows_o = idx_windows_i[:, -wdw_size_o:] + wdw_size_o
        return (
            data[idx_windows_i],
            data[idx_windows_o],
        )
