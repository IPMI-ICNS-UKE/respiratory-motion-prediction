import asyncio
import time
from dataclasses import dataclass
from io import StringIO

import aiohttp
import numpy as np
import pandas as pd
from tqdm import tqdm

from rmp.utils.logger import LoggerMixin
from rmp.utils.time_series_utils import find_peaks, resample_time_series


@dataclass
class Signal:
    id: int
    research_number_pid: int
    modality: str
    fraction: int
    is_corrupted: bool
    length_secs: float
    hash: str
    df_signal: pd.DataFrame


class RpmApiClient(LoggerMixin):
    def __init__(self, mode="train", base_url="http://localhost:8000"):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.base_url = base_url
        signals = self._query_data()
        self.signals = self._convert_signals_to_df(signals)

    def _query_data(self):
        set_url = f"{self.base_url}/signals/base/sets/{self.mode}"
        self.logger.info(f"Start querying {self.mode} dataset " f"from {set_url}...")

        start = time.time()
        responses = asyncio.run(self._fetch_all([set_url]))
        train_set = responses[0]
        urls = []
        for train_signal in tqdm(train_set):
            urls.append(
                f"{self.base_url}/signals/detail/{train_signal['research_number_pid']}/{train_signal['modality']}/{train_signal['fraction']}"  # noqa
            )
        signals = asyncio.run(self._fetch_all(urls))
        end = time.time()
        self.logger.info(
            f"Querying completed. Loaded "
            f"{len(signals)} signals in {end - start:.2f} seconds."
        )
        return signals

    def _convert_signals_to_df(self, signals):
        self.logger.info("Starting converting retrieved data to dataframes...")
        start = time.time()
        _signals = []
        for signal in tqdm(signals):
            signal["df_signal"] = pd.read_json(StringIO(signal["df_signal"]))
            _signals.append(Signal(**signal))
        self.logger.info(
            f"Converting completed " f"(took {time.time() - start: .1f} s)."
        )
        return _signals

    @staticmethod
    async def _fetch(url, session):
        async with session.get(url) as response:
            return await response.json()

    async def _fetch_all(self, urls):
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[self._fetch(url, session) for url in urls],
                return_exceptions=True,
            )
            return results

    def len(self):
        return len(self.signals)

    def get_signal(self, idx):
        signal = self.signals[idx]
        return signal

    @staticmethod
    def preprocess_signal(
        df_signal: pd.DataFrame,
        only_beam_on: bool = True,
        sampling_rate: int = 25,
        remove_offset: bool = True,
    ) -> pd.DataFrame:
        """Performs preprocessing by.

        - only using first to last beam on point (excluding potential acquisition errors)
        - resampling to given sampling_rate
        - shifting raw signal that first three minima are at zero.
        :param df_signal:
        :param only_beam_on:
        :param sampling_rate:
        :param remove_offset:
        :return: pd.Dataframe
        """
        if not isinstance(df_signal, pd.DataFrame):
            raise ValueError(
                f"df_signal should be a Dataframe but is type {type(df_signal)}"
            )
        if not {"time", "amplitude", "beam_on"}.issubset(df_signal.columns):
            raise ValueError(
                f"Dataframe does not contain all mandatory columns; {df_signal.columns}"
            )
        if (
            any(df_signal.amplitude.isna())
            or any(df_signal.time.isna())
            or any(df_signal.beam_on.isna())
        ):
            raise ValueError("Contain invalid data")
        if only_beam_on:
            beam_on_idx = np.where(df_signal.beam_on == 0)[0]
            first_beam_on, last_beam_on = min(beam_on_idx), max(beam_on_idx)
            df_signal = df_signal[first_beam_on:last_beam_on]
            df_signal.reset_index(inplace=True, drop=True)
            time_offset = df_signal.time.min()
            df_signal[:]["time"] -= time_offset
        if sampling_rate:
            t_new, a_new = resample_time_series(
                signal_time_secs=df_signal.time.values,
                signal_amplitude=df_signal.amplitude.values,
                target_samples_per_second=sampling_rate,
            )
            df_signal = pd.DataFrame.from_dict(
                dict(time=t_new, amplitude=a_new), dtype=float
            )
        if remove_offset:
            signal_subset = -1 * df_signal.amplitude[df_signal.time < 50]
            number_minima = 3
            minima_idx = find_peaks(x=signal_subset.values)
            minima = df_signal.amplitude[minima_idx].values
            df_signal.loc[:, "amplitude"] = (
                df_signal.amplitude - minima[:number_minima].mean()
            )
        return df_signal
