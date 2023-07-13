from __future__ import annotations

import numpy as np
from scipy import fftpack
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def time_series_scaling(
    time_series: np.ndarray,
    feature_range: tuple[float, float] = (-1, 1),
    scaler: str = "MinMax",
    scaling_period_s: tuple[int, int] | None = None,
    scaling_features: np.ndarray | None = None,
    return_scaler: bool | None = None,
    sampling_rate: int = 25,
) -> tuple[np.ndarray, MinMaxScaler | RobustScaler] | np.ndarray:
    assert time_series.ndim == 1
    time_series = time_series.reshape(-1, 1)
    if scaling_features is not None:
        assert scaling_features.ndim == 1
        scaling_features = scaling_features.reshape(-1, 1)
    else:
        scaling_features = time_series
    if scaler == "MinMax":
        assert feature_range is not None
        assert -2 < feature_range[0] < feature_range[1] < 2
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler == "Robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"{scaler} not supported scaler, i.e MinMax and Robust")
    if scaling_period_s:
        assert all(isinstance(el, int) for el in scaling_period_s)
        assert 0 <= scaling_period_s[0] < scaling_period_s[1]
        scaler.fit(
            time_series[
                scaling_period_s[0]
                * sampling_rate : scaling_period_s[1]
                * sampling_rate
            ]
        )  # converts period limits in index
    else:
        scaler.fit(scaling_features)
    scaled_time_series = scaler.transform(time_series)
    if return_scaler:
        return scaled_time_series[:, 0], scaler
    else:
        return scaled_time_series[:, 0]


def fourier_smoothing(
    time_series: np.ndarray,
    freq_threshold_hz: float,
    sampling_rate: int,
    return_spectrum: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """De-noises a signal using Fourier smoothing.

    Especially at the edges this approach performs poorly.
    For details look at https://en.wikipedia.org/wiki/Gibbs_phenomenon.

    :param time_series: 1-D array which is going to be de-noised
    :type time_series:
    :param freq_threshold_hz: cutoff frequency in Hz
    :type freq_threshold_hz:
    :param sampling_rate: samples per second of time_series
    :type sampling_rate:
    :param return_spectrum: if set to True, return smoothed signal, raw spectrum
    (of time_series) and new spec (of smoothed time series)
    :type return_spectrum:
    :return: smoothed signal
    :rtype:
    """

    raw_spec = fftpack.rfft(time_series)
    time_step = 1 / sampling_rate
    freq = fftpack.rfftfreq(len(raw_spec), d=time_step)

    # set all frequencies greater freq_threshold to zero
    new_spec = raw_spec.copy()
    new_spec[freq > freq_threshold_hz] = 0
    # reverse fourier transformation
    smoothed_time_series = fftpack.irfft(new_spec)

    if return_spectrum:
        return smoothed_time_series, raw_spec, new_spec, freq
    return smoothed_time_series


def add_white_noise_to_signal(
    target_snr_db: int, signal: np.ndarray, **kwargs
) -> np.ndarray:
    """
    :param target_snr_db: signal-to-noise-ratio (SNR) in dB
    :type target_snr_db:
    :param signal: 1-D array on which to add noise
    :type signal:
    :return: 1-D array. time_series with SNR target_snr_db
    :rtype:
    """
    seed = kwargs.get("seed", None)
    # Calculate signal power and convert to dB
    signal_power = signal**2
    sig_avg_power = np.mean(signal_power)
    sig_avg_db = 10 * np.log10(sig_avg_power)
    # Calculate noise according to SNR_dB = P_signal_dB - P_noise_dB
    # with P being the average signal power then convert
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_square = 10 ** (noise_avg_db / 10)
    # Generate a sample of white noise
    mean_noise = 0
    if seed:
        np.random.seed(seed)
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_square), len(signal_power))
    # Noise up the original signal
    noisy_time_series = signal + noise
    return noisy_time_series
