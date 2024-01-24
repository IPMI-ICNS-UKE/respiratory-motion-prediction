from __future__ import annotations

import numpy as np
import scipy.signal as signal
from scipy import fftpack, interpolate
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from rmp.utils.common_types import Number


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


def resample_time_series(
    signal_time_secs: np.ndarray,
    signal_amplitude: np.ndarray,
    target_samples_per_second: Number,
) -> tuple[np.ndarray, np.ndarray]:
    """Resamples a given time series to the desired samples per second (target)
    by performing a simple 1D interpolation.

    :param signal_time_secs: 1-D array containing time component
    :type signal_time_secs:
    :param signal_amplitude: 1-D array containing amplitude information
    :type signal_amplitude:
    :param target_samples_per_second:
    :type target_samples_per_second:
    :return: time and corresponding amplitude of resampled signal
    :rtype: tuple(1-D array, 1-D array)
    """
    if signal_amplitude.shape != signal_time_secs.shape:
        raise ValueError("Shape mismatch")

    # define function between time and amplitude
    sampler = interpolate.interp1d(signal_time_secs, signal_amplitude)

    # calc time_new with given samples_per_second
    step = 1 / target_samples_per_second
    t_new = np.arange(
        start=signal_time_secs.min(),
        stop=signal_time_secs.min() + max(signal_time_secs),
        step=step,
    )
    a_new = sampler(t_new)

    return t_new, a_new


def calc_derivative(
    signal: np.ndarray,
    samples_per_second: Number,
    smoothing_kernel_size: int | None = None,
) -> np.ndarray:
    """Calculates the derivative of a given time series signal. If a kernel is
    given, derivative is smoothed.

    :param signal: 1-D array containing time component
    :type signal:
    :param samples_per_second: samples per second of given signal
    :type signal:
    :param smoothing_kernel_size: size of smoothing kernel
    :type smoothing_kernel_size:

    :return: derivative
    :rtype:
    """
    dt = 1 / samples_per_second
    derivative = np.gradient(signal, dt)
    if smoothing_kernel_size:
        if not isinstance(smoothing_kernel_size, int):
            raise ValueError("smoothing_kernel_size has to be integer")
        con_vec = np.ones(smoothing_kernel_size) / smoothing_kernel_size
        derivative = np.convolve(derivative, con_vec, mode="same")
    return derivative


def find_peaks(x: np.ndarray, scale: int = None, debug: bool = False):
    """Find peaks in quasi-periodic noisy signals using AMPD algorithm.
    Extended implementation handles peaks near start/end of the signal.
    Optimized implementation by Igor Gotlibovych, 2018.

    Taken from https://github.com/ig248/pyampd

    Parameters
    ----------
    x : ndarray
        1-D array on which to find peaks
    scale : int, optional
        specify maximum scale window size of (2 * scale + 1)
    debug : bool, optional
        if set to True, return the Local Scalogram Matrix, `LSM`,
        weigted number of maxima, 'G',
        and scale at which G is maximized, `l`,
        together with peak locations
    Returns
    -------
    pks: ndarray
        The ordered array of peak indices found in `x`
    """
    x = signal.detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM[k - 1, 0 : N - k] &= x[0 : N - k] > x[k:N]  # compare to right neighbours
        LSM[k - 1, k:N] &= x[k:N] > x[0 : N - k]  # compare to left neighbours

    # Find scale with most maxima
    G = LSM.sum(axis=1)
    G = G * np.arange(
        N // 2, N // 2 - L, -1
    )  # normalize to adjust for new edge regions
    l_scale = np.argmax(G)

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)
    if debug:
        return pks, LSM, G, l_scale
    return pks


def split_into_cycles(curve: np.ndarray, peaks: np.ndarray = None) -> list[np.ndarray]:
    if peaks is None:
        peaks = find_peaks(curve)
    # discard potentially incomplete first and last cycle
    slicing = slice(None)
    if peaks[0] == 0:
        slicing = slice(1, None)
    if peaks[-1] == len(curve) - 1:
        slicing = slice(slicing.start, -1)
    return np.split(curve, peaks[slicing])
