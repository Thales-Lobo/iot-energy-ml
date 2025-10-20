"""
central/app/preprocess.py

Preprocessing utilities for converting raw IoT current measurements
into a fixed-size numerical feature vector suitable for ML models.

The preprocessing pipeline is designed to match the transformations
used during model training. It includes optional decimation, smoothing,
and feature extraction steps.

Key exported functions:
-----------------------
- decimate_if_needed(signal, original_rate=48828, target_rate=2048)
- moving_average(signal, window_size=15)
- extract_features(signal)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import decimate
from scipy.stats import skew, kurtosis


# ------------------------------------------------------------
# Utility: decimation
# ------------------------------------------------------------
def decimate_if_needed(signal: np.ndarray, original_rate: int = 48828, target_rate: int = 2048) -> np.ndarray:
    """
    Downsample the signal if its sampling rate is higher than the target rate.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array (1D).
    original_rate : int, optional
        Original sampling frequency in Hz (default: 48,828, as in Dragon_Pi dataset).
    target_rate : int, optional
        Target sampling frequency in Hz (default: 2,048, as used in many ML pipelines).

    Returns
    -------
    np.ndarray
        Decimated signal (1D array).

    Notes
    -----
    - Uses FIR filter + zero-phase decimation from SciPy.
    - If original_rate <= target_rate, returns the signal unchanged.
    """
    if original_rate <= target_rate:
        return signal

    factor = original_rate // target_rate
    if factor <= 1:
        return signal

    try:
        decimated = decimate(signal, factor, ftype="fir", zero_phase=True)
        return decimated
    except Exception:
        # fallback (avoid crash if decimation fails)
        return signal[::factor]


# ------------------------------------------------------------
# Utility: moving average smoothing
# ------------------------------------------------------------
def moving_average(signal: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Apply a simple moving average filter to reduce noise.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).
    window_size : int, optional
        Number of samples for the moving window (default: 15).

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if window_size <= 1:
        return signal

    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="same")


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract numerical features from a 1D current signal.

    The resulting features are designed to capture time-domain
    and frequency-domain characteristics relevant to anomaly
    detection in energy consumption.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array).

    Returns
    -------
    np.ndarray
        Feature vector (1D array).

    Extracted features
    ------------------
    Time-domain:
        - Mean
        - Standard deviation
        - Min
        - Max
        - Peak-to-peak
        - Root Mean Square (RMS)
        - Skewness
        - Kurtosis
    Frequency-domain:
        - Spectral centroid
        - Spectral bandwidth
        - Total spectral energy (sum of squared magnitudes)
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be 1D")

    # --- Basic stats
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    ptp_val = np.ptp(signal)
    rms_val = np.sqrt(np.mean(signal ** 2))
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)

    # --- Frequency domain (FFT-based)
    fft_vals = np.fft.rfft(signal)
    fft_magnitudes = np.abs(fft_vals)
    fft_freqs = np.fft.rfftfreq(len(signal), d=1.0)

    # Avoid division by zero
    total_energy = np.sum(fft_magnitudes ** 2)
    if total_energy == 0:
        total_energy = 1e-12

    spectral_centroid = np.sum(fft_freqs * fft_magnitudes) / np.sum(fft_magnitudes)
    spectral_bandwidth = np.sqrt(
        np.sum(((fft_freqs - spectral_centroid) ** 2) * fft_magnitudes)
        / np.sum(fft_magnitudes)
    )
    spectral_energy = np.sum(fft_magnitudes ** 2)

    features = np.array(
        [
            mean_val,
            std_val,
            min_val,
            max_val,
            ptp_val,
            rms_val,
            skew_val,
            kurt_val,
            spectral_centroid,
            spectral_bandwidth,
            spectral_energy,
        ],
        dtype=np.float32,
    )

    return features
