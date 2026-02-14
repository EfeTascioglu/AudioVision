"""
Sound localization utilities.
CSP: Cross-Power-Spectrum (Phase) comparison of two signals via a single FT.
Caller is responsible for windowing the signals before passing them in.
"""

import numpy as np
from typing import Tuple, Optional, Union

from audio_util import visualize_waveforms


def CSP(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    fs: float = 44200,
    use_phase_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-Power-Spectrum Phase (CSP): compare two sound signals using a single FT and power.

    Computes one FFT over the entire length of each signal, then the cross-power spectrum
    S_ab(f) = G_a(f) * conj(G_b(f)). Optionally uses phase-only weighting (GCC-PHAT style):
    S_ab / |S_ab|.

    Windowing must be applied by the caller before passing the signals.

    Parameters
    ----------
    signal_a, signal_b : np.ndarray
        One-dimensional sound signals (samples,). Same length recommended;
        shorter one is zero-padded to match the longer.
    fs : float, optional
        Sample rate in Hz. If given, freqs are in Hz; otherwise in cycles/sample.
    use_phase_only : bool
        If True, normalize by magnitude so only phase is used (PHAT weighting).

    Returns
    -------
    cross_power : np.ndarray
        Cross-power spectrum, shape (n_freqs,), complex.
    freqs : np.ndarray
        Frequency bin centers in Hz (if fs given) or in cycles/sample.
    """
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("signal_a and signal_b must be 1D arrays")

    assert len(a) == len(b)
    n = len(a)

    # Single FT over the entire window (caller does windowing)
    G_a = np.fft.rfft(a)
    G_b = np.fft.rfft(b)

    # Cross-power spectrum: G_a * conj(G_b)
    cross_power = G_a * np.conj(G_b)

    if use_phase_only:
        magnitude = np.abs(cross_power)
        magnitude[magnitude == 0] = 1.0  # avoid division by zero
        cross_power = cross_power / magnitude
    else:
        raise ValueError("Not Implemented")

    n_freqs = cross_power.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    
    return cross_power, freqs


def find_delay(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    window_len: int,
    step: int,
    use_phase_only: bool = True,
    fs: float = 44200,
) -> Tuple[int, float]:
    """
    Sliding-window CSP over two streams; returns the delay/shift that maximizes
    cross-power (cross-correlation peak).

    Takes a sliding window over both signals, computes CSP for each pair of
    windows, converts to cross-correlation via IFFT, and returns the delay
    (in samples and optionally seconds) that corresponds to the strongest
    aggregate correlation. Positive delay means signal_b is delayed relative
    to signal_a (shift b left by delay to align).

    Parameters
    ----------
    signal_a, signal_b : np.ndarray
        One-dimensional audio streams (same length).
    window_len : int
        Length of each sliding window in samples.
    step : int, optional
        Step between consecutive windows.
    use_phase_only : bool
        If True, use phase-only weighting in CSP (default True).
    fs : float, optional
        Sample rate in Hz for returning delay in seconds.

    Returns
    -------
    delay_samples : int
        Delay in samples (positive = b is later than a).
    delay_seconds : float or None
        Delay in seconds if fs was provided, else None.
    """
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("signal_a and signal_b must be 1D arrays")
    n = len(a)
    if len(b) != n:
        raise ValueError("signal_a and signal_b must have the same length")

    win = np.hanning(window_len)

    # Zero-pad to 2*window_len so IFFT gives full linear correlation (positive and negative lags)
    n_fft = 2 * window_len
    delays: list[int] = []
    peak_strengths: list[float] = []

    for start in range(0, n - window_len + 1, step):
        w_a = a[start : start + window_len] * win
        w_b = b[start : start + window_len] * win
        w_a_pad = np.pad(w_a, (0, window_len), mode="constant", constant_values=0.0)
        w_b_pad = np.pad(w_b, (0, window_len), mode="constant", constant_values=0.0)

        cross_power, _ = CSP(w_a_pad, w_b_pad, fs=44200, use_phase_only=use_phase_only)
        # cross_power length is n_fft//2 + 1; irfft gives n_fft samples
        corr = np.fft.irfft(cross_power, n=n_fft)
        # irfft: index 0..window_len = lags 0..window_len; index window_len+1..n_fft-1 = negative lags
        lag_idx = int(np.argmax(np.abs(corr)))
        lag_samp = lag_idx if lag_idx <= window_len else lag_idx - n_fft
        # Delay of b relative to a: positive when b is delayed (same content appears later in b)
        delay_samp = -lag_samp
        delays.append(delay_samp)
        peak_strengths.append(float(np.abs(corr[lag_idx])))

    # Return delay that maximizes aggregate confidence: use the delay from the window with strongest peak
    best_i = int(np.argmax(peak_strengths))
    delay_samples = delays[best_i]

    delay_seconds = delay_samples / fs

    return delay_samples, delay_seconds


def fake_audio_stream_test(delay=8) -> None:
    """Generate a single source, two delayed+noisy streams, and estimate delay with find_delay."""
    rng = np.random.default_rng(42)
    fs = 44200
    duration_sec = 2.0
    n_total = int(fs * duration_sec)

    # Single source: mixture of tones + noise
    t = np.linspace(0, duration_sec, n_total, endpoint=False)
    source = (
        1 * np.sin(2 * np.pi * (80) * t)
        + 0.3 * np.sin(2 * np.pi * 120 * t)
        + 0.2 * rng.standard_normal(n_total)
    )

    # Delays in samples (stream_b is delayed relative to stream_a by delay_b - delay_a)
    delay_a = 0
    delay_b = int(delay)  
    max_delay = max(delay_a, delay_b)
    n = n_total + max_delay

    stream_a = np.zeros(n)
    stream_a[delay_a : delay_a + n_total] = source
    stream_a += 0.1 * rng.standard_normal(n)

    stream_b = np.zeros(n)
    stream_b[delay_b : delay_b + n_total] = source
    stream_b += 0.15 * rng.standard_normal(n)

    stream_a = stream_a[:n_total]
    stream_b = stream_b[:n_total]

    visualize_waveforms(stream_a, stream_b, fs=fs)

    window_len = 4420  # 100ms
    delay_samples, delay_seconds = find_delay(
        stream_a, stream_b, window_len=window_len, step=window_len//2, fs=fs, use_phase_only=True
    )

    true_delay = delay_b - delay_a
    print("Generated: stream_b delayed relative to stream_a by", true_delay, "samples (", true_delay / fs * 1000, "ms )")
    print("Estimated delay (samples):", delay_samples)
    if delay_seconds is not None:
        print("Estimated delay (seconds):", delay_seconds)
        print("Estimated delay (ms):", delay_seconds * 1000)


if __name__ == "__main__":
    fake_audio_stream_test()
