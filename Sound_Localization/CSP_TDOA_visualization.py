"""
CSP/TDOA pipeline from fake_audio_stream_test with step-by-step visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sound_localization import CSP, find_delay


def fake_audio_stream_test(delay: int = 8) -> None:
    rng = np.random.default_rng(42)
    fs = 48000
    duration_sec = 2.0
    n_total = int(fs * duration_sec)

    t = np.linspace(0, duration_sec, n_total, endpoint=False)
    source = (
        1 * np.sin(2 * np.pi * 80 * t)
        + 0.3 * np.sin(2 * np.pi * 120 * t)
        + 0.2 * rng.standard_normal(n_total)
    )

    delay_a, delay_b = 0, int(delay)
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

    # --- 1. Original source ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.plot(np.arange(n_total) / fs, source)
    ax.set_xlabel("Time (s)")
    ax.set_title("Original source")
    ax.set_xlim(0, 0.05)
    plt.tight_layout()
    plt.show()

    # --- 2. Shifted streams (stream_a, stream_b) ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 3))
    t_axis = np.arange(n_total) / fs
    axes[0].plot(t_axis, stream_a, label="stream_a (delay=0)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Stream A")
    axes[0].legend()
    axes[0].set_xlim(0, 0.05)
    axes[1].plot(t_axis, stream_b, label=f"stream_b (delay={delay_b} samples)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Stream B")
    axes[1].legend()
    axes[1].set_xlim(0, 0.05)
    plt.tight_layout()
    plt.show()

    window_len = 4420
    step = window_len // 2
    win = np.hanning(window_len)
    n_fft = 2 * window_len
    delays_list: list[int] = []
    peak_strengths_list: list[float] = []

    # Pick a representative window (e.g. middle) for frequency/correlation plots
    starts = list(range(0, n_total - window_len + 1, step))
    mid_start = starts[len(starts) // 2]

    for start in starts:
        w_a = stream_a[start : start + window_len] * win
        w_b = stream_b[start : start + window_len] * win
        w_a_pad = np.pad(w_a, (0, window_len), mode="constant", constant_values=0.0)
        w_b_pad = np.pad(w_b, (0, window_len), mode="constant", constant_values=0.0)

        cross_power, freqs = CSP(w_a_pad, w_b_pad, fs=fs, use_phase_only=True)
        corr = np.fft.irfft(cross_power, n=n_fft)

        lag_idx = int(np.argmax(np.abs(corr)))
        lag_samp = lag_idx if lag_idx <= window_len else lag_idx - n_fft
        delay_samp = -lag_samp
        delays_list.append(delay_samp)
        peak_strengths_list.append(float(np.abs(corr[lag_idx])))

        if start == mid_start:
            cross_power_mid = cross_power
            freqs_mid = freqs
            corr_mid = corr

    # --- 3. Frequency domain (one window: cross-power magnitude & phase) ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    axes[0].plot(freqs_mid, np.abs(cross_power_mid))
    axes[0].set_ylabel("|Cross-power|")
    axes[0].set_title("Cross-power spectrum (frequency domain), one window")
    axes[1].plot(freqs_mid, np.angle(cross_power_mid))
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (rad)")
    plt.tight_layout()
    plt.show()

    # --- 4. Correlation domain (one window: corr vs lag) ---
    # irfft order: lag 0..window_len, then lag -window_len+1..-1
    lag_axis = np.arange(n_fft)
    lag_axis[window_len + 1 :] -= n_fft  # convert to actual lag in samples
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(lag_axis, np.abs(corr_mid))
    ax.axvline(delay_b - delay_a, color="green", linestyle="--", label=f"True delay = {delay_b} samp")
    ax.axvline(delays_list[len(starts) // 2], color="red", linestyle=":", label="Est. (this window)")
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("|Correlation|")
    ax.set_title("Cross-correlation (correlation domain), one window")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # --- 5. Lag values per window ---
    best_i = int(np.argmax(peak_strengths_list))
    delay_samples = delays_list[best_i]
    delay_seconds = delay_samples / fs

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    window_centers = [s + window_len / 2 for s in starts]
    ax.plot(np.array(window_centers) / fs, delays_list, "b.", markersize=2, label="Delay per window")
    ax.axhline(delay_b - delay_a, color="green", linestyle="--", label="True delay")
    ax.axhline(delay_samples, color="red", linestyle=":", label="Final estimate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Delay (samples)")
    ax.set_title("Estimated delay per window")
    ax.legend()
    plt.tight_layout()
    plt.show()

    true_delay = delay_b - delay_a
    print(f"True delay: {true_delay} samples ({true_delay / fs * 1000:.2f} ms)")
    print(f"Estimated: {delay_samples} samples ({delay_seconds * 1000:.2f} ms)")


if __name__ == "__main__":
    fake_audio_stream_test()
