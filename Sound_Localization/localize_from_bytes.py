"""
Sound localization from interleaved PCM bytes (e.g. from a stream or buffer)
instead of loading from a WAV file.
"""
import numpy as np

import sys
from pathlib import Path
from typing import Optional, Tuple, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sound_localization import find_delay
from TDOA import tdoa_using_grid_search, localize_sources_top3


def main(
    pcm_bytes: bytes,
    sample_rate: int = 48000,
    num_channels: int = 3,
    bytes_per_sample: int = 2,
) -> Optional[List[Tuple[Tuple[float, float, float], float]]]:
    """
    Run TDOA localization on interleaved multi-channel PCM bytes.
    Returns list of (position, strength) for top 3 sources when 3 channels; else None.

    Parameters
    ----------
    pcm_bytes : bytes
        Raw PCM, interleaved by channel (ch0, ch1, ch2, ch0, ch1, ch2, ...).
    sample_rate : int
        Sample rate in Hz.
    num_channels : int
        Number of channels (default 3 for 3-mic TDOA).
    bytes_per_sample : int
        Bytes per sample (default 2 for int16).
    """
    dtype = np.int16
    n_bytes = len(pcm_bytes)
    n_samples_total = n_bytes // bytes_per_sample
    if n_samples_total * bytes_per_sample != n_bytes:
        raise ValueError(f"pcm_bytes length {n_bytes} is not a multiple of {bytes_per_sample}")
    samples = np.frombuffer(pcm_bytes, dtype=dtype)
    n_frames = n_samples_total // num_channels
    if n_frames * num_channels != n_samples_total:
        raise ValueError(
            f"Total samples {n_samples_total} is not divisible by num_channels={num_channels}"
        )
    data = samples.reshape(n_frames, num_channels)
    data = data.astype(np.float64) / np.iinfo(dtype).max
    channels = [data[:, i] for i in range(data.shape[1])]

    window_len = int(0.1 * sample_rate)
    step = window_len // 2
    pair_delays_sec = []
    pair_powers = []
    for i in range(len(channels) - 1):
        delays_samp, delays_sec, powers = find_delay(
            channels[i], channels[i + 1], window_len, step, fs=sample_rate
        )
        pair_delays_sec.append(delays_sec)
        pair_powers.append(powers)
        print(f"Ch {i}–{i + 1}: delays {delays_samp}  powers: {powers}")
    if len(channels) == 3 and len(pair_delays_sec) == 2:
        sources = localize_sources_top3(pair_delays_sec, pair_powers, loc_fn=tdoa_using_grid_search)
        result = [((float(p[0]), float(p[1]), float(p[2])), s) for p, s in sources]
        for i, (pos, strength) in enumerate(result):
            print(f"Source {i+1} (m): ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  strength: {strength:.4f}")
        return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from scipy.io import wavfile

    # Default test file; override with first CLI arg
    wav_path = "audio_2026-02-17T21-29-06-927Z.wav"
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
    path = Path(wav_path)
    if not path.is_file():
        print(f"Test file not found: {path.resolve()}")
        sys.exit(1)

    # Load WAV and convert to PCM bytes (interleaved int16)
    fs, data = wavfile.read(str(path))
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if not np.issubdtype(data.dtype, np.integer):
        data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        data = data.astype(np.int16)
    pcm_bytes = data.tobytes()
    num_channels = data.shape[1]

    print(f"Test: {path.name} -> {len(pcm_bytes)} bytes, {fs} Hz, {num_channels} ch")
    main(pcm_bytes, sample_rate=fs, num_channels=num_channels)
