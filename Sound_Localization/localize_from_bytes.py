"""
Sound localization from interleaved PCM bytes (e.g. from a stream or buffer)
instead of loading from a WAV file.
"""
import numpy as np

import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sound_localization import find_delay
from TDOA import tdoa_using_grid_search


def main(
    pcm_bytes: bytes,
    sample_rate: int = 48000,
    num_channels: int = 3,
    bytes_per_sample: int = 2,
) -> Optional[Tuple[float, float, float]]:
    """
    Run TDOA localization on interleaved multi-channel PCM bytes.
    Returns (x, y, z) in meters when 3 channels are used; otherwise None.

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
    # Bytes -> int16 array
    dtype = np.int16
    n_bytes = len(pcm_bytes)
    n_samples_total = n_bytes // bytes_per_sample
    if n_samples_total * bytes_per_sample != n_bytes:
        raise ValueError(f"pcm_bytes length {n_bytes} is not a multiple of {bytes_per_sample}")
    samples = np.frombuffer(pcm_bytes, dtype=dtype)
    # Reshape to (n_frames, num_channels)
    n_frames = n_samples_total // num_channels
    if n_frames * num_channels != n_samples_total:
        raise ValueError(
            f"Total samples {n_samples_total} is not divisible by num_channels={num_channels}"
        )
    data = samples.reshape(n_frames, num_channels)

    # Normalize to float [-1, 1] (same as localize_from_audio_file.main)
    data = data.astype(np.float64) / np.iinfo(dtype).max
    channels = [data[:, i] for i in range(data.shape[1])]

    window_len = int(0.1 * sample_rate)
    step = window_len // 2
    tdoa_sec = []
    for i in range(len(channels) - 1):
        delay_samp, delay_sec = find_delay(
            channels[i], channels[i + 1], window_len, step, fs=sample_rate
        )
        tdoa_sec.append(delay_sec)
        print(f"Ch {i}–{i + 1}: {delay_samp} samples ({delay_sec * 1000:.2f} ms)")
    if len(channels) == 3 and len(tdoa_sec) == 2:
        pos, similarity = tdoa_using_grid_search(np.array(tdoa_sec))
        position = (float(pos[0]), float(pos[1]), float(pos[2]))
        print(f"Source position (m): {position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}")
        print(f"Similarity: {similarity:.4f}")
        return position


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
