import numpy as np
from scipy.io import wavfile

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sound_localization import find_delay
from audio_util import visualize_waveforms
from TDOA import tdoa_using_ls, tdoa_using_ls_2D, tdoa_using_grid_search
import pdb


def test_from_mono_audio(
    wav_path: str, delay_a: int = 3, delay_b: int = 0, delay_c: int = 5, noise_scale: float = 0.05
) -> None:
    fs, data = wavfile.read(wav_path)
    source = np.asarray(data, dtype=np.float64)
    if source.ndim > 1:
        source = source[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        source = source / np.iinfo(data.dtype).max
    n_total = len(source)
    rng = np.random.default_rng(42)

    max_delay = max(delay_a, delay_b, delay_c)
    n = n_total + max_delay
    stream_a = np.zeros(n)
    stream_a[delay_a : delay_a + n_total] = source
    stream_a += noise_scale * rng.standard_normal(n)
    stream_b = np.zeros(n)
    stream_b[delay_b : delay_b + n_total] = source
    stream_b += noise_scale * 1.5 * rng.standard_normal(n)
    stream_c = np.zeros(n)
    stream_c[delay_c : delay_c + n_total] = source
    stream_c += noise_scale * 1.2 * rng.standard_normal(n)
    stream_a, stream_b, stream_c = stream_a[:n_total], stream_b[:n_total], stream_c[:n_total]

    #visualize_waveforms(stream_a, stream_b, stream_c, fs=fs)

    window_len = int(0.1 * fs)
    step = window_len // 2
    tdoa_sec = []
    for s1, s2, d in [(stream_a, stream_b, delay_b), (stream_a, stream_c, delay_c)]:
        delay_samp, delay_sec = find_delay(s1, s2, window_len, step, fs=fs)
        tdoa_sec.append(delay_sec)
        print(f"True delay: {d} samples ({d / fs * 1000:.2f} ms)  Est: {delay_samp} samples ({delay_sec * 1000:.2f} ms)")
    #pos = tdoa_using_ls_2D(np.array(tdoa_sec))
    # pos = tdoa_using_ls(np.array(tdoa_sec))
    pos, similarity = tdoa_using_grid_search(np.array(tdoa_sec))

    print(f"Source position (m): {float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f}")
    print(f"Similarity: {similarity:.4f}")


def main(data, fs) -> None:
    print(f"Running localize from audio \n")

    import soundfile as sf
    #info = sf.info(wav_path)
    #print(info)
    #data, fs = sf.read(wav_path, always_2d=True)

    print(f"shape after reading {data.shape}") 

    # fs, data = wavfile.read(wav_path)
    print(f"Shape of data{data.shape}")
    if data.ndim == 1:
        data = data[:, None]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    channels = [data[:, i] for i in range(data.shape[1])]

    print(f"Number of channels {data.shape[1]} \n")
    window_len = int(0.1 * fs)
    step = window_len // 2
    tdoa_sec = []
    for i in range(len(channels) - 1):
        delay_samp, delay_sec = find_delay(
            channels[i], channels[i + 1], window_len, step, fs=fs
        )
        tdoa_sec.append(delay_sec)
        print(f"Ch {i}–{i + 1}: {delay_samp} samples ({delay_sec * 1000:.2f} ms)")
    if len(channels) == 3 and len(tdoa_sec) == 2:
        #pos = tdoa_using_ls(np.array(tdoa_sec))
        pos, similarity = tdoa_using_grid_search(np.array(tdoa_sec))
        print(f"Source position (m): {float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f}")
        print(f"Similarity: {similarity:.4f}")

    return pos

if __name__ == "__main__":
    import sys
    main("audio_2026-02-17T21-29-06-927Z.wav")
    #test_from_mono_audio("test_audio.wav")