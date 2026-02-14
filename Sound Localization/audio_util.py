import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def visualize_waveforms(*waveforms: np.ndarray, fs: float = 44200) -> None:
    n = max(len(w) for w in waveforms)
    x = np.arange(n) / fs if fs else np.arange(n)
    for w in waveforms:
        plt.plot(x[: len(w)], w)
    plt.show()
