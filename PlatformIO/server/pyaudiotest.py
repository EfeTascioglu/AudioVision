import pyaudio
import numpy as np

# Parameters
p = pyaudio.PyAudio()
volume = 0.5     # Range [0.0, 1.0]
fs = 44100       # Sampling rate, Hz
duration = 2.0   # Duration in seconds
f = 440.0        # Sine frequency, Hz

# Generate sine wave samples
samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

# Open stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# Play samples
print("Playing...")
stream.write((volume * samples).tobytes())

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
print("Done.")