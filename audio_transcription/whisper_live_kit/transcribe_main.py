
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Sound_Localization.localize_from_audio_file import main as localization_main

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args
from faster_whisper import WhisperModel


# --------------------------------
#  Helpers (same as your original)
# --------------------------------

def bytes_to_seconds(num_bytes: int, sample_rate: int = 48000, bytes_per_sample: int = 2) -> float:
    samples = num_bytes // bytes_per_sample
    return samples / sample_rate


def whisper_time_to_seconds(time_str: str) -> float:
    """Convert Whisper time format '0:00:05' to seconds (5.0)."""
    if not time_str:
        return 0.0
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(parts[0])

def pick_channel(audio_bytes: bytes, ch: int, num_channels: int = 3) -> bytes:
    s = np.frombuffer(audio_bytes, dtype=np.int16)
    frames = s.reshape(-1, num_channels)
    return frames[:, ch].astype(np.int16).tobytes()


def mix_to_mono(audio_bytes: bytes, num_channels: int = 3) -> bytes:
    """
    Convert interleaved multi-channel PCM16 audio to mono.
    Interleaved: [ch1, ch2, ch3, ch1, ch2, ch3, ...]
    """
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if samples.size == 0:
        return b""
    frames = samples.reshape(-1, num_channels)
    mono_frames = frames.mean(axis=1).astype(np.int16)
    return mono_frames.tobytes()


# -----------------------------
#  Logging
# -----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def initilaize_model() -> WhisperModel:
    '''Initilize small Whisper model'''
    model = WhisperModel("small", device="cpu", compute_type="int8") 
    return model 


def main():
    import sys

    '''
    1. Init the whisper model
    2. Check a certain directory to see if there are any .wav files inside of it
    3. pick the oldest one, and generate a .raw file from it
    4. Reshape the raw and feed it to localization_main
    5. Using the original .wav feed that into whisper_model.transcribe
    6. Format the transcription and localizaiton vector together in a json to send 
    7. Repeat step 2
    
    '''

    ## 1. Initialize the whisper model
    whisper_model = initilaize_model()

    ### 2 . Reading in the .wav # NOTE: this should be converted to a function
    #wav_path = "test_audio_stereo.wav"
    #wav_path = "audio_2026-02-17T21-29-06-927Z.wav"

    fs = 48000
    data = np.fromfile("audio.raw", dtype=np.float32)
    data = data.reshape(-1, 3)

    print(data.shape)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        wav_path = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    ### 3. Parse the arguments (should be none)
    args = parse_args()
    print(f"3. arguments {args} \n")

    ### 4. Run the localization on the .wav
    localization_vector = localization_main(data, fs)

    ### 5. Run the captioning on the .wav
    segments, info = whisper_model.transcribe("audio_3ch_declared.wav") 
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    ### 6. Format the outputs for the Quest
    
    print(localization_vector)
    '''
    1. all the wav files get put into a directory 
    2. gather all the wav files in that directory and apply this to them
    3. after reading them, delete them
    '''


if __name__ == "__main__":
    main()
