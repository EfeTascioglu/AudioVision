
import asyncio
import logging
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import os
import glob
import json

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Sound_Localization.localize_from_audio_file import main as localization_main

from faster_whisper import WhisperModel


# -----------------------------
#  Logging
# -----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# -----------------------------
# Select oldest WAV in directory
# -----------------------------
def get_oldest_wav(directory: str) -> str | None:
    files = sorted(glob.glob(os.path.join(directory, "*.wav")), key=os.path.getctime)
    return files[0] if files else None


# -----------------------------
# WAV → RAW conversion
# -----------------------------
def convert_wav_to_raw(wav_path: str, raw_path: str = "audio.raw") -> tuple[np.ndarray, int]:
    """
    Converts a 3-channel WAV file to a .raw float32 file and returns
    a reshaped array along with its sample rate.
    """
    import subprocess

    # Force 3 channels, float32, 48000 Hz
    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "3",
        "-ar", "48000",
        raw_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Load raw data into NumPy
    fs = 48000
    data = np.fromfile(raw_path, dtype=np.float32)
    data = data.reshape(-1, 3)
    return data, fs

# -----------------------------
# Format output JSON
# -----------------------------
def format_output(localization_vector: np.ndarray, transcription_segments) -> str:
    """
    Formats the localization vector and transcription segments into a JSON string.
    Transcription is concatenated into a single string.
    """
    full_text = " ".join(seg.text for seg in transcription_segments)

    output = {
        "localization": localization_vector.tolist(),  # convert NumPy array to list
        "transcription": full_text
    }
    return json.dumps(output, indent=2)


# -----------------------------
# Model Initialization
# -----------------------------
def initilaize_model() -> WhisperModel:
    '''Initilize small Whisper model'''
    model = WhisperModel("small", device="cpu", compute_type="int8") 
    return model 


def main():
    '''
    Breakdown
    1. Init the whisper model
    2. Check a certain directory to see if there are any .wav files inside of it
    3. pick the oldest one, and generate a .raw file from it
    4. Reshape the raw and feed it to localization_main
    5. Using the original .wav feed that into whisper_model.transcribe
    6. Format the transcription and localizaiton vector together in a json to send 
    7. Repeat from step 2
    '''

    ## 1. Initialize the whisper model
    whisper_model = initilaize_model()

    directory = "./wav_queue"

    while True: # infinite loop of checking for WAV files
        wav_path = get_oldest_wav(directory)
        if not wav_path:
            print("No more WAV files in directory. Waiting...")
            import time
            time.sleep(1) # NOTE: will probably have to make this smaller 
            continue
    
        print(f"Processing: {wav_path}")

        ## 2. Convert to wav to raw (so it can handle 3 channels)
        data, fs = convert_wav_to_raw(wav_path)

        ## 3. Run localization
        localization_vector = localization_main(data, fs) 
        print(f"Localization vector: {localization_vector}")

        ## 4. Run Transcription
        segments, info = whisper_model.transcribe(wav_path)

        ## 5. Format as JSON (for Quest)
        output_json = format_output(localization_vector, segments)
        print(f"Output JSON: \n{output_json}")

        ## 6. Delete Processed file
        os.remove(wav_path)
        print(f"Deleted: {wav_path} \n")



if __name__ == "__main__":
    main()
