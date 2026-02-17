"""
Function-based transcription + sound localization (no WebSockets).

- Same segment output schema as your reference.
- Same core helpers: bytes_to_seconds, whisper_time_to_seconds, mix_to_mono, etc.
- Same single entry point: run(chunks, ...)

Additionally:
- Removes global mutable state by encapsulating into a session object.
- Still uses whisperlivekit AudioProcessor/TranscriptionEngine the same way.
"""

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

from Sound_Localization.localize_from_bytes import main as localization_main

from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args



# Optional WAV loading
try:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# -----------------------------
# Localization stub
# -----------------------------

def _stub_localization(audio_bytes: bytes) -> List[float]:
    """Replace with your real localization function."""
    return [0.0, 0.0, 0.0]


# -----------------------------
# Helpers (same as your original)
# -----------------------------

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
# Logging
# -----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# -----------------------------
# Session object (no globals)
# -----------------------------

@dataclass
class AudioTimeTracker:
    total_samples: int = 0
    sample_rate: int = 48000
    bytes_per_sample: int = 2  # PCM16


@dataclass
class LocalizationEntry:
    start_audio_time: float
    end_audio_time: float
    vector: List[float]


@dataclass
class TranscriptionLocalizationSession:
    """
    Encapsulates:
    - localization buffer + lock
    - audio time tracker
    - whisperlivekit engine + processor
    - result collection
    """
    localization_fn: Callable[[bytes], List[float]] = localization_main
    num_channels: int = 3
    engine_kwargs: Dict = field(default_factory=dict)

    buffer_maxlen: int = 500
    tolerance: float = 0.1

    _buffer: Deque[LocalizationEntry] = field(init=False)
    _lock: asyncio.Lock = field(init=False)
    _time: AudioTimeTracker = field(init=False)

    _engine: Optional[TranscriptionEngine] = field(init=False, default=None)
    _processor: Optional[AudioProcessor] = field(init=False, default=None)
    _results_task: Optional[asyncio.Task] = field(init=False, default=None)

    _collected_segments: List[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.buffer_maxlen)
        self._lock = asyncio.Lock()
        self._time = AudioTimeTracker(sample_rate=48000, bytes_per_sample=2)

    # ---- buffer lookup ----

    def find_vector_for_time(self, start_time_str: str, end_time_str: str) -> Optional[List[float]]:
        start_seconds = whisper_time_to_seconds(start_time_str)
        end_seconds = whisper_time_to_seconds(end_time_str)

        snapshot = list(self._buffer)
        matching = [
            e.vector
            for e in snapshot
            if (e.start_audio_time <= end_seconds + self.tolerance
                and e.end_audio_time >= start_seconds - self.tolerance)
        ]
        if matching:
            return np.mean(matching, axis=0).tolist()
        return None

    # ---- localization processing ----

    async def _process_sound_localization(self, audio_bytes: bytes, start_audio_time: float, end_audio_time: float) -> None:
        vector = self.localization_fn(audio_bytes)
        async with self._lock:
            self._buffer.append(LocalizationEntry(start_audio_time, end_audio_time, vector))

    # ---- whisper result consumption ----

    async def _handle_results(self, results_generator) -> None:
        """
        Consumes results from the audio processor and collects speaker segments with sound_vector.
        """
        try:
            async for response in results_generator:
                response_dict = response.to_dict()
                lines = response_dict.get("lines", [])

                if not lines:
                    continue

                for line in lines:
                    text = (line.get("text", "") or "").strip()
                    speaker = line.get("speaker", 0)
                    if not text or speaker == -2:
                        continue

                    start_time = line.get("start", "") or ""
                    end_time = line.get("end", "") or ""

                    seg = {
                        "speaker_id": speaker,
                        "text": text,
                        "start": start_time,
                        "end": end_time,
                        "sound_vector": self.find_vector_for_time(start_time, end_time),
                    }
                    self._collected_segments.append(seg)
                    logger.debug(f"[speaker {seg['speaker_id']}] {seg['text']}")

            logger.info("Results generator finished")
        except Exception as e:
            logger.exception(f"Error in results handler: {e}")

    # ---- lifecycle ----

    async def start(self) -> None:
        """
        Initialize TranscriptionEngine/AudioProcessor and start the results consumer task.
        """
        self._time.total_samples = 0
        self._buffer.clear()
        self._collected_segments.clear()

        self._engine = TranscriptionEngine(**self.engine_kwargs)
        self._processor = AudioProcessor(transcription_engine=self._engine)

        results_generator = await self._processor.create_tasks()
        self._results_task = asyncio.create_task(self._handle_results(results_generator))

    async def feed_chunk(self, chunk: bytes) -> None:
        """
        Feed one multi-channel PCM16 interleaved chunk:
        - update time tracker
        - run localization on full multi-channel
        - mix to mono and feed to whisper processor
        """
        if self._processor is None:
            raise RuntimeError("Session not started. Call await session.start() first.")

        start_audio_time = self._time.total_samples / self._time.sample_rate

        chunk_duration = bytes_to_seconds(
            len(chunk),
            sample_rate=self._time.sample_rate,
            bytes_per_sample=self._time.bytes_per_sample * self.num_channels,
        )
        end_audio_time = start_audio_time + chunk_duration

        # Update tracker (divide by num_channels since samples are interleaved)
        self._time.total_samples += (len(chunk) // self._time.bytes_per_sample) // self.num_channels

        await self._process_sound_localization(chunk, start_audio_time, end_audio_time)

        #mono_audio = mix_to_mono(chunk, num_channels=self.num_channels)
        mono_audio = pick_channel(chunk, 0, num_channels=self.num_channels)
        logger.debug(
            "Feeding chunk: mono_bytes=%d, start_audio_time=%.2fs, end_audio_time=%.2fs",
            len(mono_audio), start_audio_time, end_audio_time,
        )
        await self._processor.process_audio(mono_audio)

    async def finish(self) -> List[dict]:
        """
        Cleanup processor and wait for results task. Returns collected segments.
        """
        if self._processor is None:
            return self._collected_segments

        await self._processor.cleanup()

        if self._results_task is not None:
            await self._results_task

        return list(self._collected_segments)


# -----------------------------
# Public API (same idea as your run())
# -----------------------------

def run(
    chunks: List[bytes],
    localization_fn: Optional[Callable[[bytes], List[float]]] = None,
    num_channels: int = 3,
    **engine_kwargs,
) -> List[dict]:
    """
    Single entry point: run transcription + localization on a list of multi-channel PCM chunks.
    Returns list of segments (each with speaker_id, text, start, end, sound_vector).
    """
    if localization_fn is None:
        localization_fn = localization_main

    async def _run_async() -> List[dict]:
        session = TranscriptionLocalizationSession(
            localization_fn=localization_fn,
            num_channels=num_channels,
            engine_kwargs=engine_kwargs,
        )
        await session.start()
        for ch in chunks:
            await session.feed_chunk(ch)
        return await session.finish()

    return asyncio.run(_run_async())


# -----------------------------
# WAV utilities (unchanged spirit)
# -----------------------------

def load_wav_as_chunks(
    wav_path: str,
    target_sr: int = 48000,
    num_channels: int = 3,
    chunk_duration_sec: float = 0.5,
) -> List[bytes]:
    """
    Load a WAV file and return a list of num_channels PCM16 chunks (interleaved).
    Resamples to target_sr if needed; converts mono/stereo to num_channels.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for WAV loading (pip install scipy)")

    fs, data = wavfile.read(wav_path)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n_frames, n_ch = data.shape

    # Normalize to float [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    data = np.clip(data, -1.0, 1.0)

    # Resample per channel if needed
    if fs != target_sr:
        n_new = int(n_frames * target_sr / fs)
        resampled = np.column_stack([resample_poly(data[:, c], target_sr, fs) for c in range(n_ch)])
        data = resampled
        if data.shape[0] < n_new:
            data = np.pad(data, ((0, n_new - data.shape[0]), (0, 0)), mode="constant")
        else:
            data = data[:n_new]

    # Expand/trim channels
    if data.shape[1] < num_channels:
        extra = np.tile(data[:, 0:1], (1, num_channels - data.shape[1]))
        data = np.hstack([data, extra])
    else:
        data = data[:, :num_channels]

    # Float -> PCM16
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)

    # Chunk into interleaved bytes
    chunk_frames = int(chunk_duration_sec * target_sr)
    chunks: List[bytes] = []
    for start in range(0, pcm.shape[0], chunk_frames):
        block = pcm[start : start + chunk_frames]
        if block.size == 0:
            break
        chunks.append(block.tobytes())
    return chunks


def main():
    """CLI test (optional)."""
    import sys

    num_channels = 3
    wav_path = "audio_2026-02-17T21-29-06-927Z.wav"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        wav_path = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    args = parse_args()

    path = Path(wav_path)
    if path.is_file() and _HAS_SCIPY:
        print(f"Loading WAV: {path.resolve()}")
        chunks = load_wav_as_chunks(str(path), num_channels=num_channels)
        print(f"Loaded {len(chunks)} chunks (~{len(chunks) * 0.1:.1f} s)")
    else:
        if not path.is_file():
            print(f"File not found: {wav_path}, using synthetic silence.")
        else:
            print("scipy not installed, using synthetic silence.")
        chunk_samples_per_channel = 1600
        chunks = [
            np.zeros((chunk_samples_per_channel, num_channels), dtype=np.int16).tobytes()
            for _ in range(30)
        ]

    print("Running transcription (run = single entry point)...")
    segments = run(chunks, num_channels=num_channels, **vars(args))
    print(f"Got {len(segments)} segment(s).")
    for seg in segments:
        print(f"  [speaker {seg['speaker_id']}] {seg['text']!r}  vector={seg.get('sound_vector')}")
    print("Test finished.")


if __name__ == "__main__":
    main()
