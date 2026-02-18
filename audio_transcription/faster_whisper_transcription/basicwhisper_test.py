

from faster_whisper import WhisperModel

# Initialize model and transcribe
model = WhisperModel("small", device="cpu", compute_type="int8") 
segments, info = model.transcribe("audio_2026-02-17T21-29-06-927Z.wav") 

# Print transcription
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")