# Transcription
Real-time audio transcription and sound localization using Faster Whisper. transcribe_main.py just prints and outputs while transcribe_send_main.py sends the output json with asyncio WebSockets.

## Setup

### 1. Create Python Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
brew install ffmpeg        # sudo apt install ffmpeg

pip install faster_whisper websockets numpy
```

## Running the files
Run files from AudioVision directory. Both transcribe_send_main.py and transcribe_main.py reads in .wav files in the directory **wav_queue**, processess them, and deletes them. 

Running transcribe_main.py prints transcription and localization vector to console
```bash
python3 audio_transcription/faster_whisper_transcription/transcribe_main.py
```

Running transcribe_send_main.py sends outputs in JSON format over network w/ websocket port: 8765
```bash
python3 audio_transcription/faster_whisper_transcription/transcribe_main.py
```

Each JSON output is of the format:
```python
output = {
        "localization": localization_vector,  # [x, y, z] vector
        "transcription": full_text            # concatenated text string
    }
```

## Project Structure
```
.
├── transcribe_main.py           # Sound source localization and transcription -> prints to console
├── transcribe_send_main.py      # Same as transcribe_main but with websocket output functionality
├── basicwhisper_text.py         # 2 line faster whisper test
└── README.md
```


## Fish

blub blub fish