# Quest Transcription Server

Real-time audio transcription serve using WhisperLiveKit. This server receives live audio streams via WebSocket and returns real-time transcriptions. 

## Prerequisites

- Python 3.8+
- OpenSSL (for generating SSL certificates)

## Setup

### 1. Create Python Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install whisperlivekit fastapi uvicorn websockets
```

### 3. Generate SSL Certificates

SSL certificates are required for browser microphone access. Generate self-signed certificates for localhost:
```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout localhost-key.pem \
  -out localhost-cert.pem \
  -days 365 \
  -subj "/CN=localhost"
```

This creates:
- `localhost-cert.pem` - SSL certificate
- `localhost-key.pem` - Private key

**Note:** These files are gitignored and must be generated locally.

## Running the Server
```bash
python quest_transcription.py --host 0.0.0.0 --port 8000 --ssl-certfile localhost-cert.pem --ssl-keyfile localhost-key.pem
```

The server will start on `https://localhost:8000`

### Server Endpoints

- **Web UI**: `https://localhost:8000/`
- **WebSocket**: `wss://localhost:8000/asr`

## Usage

### Testing with Web Browser

1. Navigate to `https://localhost:8000/`
2. Accept the self-signed certificate warning (click "Advanced" → "Proceed to localhost")
3. Allow microphone access when prompted
4. Start speaking - transcriptions will appear in real-time

### Connecting from Meta Quest

In your Quest application, connect to:
```
wss://YOUR_COMPUTER_IP:8000/asr
```

Replace `YOUR_COMPUTER_IP` with your computer's local IP address (e.g., `192.168.1.100`).

## WebSocket API

### Sending Audio

Send raw audio bytes to the WebSocket:
```javascript
websocket.send(audioByteArray);
```

### Receiving Transcriptions

The server sends JSON messages:
```json
{
  "type": "transcription",
  "text": "transcribed text here",
  "is_final": false,
  "status": "active_transcription",
  "lines": [
    {
      "speaker": 1,
      "text": "transcribed text here",
      "start": "0:00:05",
      "end": "0:00:13",
      "detected_language": "en"
    }
  ],
  "timestamp": 1234567890
}
```

**Fields:**
- `text`: Combined transcription text
- `is_final`: `true` when transcription is finalized
- `status`: Current status (`active_transcription`, `committed`, `final`)
- `lines`: Array of transcription segments with speaker info and timestamps

### Session End

When audio processing completes:
```json
{
  "type": "ready_to_stop"
}
```

## Configuration Options

Additional command-line arguments:
```bash
# Use different Whisper model
--model base  # Options: tiny, base, small, medium, large -> currently hardcoded to base

# Set language
--lan en  # Language code (e.g., en, es, fr) -> currently hardcoded to base

# Enable diarization (speaker detection)
--diarization True # -> currently hardedcoded to true since there were errors when it was set to False

# Disable voice activity detection
--no-vad

# Change backend
--backend faster-whisper  # Options: faster-whisper, mlx-whisper (for mac optimization), whisper
```

### Example with Options
```bash
python quest_transcription.py \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-certfile localhost-cert.pem \
  --ssl-keyfile localhost-key.pem \
  --model small \
  --lan en
```

## Troubleshooting

### Certificate Errors

If you see SSL certificate warnings in your browser:
1. Click "Advanced" or "Show Details"
2. Click "Proceed to localhost (unsafe)" or "Accept Risk"
3. This is normal for self-signed certificates

### Microphone Not Working

- Ensure you're using HTTPS (`https://localhost:8000`)
- Check that SSL certificates are properly generated
- Allow microphone permissions in browser when prompted

### No Transcription Output

- Check that you're speaking clearly
- Verify the correct language is set with `--lan`
- Try a different model with `--model small` or `--model medium`

### Quest Connection Issues

- Ensure your computer and Quest are on the same network
- Use your computer's local IP address, not `localhost`
- Verify the Quest can reach your computer (ping test)

## Project Structure
```
.
├── quest_transcription.py   # Main server file
├── localhost-cert.pem        # SSL certificate (gitignored)
├── localhost-key.pem         # SSL private key (gitignored)
├── .gitignore
└── README.md
```

## Credits

Based on [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) by Quentin Fuxa.

## License

[Your License Here]