import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from whisperlivekit import (AudioProcessor, TranscriptionEngine,
                            get_inline_ui_html, parse_args)

# Handling synchronization
from collections import deque
import time
import numpy as np


# NOTE: most of this code is a copy and paste from "basic_server.py" from the WhisperLiveKit library
# https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/basic_server.py

##################################################################################################

# - - - - - - - - - - - #
#   Sound Localization  #
# - - - - - - - - - - - #

'''
Handling the synchornization -> localization buffer
- Core idea: store localizaiton results with a timestamp, keep a small buffer of recent {timestamp: vector} mappings
- in handle_websocket_results, when storing the information, look up in the buffer the vector corresponding to the timestampe
'''
# Circular buffer for sound localization results
localization_buffer = deque(maxlen=500)  # Keep last 500 results
localization_lock = asyncio.Lock()

# Track cumulative audio time
audio_time_tracker = {
    "total_samples": 0,
    "sample_rate": 16000,  # Match your audio config
    "bytes_per_sample": 2  # PCM16 = 2 bytes per sample
}

# When stream starts, record the start
stream_start_wall_time = None
stream_start_audio_time = 0.0  # Always starts at 0

def bytes_to_seconds(num_bytes, sample_rate=16000, bytes_per_sample=2):
    """Convert byte count to audio duration in seconds"""
    samples = num_bytes // bytes_per_sample
    return samples / sample_rate

def whisper_time_to_seconds(time_str):
    """Convert Whisper time format '0:00:05' to seconds (5.0)"""
    if not time_str:
        return 0.0
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(parts[0])

def find_vector_for_time(start_time_str, end_time_str, tolerance=0.1):
    """Find localization vector matching a Whisper time range"""
    start_seconds = whisper_time_to_seconds(start_time_str)
    end_seconds = whisper_time_to_seconds(end_time_str)
    
    matching_vectors = []
    
    # Take a snapshot of the buffer to avoid async issues
    buffer_snapshot = list(localization_buffer)  # ← snapshot instead of iterating live
    
    for loc_data in buffer_snapshot:
        if (loc_data["start_audio_time"] <= end_seconds + tolerance and
            loc_data["end_audio_time"] >= start_seconds - tolerance):
            matching_vectors.append(loc_data["vector"])
    
    if matching_vectors:
        return np.mean(matching_vectors, axis=0).tolist()

async def process_sound_localization(audio_bytes, start_audio_time, end_audio_time):
    """Process multi-channel audio for sound localization"""
    # Import your localization function from the other file
    # from your_localization_file import efes_beautiful_localization_function
    
    vector = efes_beautiful_localization_function(audio_bytes)
    
    async with localization_lock:
        localization_buffer.append({
            "start_audio_time": start_audio_time,
            "end_audio_time": end_audio_time,
            "vector": vector
        })

def mix_to_mono(audio_bytes, num_channels=3, bytes_per_sample=2):
    """
    Convert interleaved multi-channel PCM16 audio to mono.
    
    Interleaved format: [ch1, ch2, ch3, ch1, ch2, ch3, ...]
    Output: [avg, avg, avg, ...]
    """
    # Convert bytes to numpy array of int16 samples
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # Reshape into (num_frames, num_channels)
    # Each row is one time frame, each column is one mic
    frames = samples.reshape(-1, num_channels)
    
    # Average across channels to get mono
    # Use float32 to avoid overflow when averaging
    mono_frames = frames.mean(axis=1).astype(np.int16)
    
    # Convert back to bytes
    return mono_frames.tobytes()

##################################################################################################

# - - - - - - - - - - - #
#   Logging and args    #
# - - - - - - - - - - - #

'''
NOTE: send format will be the following:

{
  "type": "transcription",
  "status": "active_transcription",
  "speaker_segments": [
    {
      "speaker_id": 1,
      "text": "Hello how are you?",
      "start": "0:00:00",
      "end": "0:00:03",
      "sound_vector": [0.5, 0.3, 0.8]
    },
    {
      "speaker_id": 2,
      "text": "I'm doing great!",
      "start": "0:00:03",
      "end": "0:00:06",
      "sound_vector": [-0.3, 0.7, 0.2]
    }
  ]
}


'''


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None

# Store connected Quest clients
quest_clients: Set[WebSocket] = set()

# Pass arguments to the transcription engine on setup -> realistically i want to set some of these (model = base, lang = en)
@asynccontextmanager
async def lifespan(app: FastAPI):    
    global transcription_engine

    # Override specific args before passing to TranscriptionEngine
    args.model = "base"
    args.lan = "en"
    args.diarization = True

    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##################################################################################################

# - - - - - - - - - - - #
#   audio processing    #
# - - - - - - - - - - - #

@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())

async def broadcast_to_quest_clients(message: dict):
    """Send transcription to all connected Quest clients"""
    disconnected_clients = set()
    
    for client in quest_clients:
        try:
            await client.send_json(message)
            logger.debug(f"Sent to Quest client: {message}")
        except Exception as e:
            logger.error(f"Failed to send to Quest client: {e}")
            disconnected_clients.add(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        quest_clients.discard(client)
        logger.info(f"Removed disconnected Quest client. Remaining: {len(quest_clients)}")

async def handle_websocket_results(results_generator):
    """
    Handles results from the audio processor and broadcasts to Quest clients
    
    """
    try:
        async for response in results_generator:
            response_dict = response.to_dict() # organizes outputas dictionary
            '''
            From the following (whisperlivekit / audio_processor.py)
            The format of the response outputs are the following:
            
            response = FrontData(
                    status=response_status,
                    lines=lines,
                    buffer_transcription=buffer_transcription_text,
                    buffer_diarization=buffer_diarization_text,
                    buffer_translation=buffer_translation_text,
                    remaining_time_transcription=state.remaining_time_transcription,
                    remaining_time_diarization=state.remaining_time_diarization if self.args.diarization else 0
                )
            '''

            # Extract data from response:
            lines = response_dict.get('lines', [])
            status = response_dict.get('status', '')

            # Build per speaker segments w/ localizaiton vectors
            quest_message_segments = []
            if lines:
                for line in lines:
                    text = line.get('text', '').strip()
                    speaker = line.get('speaker',0)
                    if text and speaker != -2:  # only add if there's valid text, -2 speaker is no speaker
                        start_time = line.get('start', '')
                        end_time = line.get('end', '')
                                            
                        quest_message_segments.append({
                            "speaker_id": speaker,
                            "text": text,
                            "sound_vector": find_vector_for_time(start_time, end_time)
                        })

            # Debug output
            if quest_message_segments:
                for seg in quest_message_segments:
                    print(f"\n [speaker {seg['speaker_id']}] {seg['text']}")

            quest_message = {
                "type": "transcription",
                "status": status,
                "speaker_segments": quest_message_segments
            }
            
            if quest_message_segments:
                logger.info(f" --> Broadcasting to {len(quest_clients)} Quest client(s): {len(quest_message_segments)} segment(s) (status={status})")


            # Broadcast to all Quest clients
            await broadcast_to_quest_clients(quest_message)
            
        # Signal that all audio has been processed
        print("\n[DONE] Audio processing complete.")
        logger.info("Results generator finished")
        #await broadcast_to_quest_clients({"type": "ready_to_stop"})
        
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")

        
@app.websocket("/asr")
async def audio_input_endpoint(websocket: WebSocket):
    '''
    Endpoint for receiving audio byte streams from audio handling server
    The server connected to the ras-pi sends audio here for transcription
    '''
    global transcription_engine, stream_start_wall_time  # ← ADD global declaration

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine
    )
    
    await websocket.accept()
    logger.info("Audio input WebSocket connection opened (for audio stream source).")
    
    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to audio source: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()

            # DEBUG: Inspect the incoming bitstream
            print(f"[BITSTREAM DEBUG] Chunk size: {len(message)} bytes")
            print(f"[BITSTREAM DEBUG] First 32 bytes (hex): {message[:32].hex()}")
            print(f"[BITSTREAM DEBUG] First 32 bytes (raw): {list(message[:32])}")
            
            # Check if it looks like raw PCM (values should be small integers)
            # or compressed (will look like random high values)
            import struct
            first_samples = struct.unpack(f'{min(16, len(message)//2)}h', message[:min(32, len(message))])
            print(f"[BITSTREAM DEBUG] As int16 samples: {first_samples}")


            # First chunk - record wall clock start time
            if stream_start_wall_time is None:
                stream_start_wall_time = time.time()
            
            # Calculate audio time based on bytes received (same reference as Whisper)
            start_audio_time = audio_time_tracker["total_samples"] / audio_time_tracker["sample_rate"]
            
            # IMPORTANT: message contains 3 channels of audio interleaved
            # bytes_per_sample * num_channels = total bytes per frame
            num_channels = 3
            chunk_duration = bytes_to_seconds(
                len(message), 
                sample_rate=audio_time_tracker["sample_rate"],
                bytes_per_sample=audio_time_tracker["bytes_per_sample"] * num_channels  # ← account for 3 channels
            )
            end_audio_time = start_audio_time + chunk_duration
            
            # Update tracker (divide by num_channels since samples are interleaved)
            audio_time_tracker["total_samples"] += (len(message) // audio_time_tracker["bytes_per_sample"]) // num_channels
            
            # Send FULL multi-channel audio to localization
            asyncio.create_task(
                process_sound_localization(message, start_audio_time, end_audio_time)
            )
            
            # Convert to mono for Whisper
            mono_audio = mix_to_mono(message, num_channels=num_channels)
            await audio_processor.process_audio(mono_audio)

    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Audio source closed on the connection.")
        else:
            logger.error(f"Unexpected KeyError in audio_input_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("Audio source disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error in audio_input_endpoint: {e}", exc_info=True)
    finally:
        # Reset audio time tracking for next connection
        audio_time_tracker["total_samples"] = 0
        stream_start_wall_time = None  # needs global declaration too

        logger.info("Cleaning up audio input endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")
            
        await audio_processor.cleanup()
        logger.info("Audio input endpoint cleaned up successfully.")




@app.websocket("/quest")
async def quest_output_endpoint(websocket: WebSocket):
    """
    Endpoint for Meta Quest to receive transcriptions.
    Quest clients connect here to receive real-time transcription results.
    """

    await websocket.accept()
    quest_clients.add(websocket)
    logger.info(f"✓ Quest client connected. Total Quest clients: {len(quest_clients)}")

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to transcription server",
            "timestamp": asyncio.get_running_loop().time()
        })

        # Keep connection alive and wait for messages/disconnection
        while True:
            try:
                message = await websocket.receive_text()
                logger.debug(f"Received from quest: {message}")

                # NOTE: the specific commands Quest can send can be handled here

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error receiving from Quest: {e}")
                break
    except Exception as e:
        logger.error(f"Error in quest-output_endoit: {e}", exc_info = True)
    finally:
        quest_clients.discard(websocket)
        logger.info(f"x Quest client Disconnected. Remaining Quest clients: {len(quest_clients)}")

##################################################################################################
# - - - - - - - - - - - #
#   Main                #
# - - - - - - - - - - - #

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    print("="*60)
    print("TRANSCRIPTION SERVER")
    print("="*60)
    print(f"Model: base")          # hardcoded
    print(f"Language: en")         # hardcoded  
    print(f"Diarization: True")    # hardcoded
    print(f"Host: {args.host}:{args.port}")
    print("")
    print("ENDPOINTS:")
    print(f"  Audio Input (byte stream): ws://{args.host}:{args.port}/asr")
    print(f"  Meta Quest Output:         ws://{args.host}:{args.port}/quest")
    print(f"  Web UI:                    http://{args.host}:{args.port}/")
    print("="*60)
    
    uvicorn_kwargs = {
        "app": "quest_transcription:app",
        "host": args.host, 
        "port": args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = {**uvicorn_kwargs, "forwarded_allow_ips": args.forwarded_allow_ips}

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()



