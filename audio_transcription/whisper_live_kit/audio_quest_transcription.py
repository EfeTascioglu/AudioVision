import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from whisperlivekit import (AudioProcessor, TranscriptionEngine,
                            get_inline_ui_html, parse_args)

# NOTE: most of this code is a copy and paste from "basic_server.py" from the WhisperLiveKit library
# https://github.com/QuentinFuxa/WhisperLiveKit/blob/main/whisperlivekit/basic_server.py

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

@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())

async def broadcast_to_quest_clients(message: dict):
    """Send transcription to all connected Quest clients"""
    disconnected_clients = set()
    
    for client in quest_clients:
        try:
            await client.send_json(message)
            logger.debug(f"Sent to Quest client: {message.get('type')}")
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

            # combine text from all lines & skip empty lines (speaker 2)
            transcription_text = ''
            if lines:   # text detected
                text_fragments = []
                for line in lines:
                    text = line.get('text', '').strip()
                    speaker = line.get('speaker', 0)
                    if text and speaker != -2:  # only add if there's valid text, and not -2 speaker
                        text_fragments.append(text)
                transcription_text = ' '.join(text_fragments)

            # Command line output of full transcription # NOTE: this is only for debugging
            if transcription_text:
                    print(f"\n [partial text] {transcription_text}")
            
            # Format output to send to Quest endpoint TODO: figure out exactly what to send out -> some sense of time would be good
            quest_message = {
                "type": "transcription", # either {"active_transcription", ""no_audio_detected""} 
                "text": transcription_text,
                "status": status,
                "start_time": line.get('start', '').strip(),
                "end_time": line.get('end', '').strip()
            }

            # Log and broadcast -> TODO: there shoudl be a modification here to send regardless if there's transcription_text, so the position vecotrs are sent as well
            if transcription_text:
                logger.info(f" --> Broadcasting to {len(quest_clients)} Quest client(s): '{transcription_text[:50]}...' (status={status})")
            
            # Broadcast to all Quest clients
            await broadcast_to_quest_clients(quest_message)
            
        # Signal that all audio has been processed
        print("\n[DONE] Audio processing complete.")
        logger.info("Results generator finished. Sending 'ready_to_stop' to Quest clients.")
        await broadcast_to_quest_clients({"type": "ready_to_stop"})
        
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")

        
@app.websocket("/asr")
async def audio_input_endpoint(websocket: WebSocket):
    '''
    Endpoint for receiving audio byte streams from audio handling server
    The server connected to the ras-pi sends audio here for transcription
    '''

    global transcription_engine

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine # arguments are passed into this from lifespan
        )
    
    await websocket.accept()
    logger.info("Audio input WebSocket connection opened (for audio stream source).")
    
    try: # handshake -> sending configuration for audio inputs (client configuration, argument is pcm_input aka raw PCM)
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to audio source: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(results_generator))

    try:
        while True:
            # Receive audio bytes from the audio handling server
            message = await websocket.receive_bytes() 
            logger.debug(f"Received {len(message)} bytes from audio source")
            await audio_processor.process_audio(message)

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
            "timestamp": asyncio.get_event_loop().time()
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

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    print("="*60)
    print("TRANSCRIPTION SERVER")
    print("="*60)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Host: {args.host}:{args.port}")
    print("")
    print("ENDPOINTS:")
    print(f"  Audio Input (byte stream): wss://{args.host}:{args.port}/asr")
    print(f"  Meta Quest Output:         wss://{args.host}:{args.port}/quest")
    print(f"  Web UI:                    https://{args.host}:{args.port}/")
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



