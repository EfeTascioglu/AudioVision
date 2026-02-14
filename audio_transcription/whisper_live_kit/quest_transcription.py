import asyncio
import logging
from contextlib import asynccontextmanager

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


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them to Quest"""
    try:
        async for response in results_generator:
            response_dict = response.to_dict()
            
            # Extract data from the response
            lines = response_dict.get('lines', [])
            status = response_dict.get('status', '')
            
            # Combine text from all lines (skip empty lines from speaker -2)
            transcription_text = ''
            if lines:
                text_parts = []
                for line in lines:
                    text = line.get('text', '').strip()
                    speaker = line.get('speaker', 0)
                    # Skip empty lines or silence markers (speaker -2)
                    if text and speaker != -2:
                        text_parts.append(text)
                transcription_text = ' '.join(text_parts)
            
            # Determine if this is final
            is_final = status in ['committed', 'final']
            
            # OUTPUT TO COMMAND LINE
            if transcription_text:
                if is_final:
                    print(f"\n✓ [FINAL] {transcription_text}")
                else:
                    print(f"⋯ [PARTIAL] {transcription_text}")
            
            # Format message for Quest
            quest_message = {
                "type": "transcription",
                "text": transcription_text,
                "is_final": is_final,
                "status": status,
                "lines": lines,  # Include full line data with timestamps
                "timestamp": response_dict.get('timestamp', 0)
            }
            
            # Log what we're sending
            if transcription_text:
                logger.info(f" --> Quest: '{transcription_text}' (status={status})")
            
            # Send to Quest
            await websocket.send_json(quest_message)
            
        # Signal that all audio has been processed
        print("\n[DONE] Audio processing complete.")
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    transcription_engine = TranscriptionEngine(model="base", diarization=True, lan="en")
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened from Meta Quest.")

    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")
            
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")
            
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    print("="*60)
    print("STARTING TRANSCRIPTION SERVER")
    print(f"Backend: {args.backend}")
    print(f"Host: {args.host}:{args.port}")
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