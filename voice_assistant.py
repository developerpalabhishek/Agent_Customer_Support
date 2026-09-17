import json
import base64
import asyncio
import logging

from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from google import genai
from google.genai import types

LOG_FILE = Path(__file__).parent / "voice_assistant.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("voice_assistant")

load_dotenv()

MODEL = "gemini-3.8-live"
VOICE = "Puck"  # prebuilt Gemini Live voice
INPUT_SR = 16000  # sample-rate we'll use client-side for mic audio
OUTPUT_SR = 24000  # sample-rate Gemini Live returns for assistant audio
PORT = 5050

app = FastAPI()
client = genai.Client()

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
        )
    ),
    system_instruction=types.Content(
        parts=[types.Part(text="You are a concise AI assistant.")]
    ),
)


@app.get("/")
async def index() -> HTMLResponse:
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(html)


@app.websocket("/voice")
async def voice_bridge(ws: WebSocket) -> None:
    """
    1. Browser opens ws://host:5050/voice
    2. Browser streams base64-encoded 16-bit mono PCM chunks at 16kHz: {"audio": "<b64>"}
    3. We forward chunks to Gemini Live as realtime audio input
    4. We relay assistant audio (24kHz PCM) back to the browser the same way
    5. Gemini's server-side VAD flags user interruptions via `interrupted`
       on server_content, which we forward to the browser
    """
    await ws.accept()
    logger.info("browser connected")

    try:
        async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
            logger.info("gemini live session established (model=%s)", MODEL)
            in_chunks = 0
            out_chunks = 0

            async def from_client() -> None:
                """Relay microphone PCM chunks from browser -> Gemini."""
                nonlocal in_chunks
                async for msg in ws.iter_text():
                    data = json.loads(msg)
                    pcm = base64.b64decode(data["audio"])
                    in_chunks += 1
                    if in_chunks % 20 == 0:
                        logger.info("received %d mic chunks from browser", in_chunks)
                    await session.send_realtime_input(
                        audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={INPUT_SR}")
                    )

            async def to_client() -> None:
                """Relay assistant audio + handle interruptions."""
                nonlocal out_chunks
                async for message in session.receive():
                    content = message.server_content
                    if content is None:
                        continue

                    # user started talking -> Gemini cancels its own turn
                    if content.interrupted:
                        logger.info("assistant speech interrupted by user")
                        await ws.send_json({"interrupted": True})
                        continue

                    # assistant speaks
                    if content.model_turn:
                        for part in content.model_turn.parts:
                            inline = part.inline_data
                            if inline and inline.data:
                                out_chunks += 1
                                await ws.send_json({
                                    "audio": base64.b64encode(inline.data).decode("ascii")
                                })

                    if content.turn_complete:
                        logger.info(
                            "assistant turn complete (%d audio chunks sent)", out_chunks
                        )
                        out_chunks = 0

            await asyncio.gather(from_client(), to_client())
    except Exception:
        logger.exception("voice_bridge session failed")
        await ws.send_json({"error": "voice session failed, see server logs"})
    finally:
        logger.info("browser disconnected")
        await ws.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_assistant:app", host="0.0.0.0", port=PORT)
