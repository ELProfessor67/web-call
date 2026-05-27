import logging
import os
import json
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    ChatContext,
    RoomInputOptions,
)
from livekit.plugins import openai, deepgram, elevenlabs, silero
from stt import FasterWhisperSTT
from tts import PiperTTS
from veena_tts import VeenaTTS
from parler_tts_plugin import ParlerTTS
from parler_tts_plugin import _ParlerEngine

load_dotenv()
logger = logging.getLogger("outbound-agent")
logger.setLevel(logging.INFO)

# Local pipeline settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")
PIPER_USE_CUDA = os.getenv("PIPER_USE_CUDA", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Pre-load the Parler TTS model so it doesn't block the async entrypoint
    logger.info("Preloading Parler TTS engine...")
    _ParlerEngine.get()
    logger.info("Parler TTS engine preloaded.")


async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    logger.info(f"Started voice assistant for participant {participant.identity}")

    # Default metadata
    call_context = {
        "prompt": "You are a helpful assistant.",
        "provider": "ollama",
        "model": "llama3.2:1b",
        "stt_model": "medium",
        "language": "hi",
        "voice": "Divya",
    }

    try:
        raw_meta = participant.metadata
        print(raw_meta,"raw_meta")
        if raw_meta:
            parsed_meta = json.loads(raw_meta)
            print(parsed_meta,"parsed_meta")
            call_context.update(parsed_meta)
            logger.info(f"🔧 [LIVEKIT_AGENT] Call context loaded: {call_context}")
    except Exception as e:
        logger.error(f"Error parsing participant metadata: {e}")

    # ── LLM setup ────────────────────────────────────────────────────────────
    provider = call_context.get("provider", "ollama")
    model_name = call_context.get("model", "llama3.2:1b")

    if provider == "ollama":
        llm_plugin = openai.LLM.with_ollama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
        )
    else:
        llm_plugin = openai.LLM.with_ollama(
            model="llama3.2:1b",
            base_url=OLLAMA_BASE_URL,
        )

    # ── STT setup ─────────────────────────────────────────────────────────────
    
    stt_plugin = FasterWhisperSTT(
            model_size=call_context.get("stt_model"),
            device=WHISPER_DEVICE,
            compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
        )

    # ── TTS setup ─────────────────────────────────────────────────────────────
    # voice_path = call_context.get("voice", "/voices/pratham/medium/hi_IN-pratham-medium.onnx")
    # tts_plugin = PiperTTS(
    #     model_path=voice_path,
    #     use_cuda=PIPER_USE_CUDA,
    # )

    voice = call_context.get("voice", "Divya")
    tts_plugin = ParlerTTS(voice=voice)

    # ── Agent definition ──────────────────────────────────────────────────────
    class OutboundAgent(Agent):
        def __init__(self) -> None:
            super().__init__(
                instructions=call_context.get("prompt", "You are a helpful assistant."),
            )

        async def on_enter(self) -> None:
            # Greet the user as soon as the agent enters the session
            await self.session.generate_reply(
                instructions="Greet the user. Say: Hello. I am connected and ready to help!"
            )

    # ── AgentSession ──────────────────────────────────────────────────────────
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
    )

    await session.start(
        room=ctx.room,
        agent=OutboundAgent(),
        # room_input_options=RoomInputOptions(
        #     participant=participant,  # link session to specific participant
        # ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,  # Parler model load needs ~6-8s
        )
    )