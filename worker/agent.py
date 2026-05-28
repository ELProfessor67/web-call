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
from livekit.plugins import openai, silero
from livekit.agents import MetricsCollectedEvent, UserStateChangedEvent, AgentStateChangedEvent
from faster_whisper import WhisperModel
from stt import FasterWhisperSTT
from tts import PiperTTS
# from veena_tts import VeenaTTS
# from parler_tts_plugin import ParlerTTS
# from parler_tts_plugin import _ParlerEngine

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
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration = 0.1,
        min_silence_duration = 0.1,
        activation_threshold = 0.5,
        force_cpu = True,
    )
    # Pre-load the Whisper STT model so first call doesn't have cold-start
    logger.info("Preloading FasterWhisper model...")
    proc.userdata["whisper_model"] = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
    )
    logger.info("FasterWhisper model preloaded.")


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
        "language": "en",
        "voice": "/home/web-call/worker/voices/pratham/medium/hi_IN-pratham-medium.onnx",
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
            model_size=call_context.get("stt_model", WHISPER_MODEL),
            device=WHISPER_DEVICE,
            compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
            language=call_context.get("language", "en"),
            beam_size=1,  # Greedy decoding for lowest latency
            preloaded_model=ctx.proc.userdata.get("whisper_model"),
        )

    # ── TTS setup ─────────────────────────────────────────────────────────────
    voice_path = call_context.get("voice", "/home/web-call/worker/voices/pratham/medium/hi_IN-pratham-medium.onnx")
    tts_plugin = PiperTTS(
        model_path=voice_path,
        use_cuda=True,
    )

    # voice = call_context.get("voice", "Divya")
    # tts_plugin = ParlerTTS(voice=voice)

    # ── Agent definition ──────────────────────────────────────────────────────
    class OutboundAgent(Agent):
        def __init__(self) -> None:
            # Add conciseness instruction to reduce LLM generation time
            base_prompt = call_context.get("prompt", "You are a helpful assistant.")
            optimized_prompt = base_prompt + "\nKeep responses concise and under 2 sentences unless asked for detail."
            super().__init__(
                instructions=optimized_prompt,
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
        min_endpointing_delay=0.3,  # Respond faster after user stops speaking
    )

    await session.start(
        room=ctx.room,
        agent=OutboundAgent(),
        # room_input_options=RoomInputOptions(
        #     participant=participant,  # link session to specific participant
        # ),
    )

    #print latency etc here
    _eou_delay = 0.0
    _llm_ttft = 0.0
    _tts_ttfb = 0.0

    @session.on("metrics_collected")
    def metrics_collected(event: MetricsCollectedEvent):
        nonlocal _eou_delay, _llm_ttft, _tts_ttfb
        if event.type != "metrics_collected":
            return

        try:
            if event.metrics.type == "eou_metrics":
                _eou_delay = event.metrics.end_of_utterance_delay

            if event.metrics.type == "llm_metrics":
                _llm_ttft = event.metrics.ttft

            if event.metrics.type == "tts_metrics":
                _tts_ttfb = event.metrics.ttfb
                logger.info(f"Latency: EOU Delay: {_eou_delay:.3f}s")
                logger.info(f"Latency: LLM TTFT: {_llm_ttft:.3f}s")
                logger.info(f"Latency: TTS TTFB: {_tts_ttfb:.3f}s")
                logger.info(f"Latency: Total: {_eou_delay + _llm_ttft + _tts_ttfb:.3f}s")

        except Exception as e:
            logger.debug(f"Metrics error: {e}")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,  # Parler model load needs ~6-8s
        )
    )