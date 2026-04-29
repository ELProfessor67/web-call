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

load_dotenv()
logger = logging.getLogger("outbound-agent")
logger.setLevel(logging.INFO)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    logger.info(f"Started voice assistant for participant {participant.identity}")

    # Default metadata
    call_context = {
        "prompt": "You are a helpful assistant.",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "stt_model": "nova-2-general",
        "language": "hi",
        "voice": "cgSgspJ2msm6clMCkdW9",  # Jessica
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
    provider = call_context.get("provider", "groq")
    model_name = call_context.get("model", "llama-3.3-70b-versatile")

    if provider == "groq":
        llm_plugin = openai.LLM(
            model=model_name,
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    elif provider == "openrouter":
        llm_plugin = openai.LLM(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    else:
        llm_plugin = openai.LLM(model=model_name)

    # ── STT setup ─────────────────────────────────────────────────────────────
    print(call_context,"call_context")
    stt_model = call_context.get("stt_model", "nova-2-general")
    language = call_context.get("language", "hi")
    stt_plugin = deepgram.STT(model=stt_model, language=language)

    # ── TTS setup ─────────────────────────────────────────────────────────────
    voice_id = call_context.get("voice", "cgSgspJ2msm6clMCkdW9")
    tts_plugin = elevenlabs.TTS(
        voice_id=voice_id,
        voice_settings=elevenlabs.VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
        ),
    )   

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
        )
    )