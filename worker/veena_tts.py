"""
Veena TTS Plugin for LiveKit Agents (in-process, streaming)
===========================================================

Local Hindi/English TTS using Veena (Maya Research). The model is loaded
ONCE inside the agent worker process — no separate server.

STREAMING:
  Veena emits audio tokens in groups of 7 (= one SNAC frame). We run
  generation with a token streamer and, every few frames, decode with SNAC
  and push to LiveKit immediately. So the first audio plays long before the
  full reply is finished generating.

Voice is chosen once at construction:  kavya | agastya | maitri | vinaya

Usage:
    >>> from veena_tts import VeenaTTS
    >>> tts = VeenaTTS(voice="kavya")            # loads model (slow, once)
    >>> session = AgentSession(stt=..., llm=..., tts=tts)
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from snac import SNAC
from livekit.agents import tts, APIConnectOptions

if TYPE_CHECKING:
    from livekit.agents.tts.tts import AudioEmitter

__all__ = ["VeenaTTS"]

logger = logging.getLogger(__name__)

# ---- Veena constants (from model card) ----
VEENA_SAMPLE_RATE = 24000
MODEL_ID = os.getenv("VEENA_MODEL", "maya-research/veena-tts")
VOICES = ["kavya", "agastya", "maitri", "vinaya"]
DEFAULT_VOICE = "kavya"

START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN   = 128258
START_OF_HUMAN_TOKEN  = 128259
END_OF_HUMAN_TOKEN    = 128260
START_OF_AI_TOKEN     = 128261
END_OF_AI_TOKEN       = 128262
AUDIO_CODE_BASE_OFFSET = 128266
_AUDIO_MAX = AUDIO_CODE_BASE_OFFSET + 7 * 4096


class _VeenaEngine:
    """Holds the model + SNAC. Loaded once and shared by all streams."""

    _instance: "_VeenaEngine | None" = None

    def __init__(self) -> None:
        logger.info("Loading Veena (NF4 4-bit)... (one-time, ~30-60s)")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=quant_cfg,
            device_map="auto", trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
        self._snac_dev = next(self.snac.parameters()).device
        self._gen_lock = threading.Lock()   # serialize GPU generation across streams
        logger.info("Veena ready.")
        self._warmup()

    @classmethod
    def get(cls) -> "_VeenaEngine":
        if cls._instance is None:
            cls._instance = _VeenaEngine()
        return cls._instance

    def _warmup(self) -> None:
        try:
            for _ in self.stream_pcm("नमस्ते", DEFAULT_VOICE):
                pass
            logger.info("Warmup done.")
        except Exception as e:
            logger.warning(f"Warmup skipped: {e}")

    def _decode_frame(self, seven):
        """Decode exactly 7 audio tokens (one SNAC frame) -> float audio chunk."""
        off = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
        lvl0 = [seven[0] - off[0]]
        lvl1 = [seven[1] - off[1], seven[4] - off[4]]
        lvl2 = [seven[2] - off[2], seven[3] - off[3],
                seven[5] - off[5], seven[6] - off[6]]
        codes = []
        for c in (lvl0, lvl1, lvl2):
            t = torch.tensor(c, dtype=torch.int32, device=self._snac_dev).unsqueeze(0)
            if torch.any((t < 0) | (t > 4095)):
                return None
            codes.append(t)
        with torch.no_grad():
            audio = self.snac.decode(codes)
        return audio.squeeze().clamp(-1, 1).cpu().numpy()

    def stream_pcm(self, text, voice, temperature=0.4, top_p=0.9):
        """Generator yielding int16 PCM bytes as SNAC frames become ready."""
        if voice not in VOICES:
            voice = DEFAULT_VOICE

        prompt = f"<spk_{voice}> {text}"
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tokens = [START_OF_HUMAN_TOKEN, *prompt_tokens, END_OF_HUMAN_TOKEN,
                        START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN]
        input_ids = torch.tensor([input_tokens], device=self.model.device)
        max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)

        # Lightweight custom streamer that hands us RAW token ids as they're made.
        raw_q = queue.Queue()

        class _IdStreamer:
            def put(self, value):
                try:
                    ids = value.view(-1).tolist()
                except AttributeError:
                    ids = list(value)
                for tid in ids:
                    raw_q.put(tid)
            def end(self):
                raw_q.put(None)

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
            streamer=_IdStreamer(),
        )

        def _generate():
            with self._gen_lock, torch.no_grad():
                self.model.generate(**gen_kwargs)

        threading.Thread(target=_generate, daemon=True).start()

        buf = []                 # rolling buffer of 7 audio tokens
        frames = []              # decoded frames awaiting flush
        FLUSH_EVERY = 4          # batch a few frames per chunk (latency/quality balance)
        skip_prompt = len(input_tokens)
        seen = 0

        while True:
            tid = raw_q.get()
            if tid is None:
                break
            # the streamer echoes prompt ids first; skip them
            if seen < skip_prompt:
                seen += 1
                continue
            if AUDIO_CODE_BASE_OFFSET <= tid < _AUDIO_MAX:
                buf.append(tid)
                if len(buf) == 7:
                    chunk = self._decode_frame(buf)
                    buf = []
                    if chunk is not None:
                        frames.append(chunk)
                        if len(frames) >= FLUSH_EVERY:
                            yield self._to_pcm16(frames)
                            frames = []
        if frames:
            yield self._to_pcm16(frames)

    @staticmethod
    def _to_pcm16(frames):
        audio = np.concatenate(frames)
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class _VeenaChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_plugin, input_text, conn_options):
        super().__init__(tts=tts_plugin, input_text=input_text, conn_options=conn_options)
        self._veena = tts_plugin

    async def _run(self, emitter: "AudioEmitter") -> None:
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._veena.sample_rate,
            num_channels=self._veena.num_channels,
            mime_type="audio/pcm",
        )
        engine = self._veena._engine
        voice = self._veena._voice
        text = self._input_text
        start = time.perf_counter()
        first_logged = False

        loop = asyncio.get_running_loop()
        aq: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                for pcm in engine.stream_pcm(text, voice):
                    loop.call_soon_threadsafe(aq.put_nowait, pcm)
            finally:
                loop.call_soon_threadsafe(aq.put_nowait, None)

        threading.Thread(target=_produce, daemon=True).start()

        while True:
            pcm = await aq.get()
            if pcm is None:
                break
            if not first_logged:
                logger.debug(
                    f"Veena first-audio [{voice}]: "
                    f"{(time.perf_counter()-start)*1000:.0f}ms"
                )
                first_logged = True
            emitter.push(pcm)


class VeenaTTS(tts.TTS):
    """
    In-process Veena TTS. Loads the model on first construction.

    Args:
        voice: kavya | agastya | maitri | vinaya (set once; unknown -> kavya).
    """

    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=VEENA_SAMPLE_RATE,
            num_channels=1,
        )
        self._voice = voice if voice in VOICES else DEFAULT_VOICE
        self._engine = _VeenaEngine.get()   # load (or reuse) the model
        logger.info(f"VeenaTTS ready — voice={self._voice}")

    def synthesize(self, text, *, conn_options=None) -> tts.ChunkedStream:
        if conn_options is None:
            conn_options = APIConnectOptions()
        return _VeenaChunkedStream(
            tts_plugin=self, input_text=text, conn_options=conn_options,
        )