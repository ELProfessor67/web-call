"""
Indic Parler-TTS Plugin for LiveKit Agents (in-process, part chunking)
======================================================================

Local Indic/English TTS using ai4bharat/indic-parler-tts (0.9B). Model loads
ONCE inside the agent worker process — no separate server.

Why Parler over the earlier Veena setup:
  - Smaller (0.9B) -> lighter on the L4, faster.
  - Named speakers (Hindi: Rohit, Divya, Aman, Rani) give consistent voice
    via the `description` caption, which fixes the "voice keeps changing"
    problem better than token-conditioned models.

How it works:
  Parler is encoder-decoder. It takes TWO inputs:
    1. description (caption) -> the description_tokenizer  (controls voice)
    2. prompt (text to speak) -> the prompt tokenizer
  Voice consistency comes from (a) naming the speaker in the description and
  (b) a fixed seed per part so sampling doesn't drift between chunks.

Gated model: accept conditions at
  https://huggingface.co/ai4bharat/indic-parler-tts  (login once),
and export HF_TOKEN before running.  NEVER hardcode the token.

Usage:
    >>> from parler_tts_plugin import ParlerTTS
    >>> tts = ParlerTTS(voice="Divya")          # loads model (slow, once)
    >>> session = AgentSession(stt=..., llm=..., tts=tts)
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from parler_tts import ParlerTTSForConditionalGeneration
from livekit.agents import tts, APIConnectOptions

if TYPE_CHECKING:
    from livekit.agents.tts.tts import AudioEmitter

__all__ = ["ParlerTTS"]

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("PARLER_MODEL", "ai4bharat/indic-parler-tts")
DEFAULT_VOICE = "Divya"

# Recommended Hindi speakers (from the model card). Add more as needed.
KNOWN_VOICES = {
    "Divya", "Rohit", "Aman", "Rani",   # Hindi
    "Mary", "Thoma", "Kavya",           # English (a few)
}

# A consistent, clean caption template. Speaker name first => stable voice.
def build_description(voice: str) -> str:
    return (
        f"{voice} speaks at a moderate pace with a slightly expressive tone. "
        f"The recording is very high quality, with the voice sounding clear "
        f"and very close up, with no background noise."
    )

MAX_PART_CHARS = 160


def split_into_parts(text: str, max_chars: int = MAX_PART_CHARS) -> list[str]:
    """Sentences first; long sentences split on clause marks, then spaces."""
    text = text.strip()
    if not text:
        return []
    sentences = re.findall(r'[^।.?!]+[।.?!]?', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    parts: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            parts.append(sent); continue
        clauses = re.split(r'(?<=[,;:।])\s+', sent)
        buf = ""
        for cl in clauses:
            cl = cl.strip()
            if not cl: continue
            if len(cl) > max_chars:
                for chunk in _split_on_spaces(cl, max_chars):
                    if buf: parts.append(buf); buf = ""
                    parts.append(chunk)
                continue
            if len(buf) + len(cl) + 1 <= max_chars:
                buf = f"{buf} {cl}".strip()
            else:
                if buf: parts.append(buf)
                buf = cl
        if buf: parts.append(buf)
    return parts


def _split_on_spaces(s: str, max_chars: int) -> list[str]:
    words = s.split(); out, buf = [], ""
    for w in words:
        if len(buf) + len(w) + 1 <= max_chars:
            buf = f"{buf} {w}".strip()
        else:
            if buf: out.append(buf)
            buf = w
    if buf: out.append(buf)
    return out


class _ParlerEngine:
    """Holds the model + tokenizers. Loaded once, shared by all streams."""

    _instance: "_ParlerEngine | None" = None

    def __init__(self) -> None:
        logger.info("Loading Indic Parler-TTS (0.9B, 4-bit NF4)... (one-time)")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,   # fp16 is a touch faster on L4
            bnb_4bit_use_double_quant=True,
        )
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quant_cfg,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self.sample_rate = int(self.model.config.sampling_rate)
        self._gen_lock = threading.Lock()
        logger.info(f"Parler ready — sample_rate={self.sample_rate}, dev={self.device}")

    @classmethod
    def get(cls) -> "_ParlerEngine":
        if cls._instance is None:
            cls._instance = _ParlerEngine()
        return cls._instance

    def _warmup(self) -> None:
        try:
            self.synth_part("नमस्ते", DEFAULT_VOICE)
            logger.info("Warmup done.")
        except Exception as e:
            logger.warning(f"Warmup skipped: {e}")

    @torch.no_grad()
    def synth_part(self, text: str, voice: str) -> bytes:
        """Synthesize ONE part -> int16 PCM bytes. Fixed seed = stable voice."""
        if voice not in KNOWN_VOICES:
            voice = DEFAULT_VOICE
        description = build_description(voice)

        dev = self.model.device
        desc = self.desc_tokenizer(description, return_tensors="pt").to(dev)
        prm = self.tokenizer(text, return_tensors="pt").to(dev)

        with self._gen_lock:
            torch.manual_seed(1234)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1234)
            generation = self.model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prm.input_ids,
                prompt_attention_mask=prm.attention_mask,
            )

        audio = generation.cpu().numpy().squeeze()
        if audio.size == 0:
            return b""
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class _ParlerChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_plugin, input_text, conn_options):
        super().__init__(tts=tts_plugin, input_text=input_text, conn_options=conn_options)
        self._parler = tts_plugin

    async def _run(self, emitter: "AudioEmitter") -> None:
        import asyncio

        engine = self._parler._engine
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._parler.sample_rate,
            num_channels=self._parler.num_channels,
            mime_type="audio/pcm",
        )
        voice = self._parler._voice
        parts = split_into_parts(self._input_text)
        if not parts:
            return

        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        aq: asyncio.Queue = asyncio.Queue()

        # Producer thread: generate parts back-to-back; GPU stays busy on part
        # N+1 while LiveKit plays part N (no idle gaps).
        def _produce():
            try:
                for part in parts:
                    pcm = engine.synth_part(part, voice)
                    loop.call_soon_threadsafe(aq.put_nowait, pcm)
            finally:
                loop.call_soon_threadsafe(aq.put_nowait, None)

        threading.Thread(target=_produce, daemon=True).start()

        first = True
        while True:
            pcm = await aq.get()
            if pcm is None:
                break
            if first:
                logger.debug(
                    f"Parler first-part [{voice}]: "
                    f"{(time.perf_counter()-start)*1000:.0f}ms"
                )
                first = False
            if pcm:
                emitter.push(pcm)
        emitter.flush()


class ParlerTTS(tts.TTS):
    """
    In-process Indic Parler-TTS with part chunking and stable voice.

    Args:
        voice: a named speaker, e.g. Divya | Rohit | Aman | Rani (Hindi).
               Unknown names fall back to Divya.
    """

    def __init__(self, voice: str = DEFAULT_VOICE) -> None:
        self._engine = _ParlerEngine.get()
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._engine.sample_rate,
            num_channels=1,
        )
        self._voice = voice if voice in KNOWN_VOICES else DEFAULT_VOICE
        logger.info(f"ParlerTTS ready — voice={self._voice}")

    def synthesize(self, text, *, conn_options=None) -> tts.ChunkedStream:
        if conn_options is None:
            conn_options = APIConnectOptions()
        return _ParlerChunkedStream(
            tts_plugin=self, input_text=text, conn_options=conn_options,
        )