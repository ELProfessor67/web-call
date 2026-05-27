"""
Veena TTS Plugin for LiveKit Agents (in-process, sentence/part chunking)
========================================================================

Local Hindi/English TTS using Veena (Maya Research). Model loads ONCE inside
the agent worker process — no separate server.

LATENCY APPROACH (NOT frame streaming — that sounded choppy):
  We split the incoming text into small "parts" (sentences, and long
  sentences further split on commas/clauses). Each part is synthesized fully
  with SNAC (clean, smooth audio) and pushed as soon as it's ready. Because
  the first part is short, it plays quickly; the rest follow seamlessly.

Voice is chosen once at construction:  kavya | agastya | maitri | vinaya
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

# How long (chars) a single part can be before we split it further.
MAX_PART_CHARS = 160


def split_into_parts(text: str, max_chars: int = MAX_PART_CHARS) -> list[str]:
    """
    Break text into small synthesizable parts.

    1) Split on sentence enders (Hindi danda '।', '?', '!', '.').
    2) Any sentence still longer than max_chars is split on clause marks
       (commas, semicolons, Hindi/Eng) and, if still too long, on spaces.
    """
    text = text.strip()
    if not text:
        return []

    # 1) sentences — keep the ender attached
    sentences = re.findall(r'[^।.?!]+[।.?!]?', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    parts: list[str] = []
    for sent in sentences:
        if len(sent) <= max_chars:
            parts.append(sent)
            continue
        # 2) split long sentence on clause marks
        clauses = re.split(r'(?<=[,;:।])\s+', sent)
        buf = ""
        for cl in clauses:
            cl = cl.strip()
            if not cl:
                continue
            if len(cl) > max_chars:
                # 3) hard-split very long clause on spaces
                for chunk in _split_on_spaces(cl, max_chars):
                    if buf:
                        parts.append(buf); buf = ""
                    parts.append(chunk)
                continue
            if len(buf) + len(cl) + 1 <= max_chars:
                buf = f"{buf} {cl}".strip()
            else:
                if buf:
                    parts.append(buf)
                buf = cl
        if buf:
            parts.append(buf)
    return parts


def _split_on_spaces(s: str, max_chars: int) -> list[str]:
    words = s.split()
    out, buf = [], ""
    for w in words:
        if len(buf) + len(w) + 1 <= max_chars:
            buf = f"{buf} {w}".strip()
        else:
            if buf:
                out.append(buf)
            buf = w
    if buf:
        out.append(buf)
    return out


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
        self._gen_lock = threading.Lock()   # serialize GPU work across streams
        logger.info("Veena ready.")
        self._warmup()

    @classmethod
    def get(cls) -> "_VeenaEngine":
        if cls._instance is None:
            cls._instance = _VeenaEngine()
        return cls._instance

    def _warmup(self) -> None:
        try:
            self.synth_part("नमस्ते", DEFAULT_VOICE)
            logger.info("Warmup done.")
        except Exception as e:
            logger.warning(f"Warmup skipped: {e}")

    @torch.no_grad()
    def synth_part(self, text: str, voice: str,
                   temperature: float = 0.3, top_p: float = 0.9) -> bytes:
        """Synthesize ONE short part fully -> int16 PCM bytes (clean audio).

        We set the SAME seed before every part so the speaker's voice stays
        consistent across parts. Without this, do_sample=True makes each part
        sample slightly differently -> the voice seems to drift ("kabhi kuch
        kabhi kuch").
        """
        if voice not in VOICES:
            voice = DEFAULT_VOICE

        prompt = f"<spk_{voice}> {text}"
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tokens = [START_OF_HUMAN_TOKEN, *prompt_tokens, END_OF_HUMAN_TOKEN,
                        START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN]
        input_ids = torch.tensor([input_tokens], device=self.model.device)
        max_tokens = min(int(len(text) * 1.3) * 7 + 21, 700)

        with self._gen_lock:
            torch.manual_seed(1234)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1234)
            output = self.model.generate(
                input_ids, max_new_tokens=max_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
            )

        gen = output[0][len(input_tokens):].tolist()
        snac_tokens = [t for t in gen if AUDIO_CODE_BASE_OFFSET <= t < _AUDIO_MAX]
        if not snac_tokens:
            return b""
        audio = self._decode_snac(snac_tokens)
        if audio is None:
            return b""
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def _decode_snac(self, snac_tokens):
        if not snac_tokens or len(snac_tokens) % 7 != 0:
            # drop trailing partial frame
            snac_tokens = snac_tokens[: len(snac_tokens) - (len(snac_tokens) % 7)]
            if not snac_tokens:
                return None
        off = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]
        lvl = [[], [], []]
        for i in range(0, len(snac_tokens), 7):
            lvl[0].append(snac_tokens[i]     - off[0])
            lvl[1].append(snac_tokens[i + 1] - off[1])
            lvl[1].append(snac_tokens[i + 4] - off[4])
            lvl[2].append(snac_tokens[i + 2] - off[2])
            lvl[2].append(snac_tokens[i + 3] - off[3])
            lvl[2].append(snac_tokens[i + 5] - off[5])
            lvl[2].append(snac_tokens[i + 6] - off[6])
        codes = []
        for c in lvl:
            t = torch.tensor(c, dtype=torch.int32, device=self._snac_dev).unsqueeze(0)
            if torch.any((t < 0) | (t > 4095)):
                return None
            codes.append(t)
        with torch.no_grad():
            audio = self.snac.decode(codes)
        return audio.squeeze().clamp(-1, 1).cpu().numpy()


class _VeenaChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_plugin, input_text, conn_options):
        super().__init__(tts=tts_plugin, input_text=input_text, conn_options=conn_options)
        self._veena = tts_plugin

    async def _run(self, emitter: "AudioEmitter") -> None:
        import asyncio

        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._veena.sample_rate,
            num_channels=self._veena.num_channels,
            mime_type="audio/pcm",
        )
        engine = self._veena._engine
        voice = self._veena._voice

        parts = split_into_parts(self._input_text)
        if not parts:
            return

        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        aq: asyncio.Queue = asyncio.Queue()

        # Producer thread: generate parts back-to-back and hand PCM to the loop.
        # This keeps the GPU busy on part N+1 while LiveKit plays part N,
        # so there are no idle gaps between parts.
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
                    f"Veena first-part [{voice}]: "
                    f"{(time.perf_counter()-start)*1000:.0f}ms"
                )
                first = False
            if pcm:
                emitter.push(pcm)


class VeenaTTS(tts.TTS):
    """
    In-process Veena TTS with sentence/part chunking.

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
        self._engine = _VeenaEngine.get()
        logger.info(f"VeenaTTS ready — voice={self._voice}")

    def synthesize(self, text, *, conn_options=None) -> tts.ChunkedStream:
        if conn_options is None:
            conn_options = APIConnectOptions()
        return _VeenaChunkedStream(
            tts_plugin=self, input_text=text, conn_options=conn_options,
        )