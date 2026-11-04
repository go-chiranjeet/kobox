"""REHA's voice: Kokoro-82M (ONNX, int8) running locally. Playback via macOS
`afplay`, so she can speak even when the app window is closed."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import re
import shutil
import threading
import time
import uuid
import wave
from typing import Awaitable, Callable

import numpy as np

from . import paths
from . import prosody
from .textproc import sentences

log = logging.getLogger("reha.voice")

KOKORO_MODEL = "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = "voices-v1.0.bin"


def parse_voice(spec: str) -> list[tuple[str, float]]:
    """'hf_alpha:0.55+bf_emma:0.45' -> [('hf_alpha', .55), ('bf_emma', .45)]"""
    out = []
    for part in (spec or "af_heart").split("+"):
        name, _, w = part.strip().partition(":")
        out.append((name.strip(), float(w) if w else 1.0))
    total = sum(w for _, w in out) or 1.0
    return [(n, w / total) for n, w in out]


def envelope(samples: np.ndarray, sr: int, fps: int = 30) -> list[float]:
    """Loudness curve (0..1) the UI uses to animate her while speaking."""
    hop = max(1, sr // fps)
    n = len(samples) // hop
    if n == 0:
        return []
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    peak = float(np.percentile(rms, 95)) or 1e-6
    return [round(float(min(1.0, v / peak)), 3) for v in rms]


def soften(samples: np.ndarray, sr: int, warmth: float) -> np.ndarray:
    """Warmer, less digital sound: blend in a gentle low-pass (tames sibilance and
    edge), add short fades so sentence joins don't click, normalise softly."""
    x = samples.astype(np.float32)
    if warmth > 0 and len(x):
        a = float(np.exp(-2 * np.pi * 3800 / sr))
        y = np.empty_like(x)
        acc = 0.0
        for i in range(len(x)):  # one-pole IIR; ~0.1 s per sentence
            acc = (1 - a) * x[i] + a * acc
            y[i] = acc
        x = (1 - warmth) * x + warmth * y
    f = min(len(x) // 4, int(sr * 0.012))
    if f > 0:
        x[:f] *= np.linspace(0, 1, f, dtype=np.float32)
        x[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0
    return x / peak * 0.88 if peak > 0 else x


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    return np.convolve(x, np.ones(n, np.float32) / n, mode="same").astype(np.float32) if n > 1 else x


def _air(n: int, sr: int, rnd: np.random.Generator) -> np.ndarray:
    """Noise shaped like breath, not tape hiss: most energy around 0.5-2 kHz,
    rolling off steeply above 3 kHz and gone by ~6 kHz. (Box-filtered white
    noise kept everything up to 12 kHz, which read as a 'hiss' on every
    breath, laugh and whisper.)"""
    if n <= 0:
        return np.zeros(0, np.float32)
    spec = rnd.standard_normal(n // 2 + 1) + 1j * rnd.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / sr)
    shape = (f / 350.0) ** 2 / (1 + (f / 350.0) ** 2)          # gentle high-pass: no rumble
    shape /= 1 + (f / 2200.0) ** 4                               # steep roll-off above ~2.5 kHz
    shape[f > 6000] = 0
    w = np.fft.irfft(spec * shape, n).astype(np.float32)
    rms = float(np.sqrt(np.mean(w ** 2))) or 1.0
    return w / rms


def lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Zero-phase low-pass in the frequency domain with a smooth 1 kHz shoulder."""
    if len(x) < 64 or cutoff >= sr / 2:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X *= np.clip((cutoff + 1000 - f) / 1000, 0, 1)
    return np.fft.irfft(X, len(x)).astype(np.float32)


def speaker_safe(x: np.ndarray, sr: int, ceiling: float = 0.89) -> np.ndarray:
    """Last stage before any audio reaches the speakers:
    * DC and sub-bass rumble below ~30 Hz removed (an offset makes cones sit
      off-centre and wastes excursion),
    * nothing above 10 kHz, where synthesis artefacts sound like static and
      speech carries little,
    * a soft limiter so peaks never clip, plus 5 ms fades so starts and stops
      never click,
    * silent gaps gated back to true silence after filtering."""
    x = np.nan_to_num(x.astype(np.float32))
    if len(x) < 64:
        return np.clip(x, -ceiling, ceiling)
    quiet = _smooth(np.abs(x), sr * 0.01) < 1e-4             # gaps that were silent stay silent
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X *= np.clip((f - 20) / 20, 0, 1) * np.clip((11000 - f) / 1000, 0, 1)   # 20-40 Hz up, 10-11 kHz down
    x = np.fft.irfft(X, len(x)).astype(np.float32)
    x[quiet] = 0.0
    over = np.abs(x) > 0.7 * ceiling                          # soft-knee limiter above -4 dB
    if over.any():
        k = 0.7 * ceiling
        x[over] = np.sign(x[over]) * (k + (ceiling - k) * np.tanh((np.abs(x[over]) - k) / (ceiling - k)))
    f = min(len(x) // 4, int(sr * 0.005))
    x[:f] *= np.linspace(0, 1, f, dtype=np.float32)
    x[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    return np.clip(x, -ceiling, ceiling)


def breathify(x: np.ndarray, sr: int, amount: float, rnd: np.random.Generator) -> np.ndarray:
    """Softer, lighter delivery: a little less chest and a gentle top-end roll-off,
    scaled by amount 0..1. No noise is ever added (he heard it as static), so
    softness comes only from the voice itself."""
    if amount <= 0 or len(x) < 32:
        return x
    voiced = x - _smooth(x, sr / 280) * (0.35 * amount)     # thinner, less chest
    return lowpass(voiced, sr, 7000 - 2500 * amount)        # softer = darker, never brighter


def breath_sound(sr: int, seconds: float, rnd: np.random.Generator, level: float = 0.05) -> np.ndarray:
    """A soft exhale: air with a rise-and-fall envelope."""
    n = int(sr * seconds)
    t = np.linspace(0, 1, n, dtype=np.float32)
    env = np.clip(np.sin(np.pi * t), 0, None) ** 1.6 * (1 - 0.35 * t)
    return _air(n, sr, rnd) * env * level


def chuckle_sound(sr: int, seconds: float, rnd: np.random.Generator, level: float = 0.06) -> np.ndarray:
    """A soft breathy laugh: three or four fading 'h' pulses."""
    n = int(sr * seconds)
    out = np.zeros(n, np.float32)
    pulses = 4 if seconds > 0.6 else 3
    for i in range(pulses):
        start = int(n * (0.05 + i * 0.85 / pulses))
        m = int(sr * 0.11)
        if start + m > n:
            break
        t = np.linspace(0, 1, m, dtype=np.float32)
        env = (np.clip(np.sin(np.pi * t), 0, None) ** 2) * (0.85 ** i)
        out[start:start + m] += _air(m, sr, rnd) * env * level
    return out


# voiced moments: (spoken forms, tempo, pitch, gain, breathiness)
VOCAL = {
    "laugh": (("Ha ha.", "Hah, ha.", "Ha ha ha."), 1.15, 1.06, 0.62, 0.3),
    "giggle": (("Hee hee.", "Heh heh.", "Hee hee hee."), 1.3, 1.12, 0.5, 0.35),
    "hum": (("Mmm.", "Hmm."), 0.85, 0.98, 0.55, 0.15),
}


def write_wav(path: os.PathLike, samples: np.ndarray, sr: int) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- Chatterbox (optional engine)
# Chatterbox Turbo performs real laughs and sighs from tags. It runs in its own
# venv as a long-lived worker (reha/chatterbox_worker.py); Kokoro stays the
# default and the fallback whenever Chatterbox is missing, slow or failing.
CB_VENV_PY = paths.CB_PYTHON
CB_REF = paths.VOICE_REF
_CB_EVENT = {"laugh": " [laugh] ", "giggle": " [chuckle] ", "breath": " [sigh] ", "hum": " Hmm. ",
             "gasp": " [gasp] ", "groan": " [groan] ", "sniff": " [sniff] ", "clear_throat": " [clear throat] ",
             "shush": " [shush] ", "pause": " ... ", "beat": " ... "}
# delivery styles Chatterbox performs (its own tokens); the rest (soft, warm...) are dropped, not read out
CB_STYLE = {"whisper": "[whispering]", "happy": "[happy]", "excited": "[happy]", "surprised": "[surprised]",
            "sarcastic": "[sarcastic]", "dramatic": "[dramatic]", "fear": "[fear]", "angry": "[angry]",
            "crying": "[crying]"}


def to_chatterbox(text: str) -> str:
    """REHA's cues -> Chatterbox tags. Whisper maps to [whispering]; styles it
    can't perform (soft, warm…) are dropped rather than read out; *emphasis*
    asterisks are removed."""
    out = []
    for kind, val in prosody.split_cues(text):
        if kind == "text":
            out.append(val.replace("*", ""))
        elif kind in ("event", "beat"):
            out.append(_CB_EVENT.get(val if kind == "event" else "beat", " "))
        elif kind == "style" and val in CB_STYLE:
            out.append(f" {CB_STYLE[val]} ")
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def cb_chunks(text: str, first: int = 140, rest: int = 260) -> list[str]:
    """Sentence-aligned chunks: a short first one so she starts quickly."""
    chunks, cur = [], ""
    for para in re.split(r"\n\s*\n", text.strip()):
        for sent in re.split(r"(?<=[.!?…])\s+", para.strip()):
            if not sent:
                continue
            limit = first if not chunks else rest
            if cur and len(cur) + len(sent) + 1 > limit:
                chunks.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


EXPRESSIVE_EVENTS = {"laugh", "giggle", "breath", "gasp", "groan", "sniff", "clear_throat", "shush"}


def split_runs(text: str) -> list[tuple[bool, str]]:
    """For Auto mode: cut a reply into runs of (expressive, text). A sentence is
    expressive when a laugh, giggle or sigh sits in or right before it, or it is
    whispered. Each run keeps its style cue so Kokoro's prosody still applies."""
    units: list[tuple[bool, str]] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        style, pending = "", []                       # active style cue; events waiting for a sentence
        for kind, val in prosody.split_cues(para):
            if kind == "style":
                style = "" if val == "normal" else f"[{val}] "
            elif kind in ("event", "beat"):
                pending.append(val if kind == "event" else "pause")
            else:
                for sent in re.split(r"(?<=[.!?…])\s+", val.strip()):
                    if not sent.strip():
                        continue
                    lead = "".join(f"[{_EVENT_CUE[e]}] " for e in pending)
                    expressive = bool(EXPRESSIVE_EVENTS & set(pending)) or style.strip("[] ") in CB_STYLE
                    units.append((expressive, f"{style}{lead}{sent}".strip()))
                    pending = []
        if pending:                                   # a trailing laugh belongs to the last sentence
            tail = " ".join(f"[{_EVENT_CUE[e]}]" for e in pending)
            if units:
                units[-1] = (units[-1][0] or bool(EXPRESSIVE_EVENTS & set(pending)), f"{units[-1][1]} {tail}")
            else:
                units.append((bool(EXPRESSIVE_EVENTS & set(pending)), tail))
        if units:
            units[-1] = (units[-1][0], units[-1][1] + "\n\n")
    runs: list[tuple[bool, str]] = []
    for e, t in units:
        if runs and runs[-1][0] == e and not e:       # merge plain sentences; expressive ones stay short
            runs[-1] = (e, f"{runs[-1][1]} {t}")
        else:
            runs.append((e, t))
    return [(e, t.strip()) for e, t in runs if t.strip()]


_EVENT_CUE = {"laugh": "laughs", "giggle": "giggles", "breath": "sighs", "hum": "hmm", "pause": "pause",
              "gasp": "gasps", "groan": "groans", "sniff": "sniffs", "clear_throat": "clears throat", "shush": "shh"}


def _group_text(g) -> str:
    return " ".join(seg.text for seg in g.segments if seg.kind == "speech" and seg.text).strip()


def _discard_clip(fut: asyncio.Future) -> None:
    if not fut.cancelled() and not fut.exception() and fut.result() and os.path.exists(fut.result()):
        os.unlink(fut.result())


def _offline_env() -> dict:
    """The voice worker runs fully offline: the installer downloaded the model once, so
    Hugging Face (or anything else) is never contacted while she speaks."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("ANTHROPIC", "HF_TOKEN", "HUGGING"))}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_DATASETS_OFFLINE="1", DO_NOT_TRACK="1")
    return env


class Chatterbox:
    """Client for the worker process; starts it on first use."""

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.broken = ""                    # reason it can't be used this run

    @staticmethod
    def installed() -> bool:
        return CB_VENV_PY.exists() and CB_REF.exists()

    async def _start(self) -> None:
        worker = paths.PKG / "chatterbox_worker.py"
        self.proc = await asyncio.create_subprocess_exec(
            str(CB_VENV_PY), "-I", str(worker), str(CB_REF), "0",   # 0: stay loaded while the core runs
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=_offline_env(),
            limit=1 << 20)
        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=240)   # model load
        if not line or not json.loads(line).get("ready"):
            raise RuntimeError("Chatterbox didn't start")
        log.info("chatterbox ready: %s", line.decode().strip())

    async def render(self, text: str, path: str, timeout: float = 45,
                     still_needed: Callable[[], bool] | None = None) -> bool:
        """Render text to path. Returns False without rendering if `still_needed`
        says no once the worker is free (Kobox: the line was already spoken)."""
        async with self.lock:
            if still_needed is not None and not still_needed():
                return False
            if self.proc is None or self.proc.returncode is not None:
                await self._start()
            self.proc.stdin.write((json.dumps({"text": text, "out": path}) + "\n").encode())
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=timeout)
            res = json.loads(line or b"{}")
            if not res.get("ok"):
                raise RuntimeError(res.get("error") or "Chatterbox worker stopped")
            return True

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr, n, width = w.getframerate(), w.getnframes(), w.getsampwidth()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    else:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648
    return speaker_safe(x, sr), sr


class Voice:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._kokoro = None
        self._lock = threading.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel = asyncio.Event()
        self.speaking = False
        self.cb = Chatterbox()
        from .engine import Kobox
        self.kobox = Kobox(self)
        # what's left of the current utterance, so an interruption can resume it
        self._head = lambda: ""
        self._tail = ""

    # ---------- synthesis ----------
    def _engine(self):
        with self._lock:
            if self._kokoro is None:
                from kokoro_onnx import Kokoro  # heavy import, deferred
                self._kokoro = Kokoro(str(paths.MODELS / KOKORO_MODEL), str(paths.MODELS / KOKORO_VOICES))
            return self._kokoro

    def available(self) -> bool:
        return (paths.MODELS / KOKORO_MODEL).exists() and (paths.MODELS / KOKORO_VOICES).exists()

    def voices(self) -> list[str]:
        return sorted(self._engine().get_voices())

    def _style(self, spec: str):
        k = self._engine()
        mix = parse_voice(spec)
        if len(mix) == 1:
            return mix[0][0]
        return sum(k.get_voice_style(n) * w for n, w in mix)

    def _lexicon(self) -> dict[str, str]:
        try:
            data = json.loads((paths.CONFIG_DIR / "pronunciations.json").read_text("utf-8"))
            return {k.lower(): v for k, v in data.items() if not k.startswith("_")}
        except Exception:
            return {}

    def phonemes(self, text: str) -> str | None:
        """Phoneme string with lexicon words spliced in, or None if no lexicon word occurs."""
        lex = self._lexicon()
        if not lex:
            return None
        rx = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(lex, key=len, reverse=True)) + r")('s)?\b", re.I)
        if not rx.search(text):
            return None
        k, lang = self._engine(), self.cfg.get("lang", "en-us")
        out, pos = [], 0
        for m in rx.finditer(text):
            before = text[pos:m.start()]
            if before.strip():
                out.append(k.tokenizer.phonemize(before, lang))
            ipa = lex[m.group(1).lower()]
            if m.group(2):  # possessive: voiceless endings take /s/, the rest /z/
                ipa += "s" if ipa[-1] in "tkpfθ" else "z"
            out.append(ipa)
            pos = m.end()
        tail = text[pos:]
        if tail.strip():
            out.append(k.tokenizer.phonemize(tail, lang))
        joined = " ".join(p.strip() for p in out if p.strip())
        return re.sub(r"\s+([,.!?;:])", r"\1", joined)

    def _raw(self, text: str, style, speed: float) -> tuple[np.ndarray, int]:
        k = self._engine()
        ph = self.phonemes(text)
        if ph:
            return k.create(ph, voice=style, speed=speed, is_phonemes=True, sentence_pause=0.0, clause_pause=0.0)
        return k.create(text, voice=style, speed=speed, lang=self.cfg.get("lang", "en-us"),
                        sentence_pause=0.0, clause_pause=0.0)

    def _mood_style(self, voice: str | None, mood: str):
        base = self._style(voice or self.cfg["voice"])
        bias = prosody.profile(mood)[4]
        if not bias or not self.cfg.get("expressive", True):
            return base
        k = self._engine()
        if isinstance(base, str):
            base = k.get_voice_style(base)
        name, w = bias
        return base * (1 - w) + k.get_voice_style(name) * w

    def _biased(self, style, bias: tuple[str, float] | None):
        """Lean the voice toward another voice's colour (e.g. breathier for a whisper)."""
        if not bias or not self.cfg.get("expressive", True):
            return style
        k = self._engine()
        base = k.get_voice_style(style) if isinstance(style, str) else style
        name, w = bias
        return base * (1 - w) + k.get_voice_style(name) * w

    def _vocal(self, kind: str, style, base_speed: float, rnd: np.random.Generator) -> np.ndarray | None:
        """Render a laugh, giggle or hum with her own voice. None if synthesis fails."""
        texts, tempo, pitch, _, breath = VOCAL[kind]
        try:
            x, _sr = self._raw(texts[int(rnd.integers(len(texts)))], self._biased(style, ("af_bella", 0.3)),
                               base_speed * tempo / pitch)
        except Exception:
            return None
        x = x.astype(np.float32)
        if len(x) < 10:
            return None
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) / pitch)), np.arange(len(x)), x).astype(np.float32)
        return breathify(x, 24000, breath, rnd)

    def render_group(self, group: "prosody.Group", style, base_speed: float) -> tuple[np.ndarray, int]:
        """Synthesise one breath-group plan: per-clause tempo, pitch, pauses and
        delivery (gain, breathiness, voice colour). Cue moments (breath, laugh,
        pause) are placed after normalisation so they sit at natural levels."""
        pieces, gains, moments, sr = [], [], [], 24000
        rnd = np.random.default_rng()
        pos = 0

        def add(x: np.ndarray, gain: float) -> None:
            nonlocal pos
            pieces.append(x)
            gains.append(np.full(len(x), gain, np.float32))
            pos += len(x)

        for seg in group.segments:
            if seg.kind != "speech":
                n = int(sr * seg.pause_before)
                if seg.kind in VOCAL:
                    # a voiced moment (laugh, giggle, hum): Kokoro says it, lifted and lighter
                    x = self._vocal(seg.kind, style, base_speed, rnd)
                    if x is not None:
                        sr = 24000
                        add(x, VOCAL[seg.kind][3] * seg.gain)
                        add(np.zeros(int(sr * 0.12), np.float32), 1.0)
                        continue
                # a sigh (or a laugh that couldn't be voiced) is a quiet beat here, never
                # synthetic noise; real sighs come from Chatterbox in Auto mode
                add(np.zeros(n, np.float32), 1.0)
                continue
            seg_style = self._biased(style, seg.bias)
            # pitch trick: synthesise slower by the pitch ratio, then resample faster by
            # the same ratio -> pitch moves, duration stays put
            samples, sr = self._raw(seg.text, seg_style, base_speed * seg.tempo / seg.pitch)
            x = samples.astype(np.float32)
            if abs(seg.pitch - 1.0) > 0.002 and len(x) > 10:
                n_new = int(len(x) / seg.pitch)
                x = np.interp(np.linspace(0, len(x) - 1, n_new), np.arange(len(x)), x).astype(np.float32)
            x = breathify(x, sr, seg.breath, rnd)
            if seg.pause_before:
                add(np.zeros(int(sr * seg.pause_before), np.float32), seg.gain)
            add(x, seg.gain)
            add(np.zeros(int(sr * seg.pause_after), np.float32), seg.gain)
        if not pieces:
            return np.zeros(1, np.float32), sr
        audio = soften(np.concatenate(pieces), sr, float(self.cfg.get("warmth", 0.35)))
        g = _smooth(np.concatenate(gains), sr * 0.03)          # no clicks where the delivery changes
        audio = audio * g
        for at, snd in moments:
            end = min(len(audio), at + len(snd))
            audio[at:end] += snd[: end - at]
        return speaker_safe(audio, sr), sr

    def synth(self, text: str, voice: str | None = None, speed: float | None = None,
              mood: str = "neutral") -> tuple[np.ndarray, int]:
        """Whole text as one clip (previews, tests)."""
        style = self._mood_style(voice, mood)
        spd = speed or self.cfg.get("speed", 1.0)
        out, sr = [], 24000
        for g in prosody.plan(text, mood):
            a, sr = self.render_group(g, style, spd)
            out.append(a)
        return (np.concatenate(out) if out else np.zeros(1, np.float32)), sr

    # ---------- playback ----------
    async def _play_file(self, path: str, duration: float) -> None:
        player = self.cfg.get("player", "afplay")
        exe = shutil.which(player)
        if not exe:  # headless/test machines: simulate playback time
            try:
                await asyncio.wait_for(self._cancel.wait(), timeout=duration)
            except asyncio.TimeoutError:
                pass
            return
        self._proc = await asyncio.create_subprocess_exec(exe, path)
        try:
            await self._proc.wait()
        finally:
            self._proc = None

    def remaining(self) -> str:
        """The rest of what she was saying, from the sentence that was playing."""
        try:
            return f"{self._head()} {self._tail}".strip()
        except Exception:
            return ""

    def engine(self) -> str:
        """The engine that will actually speak: kobox (the combined engine), auto
        (Kokoro plus Chatterbox for expressive moments) or chatterbox, falling back
        to kokoro if unusable. Kobox stays kobox while Chatterbox recovers: it
        speaks Kokoro lines meanwhile and retries the worker on its own schedule."""
        want = self.cfg.get("engine", "kokoro")
        if want == "kobox" and Chatterbox.installed():
            return "kobox"
        if want in ("auto", "chatterbox") and Chatterbox.installed() and not self.cb.broken:
            return want
        return "kokoro"

    async def warm(self) -> None:
        """Load the Chatterbox model ahead of time so her first laugh isn't late."""
        if self.engine() == "kobox":
            # Kokoro's first render and the audio process start are the slow part of a cold first reply
            loop = asyncio.get_running_loop()
            try:
                style = await loop.run_in_executor(None, self._mood_style, None, "neutral")
                await asyncio.wrap_future(self.kobox.pool.submit(self.kobox._kokoro, "Okay.", style, 1.0, "neutral"))
                s = sys.modules[type(self.kobox).__module__].settings(self.cfg)     # works as reha.kobox and kobox.engine
                if s.get("stream", True) and s.get("process_player", True):
                    await self.kobox._new_player(loop, s).ensure()
            except Exception as e:
                log.warning("kobox warm-up failed: %s", e)
        if self.engine() == "kobox" and self.kobox.ears is not None and self.kobox.ears.available():
            # the accuracy check's Whisper model, so the first checked line isn't late
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self.kobox.ears.verify, np.zeros(16000, np.float32), 5.0)
            except Exception as e:
                log.warning("kobox whisper warm-up failed: %s", e)
        if self.engine() != "kokoro":
            try:
                async with self.cb.lock:
                    if self.cb.proc is None or self.cb.proc.returncode is not None:
                        await self.cb._start()
            except Exception as e:
                log.warning("chatterbox warm-up failed: %s", e)
                self.cb.broken = str(e)[:200]
        if self.engine() == "kobox" and not self.cb.broken:
            # the first renders after loading are several times slower (GPU kernels compile):
            # do them now, not on her first real laugh
            for i, t in enumerate(("[laugh] Okay, that's fine.",)):
                f = paths.AUDIO_TMP / f"kobox-warm-{i}.wav"
                try:
                    await self.cb.render(t, str(f), timeout=90)
                except Exception as e:
                    log.warning("chatterbox warm render failed: %s", e)
                    break
                finally:
                    f.unlink(missing_ok=True)
            asyncio.ensure_future(self.kobox.prepare_bank())   # background; only renders what's missing

    async def speak(self, text: str, on_chunk: Callable[[list[float], float], Awaitable[None]] | None = None,
                    voice: str | None = None, mood: str = "neutral") -> bool:
        self._head, self._tail = (lambda: ""), ""
        eng = self.engine() if text.strip() else "kokoro"
        if eng == "kobox":
            # fallback chain Kobox -> Auto -> Kokoro: she never goes silent because of Kobox
            try:
                done = await self.kobox.speak(text, on_chunk, voice, mood)
            except Exception:
                log.exception("kobox failed")
                done = None
            if done is not None:
                return done
            self.kobox.last = {**self.kobox.last, "fallback": self.kobox.last.get("fallback") or "Kobox failed; Auto spoke"}
            try:
                return await self._speak_auto(text, on_chunk, voice, mood)
            except Exception:
                log.exception("auto fallback failed")
                return await self._speak_kokoro(text, on_chunk, voice, mood)
        if eng == "chatterbox":
            done = await self._speak_chatterbox(text, on_chunk)
            if done is not None:
                return done
        elif eng == "auto":
            return await self._speak_auto(text, on_chunk, voice, mood)
        return await self._speak_kokoro(text, on_chunk, voice, mood)

    async def _cb_file(self, text: str) -> str | None:
        f = str(paths.AUDIO_TMP / f"cb-{uuid.uuid4().hex[:8]}.wav")
        try:
            await self.cb.render(text, f)
            return f
        except Exception as e:
            log.warning("chatterbox failed, using kokoro: %s", e)
            self.cb.broken = str(e)[:200]
            return None

    async def _speak_auto(self, text: str, on_chunk, voice: str | None, mood: str) -> bool:
        """Kokoro for plain lines (instant); Chatterbox for laughs, giggles, sighs and
        whispers. Every run after the first is rendered in the background while the
        one before it plays. A Chatterbox line that isn't ready within `cb_wait`
        seconds of its turn is spoken by Kokoro instead, so a gap never grows long."""
        self._cancel.clear()
        runs = split_runs(text)
        if not any(e for e, _ in runs):
            return await self._speak_kokoro(text, on_chunk, voice, mood)
        loop = asyncio.get_running_loop()
        style = await loop.run_in_executor(None, self._mood_style, voice, mood)
        spd = float(self.cfg.get("speed", 1.0))
        wait = float(self.cfg.get("cb_wait", 1.5))

        def kokoro_clip(t: str) -> tuple[np.ndarray, int]:
            out, sr = [], 24000
            for g in prosody.plan(t, mood):
                a_, sr = self.render_group(g, style, spd)
                out.append(a_)
            return (np.concatenate(out) if out else np.zeros(1, np.float32)), sr

        # Chatterbox renders start now, one after another in the worker. They are
        # never cancelled mid-request (that would desynchronise the worker), only
        # discarded if unused.
        cb = {i: asyncio.ensure_future(self._cb_file(to_chatterbox(t))) for i, (e, t) in enumerate(runs) if e}
        kk: dict[int, asyncio.Future] = {}

        def prefetch(i: int) -> None:
            # Kokoro clips for the next run; for an expressive run this is the instant
            # backup in case Chatterbox misses its deadline
            if i < len(runs) and i not in kk:
                kk[i] = loop.run_in_executor(None, kokoro_clip, runs[i][1])

        async def play(samples: np.ndarray, sr: int) -> None:
            f = paths.AUDIO_TMP / f"say-{uuid.uuid4().hex[:8]}.wav"
            write_wav(f, samples, sr)
            dur = len(samples) / sr
            if on_chunk:
                await on_chunk(envelope(samples, sr), dur)
            await self._play_file(str(f), dur)
            f.unlink(missing_ok=True)

        self.speaking = True
        try:
            for i, (e, t) in enumerate(runs):
                self._tail = " ".join(tt for _, tt in runs[i + 1:])
                self._head = (lambda tt=t: tt)
                if self._cancel.is_set():
                    return False
                if i == 0 and not e:
                    prefetch(1)
                    if not await self._speak_kokoro(t, on_chunk, voice, mood):
                        return False
                    self.speaking = True
                    continue
                if e:
                    try:
                        f = await asyncio.wait_for(asyncio.shield(cb[i]), timeout=wait)
                    except asyncio.TimeoutError:
                        f = None
                        log.info("chatterbox line not ready in %.1fs; kokoro speaks it", wait)
                    if f:
                        samples, sr = read_wav(f)
                        os.unlink(f)
                    else:
                        prefetch(i)
                        samples, sr = await kk[i]
                else:
                    prefetch(i)
                    samples, sr = await kk[i]
                prefetch(i + 1)
                if self._cancel.is_set():
                    return False
                await play(samples, sr)
            return not self._cancel.is_set()
        finally:
            self.speaking = False
            for fut in cb.values():                     # tidy up renders nobody played
                fut.add_done_callback(_discard_clip)

    async def _speak_chatterbox(self, text: str, on_chunk) -> bool | None:
        """Expressive path. Returns None (nothing played) if Chatterbox fails before
        the first chunk, so Kokoro can take over; later failures fall back per chunk."""
        self._cancel.clear()
        self.speaking = True
        originals = [x for x in cb_chunks(text) if to_chatterbox(x).strip(" .")]
        chunks = [to_chatterbox(x) for x in originals]
        if not chunks:
            self.speaking = False
            return True

        async def render(c: str) -> str | None:
            f = str(paths.AUDIO_TMP / f"cb-{uuid.uuid4().hex[:8]}.wav")
            try:
                await self.cb.render(c, f)
                return f
            except Exception as e:
                log.warning("chatterbox failed, using kokoro: %s", e)
                self.cb.broken = str(e)[:200]
                return None
        try:
            nxt = asyncio.create_task(render(chunks[0]))
            for i, c in enumerate(chunks):
                self._head = (lambda j=i: " ".join(originals[j:]))
                f = await nxt
                if self._cancel.is_set():
                    return False
                if i + 1 < len(chunks):
                    nxt = asyncio.create_task(render(chunks[i + 1]))
                if f is None:
                    if i == 0:
                        if i + 1 < len(chunks):
                            nxt.cancel()
                        return None                       # nothing played yet: Kokoro speaks it all
                    rest = " ".join(chunks[i:])
                    if i + 1 < len(chunks):
                        nxt.cancel()
                    return await self._speak_kokoro(rest, on_chunk)
                samples, sr = read_wav(f)              # passes through speaker_safe
                write_wav(f, samples, sr)              # play the safe version, not the raw file
                dur = len(samples) / sr
                if on_chunk:
                    await on_chunk(envelope(samples, sr), dur)
                await self._play_file(f, dur)
                os.unlink(f)
                if self._cancel.is_set():
                    return False
            return True
        finally:
            self.speaking = False

    async def _speak_kokoro(self, text: str, on_chunk: Callable[[list[float], float], Awaitable[None]] | None = None,
                            voice: str | None = None, mood: str = "neutral") -> bool:
        """Speak with human-like prosody, one sentence group at a time, rendering
        the next group while the current one plays. Returns False if interrupted."""
        if not text.strip() or not self.available():
            return True
        self._cancel.clear()
        self.speaking = True
        loop = asyncio.get_running_loop()
        groups = prosody.plan(text, mood) if self.cfg.get("expressive", True) else [
            prosody.Group([prosody.Segment(c, 1.0, 1.0, 0.35)]) for c in sentences(text)]
        if not groups:
            self.speaking = False
            return True
        spd = float(self.cfg.get("speed", 1.0))
        try:
            style = await loop.run_in_executor(None, self._mood_style, voice, mood)
            nxt = loop.run_in_executor(None, self.render_group, groups[0], style, spd)
            self._head = lambda: " ".join(_group_text(g) for g in groups[self._gi:])
            for i in range(len(groups)):
                self._gi = i
                samples, sr = await nxt
                if self._cancel.is_set():
                    return False
                if i + 1 < len(groups):
                    nxt = loop.run_in_executor(None, self.render_group, groups[i + 1], style, spd)
                f = paths.AUDIO_TMP / f"say-{uuid.uuid4().hex[:8]}.wav"
                write_wav(f, samples, sr)
                dur = len(samples) / sr
                if on_chunk:
                    await on_chunk(envelope(samples, sr), dur)
                await self._play_file(str(f), dur)
                f.unlink(missing_ok=True)
                if self._cancel.is_set():
                    return False
            return True
        except Exception:
            log.exception("speech failed")
            return True
        finally:
            self.speaking = False

    async def stop(self) -> None:
        self._cancel.set()
        self.kobox.stop()
        p = self._proc
        if p and p.returncode is None:
            try:
                p.terminate()
            except ProcessLookupError:
                pass

    def preview_file(self, text: str, voice: str, mood: str = "warm") -> str:
        samples, sr = self.synth(text, voice, mood=mood)
        f = paths.AUDIO_TMP / f"preview-{int(time.time())}.wav"
        write_wav(f, samples, sr)
        return str(f)
