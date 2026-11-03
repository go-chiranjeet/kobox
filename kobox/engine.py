"""Kobox: REHA's combined voice engine (Kokoro + Chatterbox), spec in docs/kobox/.

Kokoro speaks plain lines instantly; Chatterbox performs the laughs, sighs and
whispers. Kobox makes the two sound like one person and one continuous take:

* plan     every reply is cut into lines, each marked plain or expressive
* render   Chatterbox starts on every expressive line at once (in order, in its
           own worker), Kokoro renders every line, expressive ones as backup
* match    Chatterbox lines are trimmed, toned and levelled to Kokoro's sound
* check    Whisper listens to each Chatterbox line; wrong words -> Kokoro's version
* deadline a Chatterbox line that isn't ready when its turn comes (plus a grace
           period learned from this Mac's real speed) is spoken by Kokoro
* play     one continuous output stream with short equal-power crossfades

It never raises into the caller: speak() returns True (finished), False
(interrupted) or None (failed before any sound, so the caller falls back).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Awaitable, Callable

import numpy as np

from . import paths

log = logging.getLogger("reha.kobox")

SR = 24000
DEFAULTS = {
    "grace_max_s": 4.0,         # longest silence Kobox will hold for a late optional (mood) line
    "expression_wait_s": 8.0,   # a laugh, sigh, hum or whisper the reply asked for: wait for the real
                                # performance up to this (a safety cap; normally 1-2 s), never swap it
    "retries": 1,               # re-render a take the accuracy check rejected, this many times
    "start_wait_s": 1.5,        # ...but at the very start of a reply (nothing playing yet) wait at most this
    "wer_max": 0.25,            # accuracy check: highest word-error rate accepted
    "check": True,              # Whisper accuracy check on every Chatterbox line
    "crossfade_ms": 20,
    "cb_pause_s": 0.32,         # pause after a Chatterbox line (Kokoro lines carry their own)
    "eq_strength": 0.35,        # how far Chatterbox's brighter top end is pulled toward Kokoro's
    "mood_lines": True,         # amused/playful/tender/concerned replies may give a short plain line to Chatterbox
    "instant_start": True,      # the first line is always Kokoro, so she starts at once
    "stream": True,             # one continuous output stream (False: one file per line via afplay)
    "restart_every_s": 600,     # a failed Chatterbox worker is retried at most this often
}
MOOD_LINE_MOODS = {"amused", "playful", "tender", "concerned"}
LOG_KEEP_DAYS = 14


def settings(voice_cfg: dict) -> dict:
    return {**DEFAULTS, **(voice_cfg.get("kobox") or {})}


# ---------------------------------------------------------------- planning
@dataclass
class Line:
    idx: int
    text: str                   # REHA's cue text (Kokoro reads this through prosody)
    expressive: bool
    cb_text: str = ""           # what Chatterbox is asked to say
    engine: str = "kokoro"      # who actually spoke it
    reason: str = ""            # why Kokoro spoke an expressive line
    spoken: bool = False
    kk_text: str | None = None  # what Kokoro says if it differs (hums removed: Kokoro can't hum)
    lead: str | None = None     # bank sound before the words (fallback: a real laugh, then Kokoro)
    tail: str | None = None     # bank sound after the words
    instant: bool = False       # the whole line is one bank sound: plays at once, no render
    hum: bool = False           # a hum/"mm-hmm" on its own, performed by Chatterbox
    optional: bool = False      # chosen by Kobox for the mood, not asked for by the reply


# hums: a [hmm] cue, or a bare "Mmm", "Hmm", "Mm-hmm", "Mmm hmmm" in the text (not "5 mm")
_HUM_TOKEN = r"(?:m+-?h+m+|h+m+|m{2,}|mh+m+)"
HUM_RE = re.compile(r"\[hmm\]\s*|(?<!\d )(?<!\d)\b" + _HUM_TOKEN + r"\b(?:[\s,-]+" + _HUM_TOKEN + r"\b)*[,.!?…]*\s*", re.I)


def _hum_say(tok: str) -> str:
    words = re.findall(r"[a-z]+", tok.lower())
    if not words or words == ["hmm"] or (len(words) == 1 and words[0].startswith("h")):
        return "Hmm."
    if len(words) > 1 or "h" in words[0]:
        return "Mm-hmm."
    return "Mmm."


# a sentence that opens with an interjection is performed by Chatterbox (Kokoro reads them flatly)
INTERJECTION_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)*(?:ooh+|oo+h|aww+|ugh+|whoa+|pfft+|heh+|uh-oh|yay+|oh no|"
                             r"ahh+|oops|wow|ouch|eek|phew|aha|yikes)\b", re.I)


def shorten_expressive(text: str, longest: int = 80) -> tuple[str, str, str]:
    """A long expressive sentence -> (before, performed, after): Chatterbox performs only
    the clause the laugh or sigh belongs to (fast to render), Kokoro says the rest."""
    from .prosody import strip_cues
    if len(strip_cues(text)) <= longest or re.search(r"\[(whisper|whispering)\]", text, re.I):
        return "", text, ""
    cuts = [m.end() for m in re.finditer(r"[,;:—–]\s+", text)]
    if re.match(r"^\s*\[[^\]]+\]", text):                      # cue up front: keep the first clause
        for c in cuts:
            if 12 <= len(strip_cues(text[:c])) <= longest:
                return "", text[:c].strip(), text[c:].strip()
    elif re.search(r"\[[^\]]+\]\s*$", text):                  # cue at the end: keep the last clause
        for c in reversed(cuts):
            if 12 <= len(strip_cues(text[c:])) <= longest:
                return text[:c].strip(), text[c:].strip(), ""
    return "", text, ""


def split_hums(text: str) -> list[tuple[bool, str]]:
    """'Mmm hmmm, that sounds right.' -> [(hum, 'Mm-hmm.'), (plain, 'that sounds right.')]."""
    out, pos = [], 0
    for m in HUM_RE.finditer(text):
        if text[pos:m.start()].strip():
            out.append((False, text[pos:m.start()].strip()))
        out.append((True, _hum_say(m.group())))
        pos = m.end()
    if text[pos:].strip():
        out.append((False, text[pos:].strip()))
    return out or [(False, text)]


_IPA = [("eɪ", "ay"), ("aɪ", "eye"), ("ɔɪ", "oy"), ("aʊ", "ow"), ("oʊ", "oh"), ("əʊ", "oh"), ("tʃ", "ch"),
        ("dʒ", "j"), ("ɪə", "eer"), ("eə", "air"), ("ʊə", "oor"), ("iː", "ee"), ("uː", "oo"), ("ɑː", "ah"),
        ("ɔː", "aw"), ("ɜː", "er"), ("ɚ", "er"), ("ɝ", "er"), ("ə", "uh"), ("ɪ", "i"), ("ʊ", "u"), ("æ", "a"),
        ("ɛ", "e"), ("ʌ", "u"), ("ɑ", "ah"), ("ɒ", "o"), ("ɔ", "aw"), ("i", "ee"), ("u", "oo"), ("ɹ", "r"),
        ("ʃ", "sh"), ("ʒ", "zh"), ("θ", "th"), ("ð", "th"), ("ŋ", "ng"), ("ɡ", "g"), ("j", "y"), ("ɾ", "t"),
        ("ʔ", ""), ("ː", ""), ("ˈ", ""), ("ˌ", ""), (".", "")]


def respell(ipa: str) -> str:
    """IPA from the pronunciation list -> a spelling Chatterbox reads close to it
    ('ɹˈeɪhə' -> 'Rayhuh'). An explicit "_spoken" entry in the list wins."""
    table = dict(_IPA)
    rx = re.compile("|".join(re.escape(k) for k in sorted(table, key=len, reverse=True)))
    out = rx.sub(lambda m: table[m.group()], ipa)          # one pass: outputs are never re-mapped
    out = re.sub(r"[^A-Za-z\- ]", "", out)
    return out[:1].upper() + out[1:] if out else ipa


def spoken_lexicon() -> dict[str, str]:
    """{word: how Chatterbox should spell it} from config/pronunciations.json."""
    try:
        data = json.loads((paths.CONFIG_DIR / "pronunciations.json").read_text("utf-8"))
    except Exception:
        return {}
    explicit = {k.lower(): v for k, v in (data.get("_spoken") or {}).items()}
    out = {}
    for k, v in data.items():
        if k.startswith("_") or not isinstance(v, str):
            continue
        out[k] = explicit.get(k.lower()) or respell(v)
    return out


def apply_lexicon(text: str, lex: dict[str, str]) -> str:
    if not lex:
        return text
    rx = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(lex, key=len, reverse=True)) + r")\b")
    return rx.sub(lambda m: lex.get(m.group(1), m.group(1)), text)


def plan(text: str, mood: str, s: dict, lex: dict[str, str] | None = None) -> list[Line]:
    from .speech import split_runs, to_chatterbox
    runs = split_runs(text)
    mood_lines = bool(s.get("mood_lines")) and mood in MOOD_LINE_MOODS
    # plain runs back into sentences: each is rendered and played as soon as it's ready, so a
    # long reply starts at once (not after the whole run is rendered), and a mood line can be chosen
    runs = [(e, part) for e, t in runs
            for part in ([t] if e else [p for p in re.split(r"(?<=[.!?…])\s+|\n+", t) if p.strip()])]
    lines: list[Line] = []
    for e, t in runs:
        if e:                          # expressive already: Chatterbox gets the hum, Kokoro's backup drops it
            before, t, after = shorten_expressive(t)
            if before:
                lines.append(Line(len(lines), before, False))
            ln = Line(len(lines), t, True)
            if HUM_RE.search(t):
                ln.kk_text = HUM_RE.sub(" ", t).strip()
            lines.append(ln)
            if after:
                lines.append(Line(len(lines), after, False))
            continue
        for hum, part in split_hums(t):
            ln = Line(len(lines), part, hum or bool(INTERJECTION_RE.match(part)))
            if hum:
                ln.hum, ln.kk_text, ln.reason = True, "", "hum"
            lines.append(ln)
    # mood lines: a short plain line in a clearly emotional reply, at most 1 in 3 lines
    if mood_lines and len(lines) > 1:
        budget = len(lines) // 3 - sum(1 for ln in lines if ln.expressive)
        for ln in lines[1:]:
            if budget <= 0:
                break
            plain = re.sub(r"\[[^\]]*\]", "", ln.text).strip()
            if not ln.expressive and len(plain) <= 120 and len(re.findall(r"[.!?…]", plain)) <= 1:
                ln.expressive, ln.optional, budget = True, True, budget - 1
    if s.get("instant_start") and lines and lines[0].expressive and not lines[0].hum:
        lines[0].expressive, lines[0].reason = False, "first line (instant start)"
    for ln in lines:
        if ln.expressive:
            ln.cb_text = ln.text if ln.hum else apply_lexicon(to_chatterbox(ln.text), lex or {})
            if not ln.cb_text.strip(" ."):
                ln.expressive = False
    return lines


# ---------------------------------------------------------------- matching
def _frames_db(x: np.ndarray, sr: int = SR) -> np.ndarray:
    hop = max(1, int(sr * 0.02))
    n = len(x) // hop
    if n == 0:
        return np.array([-120.0])
    f = x[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt(np.mean(f.astype(np.float64) ** 2, axis=1)) + 1e-9)


def speech_level_db(x: np.ndarray, sr: int = SR) -> float:
    """Loudness of the speech itself (frames within 25 dB of the loudest), so
    pauses don't drag the figure down."""
    db = _frames_db(x, sr)
    act = db[db > db.max() - 25]
    return float(10 * np.log10(np.mean(10 ** (act / 10)))) if len(act) else -120.0


def trim(x: np.ndarray, sr: int = SR, floor_db: float = -40.0, margin_s: float = 0.012) -> np.ndarray:
    """Drop leading and trailing silence (Chatterbox adds 90-260 ms before speaking)."""
    if len(x) < 64:
        return x
    peak = float(np.abs(x).max()) or 1e-9
    env = np.convolve(np.abs(x), np.ones(int(sr * 0.005)) / int(sr * 0.005), mode="same")
    loud = np.nonzero(env > peak * 10 ** (floor_db / 20))[0]
    if not len(loud):
        return x[:0]
    m = int(sr * margin_s)
    return x[max(0, loud[0] - m): min(len(x), loud[-1] + m)]


def match(x: np.ndarray, target_db: float, s: dict, warmth: float) -> np.ndarray:
    """A Chatterbox clip -> Kokoro's sound: trimmed, top end pulled down, same
    finishing chain (warmth, speaker-safe limiter) and the same speech level."""
    from .speech import lowpass, soften, speaker_safe
    x = trim(np.asarray(x, np.float32))
    if len(x) < 64:
        return x
    k = float(s.get("eq_strength", 0.35))
    if k > 0:
        x = (x - k * (x - lowpass(x, SR, 3500))).astype(np.float32)
    x = soften(x, SR, warmth)
    gain = 10 ** ((target_db - speech_level_db(x)) / 20)
    x = x * min(gain, 8.0)
    x = speaker_safe(x, SR)
    pause = np.zeros(int(SR * float(s.get("cb_pause_s", 0.32))), np.float32)
    return np.concatenate([x, pause])


def to16k(x: np.ndarray, sr: int = SR) -> np.ndarray:
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32) if n else x[:0]


# ---------------------------------------------------------------- playback
class StreamPlayer:
    """One continuous output stream for a whole reply. Lines are appended as they
    become ready; each join gets an equal-power crossfade; `on_start` fires from
    the audio thread (via the event loop) when a line actually reaches the speaker."""

    def __init__(self, loop: asyncio.AbstractEventLoop, crossfade_ms: float, sr: int = SR, stream=None):
        self.loop, self.sr = loop, sr
        self.xf = int(sr * crossfade_ms / 1000)
        self.buf = np.zeros(0, np.float32)
        self.pos = 0
        self.lock = threading.Lock()
        self.marks: list[tuple[int, Callable[[], None]]] = []
        self.closed = False
        self._drained = asyncio.Event()
        if stream is None:
            import sounddevice as sd
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", blocksize=1024,
                                     latency="low", callback=self._callback)
        self.stream = stream
        self.stream.start()

    def _callback(self, outdata, frames, _time, _status) -> None:
        fire = []
        with self.lock:
            take = max(0, min(frames, len(self.buf) - self.pos))
            outdata[:take, 0] = self.buf[self.pos:self.pos + take]
            outdata[take:, 0] = 0
            self.pos += take
            while self.marks and self.marks[0][0] <= self.pos:
                fire.append(self.marks.pop(0)[1])
            empty = self.pos >= len(self.buf)
        for fn in fire:
            self.loop.call_soon_threadsafe(fn)
        if empty:
            self.loop.call_soon_threadsafe(self._drained.set)

    def remaining_s(self) -> float:
        with self.lock:
            return (len(self.buf) - self.pos) / self.sr

    def append(self, x: np.ndarray) -> int:
        """Queue a clip; returns the sample index where it starts."""
        x = np.asarray(x, np.float32).copy()
        n = min(self.xf, len(x) // 4)
        with self.lock:
            pending = len(self.buf) - self.pos
            if n > 0 and pending >= n:
                t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
                head = self.buf[:-n]
                mixed = self.buf[-n:] * np.cos(t) + x[:n] * np.sin(t)
                start = len(self.buf) - n
                self.buf = np.concatenate([head, mixed, x[n:]])
            else:
                if n > 0:
                    x[:n] *= np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32))
                start = len(self.buf)
                self.buf = np.concatenate([self.buf, x])
            self._drained.clear()
        return start

    async def play(self, x: np.ndarray, on_start: Callable[[], None]) -> None:
        start = self.append(x)
        with self.lock:
            if start <= self.pos:
                self.loop.call_soon(on_start)
            else:
                self.marks.append((start, on_start))
                self.marks.sort(key=lambda m: m[0])

    async def drain(self, cancel: asyncio.Event) -> None:
        while not cancel.is_set() and self.remaining_s() > 0:
            self._drained.clear()
            if self.remaining_s() <= 0:
                break
            try:
                await asyncio.wait_for(self._drained.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass
        if not cancel.is_set():
            await asyncio.sleep(0.12)               # let the device play out its last block

    def abort(self) -> None:
        with self.lock:
            self.buf, self.pos, self.marks = np.zeros(0, np.float32), 0, []
        self.close(hard=True)

    def close(self, hard: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            (self.stream.abort if hard else self.stream.stop)()
            self.stream.close()
        except Exception:
            pass


class FilePlayer:
    """Fallback: one file per line through the Voice's normal player (afplay)."""

    def __init__(self, voice):
        self.v = voice

    def remaining_s(self) -> float:
        return 0.0

    async def play(self, x: np.ndarray, on_start: Callable[[], None]) -> None:
        from .speech import write_wav
        f = paths.AUDIO_TMP / f"kobox-{uuid.uuid4().hex[:8]}.wav"
        write_wav(f, x, SR)
        on_start()
        try:
            await self.v._play_file(str(f), len(x) / SR)
        finally:
            f.unlink(missing_ok=True)

    async def drain(self, cancel) -> None:
        return None

    def abort(self) -> None:
        return None

    def close(self, hard: bool = False) -> None:
        return None


# ---------------------------------------------------------------- the engine
@dataclass
class Outcome:
    samples: np.ndarray | None
    reason: str = ""
    ready_at: float = 0.0
    wer: float | None = None
    heard: str = ""


@dataclass
class Report:
    started: float = field(default_factory=time.time)
    first_sound_s: float | None = None
    lines: list[dict] = field(default_factory=list)
    fallback: str = ""
    takes: list[str] = field(default_factory=list)


class Kobox:
    def __init__(self, voice):
        self.v = voice
        self.ears = None                    # set by the server: the shared Whisper instance
        self.lateness = 0.5                 # learned: how late Chatterbox lines are vs their turn (s)
        self.cb_failed_at = 0.0
        self.player = None
        self.last: dict = {}
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kobox-kokoro")
        self._make_player = None            # tests inject a fake player
        from .expressions import Bank
        self.bank = Bank()                  # real nonverbal takes in her voice (laughs, sighs, aww, hums...)

    async def prepare_bank(self) -> None:
        """Background: render missing takes and the weekly fresh ones."""
        try:
            n = await self.bank.fill(self.v.cb, self.ears)
            if n:
                log.info("expression bank: %d new takes (%s)", n, self.bank.summary())
        except Exception:
            log.exception("expression bank fill failed")

    def _bank_take(self, sound: str, level: float, s: dict, pause: float, rep: "Report") -> np.ndarray | None:
        got = self.bank.pick(sound)
        if got is None:
            return None
        take, x = got
        rep.takes.append(take)
        return match(x, level, {**s, "cb_pause_s": pause}, float(self.v.cfg.get("warmth", 0.35)))

    # -- engine health
    def cb_usable(self) -> bool:
        from .speech import Chatterbox
        if not Chatterbox.installed():
            return False
        if self.v.cb.broken:
            if not self.cb_failed_at:
                self.cb_failed_at = time.time()
            if time.time() - self.cb_failed_at < float(settings(self.v.cfg)["restart_every_s"]):
                return False
            log.info("kobox: retrying Chatterbox after %s", self.v.cb.broken)
            self.v.cb.broken, self.cb_failed_at = "", 0.0
        return True

    def grace(self, s: dict) -> float:
        g = float(s["grace_max_s"])
        if self.lateness > g:
            return 0.3                      # it won't make it anyway: don't hold silence for it
        return float(np.clip(self.lateness + 0.4, 0.3, g))

    def _learn(self, late: float) -> None:
        self.lateness = 0.7 * self.lateness + 0.3 * float(np.clip(late, -1.0, 10.0))

    def status(self) -> dict:
        return {"lateness_s": round(self.lateness, 2), "chatterbox": self.cb_usable(),
                "whisper": bool(self.ears and self.ears.available()), "last": self.last}

    # -- rendering
    def _kokoro(self, text: str, style, spd: float, mood: str) -> np.ndarray:
        from . import prosody
        text = re.sub(r"\[(laughs?|laughing|giggles?|chuckles?|hmm)\]", "[pause]", text, flags=re.I)
        if not prosody.strip_cues(text).strip(" .,!?…-"):
            return np.zeros(int(SR * 0.3), np.float32)      # a hum Kokoro can't perform: a short pause
        out = []
        for g in prosody.plan(text, mood):
            a, sr = self.v.render_group(g, style, spd)
            out.append(a if sr == SR else np.interp(np.linspace(0, len(a) - 1, int(len(a) * SR / sr)),
                                                    np.arange(len(a)), a).astype(np.float32))
        return np.concatenate(out) if out else np.zeros(1, np.float32)

    async def _chatterbox(self, ln: Line, s: dict, level: Callable[[], float]) -> Outcome:
        from .speech import read_wav
        f = str(paths.AUDIO_TMP / f"kobox-cb-{uuid.uuid4().hex[:8]}.wav")
        try:
            ok = await self.v.cb.render(ln.cb_text, f, still_needed=lambda: not ln.spoken)
            if ok is False:
                return Outcome(None, "no longer needed", time.time())
            x, sr = read_wav(f)
        except Exception as e:
            self.v.cb.broken = str(e)[:200]
            self.cb_failed_at = time.time()
            log.warning("kobox: chatterbox failed: %s", e)
            return Outcome(None, f"Chatterbox failed: {str(e)[:80]}", time.time())
        finally:
            if os.path.exists(f):
                os.unlink(f)
        if sr != SR:
            x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x).astype(np.float32)
        loop = asyncio.get_running_loop()
        y = await loop.run_in_executor(None, match, x, level(), s, float(self.v.cfg.get("warmth", 0.35)))
        if len(y) < SR * 0.1:
            return Outcome(None, "Chatterbox returned silence", time.time())
        if s.get("check", True) and self.ears is not None and self.ears.available():
            heard = await loop.run_in_executor(None, self.ears.verify, to16k(trim(x)))
            if heard is None:
                return Outcome(None, "accuracy check busy (listening)", time.time())
            ok, wer = judge(ln.text, heard, s, set(spoken_lexicon()))
            if not ok:
                return Outcome(None, "accuracy check failed", time.time(), wer, heard)
            return Outcome(y, "", time.time(), wer, heard)
        return Outcome(y, "", time.time())

    async def _performed(self, ln: Line, s: dict, level: Callable[[], float]) -> Outcome:
        """Chatterbox's take, re-rendered if the accuracy check rejected it (garbled words
        are worse than a plain line, so after the retries Kokoro's version plays)."""
        out = await self._chatterbox(ln, s, level)
        for _ in range(int(s.get("retries", 1))):
            if out.reason != "accuracy check failed" or ln.spoken:
                break
            log_rejection(ln.text, out.heard, out.wer)
            out = await self._chatterbox(ln, s, level)
        return out

    # -- speaking
    async def speak(self, text: str, on_chunk: Callable[[list[float], float], Awaitable[None]] | None = None,
                    voice: str | None = None, mood: str = "neutral") -> bool | None:
        from .speech import envelope
        v, s = self.v, settings(self.v.cfg)
        loop = asyncio.get_running_loop()
        rep = Report()
        try:
            lines = plan(text, mood, s, spoken_lexicon())
        except Exception as e:
            log.exception("kobox plan failed")
            self.last = {"fallback": f"planning failed: {e}"[:200]}
            return None
        if not lines:
            return True
        use_cb = self.cb_usable()
        style = await loop.run_in_executor(None, v._mood_style, voice, mood)
        spd = float(v.cfg.get("speed", 1.0))
        levels: list[float] = []

        def level() -> float:
            return float(np.mean(levels[-4:])) if levels else -19.0

        # the expression bank: a line that is only a sound plays a real take at once; an expressive
        # line's Kokoro backup leaves its lead/tail sound to a real take instead of reading it out
        from .expressions import parts
        for ln in lines:
            if ln.hum:
                ln.instant = self.bank.has(ln.cb_text)
                ln.lead = ln.cb_text if ln.instant else None
                continue
            opening = ln.idx == 0 and ln.reason.startswith("first line")
            if not ln.expressive and not opening:
                continue
            lead, core, tail = parts(ln.text)
            ln.lead = lead if self.bank.has(lead) else None
            ln.tail = tail if self.bank.has(tail) else None
            from .prosody import strip_cues
            if (ln.lead or ln.tail) and not strip_cues(core).strip(" .,!?…-"):
                ln.instant = True
            elif ln.lead or ln.tail:
                ln.kk_text = core if ln.kk_text is None else ln.kk_text

        # Kokoro for every line, in order, on one thread (expressive ones are the backup)
        kk = [asyncio.wrap_future(self.pool.submit(self._kokoro, ln.text if ln.kk_text is None else ln.kk_text, style, spd, mood)) for ln in lines]
        cb: dict[int, asyncio.Task] = {}
        if use_cb:
            for ln in lines:
                if ln.expressive and not ln.instant:
                    cb[ln.idx] = asyncio.ensure_future(self._performed(ln, s, level))
        need_at: dict[int, float] = {}

        def learn_when_done(i: int, t: asyncio.Task) -> None:
            if t.cancelled() or t.exception() or i not in need_at:
                return
            self._learn(t.result().ready_at - need_at[i])
        for i, t in cb.items():
            t.add_done_callback(lambda t, i=i: learn_when_done(i, t))

        try:
            self.player = (self._make_player or self._new_player)(loop, s)
        except Exception as e:
            log.warning("kobox: output stream unavailable (%s); playing line by line", e)
            self.player = FilePlayer(v)
        player = self.player
        v._cancel.clear()
        v.speaking = True
        started = False
        try:
            for i, ln in enumerate(lines):
                if v._cancel.is_set():
                    return False
                x = None
                if ln.instant:
                    sound = ln.lead or ln.tail
                    x = await loop.run_in_executor(None, self._bank_take, sound, level(), s,
                                                   float(s["cb_pause_s"]), rep)
                    ln.engine = "bank" if x is not None else ln.engine
                elif ln.expressive and i in cb:
                    need_at[i] = time.time() + player.remaining_s()
                    wait = player.remaining_s() + (self.grace(s) if ln.optional
                                                   else float(s.get("expression_wait_s", 8.0)))
                    if not started:
                        wait = min(wait, float(s.get("start_wait_s", 1.5)))
                    if ln.lead or ln.tail:          # a real take is ready in the bank: don't hold a long silence
                        wait = min(wait, player.remaining_s() + float(s.get("bank_wait_s", 1.2)))
                    try:
                        out = await asyncio.wait_for(asyncio.shield(cb[i]), timeout=wait)
                        x, ln.reason = out.samples, out.reason
                        rep.lines.append({"i": i, "wer": out.wer, "reason": out.reason} if out.reason else {"i": i, "wer": out.wer})
                        if out.reason == "accuracy check failed":
                            log_rejection(ln.text, out.heard, out.wer)
                    except asyncio.TimeoutError:
                        ln.reason = f"Chatterbox not ready within {wait:.1f}s"
                    if x is not None:
                        ln.engine = "chatterbox"
                elif ln.expressive:
                    ln.reason = ln.reason or "Chatterbox unavailable"
                if x is None and (ln.lead or ln.tail):
                    # the live take didn't make it: her real laugh/sigh from the bank, Kokoro for the words
                    pre = (await loop.run_in_executor(None, self._bank_take, ln.lead, level(), s, 0.12, rep)
                           if ln.lead else None)
                    if pre is not None and not kk[i].done():
                        # her laugh starts now; the words follow as soon as Kokoro has them
                        def first(t=ln.text):
                            if rep.first_sound_s is None:
                                rep.first_sound_s = round(time.time() - rep.started, 3)
                            v._head = (lambda: t)
                        await player.play(pre, first)
                        started, pre = True, None
                        ln.engine = "bank"
                    words = await kk[i]
                    post = (await loop.run_in_executor(None, self._bank_take, ln.tail, level(), s,
                                                       float(s["cb_pause_s"]), rep) if ln.tail else None)
                    if pre is not None or post is not None or ln.engine == "bank":
                        x = np.concatenate([p for p in (pre, words, post) if p is not None])
                        ln.engine = "bank"
                if x is None:
                    x = await kk[i]
                    if (not ln.expressive or ln.engine != "chatterbox") and speech_level_db(x) > -60:
                        levels.append(speech_level_db(x))
                if v._cancel.is_set():
                    return False
                ln.spoken = True
                env, dur = envelope(x, SR), len(x) / SR
                rest = " ".join(l2.text for l2 in lines[i + 1:])

                def on_start(env=env, dur=dur, t=ln.text, rest=rest):
                    if rep.first_sound_s is None:
                        rep.first_sound_s = round(time.time() - rep.started, 3)
                    v._head, v._tail = (lambda: t), rest
                    if on_chunk:
                        asyncio.ensure_future(on_chunk(env, dur))
                await player.play(x, on_start)
                started = True
            await player.drain(v._cancel)
            return not v._cancel.is_set()
        except Exception as e:
            log.exception("kobox failed mid-reply")
            rep.fallback = f"Kobox error: {str(e)[:120]}"
            if not started:
                return None
            done = [ln for ln in lines if ln.spoken]
            rest = " ".join(ln.text for ln in lines[len(done):])
            await player.drain(v._cancel)
            return await v._speak_kokoro(rest, on_chunk, voice, mood) if rest else True
        finally:
            v.speaking = False
            player.close()
            self.player = None
            for ln in lines:
                ln.spoken = True                    # queued Chatterbox renders can be skipped now
            for t in cb.values():
                if not t.done():
                    t.add_done_callback(lambda t: None)
            self._record(lines, rep)

    def _new_player(self, loop, s):
        if not s.get("stream", True):
            return FilePlayer(self.v)
        return StreamPlayer(loop, float(s["crossfade_ms"]))

    def stop(self) -> None:
        p = self.player
        if p is not None:
            p.abort()

    def _record(self, lines: list[Line], rep: Report) -> None:
        exp = [ln for ln in lines if ln.expressive or ln.engine == "bank"]
        reasons = [ln.reason for ln in exp if ln.engine not in ("chatterbox", "bank") and ln.reason]
        self.last = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "lines": len(lines), "expressive": len(exp),
            "by_chatterbox": sum(1 for ln in exp if ln.engine == "chatterbox"),
            "by_bank": sum(1 for ln in exp if ln.engine == "bank"),
            "takes": list(rep.takes),
            "first_sound_s": rep.first_sound_s,
            "fallback": rep.fallback or (reasons[-1] if reasons else ""),
            "grace_s": round(self.grace(settings(self.v.cfg)), 2),
        }
        write_log({"kind": "reply", **self.last, "checks": rep.lines})

    def feedback(self, value: str, sound: str | None = None, takes: list[str] | None = None) -> dict:
        """The listener's reaction to the last reply (or to `takes`): 'up' uses those takes more,
        'down' retires them, 'less'/'more' changes how often expressions should be written
        (see prompt_guide). Returns what happened, with a sentence the host can say."""
        if value in ("less", "more"):
            lvl = self.bank.set_expressiveness(-1 if value == "less" else 1)
            return {"value": value, "expressiveness": lvl, "takes": 0, "retired": [],
                    "note": f"{'Fewer' if value == 'less' else 'More'} expressions from now on ({lvl}/4)."}
        if value not in ("up", "down"):
            raise ValueError("feedback is up, down, less or more")
        takes = list(self.last.get("takes", []) if takes is None else takes)
        had_bank_takes = bool(takes)
        if sound and SOUND_WORDS.get(sound, ()) is not None:
            want = SOUND_WORDS.get(sound) or (sound,)
            takes = [t for t in takes if (self.bank.index.get(t) or {}).get("sound") in want]
        what = f"that {sound}" if sound and sound not in ("expression", "sound") else "those"
        if not takes and had_bank_takes and sound:
            return {"value": value, "takes": 0, "retired": [], "live": True,
                    "note": f"Thanks. There wasn't a {sound} from my bank in that reply, so there's nothing stored to change."}
        if not takes:
            note = ("Thanks. That one was performed live, so there's no stored take to keep, but I'll keep doing it that way."
                    if value == "up" else "Got it. That one was performed live, so there's no stored take to drop.")
            return {"value": value, "takes": 0, "retired": [], "live": True, "note": note}
        retired = self.bank.feedback(takes, 0.3 if value == "up" else -0.6)
        note = (f"Glad you liked {what}. I'll use it more." if value == "up" else
                f"Got it, I've dropped {what}." if retired else f"Got it, I'll use {what} less.")
        return {"value": value, "takes": len(takes), "retired": retired, "note": note}

    def hear(self, user_text: str) -> dict | None:
        """Pass every message from the listener. Kobox learns from it:
        * a feedback phrase ('that laugh sounded off', 'fewer expressions') -> feedback()
        * a laugh back ('haha', 'lol', an emoji) within 2 minutes of a reply -> those takes score up
        Returns what it did (with a sentence the host may say), or None if the message was
        ordinary conversation, which the host then handles as usual."""
        fb = parse_feedback(user_text)
        if fb:
            value, sound = fb
            fresh = True
            if value in ("up", "down"):
                try:
                    fresh = (datetime.now() - datetime.fromisoformat(self.last.get("at", ""))).total_seconds() < 600
                except ValueError:
                    fresh = False
            if fresh:
                return {"kind": "feedback", **self.feedback(value, sound)}
            return None
        if self.last.get("takes") and re.search(r"\b(ha(ha)+|lol|lmao|hehe+)\b|😂|🤣|😆|😄", user_text or "", re.I):
            try:
                if (datetime.now() - datetime.fromisoformat(self.last["at"])).total_seconds() < 120:
                    self.bank.feedback(self.last["takes"], 0.15)
                    return {"kind": "laughed_back", "takes": len(self.last["takes"]), "note": ""}
            except (KeyError, ValueError):
                pass
        return None

    def label(self) -> str:
        """Shown under the spoken message: which voice, and how the expressions went."""
        d = self.last or {}
        if not d.get("lines"):
            return "Kobox"
        parts = ["Kobox"]
        if d.get("expressive"):
            done = d["by_chatterbox"] + d.get("by_bank", 0)
            parts.append(f"{done} of {d['expressive']} expressions performed"
                         + (f" ({d['by_bank']} from her expression bank)" if d.get("by_bank") else ""))
            if d.get("fallback") and done < d["expressive"]:
                parts.append(f"missed: {d['fallback']}")
        else:
            parts.append("no expressions in this reply")
        if d.get("first_sound_s") is not None:
            parts.append(f"started {d['first_sound_s']:.1f} s")
        return " · ".join(parts)


# ---------------------------------------------------------------- feedback (buttons, voice, any host app)
_SOUND = r"(?P<sound>laugh|laughter|laughing|giggle|chuckle|sigh|gasp|groan|hum|humming|aww|ooh|ugh|whoa|sniff|expression|sound)s?"
FEEDBACK_PATTERNS = [
    ("down", re.compile(r"\b(that|the|your|this)\s+(little\s+|last\s+)?" + _SOUND + r"\s+(sounded|sounds|was|is|felt|seemed)\s+"
                        r"(a bit\s+|really\s+|so\s+|too\s+|kind of\s+)?(off|weird|wrong|fake|bad|strange|robotic|creepy|awkward|"
                        r"forced|unnatural|cringe|odd)\b")),
    ("down", re.compile(r"\bdon'?t\s+(?P<sound>laugh|sigh|gasp|hum|giggle|groan)\s+like that\b")),
    ("up", re.compile(r"\b(that|the|your|this)\s+(little\s+|last\s+)?" + _SOUND + r"\s+(sounded|sounds|was|is|felt)\s+"
                      r"(really\s+|so\s+|very\s+)?(great|nice|good|perfect|lovely|natural|real|cute|amazing|beautiful)\b")),
    ("up", re.compile(r"\bi\s+(love|loved|like|liked)\s+(that|your|the)\s+(little\s+)?" + _SOUND + r"\b")),
    ("less", re.compile(r"\b(fewer|less)\s+(expressions?|laughing|laughs|sighs|reactions?|sound effects)\b|\btone (it|them|the expressions?) down\b|"
                        r"\bbe less expressive\b|\bless dramatic\b")),
    ("more", re.compile(r"\b(more expressive|more expressions?|more reactions?)\b|\bexpress yourself more\b")),
]
SOUND_WORDS = {"hum": ("Mm-hmm.", "Hmm.", "Mmm."), "expression": None, "sound": None}
_SOUND_ALIAS = {"laughter": "laugh", "laughing": "laugh", "giggle": "chuckle", "humming": "hum"}


def parse_feedback(text: str) -> tuple[str, str | None] | None:
    """'That laugh sounded off' -> ('down', 'laugh'); 'fewer expressions' -> ('less', None).
    Only short messages that match closely, so ordinary conversation never counts."""
    t = (text or "").strip().lower()
    words = len(t.split())
    if words > 12:
        return None
    for value, rx in FEEDBACK_PATTERNS:
        m = rx.search(t)
        if m:
            if value in ("up", "down") and words > 8:
                return None
            sound = m.groupdict().get("sound")
            return value, _SOUND_ALIAS.get(sound, sound)
    return None


def prompt_guide(expressiveness: int | None = None, about_user: str = "") -> str:
    """Instructions for the host assistant's LLM: the cues Kobox performs, how often to use
    them, and to respond to the user's emotions rather than mirror them. Append it to the
    system prompt."""
    from .expressions import EXPRESSIVENESS_TEXT
    lvl = 2 if expressiveness is None else int(np.clip(expressiveness, 0, 4))
    who = f" {about_user.strip()}" if about_user.strip() else ""
    return (
        "Your voice can perform expressions for real. Put a cue in square brackets right before the words it shapes.\n"
        "Delivery: [warmly] [softly] [seriously] [slowly] [playfully] [happy] [excited] [surprised] [sarcastic] [dramatic] "
        "[whispers] [nervously] [annoyed] [tearfully] [normal voice].\n"
        "Moments: [laughs] [chuckles] [sighs] [gasps] [groans] [sniffs] [clears throat] [shh] [hmm] [pause].\n"
        "Interjections are performed too; write them as words: Ooh, Aww, Ugh, Whoa, Pfft, Heh, Uh-oh, Yay, Oops, Phew, Mm-hmm.\n"
        f"Use them when you mean them: in a light or emotional exchange, {EXPRESSIVENESS_TEXT[lvl]}. Vary them. "
        "In serious work or bad news, use none. [annoyed], [nervously] and [tearfully] are for situations, never aimed at the user; "
        "sarcasm stays light and never at their expense.\n"
        f"Responding to the user's emotions{who}: don't copy them back. Respond the way a warm, emotionally intelligent person "
        "would. They laugh or joke: enjoy it with them, a real laugh or a fond chuckle. They sigh or sound tired: gentle and "
        "steady, [softly], no jokes. They're frustrated: on their side, a light [groans] at the situation at most, then help. "
        "They're angry: calm and steady, never angry back. They're excited or proud: genuinely happy for them, [happy], Ooh, Yay, "
        "a [gasps] at big news. They're sad: tender and quiet, [softly]. They're anxious: unhurried and confident. "
        "They tease: playful back, warm, never flirtatious.")


def judge(intended: str, heard: str, s: dict, names: set[str]) -> tuple[bool, float]:
    """Accept a Chatterbox line only if Whisper heard (close to) the intended words
    and every number and listed name in it."""
    from .prosody import strip_cues
    from .textproc import normalise_for_wer, required_words, word_error_rate
    plain = strip_cues(intended)
    wer = word_error_rate(plain, heard)
    missing = required_words(plain, names) - set(normalise_for_wer(heard))
    return (wer <= float(s.get("wer_max", 0.25)) and not missing), round(wer, 3)


def health_line(cfg: dict) -> str:
    """One line for `reha-pro doctor`: can Kobox run, and how did the last reply go."""
    from .hearing import Ears
    from .speech import Chatterbox
    chosen = cfg["voice"].get("engine") == "kobox"
    cb, ears = Chatterbox.installed(), Ears(cfg["hearing"]).available()
    last = {}
    for f in sorted(paths.LOGS.glob("kobox-*.jsonl"), reverse=True)[:2]:
        for row in reversed(f.read_text("utf-8").splitlines()):
            try:
                d = json.loads(row)
            except ValueError:
                continue
            if d.get("kind") == "reply":
                last = d
                break
        if last:
            break
    mark = "  ✓ " if cb and ears else "  · "
    parts = [f"Kobox voice {'(selected)' if chosen else '(not selected)'}",
             "Chatterbox ready" if cb else "needs Chatterbox (Install Expressive Voice.command)",
             "accuracy check ready" if ears else "accuracy check off (no Whisper model)"]
    try:
        from .expressions import Bank
        b = Bank().summary()
        parts.append(f"expression bank {b['takes']} takes for {b['sounds']} sounds"
                     + (f", {b['converted']} learned from recordings" if b["converted"] else "")
                     + f", expressiveness {b['expressiveness']}/4")
    except Exception:
        pass
    if last:
        parts.append(f"last reply {last.get('at', '')}: {last.get('by_chatterbox', 0)}/{last.get('expressive', 0)} "
                     f"expressive lines by Chatterbox, first sound {last.get('first_sound_s')} s"
                     + (f", fallback: {last['fallback']}" if last.get("fallback") else ""))
    return mark + "; ".join(parts)


# ---------------------------------------------------------------- local log (F5.5, F8.1)
def _log_dir():
    d = paths.LOGS
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_log(entry: dict) -> None:
    try:
        d = _log_dir()
        f = d / f"kobox-{datetime.now():%Y-%m-%d}.jsonl"
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        os.chmod(f, 0o600)
        cutoff = datetime.now() - timedelta(days=LOG_KEEP_DAYS)
        for old in d.glob("kobox-*.jsonl"):
            try:
                if datetime.strptime(old.stem[6:], "%Y-%m-%d") < cutoff:
                    old.unlink()
            except ValueError:
                pass
    except OSError:
        log.warning("kobox log write failed")


def redact(text: str) -> str:
    """Sensitive values (client names, emails, numbers on the privacy list) -> [kind],
    without creating placeholders in the privacy vault."""
    try:
        from .privacyfilter import find
        spans = find(text)
    except Exception:
        return "[withheld]"
    out, pos = [], 0
    for a, b, kind in spans:
        out.append(text[pos:a] + f"[{kind}]")
        pos = b
    return "".join(out) + text[pos:]


def log_rejection(intended: str, heard: str, wer: float | None) -> None:
    write_log({"kind": "rejected", "at": datetime.now().isoformat(timespec="seconds"),
               "intended": redact(intended), "heard": redact(heard), "wer": wer})
