"""REHA's ears: whisper.cpp (Metal) via pywhispercpp, fully on-device.
The model loads on first use and unloads after idle minutes to keep RAM free."""
from __future__ import annotations

import io
import logging
import os
import threading
import time
import wave

import numpy as np

from . import paths

log = logging.getLogger("reha.hearing")
SR = 16000


def decode_wav(data: bytes) -> np.ndarray:
    """WAV bytes (PCM16/float32, any rate, mono/stereo) -> float32 mono 16 kHz."""
    with wave.open(io.BytesIO(data)) as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SR and len(x):
        t_old = np.linspace(0, 1, num=len(x), endpoint=False)
        t_new = np.linspace(0, 1, num=int(len(x) * SR / rate), endpoint=False)
        x = np.interp(t_new, t_old, x).astype(np.float32)
    return x


class Ears:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        # learned speed: seconds of work per second of audio, and model load time (cold vs warm)
        self.rtf = 0.15
        self.load_s = 1.0
        self.progress = 0.0           # 0..1 for the transcription in flight

    @property
    def model_path(self) -> str:
        return str(paths.MODELS / self.cfg["model_file"])

    def available(self) -> bool:
        return os.path.exists(self.model_path)

    def loaded(self) -> bool:
        return self._model is not None

    def _get(self):
        if self._model is None:
            from pywhispercpp.model import Model
            t = time.time()
            self._model = Model(self.model_path, n_threads=int(self.cfg.get("threads", 6)),
                                print_progress=False, print_realtime=False, print_timestamps=False,
                                redirect_whispercpp_logs_to=None)
            self.load_s = time.time() - t
            log.info("whisper loaded in %.1fs", self.load_s)
        return self._model

    def estimate(self, seconds: float) -> float:
        """Expected wall time to transcribe this much audio, from what this Mac has done before."""
        return (0 if self.loaded() else max(0.7, self.load_s)) + 0.35 + seconds * self.rtf

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < SR * 0.3:
            return ""
        dur = len(audio) / SR
        self.progress = 0.0

        def on_segment(seg) -> None:            # real progress: where whisper has got to in the audio
            self.progress = max(self.progress, min(0.99, (seg.t1 / 100) / dur))

        with self._lock:
            m = self._get()
            self._last_used = time.time()
            t = time.time()
            segs = m.transcribe(audio.astype(np.float32), language=self.cfg.get("language", "en"),
                                **({"initial_prompt": self.cfg["vocabulary"]} if self.cfg.get("vocabulary") else {}),
                                no_context=True, suppress_blank=True, new_segment_callback=on_segment)
            took = time.time() - t
            self.rtf = 0.7 * self.rtf + 0.3 * max(0.02, took / max(1.0, dur))   # keep learning this Mac's speed
            self.progress = 1.0
            self._last_used = time.time()
        text = " ".join(s.text.strip() for s in segs).strip()
        # whisper's stock hallucinations on silence
        if text.lower().strip(" .!") in {"", "thank you", "thanks for watching", "you", "bye"}:
            return ""
        return text

    def verify(self, audio: np.ndarray, wait: float = 0.25) -> str | None:
        """Kobox's accuracy check: what Whisper hears in REHA's own synthesised line
        (16 kHz float32). Live listening has priority, so this never queues behind
        a transcription in flight: if the model is busy for more than `wait`
        seconds it returns None (unchecked) and the caller treats the line as late.
        It doesn't touch the progress bar or the learned speed of real listening."""
        if len(audio) < SR * 0.2 or not self.available():
            return ""
        if not self._lock.acquire(timeout=wait):
            return None
        try:
            m = self._get()
            self._last_used = time.time()
            segs = m.transcribe(audio.astype(np.float32), language=self.cfg.get("language", "en"),
                                **({"initial_prompt": self.cfg["vocabulary"]} if self.cfg.get("vocabulary") else {}),
                                no_context=True, suppress_blank=True)
        finally:
            self._lock.release()
        return " ".join(s.text.strip() for s in segs).strip()

    def maybe_unload(self) -> bool:
        mins = float(self.cfg.get("unload_after_minutes", 15))
        if self._model is not None and time.time() - self._last_used > mins * 60:
            with self._lock:
                self._model = None
            log.info("whisper unloaded after idle")
            return True
        return False
