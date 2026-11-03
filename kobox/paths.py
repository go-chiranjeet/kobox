"""Where Kobox keeps its models, logs, expression bank and settings.

Everything lives under KOBOX_HOME (default ~/.kobox), laid out like the app Kobox
was built in, so the modules are the same code that runs there.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
HOME = Path(os.environ.get("KOBOX_HOME") or Path.home() / ".kobox")
RUNTIME = HOME / ".runtime"
MODELS = RUNTIME / "models"             # kokoro-v1.0.int8.onnx, voices-v1.0.bin, ggml-large-v3-turbo-q5_0.bin
LOGS = RUNTIME / "logs"
STATE = RUNTIME / "state"               # expression bank lives in state/expressions
AUDIO_TMP = RUNTIME / "audio"
CONFIG_DIR = HOME / "config"            # optional pronunciations.json

# Chatterbox runs in its own Python environment (torch); point at it if it isn't the default
CB_PYTHON = Path(os.environ.get("KOBOX_CHATTERBOX_PYTHON") or RUNTIME / "chatterbox-venv" / "bin" / "python")
# the voice Chatterbox speaks in (and the target for converted expressions)
VOICE_REF = Path(os.environ.get("KOBOX_VOICE_REF") or PKG / "voices" / "reha-ref.wav")


def ensure_dirs() -> None:
    for d in (MODELS, LOGS, STATE, AUDIO_TMP, CONFIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
