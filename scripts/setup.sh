#!/usr/bin/env bash
# Kobox setup for macOS on Apple Silicon (also works on Linux with CPU-only Chatterbox).
#   ./scripts/setup.sh            -> .venv (Kobox), ~/.kobox/.runtime/chatterbox-venv, models (~700 MB)
set -euo pipefail
HOME_DIR="${KOBOX_HOME:-$HOME/.kobox}"
RT="$HOME_DIR/.runtime"; MODELS="$RT/models"; mkdir -p "$MODELS"
PY312="${PY312:-python3.12}"; PY311="${PY311:-python3.11}"

echo "1/3  Kobox environment (.venv, Python 3.12)"
"$PY312" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev]"

echo "2/3  Chatterbox environment ($RT/chatterbox-venv, Python 3.11, torch)"
"$PY311" -m venv "$RT/chatterbox-venv"
"$RT/chatterbox-venv/bin/pip" install -q --upgrade pip
"$RT/chatterbox-venv/bin/pip" install -q "chatterbox-tts==0.1.7" "librosa>=0.11" soundfile

echo "3/3  Models"
get() { [ -s "$MODELS/$2" ] || curl -L --fail -o "$MODELS/$2" "$1"; }
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx kokoro-v1.0.int8.onnx
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin voices-v1.0.bin
get https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin ggml-large-v3-turbo-q5_0.bin
# Chatterbox Turbo downloads itself from Hugging Face (ResembleAI/chatterbox-turbo) on first use.

echo "Done. Try:  .venv/bin/python -m kobox doctor   then   .venv/bin/python -m kobox say \"Okay. [laughs] You actually did it.\""
