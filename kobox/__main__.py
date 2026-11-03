"""Command line.

    python -m kobox say "Okay. [laughs] You actually did it." [--mood amused]
    python -m kobox doctor
    python -m kobox fill-bank            # render the expression bank (first run: ~2 min)
    python -m kobox learn recording.wav  # someone else's laughs/sighs -> expressions in her voice
"""
from __future__ import annotations

import argparse
import asyncio
import json

VOICE = {"enabled": True, "voice": "hf_beta:0.6+af_heart:0.4", "speed": 1.0, "warmth": 0.25, "expressive": True,
         "lang": "en-us", "max_spoken_chars": 0, "player": "afplay", "engine": "kobox"}
HEARING = {"model_file": "ggml-large-v3-turbo-q5_0.bin", "language": "en", "threads": 6, "unload_after_minutes": 10}


def _voice():
    from .hearing import Ears
    from .speech import Voice
    v = Voice(dict(VOICE))
    v.kobox.ears = Ears(dict(HEARING))
    return v


async def _say(text: str, mood: str) -> None:
    v = _voice()
    await v.warm()
    await v.speak(text, mood=mood)
    print(v.kobox.label())
    await v.cb.close()


async def _fill() -> None:
    v = _voice()
    await v.warm()
    n = await v.kobox.bank.fill(v.cb, v.kobox.ears)
    print(f"added {n} takes:", json.dumps(v.kobox.bank.summary()))
    await v.cb.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="kobox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("say")
    s.add_argument("text")
    s.add_argument("--mood", default="neutral")
    sub.add_parser("doctor")
    sub.add_parser("fill-bank")
    ln = sub.add_parser("learn")
    ln.add_argument("recording")
    a = ap.parse_args()
    if a.cmd == "say":
        asyncio.run(_say(a.text, a.mood))
    elif a.cmd == "doctor":
        from .engine import health_line
        print(health_line({"voice": VOICE, "hearing": HEARING}))
    elif a.cmd == "fill-bank":
        asyncio.run(_fill())
    elif a.cmd == "learn":
        from .expressions import import_recording
        from .hearing import Ears
        for row in import_recording(a.recording, Ears(dict(HEARING))):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
