"""Command line and local service.

    python -m kobox say "Okay. [laughs] You actually did it." [--mood amused]
    python -m kobox serve [--port 7766]          # local HTTP service for any assistant, any language
    python -m kobox prompt                       # the expressions guide to add to your LLM's system prompt
    python -m kobox feedback down --sound laugh  # or: up | less | more  (acts on the last reply)
    python -m kobox doctor
    python -m kobox update-bank                  # after `git pull`: add her newer shipped takes, keep your scores
    python -m kobox fill-bank                    # render more takes (the shipped bank already works)
    python -m kobox learn recording.wav          # someone else's laughs/sighs -> expressions in her voice

Settings: ~/.kobox/config/kobox.json (optional) {"voice": {...}, "hearing": {...}} overrides the
defaults below, which are exactly how REHA runs Kobox.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

# REHA's own settings: the defaults make Kobox sound exactly as it does in REHA
VOICE = {"enabled": True, "voice": "hf_beta:0.6+af_heart:0.4", "speed": 1.0, "warmth": 0.25, "expressive": True,
         "lang": "en-us", "max_spoken_chars": 0, "player": "afplay", "engine": "kobox"}
HEARING = {"model_file": "ggml-large-v3-turbo-q5_0.bin", "language": "en", "threads": 6, "unload_after_minutes": 10}


def _config() -> tuple[dict, dict]:
    from . import paths
    voice, hearing = dict(VOICE), dict(HEARING)
    try:
        cfg = json.loads((paths.CONFIG_DIR / "kobox.json").read_text("utf-8"))
        voice.update(cfg.get("voice", {}))
        hearing.update(cfg.get("hearing", {}))
    except (OSError, ValueError):
        pass
    voice["engine"] = "kobox"
    return voice, hearing


def _voice():
    from .hearing import Ears
    from .speech import Voice
    voice, hearing = _config()
    v = Voice(voice)
    v.kobox.ears = Ears(hearing)
    return v


def _last_reply_takes() -> list[str]:
    from . import paths
    for f in sorted(paths.LOGS.glob("kobox-*.jsonl"), reverse=True):
        for row in reversed(f.read_text("utf-8").splitlines()):
            try:
                d = json.loads(row)
            except ValueError:
                continue
            if d.get("kind") == "reply":
                return d.get("takes") or []
    return []


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


# ---------------------------------------------------------------- local service
MAX_BODY = 64 * 1024
REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
           415: "Unsupported Media Type"}


async def _serve(port: int, token: str) -> None:
    """HTTP on 127.0.0.1 only. Browsers can't use it: requests with an Origin or Referer
    header, or a non-JSON body, are refused, so a web page can't make Kobox speak or
    change her bank."""
    from .engine import parse_feedback, prompt_guide
    v = _voice()
    await v.warm()
    lock = asyncio.Lock()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        code, out = 200, {}
        try:
            head = (await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)).decode("latin-1")
            first, *rest = head.split("\r\n")
            method, path, _ = first.split(" ", 2)
            hdr = {k.strip().lower(): val.strip() for k, _, val in (ln.partition(":") for ln in rest if ln)}
            n = int(hdr.get("content-length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            raw = await asyncio.wait_for(reader.readexactly(n), 10) if n else b""
            if "origin" in hdr or "referer" in hdr:
                code, out = 403, {"error": "browser requests are not accepted"}
            elif token and hdr.get("authorization") != f"Bearer {token}":
                code, out = 401, {"error": "missing or wrong token"}
            elif method == "POST" and not hdr.get("content-type", "").startswith("application/json"):
                code, out = 415, {"error": "send JSON (Content-Type: application/json)"}
            else:
                body = json.loads(raw or b"{}")
                if method == "GET" and path == "/status":
                    out = {"engine": v.engine(), "bank": v.kobox.bank.summary(), "last": v.kobox.last,
                           "speaking": bool(v.speaking)}
                elif method == "GET" and path == "/prompt":
                    out = {"prompt": prompt_guide(v.kobox.bank.prefs.get("expressiveness", 2))}
                elif method == "POST" and path == "/speak":
                    text = str(body.get("text", ""))[:20000]
                    async with lock:
                        done = await v.speak(text, mood=str(body.get("mood", "neutral")))
                    out = {"done": done, "label": v.kobox.label(), "report": v.kobox.last}
                elif method == "POST" and path == "/heard":
                    out = v.kobox.hear(str(body.get("text", ""))[:2000]) or {"kind": "conversation"}
                elif method == "POST" and path == "/stop":
                    await v.stop()
                    out = {"stopped": True}
                elif method == "POST" and path == "/feedback":
                    value, sound = body.get("value"), body.get("sound")
                    if body.get("text"):
                        parsed = parse_feedback(str(body["text"]))
                        if not parsed:
                            out = {"matched": False}
                        else:
                            value, sound = parsed
                    if not out:
                        out = {"matched": True, **v.kobox.feedback(str(value), sound)}
                else:
                    code, out = 404, {"error": "unknown endpoint"}
        except Exception as e:                       # malformed request: answer, never crash the service
            code, out = 400, {"error": str(e)[:200]}
        data = json.dumps(out).encode()
        writer.write(f"HTTP/1.1 {code} {REASONS[code]}\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
        try:
            await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    print(f"Kobox listening on http://127.0.0.1:{port}" + ("  (token required)" if token else ""), flush=True)
    async with server:
        await server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(prog="kobox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("say")
    s.add_argument("text")
    s.add_argument("--mood", default="neutral")
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=7766)
    sv.add_argument("--token", default=os.environ.get("KOBOX_TOKEN", ""), help="or set KOBOX_TOKEN")
    pr = sub.add_parser("prompt")
    pr.add_argument("--about-user", default="")
    fb = sub.add_parser("feedback")
    fb.add_argument("value", choices=["up", "down", "less", "more"])
    fb.add_argument("--sound", default=None)
    sub.add_parser("doctor")
    sub.add_parser("update-bank")
    sub.add_parser("fill-bank")
    ln = sub.add_parser("learn")
    ln.add_argument("recording")
    a = ap.parse_args()
    if a.cmd == "say":
        asyncio.run(_say(a.text, a.mood))
    elif a.cmd == "serve":
        asyncio.run(_serve(a.port, a.token))
    elif a.cmd == "prompt":
        from .engine import prompt_guide
        from .expressions import Bank
        print(prompt_guide(Bank().prefs.get("expressiveness", 2), a.about_user))
    elif a.cmd == "feedback":
        from .engine import Kobox
        from .expressions import Bank
        k = Kobox.__new__(Kobox)                      # feedback needs only the bank and the last reply
        k.bank, k.last = Bank(), {"takes": _last_reply_takes()}
        print(json.dumps(Kobox.feedback(k, a.value, a.sound)))
    elif a.cmd == "doctor":
        from .engine import health_line
        voice, hearing = _config()
        print(health_line({"voice": voice, "hearing": hearing}))
    elif a.cmd == "update-bank":
        from .expressions import Bank
        print(f"added {Bank().seed(merge=True)} new takes from the shipped bank")
    elif a.cmd == "fill-bank":
        asyncio.run(_fill())
    elif a.cmd == "learn":
        from .expressions import import_recording
        from .hearing import Ears
        for row in import_recording(a.recording, Ears(_config()[1])):
            print(json.dumps(row))


if __name__ == "__main__":
    main()
