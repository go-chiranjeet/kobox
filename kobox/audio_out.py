"""Kobox's audio output, in its own process (python -I audio_out.py <samplerate> <crossfade_ms>).

Why a separate process: Kokoro's rendering holds Python's interpreter lock for up to ~200 ms.
An audio callback in the same process then misses its deadline and the speaker runs dry
(a stutter). Here nothing else runs, so playback is never starved.

stdin, binary frames:   b"A" + uint32 id + uint32 n + n float32 samples   append a clip
                        b"X"                                                 abort: silence now, drop all
                        b"Q"                                                 quit
stdout, text lines:     "M <id>"     the clip has reached the speaker (line-start marker)
                        "R <n>"      samples still queued (sent ~10x a second while playing)
                        "E <id> <glitches> <worst_late_ms>"   queue empty; <id> is the last clip
                                     finished; glitches = device underflows or callbacks late by
                                     more than one block since the last "E"
Every clip goes into ONE queue, so two replies can never play over each other.
KOBOX_SILENT=1 plays zeros (tests use the real device without making a sound).
The stream opens on the first clip and closes after a short idle, so the Mac can still sleep.
"""
import struct
import sys
import threading
import time
from collections import deque

import os

import numpy as np
import sounddevice as sd

SR = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
XF = int(SR * float(sys.argv[2] if len(sys.argv) > 2 else 20) / 1000)
IDLE_CLOSE_S = 20.0             # keep the device open between turns of a conversation
BLOCK = 512
SILENT = os.environ.get("KOBOX_SILENT") == "1"

lock = threading.Lock()
queue: deque = deque()          # [array, read offset, clip id or None]
queued = 0                      # samples not yet played
out_lock = threading.Lock()
stream = None
last_audio = time.monotonic()
reported_empty = True
done_id = 0                     # last clip fully played
glitches, worst_late, last_cb = 0, 0.0, None


def say(line: str) -> None:
    with out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def callback(outdata, frames, _t, status) -> None:
    global queued, last_audio, reported_empty, done_id, glitches, worst_late, last_cb
    now = time.monotonic()
    if last_cb is not None and queued > 0:
        late = now - last_cb - frames / SR
        worst_late = max(worst_late, late)
        if late > frames / SR or (status and status.output_underflow):
            glitches += 1
    last_cb = now
    marks = []
    filled = 0
    with lock:
        while filled < frames and queue:
            item = queue[0]
            arr, off, cid, _ = item
            if cid is not None and off == 0:
                marks.append(cid)
                item[2] = None
            take = min(frames - filled, len(arr) - off)
            outdata[filled:filled + take, 0] = arr[off:off + take]
            filled += take
            item[1] = off + take
            if item[1] >= len(arr):
                done_id = queue.popleft()[3]
        queued -= filled
        left = queued
    if filled < frames:
        outdata[filled:, 0] = 0.0
    if SILENT:
        outdata[:] = 0.0
    if filled:
        last_audio = time.monotonic()
    for cid in marks:
        say(f"M {cid}")
    if left <= 0 and not reported_empty:
        reported_empty = True
        say(f"E {done_id} {glitches} {worst_late * 1000:.1f}")
        glitches, worst_late = 0, 0.0


def ensure_stream() -> None:
    global stream
    if stream is None:
        stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK,
                                 latency="low", callback=callback)
        stream.start()


def append(cid: int, x: np.ndarray) -> None:
    """Queue a clip; an equal-power crossfade with what's still queued, else a short fade-in."""
    global queued, reported_empty
    x = x.astype(np.float32, copy=True)
    n = min(XF, len(x) // 4)
    with lock:
        tail = queue[-1] if queue else None
        if n > 0 and tail is not None and len(tail[0]) - tail[1] >= n:
            t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
            arr = tail[0]
            arr[-n:] = arr[-n:] * np.cos(t) + x[:n] * np.sin(t)
            x = x[n:]
            queue.append([x, 0, cid, cid])         # its marker fires when it reaches the speaker
        else:
            if n > 0:
                x[:n] *= np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32))
            queue.append([x, 0, cid, cid])
        queued += len(x)
        reported_empty = False
    ensure_stream()


def abort() -> None:
    global queued, reported_empty, done_id
    with lock:
        if queue:
            done_id = max(done_id, max(item[3] for item in queue))
        queue.clear()
        queued = 0
    if not reported_empty:
        reported_empty = True
        say(f"E {done_id} {glitches} {worst_late * 1000:.1f}")


def reporter() -> None:
    """Queue level ~10x a second while playing; close the stream when idle."""
    global stream
    while True:
        time.sleep(0.1)
        with lock:
            left = queued
        if left > 0:
            say(f"R {left}")
        elif stream is not None and time.monotonic() - last_audio > IDLE_CLOSE_S:
            try:
                stream.stop()
                stream.close()
            finally:
                stream = None


def read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def main() -> None:
    threading.Thread(target=reporter, daemon=True).start()
    say("READY")
    try:
        while True:
            op = read_exact(1)
            if op == b"A":
                cid, n = struct.unpack("<II", read_exact(8))
                append(cid, np.frombuffer(read_exact(4 * n), dtype="<f4"))
            elif op == b"X":
                abort()
            elif op == b"Q":
                break
    except EOFError:
        pass
    finally:
        abort()
        if stream is not None:
            stream.abort()
            stream.close()


if __name__ == "__main__":
    main()
