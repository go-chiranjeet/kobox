"""Expression import, stage 1 (runs in the Chatterbox venv): someone else's expressions -> REHA's voice.

    python -I reha/expression_convert.py <recording> <out_dir> <reha_reference.wav>

Cuts the recording into short vocal moments (0.2-2.5 s, split at silences), converts
each with Chatterbox's voice converter into REHA's voice (the target is her reference
clip, so her voice never changes: only the expression is borrowed), and writes
<out_dir>/NN.wav plus index.json. The original moments are not kept.
Stage 2 (expressions.import_converted) checks and classifies them into her bank.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_mono(path: str, sr: int) -> np.ndarray:
    import librosa
    x, _ = librosa.load(path, sr=sr, mono=True)
    return x.astype(np.float32)


def moments(x: np.ndarray, sr: int, lo=0.2, hi=2.5) -> list[tuple[int, int]]:
    """Voiced stretches between silences (frames 25 dB under the loudest are silence)."""
    hop = int(sr * 0.02)
    n = len(x) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    on = db > db.max() - 25
    out, i = [], 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j < n and gap < 8:                 # bridge pauses shorter than 160 ms
            gap = 0 if on[j] else gap + 1
            j += 1
        a, b = max(0, i - 2) * hop, min(n, j - gap + 2) * hop
        if lo * sr <= b - a <= hi * sr:
            out.append((a, b))
        elif b - a > hi * sr:                    # long stretch (singing, talking): cut into pieces
            for s in range(a, b - int(lo * sr), int(hi * sr)):
                out.append((s, min(b, s + int(hi * sr))))
        i = j
    return out


def sung(x: np.ndarray, sr: int) -> bool:
    """A held musical note: long voiced stretches whose pitch stays on a note (singing),
    unlike a laugh, sigh or spoken 'ooh', whose pitch slides or breaks up."""
    import librosa
    f0, voiced, _ = librosa.pyin(x, fmin=70, fmax=1000, sr=sr, frame_length=1024, hop_length=160)
    if voiced is None or voiced.mean() < 0.5:
        return False                              # mostly breath/noise: laughs, sighs, gasps
    semis = 12 * np.log2(f0[voiced] / 110.0)
    # longest run where pitch stays within half a semitone of where it started
    best, start = 1, 0
    for i in range(1, len(semis)):
        if abs(semis[i] - semis[start]) >= 0.5:
            start = i
        best = max(best, i - start + 1)
    return best * 160 / sr >= 0.35


def main(src: str, out: str, ref: str) -> None:
    from chatterbox.vc import ChatterboxVC
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    snap = Path(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/*"))[0])
    vc = ChatterboxVC.from_local(snap, dev)
    vc.set_target_voice(ref)
    os.makedirs(out, exist_ok=True)
    x = load_mono(src, 16000)
    rows = []
    for k, (a, b) in enumerate(moments(x, 16000)):
        if sung(x[a:b], 16000):
            print(json.dumps({"skipped": "sung (held musical pitch)", "start_s": round(a / 16000, 2)}), flush=True)
            continue
        tmp = os.path.join(out, f"_src{k}.wav")
        sf.write(tmp, x[a:b], 16000)
        try:
            with torch.inference_mode():
                w = vc.generate(tmp).squeeze(0).cpu().numpy()
        finally:
            os.unlink(tmp)                       # the original voice is not kept
        f = f"{k:03d}.wav"
        sf.write(os.path.join(out, f), w, vc.sr)
        rows.append({"file": f, "start_s": round(a / 16000, 2), "dur_s": round((b - a) / 16000, 2), "sr": vc.sr})
        print(json.dumps(rows[-1]), flush=True)
    json.dump({"source": os.path.basename(src), "moments": rows}, open(os.path.join(out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
