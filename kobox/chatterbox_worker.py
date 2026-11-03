"""Chatterbox Turbo voice worker. Runs in its own venv (.runtime/chatterbox-venv),
never imports the reha package, and talks JSON lines over stdin/stdout:

  -> {"text": "[laugh] Okay...", "out": "/path/to.wav"}
  <- {"ok": true, "seconds": 2.4}      or {"ok": false, "error": "..."}

The first line it prints is {"ready": true, "sr": 24000} once the model is
loaded. It clones REHA's voice from a reference clip (argv[1]) so she sounds
like herself, and exits after `idle` seconds without work (argv[2]; 0 = never)
to give the memory back. REHA passes 0: a cold reload takes ~35 s.
"""
import json
import os
import select
import sys
import time


def main() -> None:
    ref, idle = sys.argv[1], float(sys.argv[2] if len(sys.argv) > 2 else 600)
    out = sys.stdout
    sys.stdout = sys.stderr            # library chatter must not corrupt the protocol
    import warnings
    warnings.filterwarnings("ignore")
    import soundfile as sf
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=dev)
    model.prepare_conditionals(ref)
    out.write(json.dumps({"ready": True, "sr": model.sr, "device": dev}) + "\n")
    out.flush()
    last = time.time()
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 5)
        if not r:
            if idle > 0 and time.time() - last > idle:
                return
            continue
        line = sys.stdin.readline()
        if not line:
            return
        last = time.time()
        try:
            job = json.loads(line)
            with torch.inference_mode():
                # top_k 50 measured ~25% faster than the default 1000 on an M1 Pro
                wav = model.generate(job["text"], top_k=50, top_p=1.0).squeeze(0).numpy()
            tmp = job["out"] + ".part"
            sf.write(tmp, wav, model.sr, format="WAV")
            os.replace(tmp, job["out"])
            res = {"ok": True, "seconds": round(len(wav) / model.sr, 2)}
        except Exception as e:  # report and keep serving
            res = {"ok": False, "error": str(e)[:300]}
        out.write(json.dumps(res) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
