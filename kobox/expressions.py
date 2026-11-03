"""REHA's expression bank (Kobox E2/E3): real nonverbal moments in her own voice.

Short sounds (laughs, sighs, gasps, "aww", "ugh", hums...) are rendered by Chatterbox
ahead of time, checked, matched and stored as several takes each. Kobox plays them
instantly, picks a different take each time, and uses them whenever a live
Chatterbox line isn't ready (a real laugh plus Kokoro's words, never a fake one).

The bank evolves: takes the owner likes are used more, disliked ones are retired,
and fresh takes are rendered every week. Takes converted from recordings the owner
provides (someone else's expressions, turned into REHA's voice) join as candidates.

Stored in .runtime/state/expressions/ (REHA's voice only; never the owner's audio).
"""
from __future__ import annotations

import json
import os
import logging
import random
import re
import time
import uuid
from pathlib import Path

import numpy as np

from . import paths

log = logging.getLogger("reha.expressions")
SR = 24000

# sound -> (Chatterbox prompts tried in turn, what Whisper may hear (regex) or None for anything)
SOUNDS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "laugh": (("[laugh]", "[laugh] Ha."), None),
    "chuckle": (("[chuckle] Heh.", "[chuckle]"), None),
    "sigh": (("[sigh] Mm.", "[sigh] Hm."), None),
    "gasp": (("[gasp] Oh!",), r"\boh"),
    "groan": (("[groan]", "[groan] Ugh."), None),
    "sniff": (("[sniff]",), None),
    "clear_throat": (("[clear throat]",), None),
    "shush": (("[shush] Shh.",), r"sh"),
    "Mm-hmm.": (("Mm-hmm.",), None),
    "Hmm.": (("Hmm.",), None),
    "Mmm.": (("Mmm.",), None),
    "aww": (("Aww.", "Awww."), r"\baw"),
    "ooh": (("Ooh.", "Oooh."), r"\boo"),
    "ugh": (("Ugh.", "Ughh."), r"\b(ugh|uh|ah)"),
    "whoa": (("Whoa.", "Whoa!"), r"\bwho?a"),
    "yay": (("Yay!",), r"\byay"),
    "phew": (("Phew.",), r"\b(phew|few)"),
    "oops": (("Oops.",), r"\boops"),
    "uh-oh": (("Uh-oh.",), r"\buh"),
    "heh": (("Heh.",), r"\b(heh|he)"),
}
# a cue or interjection in REHA's text -> the bank sound that performs it
INTERJECTION_SOUND = {"aww": "aww", "awww": "aww", "ooh": "ooh", "oooh": "ooh", "ugh": "ugh", "ughh": "ugh",
                      "whoa": "whoa", "yay": "yay", "phew": "phew", "oops": "oops", "uh-oh": "uh-oh", "heh": "heh"}
EVENT_SOUND = {"laugh": "laugh", "giggle": "chuckle", "breath": "sigh", "gasp": "gasp", "groan": "groan",
               "sniff": "sniff", "clear_throat": "clear_throat", "shush": "shush"}
TAKES = 3                      # takes kept per sound
FRESH_DAYS = 7                 # a fresh take per sound each week, the weakest one retires
RETIRE_BELOW = 0.5             # one thumbs-down retires a take


def bank_dir() -> Path:
    return paths.STATE / "expressions"


class Bank:
    def __init__(self, root: Path | None = None):
        self.root = root or bank_dir()
        self.index: dict[str, dict] = {}
        self.prefs: dict = {"expressiveness": 2}       # 0 (rare) .. 4 (lots); 2 = default
        self._audio: dict[str, np.ndarray] = {}
        self._last: dict[str, str] = {}
        self.filling = False
        self.load()
        if not self.index and not os.environ.get("KOBOX_NO_SEED"):
            self.seed()

    # -- the starter bank shipped with the package (Kobox repo: kobox/bank/)
    def seed(self, src: Path | None = None, merge: bool = False) -> int:
        """First run: copy the shipped bank (her real, checked takes and their scores) so
        expressions work at once, before any rendering. merge=True (after updating the
        package): add only the shipped takes this bank doesn't have, keeping its own scores."""
        src = src or Path(__file__).with_name("bank")
        try:
            shipped = json.loads((src / "bank.json").read_text("utf-8"))
        except (OSError, ValueError):
            return 0
        (self.root / "takes").mkdir(parents=True, exist_ok=True)
        added = 0
        for take, d in shipped.items():
            if merge and take in self.index:
                continue
            f = src / d["file"]
            if f.exists():
                added += 1
                (self.root / d["file"]).write_bytes(f.read_bytes())
                self.index[take] = {**d, "uses": 0, "last_used": 0}
        if not merge:
            try:
                self.prefs.update(json.loads((src / "prefs.json").read_text("utf-8")))
            except (OSError, ValueError):
                pass
        self.save()
        return added

    # -- storage
    def load(self) -> None:
        try:
            self.index = json.loads((self.root / "bank.json").read_text("utf-8"))
        except (OSError, ValueError):
            self.index = {}
        try:
            self.prefs.update(json.loads((self.root / "prefs.json").read_text("utf-8")))
        except (OSError, ValueError):
            pass

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "bank.json.tmp"
        tmp.write_text(json.dumps(self.index, indent=1), "utf-8")
        tmp.replace(self.root / "bank.json")
        (self.root / "prefs.json").write_text(json.dumps(self.prefs), "utf-8")

    def active(self, sound: str) -> list[str]:
        return [k for k, d in self.index.items() if d["sound"] == sound and not d.get("retired")]

    def has(self, sound: str | None) -> bool:
        return bool(sound) and bool(self.active(sound))

    def audio(self, take: str) -> np.ndarray | None:
        if take not in self._audio:
            from .speech import read_wav
            try:
                x, _ = read_wav(str(self.root / self.index[take]["file"]))
                self._audio[take] = x.astype(np.float32)
            except Exception:
                return None
        return self._audio[take]

    def add(self, sound: str, x: np.ndarray, source: str = "rendered", note: str = "", score: float = 1.0) -> str:
        from .speech import write_wav
        take = f"{sound.strip('.!').lower().replace(' ', '_')}-{uuid.uuid4().hex[:8]}"
        (self.root / "takes").mkdir(parents=True, exist_ok=True)
        f = f"takes/{take}.wav"
        write_wav(self.root / f, x, SR)
        self.index[take] = {"sound": sound, "file": f, "source": source, "note": note, "score": score,
                            "uses": 0, "created": time.time(), "last_used": 0}
        self._audio[take] = x.astype(np.float32)
        self.save()
        return take

    # -- use
    def pick(self, sound: str) -> tuple[str, np.ndarray] | None:
        """A take for this sound: weighted by how well liked it is, never the same as last time."""
        takes = self.active(sound)
        if len(takes) > 1 and self._last.get(sound) in takes:
            takes.remove(self._last[sound])
        if not takes:
            return None
        w = [max(0.05, self.index[t]["score"]) for t in takes]
        take = random.choices(takes, weights=w)[0]
        x = self.audio(take)
        if x is None:
            return None
        self._last[sound] = take
        d = self.index[take]
        d["uses"], d["last_used"] = d["uses"] + 1, time.time()
        return take, x

    # -- learning
    def feedback(self, takes: list[str], delta: float) -> list[str]:
        """Owner's reaction to a reply: liked takes are used more, disliked ones retire."""
        retired = []
        for t in takes:
            d = self.index.get(t)
            if not d:
                continue
            d["score"] = float(np.clip(d["score"] + delta, 0.0, 3.0))
            if d["score"] < RETIRE_BELOW and not d.get("retired"):
                d["retired"], d["retired_at"] = True, time.time()
                retired.append(t)
        self.save()
        return retired

    def set_expressiveness(self, step: int) -> int:
        self.prefs["expressiveness"] = int(np.clip(int(self.prefs.get("expressiveness", 2)) + step, 0, 4))
        self.save()
        return self.prefs["expressiveness"]

    def wanted(self) -> list[str]:
        """Sounds that need a take now: fewer than TAKES, or none fresher than a week."""
        now, out = time.time(), []
        for sound in SOUNDS:
            act = self.active(sound)
            newest = max((self.index[t]["created"] for t in act), default=0)
            if len(act) < TAKES or now - newest > FRESH_DAYS * 86400:
                out.append(sound)
        return out

    def _retire_extra(self, sound: str) -> None:
        act = sorted(self.active(sound), key=lambda t: (self.index[t]["score"], self.index[t]["created"]))
        while len(act) > TAKES:
            t = act.pop(0)
            self.index[t]["retired"], self.index[t]["retired_at"] = True, time.time()

    async def fill(self, cb, ears=None, still_ok=lambda: True, limit: int | None = None) -> int:
        """Render missing or weekly fresh takes in the background. Returns how many were added."""
        from . import engine as kobox
        from .speech import read_wav
        if self.filling:
            return 0
        self.filling, added = True, 0
        try:
            for sound in self.wanted():
                prompts, accept = SOUNDS[sound]
                need = max(1, TAKES - len(self.active(sound)))
                tries = 0
                while need > 0 and tries < need * 3 and still_ok():
                    if limit is not None and added >= limit:
                        return added
                    prompt = prompts[tries % len(prompts)]
                    tries += 1
                    f = str(paths.AUDIO_TMP / f"bank-{uuid.uuid4().hex[:8]}.wav")
                    try:
                        if not await cb.render(prompt, f, timeout=60):
                            return added
                        x, sr = read_wav(f)
                    except Exception as e:
                        log.warning("expression bank: render failed (%s): %s", sound, e)
                        return added
                    finally:
                        Path(f).unlink(missing_ok=True)
                    if sr != SR:
                        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x).astype(np.float32)
                    y = kobox.trim(x)
                    if not (0.15 <= len(y) / SR <= 2.5) or kobox.speech_level_db(y) < -60:
                        continue
                    if accept and ears is not None and ears.available():
                        import asyncio
                        heard = await asyncio.get_running_loop().run_in_executor(None, ears.verify, kobox.to16k(y), 2.0)
                        if heard is not None and not re.search(accept, heard.lower()):
                            continue
                    self.add(sound, y, "rendered", note=prompt)
                    added += 1
                    need -= 1
                self._retire_extra(sound)
            self.save()
            return added
        finally:
            self.filling = False

    def export(self, dst: Path) -> int:
        """Write the shareable bank to dst (bank.json + takes/): active takes rendered in her
        voice, with their learned scores. Takes converted from other people's recordings
        are never exported (their rights, not ours to publish)."""
        dst = Path(dst)
        (dst / "takes").mkdir(parents=True, exist_ok=True)
        keep = {t: d for t, d in self.index.items() if not d.get("retired") and d.get("source") == "rendered"}
        out = {}
        for t, d in sorted(keep.items()):
            src = self.root / d["file"]
            if not src.exists():
                continue
            (dst / d["file"]).write_bytes(src.read_bytes())
            out[t] = {"sound": d["sound"], "file": d["file"], "source": "rendered", "note": d.get("note", ""),
                      "score": round(float(d.get("score", 1.0)), 3), "created": int(d.get("created", 0))}
        for f in (dst / "takes").glob("*.wav"):
            if f"takes/{f.name}" not in {d["file"] for d in out.values()}:
                f.unlink()
        (dst / "bank.json").write_text(json.dumps(out, indent=1, sort_keys=True), "utf-8")
        (dst / "prefs.json").write_text(json.dumps({"expressiveness": int(self.prefs.get("expressiveness", 2))}), "utf-8")
        return len(out)

    def summary(self) -> dict:
        act = [d for d in self.index.values() if not d.get("retired")]
        return {"sounds": len({d["sound"] for d in act}), "takes": len(act),
                "retired": sum(1 for d in self.index.values() if d.get("retired")),
                "converted": sum(1 for d in act if d["source"] == "converted"),
                "expressiveness": self.prefs.get("expressiveness", 2)}


EXPRESSIVENESS_TEXT = {
    0: "rarely: only when a moment really calls for it, at most one per reply",
    1: "now and then: at most one every three or four sentences",
    2: "up to one every two or three sentences",
    3: "freely: most light or emotional replies carry one or two",
    4: "generously: a light or emotional reply usually carries two or more, varied",
}


def parts(text: str) -> tuple[str | None, str, str | None]:
    """'[laughs] Okay, fine.' -> ('laugh', 'Okay, fine.', None); 'Ugh, not again.' -> ('ugh', 'not again.', None);
    'That's it. [sighs]' -> (None, "That's it.", 'sigh'). Style cues stay with the words."""
    from .prosody import classify_cue
    lead = tail = None
    m = re.match(r"^((?:\s*\[[^\]]+\]\s*)*)", text)
    head, rest = m.group(1), text[m.end():]
    keep = []
    for c in re.findall(r"\[([^\]]+)\]", head):
        cue = classify_cue(c)
        if cue and cue[0] == "event" and cue[1] in EVENT_SOUND and lead is None:
            lead = EVENT_SOUND[cue[1]]
        else:
            keep.append(f"[{c}]")
    if lead is None:
        mi = re.match(r"^(uh-oh|[a-z]+)[,.!…]*\s*", rest, re.I)
        if mi and mi.group(1).lower() in INTERJECTION_SOUND:
            lead, rest = INTERJECTION_SOUND[mi.group(1).lower()], rest[mi.end():]
            rest = rest[:1].upper() + rest[1:]
    mt = re.search(r"\s*\[([^\]]+)\]\s*$", rest)
    if mt:
        cue = classify_cue(mt.group(1))
        if cue and cue[0] == "event" and cue[1] in EVENT_SOUND:
            tail, rest = EVENT_SOUND[cue[1]], rest[:mt.start()]
    core = (" ".join(keep) + " " + rest).strip()
    return lead, core, tail


# ---------------------------------------------------------------- learning from recordings (E4)
# what Whisper hears in a converted moment -> the bank sound it is (first match wins)
HEARD_SOUND = [
    (r"\b(mm-?hmm|mhm|mm hmm)\b", "Mm-hmm."), (r"\b(laugh\w*|haha\w*|(ha|hah)(\W+(ha|hah))+)\b", "laugh"),
    (r"\b(he){2,}\b|\bheh\b|chuckl", "chuckle"), (r"sigh", "sigh"), (r"gasp", "gasp"), (r"groan", "groan"),
    (r"sniff", "sniff"), (r"ahem|clears? throat", "clear_throat"), (r"\bsh+\b|shush", "shush"),
    (r"\buh-? ?oh\b", "uh-oh"), (r"\baw+\b", "aww"), (r"\bo+h\b|\booh\b", "ooh"), (r"\bugh+\b", "ugh"),
    (r"\bwhoa+\b|\bwoah\b", "whoa"), (r"\byay+\b", "yay"), (r"\bphew\b", "phew"), (r"\boops\b", "oops"),
    (r"\bhm+\b", "Hmm."), (r"\bm{2,}\b", "Mmm."),
]


def classify_heard(heard: str) -> str | None:
    """A moment counts only if all Whisper heard is an expression: no real words (lyrics, talk)."""
    from .textproc import normalise_for_wer
    h = (heard or "").lower().strip()
    if not h:
        return None
    words = [w for w in re.sub(r"[\[\]()*]", " ", h).split() if w]
    labels = r"\b(sighs?|sighing|gasps?|laugh\w*|groans?|sniff\w*|chuckl\w*|clears?|throat|giggl\w*)\b"
    plain = re.sub(labels, " ", " ".join(normalise_for_wer(h))).strip()     # Whisper's *sigh* etc. are sounds
    if plain and not re.fullmatch(r"((ha|he|hah)+\s*)+", plain):
        return None                                   # real words in it: not an expression
    for rx, sound in HEARD_SOUND:
        if re.search(rx, " ".join(words)):
            return sound
    return None


def import_converted(folder: Path, ears, source: str, bank: Bank | None = None) -> list[dict]:
    """Stage 2: check converted moments and add the expressions to the bank as candidates
    (score 0.8: used a little less than proven takes until the owner reacts)."""
    from . import engine as kobox
    from .speech import read_wav
    bank = bank or Bank()
    out = []
    for f in sorted(Path(folder).glob("[0-9]*.wav")):
        x, sr = read_wav(str(f))
        if sr != SR:
            x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x).astype(np.float32)
        y = kobox.trim(x)
        row = {"file": f.name, "dur_s": round(len(y) / SR, 2)}
        if not (0.15 <= len(y) / SR <= 2.5) or kobox.speech_level_db(y) < -55:
            row["skipped"] = "too short, long or quiet"
        else:
            heard = ears.verify(kobox.to16k(y), 5.0) if ears is not None else ""
            sound = classify_heard(heard or "")
            row["heard"] = heard
            if sound:
                row["sound"], row["take"] = sound, bank.add(sound, y, "converted", note=f"from {source}", score=0.8)
            else:
                row["skipped"] = "not an expression (words, or unclear)"
        out.append(row)
    return out


def import_recording(path: str, ears, source: str | None = None) -> list[dict]:
    """Someone else's recording -> REHA's expressions. Her voice never changes:
    only the expression is borrowed, re-voiced from her own reference clip."""
    import shutil
    import subprocess
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="reha-expr-", dir=paths.AUDIO_TMP))
    try:
        py = paths.CB_PYTHON
        ref = paths.VOICE_REF
        script = Path(__file__).with_name("expression_convert.py")
        subprocess.run([str(py), "-I", str(script), str(path), str(work), str(ref)],
                       check=True, capture_output=True, timeout=1800)
        return import_converted(work, ears, source or Path(path).name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
