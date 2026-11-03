"""Human-like delivery for REHA's voice.

Kokoro reads text flatly when given whole sentences. This planner breaks a reply
into breath groups (clauses) and gives each its own tempo, pitch and pause, the
way people actually talk:

* declination: pitch starts a little high in a sentence and settles at the end
* questions lift at the end; exclamations quicken and lift
* burstiness: clause tempo drifts in runs (quick asides, slower key points),
  short clauses ("Honestly?") get room around them
* pauses vary: micro-pauses at commas, longer at full stops, a held beat before
  "but / honestly / look / actually", longer still at ellipses and paragraphs
* *word* in a reply marks vocal emphasis: that phrase slows and lifts slightly
* mood (from her hidden mood tag) sets the baseline: tempo, pitch, range, pause
  length and a slight shift of voice colour
* delivery cues, ElevenLabs-style, are performed and never read aloud:
  [whispers] / *softly* / (playfully) set a delivery style for the words that
  follow (volume, breathiness, tempo, pitch, voice colour) until the next style
  cue or the paragraph ends; [pause] / *sighs* / (laughs) are moments: real
  silence, a breath, a soft breathy chuckle. An unknown stage direction such as
  *leans closer* becomes a short silent beat.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

# mood: (tempo, pitch, pitch_range, pause_scale, (bias_voice, bias_weight))
MOOD_PROFILES: dict[str, tuple[float, float, float, float, tuple[str, float] | None]] = {
    "neutral":    (1.00, 1.000, 0.020, 1.00, None),
    "warm":       (0.95, 0.990, 0.025, 1.10, ("af_heart", 0.15)),
    "tender":     (0.88, 0.975, 0.020, 1.40, ("af_heart", 0.30)),
    "amused":     (1.04, 1.020, 0.040, 0.90, ("af_bella", 0.15)),
    "playful":    (1.05, 1.030, 0.045, 0.85, ("af_bella", 0.20)),
    "curious":    (1.00, 1.020, 0.035, 1.00, None),
    "focused":    (1.02, 0.990, 0.015, 0.85, ("bf_emma", 0.15)),
    "skeptical":  (0.97, 0.980, 0.030, 1.15, None),
    "concerned":  (0.90, 0.970, 0.015, 1.30, ("af_heart", 0.20)),
    "pleased":    (1.02, 1.020, 0.030, 1.00, ("af_heart", 0.10)),
    "reflective": (0.90, 0.980, 0.020, 1.35, ("af_heart", 0.20)),
}

_DISCOURSE = re.compile(r"^(but|so|and honestly|honestly|look|actually|okay|ok|well|still|now|listen|mm+|hmm+)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"(\.\.\.|…|—|–| - |,|;|:)")


@dataclass
class Segment:
    text: str
    tempo: float          # multiplier on base speed
    pitch: float          # pitch ratio (1.0 = unchanged)
    pause_after: float    # seconds of silence after this segment
    pause_before: float = 0.0
    gain: float = 1.0     # loudness after normalisation (whisper ~0.5)
    breath: float = 0.0   # 0..1 air mixed into the voice
    bias: tuple[str, float] | None = None   # nudge toward another voice's colour
    kind: str = "speech"  # speech | pause | breath | hum | laugh | giggle


# delivery styles: (tempo, pitch, gain, breath, pause_scale, voice bias)
STYLES: dict[str, tuple[float, float, float, float, float, tuple[str, float] | None]] = {
    "normal":  (1.00, 1.000, 1.00, 0.00, 1.00, None),
    # softer styles lean toward af_heart (warm, clean) rather than af_nicole, whose
    # breathiness read as static at low volume
    "whisper": (0.94, 0.985, 0.55, 0.65, 1.15, ("af_heart", 0.35)),
    "soft":    (0.95, 0.990, 0.75, 0.22, 1.15, ("af_heart", 0.30)),
    "warm":    (0.95, 0.990, 0.92, 0.08, 1.10, ("af_heart", 0.25)),
    "playful": (1.05, 1.025, 1.00, 0.05, 0.90, ("af_bella", 0.25)),
    "excited": (1.09, 1.035, 1.00, 0.00, 0.80, ("af_bella", 0.20)),
    "sad":     (0.92, 0.975, 0.78, 0.12, 1.20, ("af_heart", 0.20)),
    "serious": (0.96, 0.985, 0.95, 0.00, 1.10, ("bf_emma", 0.20)),
    "slow":    (0.84, 1.000, 1.00, 0.00, 1.30, None),
    "fast":    (1.12, 1.010, 1.00, 0.00, 0.80, None),
    # performed for real by Chatterbox in Kobox/Auto; these are Kokoro's closest fallbacks
    "happy":     (1.07, 1.030, 1.00, 0.00, 0.85, ("af_bella", 0.20)),
    "surprised": (1.06, 1.040, 1.00, 0.00, 0.90, ("af_bella", 0.15)),
    "sarcastic": (0.96, 0.985, 0.95, 0.00, 1.10, None),
    "dramatic":  (0.88, 0.980, 0.95, 0.05, 1.35, None),
    "fear":      (1.04, 1.020, 0.80, 0.15, 0.90, ("af_heart", 0.20)),
    "angry":     (1.03, 0.980, 1.00, 0.00, 0.90, ("bf_emma", 0.20)),
    "crying":    (0.88, 0.970, 0.75, 0.20, 1.30, ("af_heart", 0.30)),
}
_STYLE_WORDS = {
    "whisper": ("whisper", "whispers", "whispering", "whispered", "in a whisper", "hushed", "under her breath",
                "under my breath", "barely audible", "very quietly", "breathlessly"),
    "soft": ("soft", "softly", "softer", "quietly", "quiet", "gently", "gentle", "tenderly", "tender", "lovingly",
             "murmurs", "murmuring", "murmured", "in a soft voice", "softly smiling", "low voice", "in a low voice",
             "lowers her voice", "lowers voice",
             # REHA Pro never performs a sultry or seductive delivery: those cues get a plain soft one
             "sultry", "sultrily", "seductively", "huskily", "husky", "breathy", "breathily", "purrs", "purring"),
    "warm": ("warm", "warmly", "smiles", "smiling", "with a smile", "affectionately", "sweetly", "fondly"),
    "playful": ("playful", "playfully", "teasing", "teasingly", "teases", "mischievously", "grins", "grinning",
                "smirks", "smirking", "with a grin", "cheekily", "flirty", "flirtatiously", "flirting"),   # flirty cues: plain playful
    "excited": ("excited", "excitedly", "cheerfully", "cheerful", "happily", "brightly", "enthusiastically",
                "eagerly", "delighted"),
    "sad": ("sad", "sadly", "sorrowfully", "wistfully", "softly sad", "regretfully", "quietly sad"),
    "serious": ("serious", "seriously", "firmly", "firm", "sternly", "calmly", "calm", "steadily", "matter-of-factly"),
    "slow": ("slowly", "slow", "slower", "deliberately", "drawn out"),
    "fast": ("quickly", "fast", "rushed", "hurriedly", "rapidly"),
    "happy": ("happy", "joyfully", "joyful", "beaming", "gleefully", "overjoyed"),
    "surprised": ("surprised", "surprise", "in surprise", "astonished", "amazed", "shocked", "stunned"),
    "sarcastic": ("sarcastic", "sarcastically", "dryly", "deadpan", "wryly", "ironically"),
    "dramatic": ("dramatic", "dramatically", "theatrically", "ominously", "suspensefully"),
    "fear": ("fear", "fearful", "afraid", "scared", "nervously", "nervous", "anxiously", "anxious", "worried"),
    "angry": ("angry", "angrily", "annoyed", "irritated", "exasperated", "indignant", "indignantly"),
    "crying": ("crying", "cries", "tearfully", "teary", "through tears", "sobbing", "sobs", "choked up",
               "voice breaking"),
    "normal": ("normal", "normally", "normal voice", "back to normal", "speaking normally", "out loud", "aloud"),
}
_EVENT_WORDS = {
    "pause": ("pause", "pauses", "pausing", "beat", "a beat", "long pause", "short pause", "brief pause",
              "silence", "a moment", "waits", "hesitates", "hesitating"),
    "breath": ("sigh", "sighs", "sighing", "contented sigh", "soft sigh", "breathes", "breath", "breathing",
               "inhales", "exhales", "takes a breath", "deep breath", "breathes in", "breathes out"),
    "hum": ("hums", "hum", "humming", "hmm", "hmmm", "mmm", "mm", "mhm"),
    "laugh": ("laugh", "laughs", "laughing", "soft laugh", "chuckle", "chuckles", "chuckling", "laughs softly",
              "light laugh", "small laugh"),
    "giggle": ("giggle", "giggles", "giggling", "soft giggle", "little giggle", "snickers", "titters"),
    "gasp": ("gasp", "gasps", "gasping", "little gasp", "sharp breath", "sharp intake of breath"),
    "groan": ("groan", "groans", "groaning", "grumbles", "grumbling", "moans"),
    "sniff": ("sniff", "sniffs", "sniffles", "sniffling", "sniffle"),
    "clear_throat": ("clears throat", "clears her throat", "clear throat", "clearing throat", "ahem"),
    "shush": ("shush", "shushes", "shh", "shhh", "hush"),
}
_CUE_LOOKUP: dict[str, tuple[str, str]] = {}
for _k, _ws in _STYLE_WORDS.items():
    for _w in _ws:
        _CUE_LOOKUP[_w] = ("style", _k)
for _k, _ws in _EVENT_WORDS.items():
    for _w in _ws:
        _CUE_LOOKUP[_w] = ("event", _k)

_CUE_SPAN = re.compile(r"\*([^*\n]{1,48})\*|\[([^\[\]\n]{1,40})\]|\(([^()\n]{1,40})\)")


def classify_cue(phrase: str) -> tuple[str, str] | None:
    """'whispers' -> ('style','whisper'); 'laughs softly' -> ('event','laugh'); else None."""
    p = re.sub(r"[^a-z' -]", "", phrase.lower()).strip()
    p = re.sub(r"^(she |reha )", "", p)
    if p in _CUE_LOOKUP:
        return _CUE_LOOKUP[p]
    words = p.split()
    for n in (3, 2, 1):                       # "whispers into your ear" -> whispers
        if len(words) >= n and " ".join(words[:n]) in _CUE_LOOKUP:
            return _CUE_LOOKUP[" ".join(words[:n])]
    return None


def split_cues(text: str) -> list[tuple[str, str]]:
    """Split text into ('text', chunk) and ('style'|'event'|'beat', name) items.

    *x* that isn't a known cue stays as emphasis unless it reads like a stage
    direction on its own (several words, standing between sentences)."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _CUE_SPAN.finditer(text):
        star, square, paren = m.group(1), m.group(2), m.group(3)
        phrase = star or square or paren
        cue = classify_cue(phrase)
        if not cue and star is not None:
            before = text[:m.start()].rstrip()
            after = text[m.end():].lstrip()
            standalone = (not before or before[-1] in ".!?…\n") and (not after or after[0].isupper() or after[0] in "*\n")
            if standalone and (len(phrase.split()) >= 2 or re.search(r"(s|ing)$", phrase.strip().lower())):
                cue = ("beat", "beat")
        if not cue:
            continue                          # emphasis or ordinary brackets: leave it in the text
        if text[pos:m.start()].strip():
            out.append(("text", text[pos:m.start()]))
        out.append(cue)
        pos = m.end()
    if text[pos:].strip():
        out.append(("text", text[pos:]))
    return out


def strip_cues(text: str) -> str:
    """Text with performed cues removed (for anything that must not hear them)."""
    return re.sub(r"\s{2,}", " ", "".join(v for k, v in split_cues(text) if k == "text")).strip()


@dataclass
class Group:
    """One sentence (or a short pair) synthesised and played as a unit."""
    segments: list[Segment]


def profile(mood: str):
    return MOOD_PROFILES.get(mood, MOOD_PROFILES["neutral"])


def _clauses(sentence: str) -> list[tuple[str, str]]:
    """Split a sentence into (clause, trailing punctuation) pairs, merging scraps."""
    parts = _CLAUSE_SPLIT.split(sentence)
    out: list[tuple[str, str]] = []
    buf = ""
    for i in range(0, len(parts), 2):
        txt = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        buf = f"{buf} {txt}".strip() if buf else txt.strip()
        if not buf:
            continue
        if len(buf) < 14 and punct in (",", " - ") and i + 2 < len(parts):
            buf += punct  # too short to stand alone; keep the comma inside
            continue
        out.append((buf, punct.strip()))
        buf = ""
    if buf:
        out.append((buf, ""))
    return out


EVENT_SECONDS = {"pause": (0.55, 0.85), "beat": (0.30, 0.45), "breath": (0.5, 0.7), "laugh": (0.55, 0.75),
                 "giggle": (0.45, 0.6), "hum": (0.35, 0.5), "gasp": (0.3, 0.45), "groan": (0.4, 0.55),
                 "sniff": (0.3, 0.4), "clear_throat": (0.35, 0.5), "shush": (0.3, 0.45)}


def plan(text: str, mood: str = "neutral", seed: int | None = None) -> list[Group]:
    rnd = random.Random(seed)
    tempo0, pitch0, rng, pscale, _ = profile(mood)
    groups: list[Group] = []
    burst = 0.0  # slowly wandering tempo offset -> runs of quicker / slower phrases
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    for pi, para in enumerate(paragraphs):
        style = "normal"                      # a delivery cue lasts to the paragraph's end
        items = split_cues(para)
        for ii, (kind, val) in enumerate(items):
            if kind == "style":
                style = val
                continue
            if kind in ("event", "beat"):
                lo, hi = EVENT_SECONDS.get(val, (0.35, 0.5))
                seg = Segment(text="", tempo=1.0, pitch=1.0, pause_after=0.0, kind=val if kind == "event" else "pause",
                              gain=STYLES[style][2])
                seg.pause_before = rnd.uniform(lo, hi) * (STYLES[style][4] if val in ("pause", "beat") else 0.5)
                if groups and groups[-1].segments and groups[-1].segments[-1].kind == "speech" and ii > 0:
                    groups[-1].segments.append(seg)      # a moment belongs to the line before it
                else:
                    groups.append(Group([seg]))
                continue
            s_tempo, s_pitch, s_gain, s_breath, s_pause, s_bias = STYLES[style]
            sents = [s.strip() for s in _SENT_SPLIT.split(val) if s and s.strip()]
            for si, sent in enumerate(sents):
                end = sent.rstrip()[-1:] if sent.rstrip() else "."
                clauses = _clauses(sent)
                segs: list[Segment] = []
                n = len(clauses)
                for ci, (cl, punct) in enumerate(clauses):
                    emph = "*" in cl
                    clean = cl.replace("*", "").strip()
                    if not clean:
                        continue
                    words = len(clean.split())
                    burst = max(-0.07, min(0.07, burst * 0.6 + rnd.uniform(-0.035, 0.035)))
                    tempo = tempo0 + burst
                    # declination across the sentence: start a touch high, settle low
                    pos = ci / max(1, n - 1) if n > 1 else 0.5
                    pitch = pitch0 + rng * (0.6 - 1.2 * pos) * 0.5 + rnd.uniform(-rng, rng) * 0.35
                    if words <= 3:                       # "Honestly?" / "Look."
                        tempo -= 0.06
                    if re.search(r"\d", clean):          # numbers/lists run quicker
                        tempo += 0.03
                    if emph:
                        tempo -= 0.07
                        pitch += rng * 0.6
                    last = ci == n - 1
                    if last and end == "?":
                        pitch += rng * 1.2
                        tempo += 0.02
                    elif last and end == "!":
                        pitch += rng * 0.9
                        tempo += 0.05
                    elif last:
                        pitch -= rng * 0.4
                    # pauses
                    if punct in ("...", "…"):
                        pause = rnd.uniform(0.38, 0.55)
                    elif punct in ("—", "–", "-"):
                        pause = rnd.uniform(0.18, 0.30)
                    elif punct in (";", ":"):
                        pause = rnd.uniform(0.20, 0.30)
                    elif punct == ",":
                        pause = rnd.uniform(0.09, 0.20)
                    elif last:
                        pause = rnd.uniform(0.32, 0.55)
                        if si == len(sents) - 1 and pi < len(paragraphs) - 1:
                            pause += 0.25            # paragraph break
                    else:
                        pause = 0.08
                    if words <= 3 and last:
                        pause += 0.12                # let a short line land
                    pre = rnd.uniform(0.12, 0.28) if (_DISCOURSE.match(clean) and (ci > 0 or si > 0)
                                                     and rnd.random() < 0.6) else 0.0
                    segs.append(Segment(text=clean + (end if last and end in ".?!" and not clean.endswith(end) else ""),
                                        tempo=max(0.75, min(1.25, tempo * s_tempo)),
                                        pitch=max(0.94, min(1.06, pitch * s_pitch)),
                                        pause_after=pause * pscale * s_pause, pause_before=pre * pscale * s_pause,
                                        gain=s_gain, breath=s_breath, bias=s_bias))
                if segs:
                    groups.append(Group(segs))
    return groups
