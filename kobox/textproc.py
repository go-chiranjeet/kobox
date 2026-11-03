"""Text helpers: hidden tags, speech-safe text, sentence chunks."""
from __future__ import annotations

import re

MOODS = ("neutral", "warm", "tender", "amused", "curious", "focused", "skeptical",
         "concerned", "pleased", "playful", "reflective")

_MOOD_RE = re.compile(r"\[\[\s*mood\s*:\s*([a-z]+)\s*\]\]", re.I)
_REVERIFY_RE = re.compile(r"\[\[\s*reverify\s*\]\]", re.I)
_PRIVATE_NOTE_RE = re.compile(r"<private_note>(.*?)</private_note>", re.S | re.I)
_LEARNED_RE = re.compile(r"<learned>(.*?)</learned>", re.S | re.I)


_LESSON_RE = re.compile(r"\[\[lesson:\s*(.+?)\]\]", re.I | re.S)   # what she learned from a mistake or a correction


def extract_tags(text: str) -> dict:
    """Strip REHA's hidden control tags from a reply.

    Returns {text, mood, private_notes, learned, reverify}."""
    t = text or ""
    mood = "neutral"
    m = _MOOD_RE.findall(t)
    if m:
        cand = m[-1].lower()
        mood = cand if cand in MOODS else "neutral"
    notes = [n.strip() for n in _PRIVATE_NOTE_RE.findall(t) if n.strip()]
    learned = [n.strip() for n in _LEARNED_RE.findall(t) if n.strip()]
    reverify = bool(_REVERIFY_RE.search(t))
    lessons = [n.strip() for n in _LESSON_RE.findall(t) if n.strip()]
    for rx in (_PRIVATE_NOTE_RE, _LEARNED_RE, _MOOD_RE, _REVERIFY_RE, _LESSON_RE):
        t = rx.sub("", t)
    return {"text": re.sub(r"\n{3,}", "\n\n", t).strip(), "mood": mood,
            "private_notes": notes, "learned": learned, "reverify": reverify, "lessons": lessons}


def _speak_block(m: re.Match) -> str:
    """Short 'label: value' readouts (e.g. memory stats) are read aloud; real code stays on screen."""
    lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
    if 0 < len(lines) <= 6 and all(re.match(r"^[\w ()%/.-]{1,40}:\s*\S", l) for l in lines):
        return " " + ". ".join(lines) + ". "
    return " I've put the code on screen. "


def _speak_tables(t: str) -> str:
    """Markdown tables become spoken rows ("CPU load: 35%"), up to 8; the header row
    labels the cells when there are more than two columns."""
    out, rows = [], []

    def flush() -> None:
        if not rows:
            return
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        body = [c for c in cells[1:] if not all(re.fullmatch(r":?-{2,}:?", x) for x in c)] if len(cells) > 1 else cells
        head = cells[0] if len(cells) > 1 else []
        spoken = []
        for c in body[:40]:
            if len(c) == 2 or not head:
                spoken.append(": ".join(x for x in c[:2] if x))
            else:
                spoken.append(c[0] + ": " + ", ".join(f"{h} {v}" for h, v in zip(head[1:], c[1:]) if v))
        if len(body) > 40:
            spoken.append("the rest is on screen")
        out.append(". ".join(spoken) + ".")
        rows.clear()

    for line in t.splitlines():
        if re.match(r"^\s*\|.*\|\s*$", line):
            rows.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def to_speech(text: str, max_chars: int = 0) -> str:
    """Turn a markdown reply into something pleasant to hear. Everything on screen is
    said; only the form changes (tables become rows, code stays on screen and she says
    so). max_chars > 0 is a hard ceiling for runaway text, never a summary."""
    t = text or ""
    t = re.sub(r"```[\w-]*\n?(.*?)```", _speak_block, t, flags=re.S)
    t = _speak_tables(t)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)              # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)           # links -> label
    t = re.sub(r"https?://\S+", "the link on screen", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)      # headings
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)           # bullets
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__)([^*_]+)\1", r"\2", t)            # bold -> plain
    t = re.sub(r"(?<![*\w])_([^_]+)_(?!\w)", r"*\1*", t)       # _x_ -> spoken emphasis
    t = re.sub(r"~~([^~]+)~~", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if max_chars and len(t) > max_chars:
        cut = t[:max_chars]
        stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("\n"))
        t = (cut[: stop + 1] if stop > 120 else cut).strip() + " The rest is on screen."
    return t


def sentences(text: str) -> list[str]:
    """Split into chunks of roughly one or two sentences (>= 60 chars) for TTS."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip()) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        buf = f"{buf} {p}".strip()
        if len(buf) >= 60:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


# ---------------------------------------------------------------- acknowledgements
# What she says the moment real work starts: quick, casual or big (long or complex).
# Warm and human, never personal or flirtatious: this is a workplace.
ACKS = {
    "quick": ["Got it.", "One sec.", "Sure thing.", "On it."],
    "casual": ["Easy one. Give me a second.", "Consider it handled.", "Okay, let's see.", "Doing it now."],
    "big": ["Right, let me dig into this.", "Okay, this needs a proper look. Give me a minute.",
            "Big one. I'll take it step by step.", "Leave it with me."],
}
_CASUAL = re.compile(r"\b(play|song|music|joke|weather|coffee|lunch|dinner|weekend|open|launch|remind)\b", re.I)
_last_ack: dict[str, str] = {}


def ack_kind(text: str, tier: str, unlocked: bool) -> str:
    t = text or ""
    if tier == "deep" or len(t) > 160 or len(re.findall(r"[.!?]\s", t)) >= 2 or re.search(r"\b(and then|step by step|plan|research|compare|analy[sz]e)\b", t, re.I):
        return "big"
    if _CASUAL.search(t):
        return "casual"
    return "quick"


def pick_ack(text: str, tier: str, unlocked: bool, pools: dict | None = None) -> str:
    """An acknowledgement that fits the request; never the same one twice running."""
    import random
    kind = ack_kind(text, tier, unlocked)
    pool = (pools or ACKS).get(kind) or ACKS[kind]
    opts = [a for a in pool if a != _last_ack.get("any")] or pool
    pick = random.choice(opts)
    _last_ack["any"] = pick
    return pick



# ---------------------------------------------------------------- spoken-accuracy check (Kobox)
# Compares what a voice engine was asked to say with what Whisper heard it say.
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
FILLERS = {"hm", "hmm", "mm", "mmm", "huh", "ha", "hah", "haha", "hahaha", "heh", "hehe", "hee", "ah", "oh",
           "uh", "um", "er", "aw", "ooh", "whew", "phew", "aww", "awww", "ugh", "ughh", "whoa", "pfft", "yay", "ahh",
           "ahhh", "eek", "oops", "ouch", "yikes", "aha", "wow", "oh", "shh", "shhh", "ahem"}


def number_words(n: int) -> str:
    """0..9999 -> words ('15' -> 'fifteen', '1200' -> 'one thousand two hundred')."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 10000:
        return _ONES[n // 1000] + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return str(n)


def normalise_for_wer(text: str) -> list[str]:
    """Words only: lower case, digits spelled out, apostrophes, hyphens and
    punctuation dropped, laughs and filler sounds removed."""
    t = (text or "").lower().replace("’", "'").replace("'", "")
    t = re.sub(r"(\d),(\d{3})\b", r"\1\2", t)
    t = re.sub(r"\d+", lambda m: f" {number_words(int(m.group()))} " if len(m.group()) <= 4 else m.group(), t)
    words = re.sub(r"[^a-z0-9 ]+", " ", t).split()
    return [w for w in words if w not in FILLERS and not re.fullmatch(r"(ha|he|hee)+h?|h*m+(h+m+)?|uh+(huh)?", w)]


def _merge_compounds(a: list[str], b: list[str]) -> list[str]:
    """Join neighbouring words in `a` when `b` spells them as one ('cool down' / 'cooldown')."""
    vocab, out, i = set(b), [], 0
    while i < len(a):
        if i + 1 < len(a) and a[i] + a[i + 1] in vocab and a[i] not in vocab:
            out.append(a[i] + a[i + 1])
            i += 2
        else:
            out.append(a[i])
            i += 1
    return out


def word_error_rate(intended: str, heard: str) -> float:
    ref, hyp = normalise_for_wer(intended), normalise_for_wer(heard)
    hyp, ref = _merge_compounds(hyp, ref), _merge_compounds(ref, hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / len(ref)


def required_words(intended: str, extra: set[str] | None = None) -> set[str]:
    """Words that must be heard exactly: every number, and any listed name."""
    req = set()
    for m in re.finditer(r"\d+", intended or ""):
        if len(m.group()) <= 4:
            req.update(number_words(int(m.group())).split())
    num = set(_ONES) | set(_TENS[2:]) | {"hundred", "thousand"}
    req.update(w for w in normalise_for_wer(intended) if w in num)
    if extra:
        words = set(normalise_for_wer(intended))
        req.update(w for w in (normalise_for_wer(" ".join(extra))) if w in words)
    return req
