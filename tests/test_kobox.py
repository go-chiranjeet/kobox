"""Kobox (combined voice engine) tests: run anywhere, no models or audio device needed.
Spec: docs/kobox/FRD.md (requirement ids in comments)."""
import asyncio
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 24000


def tone(seconds=1.0, db=-20.0, f=220.0, lead=0.0):
    t = np.arange(int(SR * seconds)) / SR
    x = np.sin(2 * np.pi * f * t).astype(np.float32) * (10 ** (db / 20) * np.sqrt(2))
    return np.concatenate([np.zeros(int(SR * lead), np.float32), x])


def write(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- accuracy check (F5)
def test_wer_normalisation_forgives_formatting_not_words():
    from kobox.textproc import word_error_rate as wer
    assert wer("We'll wait for the cooldown.", "We'll wait for the cool down.") == 0
    assert wer("Before the ten o'clock review.", "Before the 10 o'clock review.") == 0
    assert wer("Okay, that made me smile.", "Hahaha. Hm, okay, that made me smile.") == 0
    assert wer("Another rate limit from Figma.", "Another rate limit from Sigma.") == 0.2


def test_judge_requires_numbers_and_names():
    from kobox.engine import judge, settings
    s = settings({})
    assert judge("Starts in fifteen minutes.", "Starts in 15 minutes.", s, set())[0]
    assert not judge("Starts in fifteen minutes.", "Starts in 50 minutes.", s, set())[0]      # phase 0 case
    assert not judge("[laughs] REHA has it.", "Rita has it.", s, {"REHA"})[0]
    assert judge("[laughs] REHA has it.", "Ha ha. Reha has it.", s, {"REHA"})[0]
    assert not judge("The second option is much stronger.", "The second is stronger.", s, set())[0]


# ---------------------------------------------------------------- planning (F2, F3.5)
def test_plan_marks_expressive_lines_and_starts_instantly():
    from kobox.engine import plan, settings
    s = settings({})
    lines = plan("[laughs] Okay. Here's the plan. [sighs] It changed again. Fine.", "neutral", s)
    assert not lines[0].expressive and "instant" in lines[0].reason            # F3.5
    exp = [ln for ln in lines if ln.expressive]
    assert len(exp) == 1 and exp[0].cb_text.startswith("[sigh]")
    off = plan("[laughs] Okay.", "neutral", {**s, "instant_start": False})
    assert off[0].expressive


def test_mood_lines_capped_at_one_in_three():
    from kobox.engine import plan, settings
    text = "\n\n".join(f"Line number {i} is here." for i in range(6))
    lines = plan(text, "amused", settings({}))
    assert 1 <= sum(ln.expressive for ln in lines) <= len(lines) // 3
    assert not any(ln.expressive for ln in plan(text, "focused", settings({})))


def test_pronunciation_list_reaches_chatterbox(isolated_home):
    (isolated_home / "config" / "pronunciations.json").write_text(
        '{"REHA": "ɹˈeɪhə", "Zenith": "zˈiːnɪθ", "_spoken": {"Zenith": "Zee-nith"}}')
    from kobox.engine import plan, settings, spoken_lexicon
    lex = spoken_lexicon()
    assert lex["REHA"] == "Rayhuh" and lex["Zenith"] == "Zee-nith"
    ln = plan("Hi. [laughs] REHA and Zenith.", "neutral", settings({}), lex)[1]
    assert ln.cb_text == "[laugh] Rayhuh and Zee-nith."


# ---------------------------------------------------------------- matching (F4)
def test_match_levels_trims_and_pauses():
    from kobox.engine import match, settings, speech_level_db, trim
    s = settings({})
    loud_kokoro = -19.0
    quiet_cb = tone(1.0, db=-27.0, lead=0.2)                                   # phase 0: 7-9 dB quieter, late start
    y = match(quiet_cb, loud_kokoro, s, warmth=0.25)
    speech = trim(y)
    assert abs(speech_level_db(speech) - loud_kokoro) <= 1.0                   # F4.1
    assert np.argmax(np.abs(y) > 0.01) < SR * 0.03                             # F4.2 lead silence gone
    assert np.all(y[-int(SR * 0.3):] == 0)                                     # planned pause after the line
    assert np.abs(y).max() <= 0.9                                              # speaker-safe


# ---------------------------------------------------------------- playback (F6)
class FakeStream:
    def __init__(self):
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def abort(self):
        self.stopped = True

    def close(self):
        pass


def test_stream_crossfade_has_no_clicks_and_fires_line_starts():
    from kobox.engine import StreamPlayer

    async def run():
        loop = asyncio.get_running_loop()
        p = StreamPlayer(loop, 20, stream=FakeStream())
        started = []
        await p.play(tone(0.5, -12, 300), lambda: started.append(1))
        await p.play(-tone(0.5, -12, 300), lambda: started.append(2))           # worst case: phase flip at the join
        out = np.zeros((len(p.buf), 1), np.float32)
        p._callback(out, len(out), None, None)
        await asyncio.sleep(0.01)
        jumps = np.abs(np.diff(out[:, 0]))
        assert jumps.max() < 0.08                                              # no step at the join (F6.2)
        assert started == [1, 2]                                               # on_chunk timing (F6.3)
        assert p.remaining_s() == 0
    asyncio.run(run())


# ---------------------------------------------------------------- the engine (F3, F5, F7)
class FakeCB:
    def __init__(self, delay=0.0, fail=False, clip=None):
        self.delay, self.fail, self.clip = delay, fail, clip
        self.broken, self.calls = "", []
        self.lock = asyncio.Lock()

    async def render(self, text, path, timeout=45, still_needed=None):
        async with self.lock:
            if still_needed and not still_needed():
                return False
            self.calls.append(text)
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("worker died")
            write(path, self.clip if self.clip is not None else tone(0.8, -27, 330))
            return True


class FakeEars:
    def __init__(self, heard=None):
        self.heard = heard

    def available(self):
        return True

    def verify(self, audio, wait=0.25):
        return self.heard


class FakePlayer:
    def __init__(self):
        self.clips, self.aborted = [], False

    def remaining_s(self):
        return 0.0

    async def play(self, x, on_start):
        self.clips.append(x)
        on_start()

    async def drain(self, cancel):
        pass

    def abort(self):
        self.aborted = True

    def close(self, hard=False):
        pass


KOKORO_HZ = 220


def make_voice(monkeypatch, cb, ears=None, cfg=None):
    from kobox import engine as kb
    from kobox import speech
    monkeypatch.setattr(speech.Chatterbox, "installed", staticmethod(lambda: True))
    v = speech.Voice({"engine": "kobox", "speed": 1.0, "warmth": 0.25, **(cfg or {})})
    v.cb = cb
    v.kobox.ears = ears
    v._mood_style = lambda voice, mood: "style"
    monkeypatch.setattr(kb.Kobox, "_kokoro", lambda self, text, style, spd, mood: tone(0.6, -19, KOKORO_HZ))
    player = FakePlayer()
    v.kobox._make_player = lambda loop, s: player
    return v, player


def dominant_hz(x):
    x = x[np.abs(x) > 1e-4]
    spec = np.abs(np.fft.rfft(x))
    return np.fft.rfftfreq(len(x), 1 / SR)[np.argmax(spec)]


TEXT = "Here's the summary. [sighs] The client changed the brief again. I'll draft options."


def test_kobox_uses_checked_chatterbox_line(isolated_home, monkeypatch):
    v, player = make_voice(monkeypatch, FakeCB(), FakeEars("The client changed the brief again."))
    assert v.engine() == "kobox"
    assert asyncio.run(v.speak(TEXT)) is True
    assert len(player.clips) == 3
    assert abs(dominant_hz(player.clips[1]) - 330) < 10                        # the Chatterbox take
    assert abs(dominant_hz(player.clips[0]) - KOKORO_HZ) < 10
    assert v.kobox.last["by_chatterbox"] == 1 and not v.kobox.last["fallback"]


def test_kobox_rejects_garbled_line(isolated_home, monkeypatch):
    v, player = make_voice(monkeypatch, FakeCB(), FakeEars("The giant ranged the beef again."))
    assert asyncio.run(v.speak(TEXT)) is True
    assert abs(dominant_hz(player.clips[1]) - KOKORO_HZ) < 10                  # F5.3: Kokoro's version played
    assert v.kobox.last["fallback"] == "accuracy check failed"
    logs = list((isolated_home / ".runtime" / "logs").glob("kobox-*.jsonl"))
    assert logs and '"rejected"' in logs[0].read_text()                        # F5.5


def test_kobox_late_line_falls_back_and_learns(isolated_home, monkeypatch):
    v, player = make_voice(monkeypatch, FakeCB(delay=1.2), FakeEars("The client changed the brief again."),
                           {"kobox": {"grace_max_s": 0.5, "expression_wait_s": 0.5}})
    v.kobox.lateness = 0.0
    t = time.time()
    assert asyncio.run(v.speak(TEXT)) is True
    assert time.time() - t < 1.2                                               # never waited past the grace
    assert abs(dominant_hz(player.clips[1]) - KOKORO_HZ) < 10                  # F3.4
    assert "not ready" in v.kobox.last["fallback"]


def test_kobox_survives_dead_worker_and_retries_later(isolated_home, monkeypatch):
    cb = FakeCB(fail=True)
    v, player = make_voice(monkeypatch, cb, FakeEars("x"))
    assert asyncio.run(v.speak(TEXT)) is True and len(player.clips) == 3      # F7.3: still speaks
    assert cb.broken and not v.kobox.cb_usable()
    v.kobox.cb_failed_at = time.time() - 601
    assert v.kobox.cb_usable() and cb.broken == ""                             # retried after 10 min


def test_fallback_chain_kobox_then_auto(isolated_home, monkeypatch):
    v, _ = make_voice(monkeypatch, FakeCB(), FakeEars())
    order = []

    async def broken(*a, **k):
        order.append("kobox")
        return None

    async def auto(*a, **k):
        order.append("auto")
        return True
    v.kobox.speak = broken
    v._speak_auto = auto
    assert asyncio.run(v.speak("Hello there.")) is True
    assert order == ["kobox", "auto"]                                          # F7.1


def test_stop_aborts_the_stream(isolated_home, monkeypatch):
    v, player = make_voice(monkeypatch, FakeCB(delay=0.3), FakeEars("x"))

    async def run():
        task = asyncio.ensure_future(v.speak("One. [sighs] Two. Three. [laughs] Four. Five."))
        await asyncio.sleep(0.05)
        v.kobox.player = player
        await v.stop()
        return await task
    assert asyncio.run(run()) is False and player.aborted                      # F6.4


def test_other_engines_untouched(isolated_home, monkeypatch):
    from kobox import speech
    monkeypatch.setattr(speech.Chatterbox, "installed", staticmethod(lambda: True))
    for e in ("auto", "kokoro", "chatterbox"):
        assert speech.Voice({"engine": e}).engine() == e
    monkeypatch.setattr(speech.Chatterbox, "installed", staticmethod(lambda: False))
    assert speech.Voice({"engine": "kobox"}).engine() == "kokoro"              # F1.3


def test_long_plain_reply_starts_with_one_sentence():
    # a typed reply with no cues used to be one huge line: silent until all of it was rendered
    from kobox.engine import plan, settings
    text = " ".join(f"This is sentence number {i} of a long typed reply." for i in range(40))
    lines = plan(text, "neutral", settings({}))
    assert len(lines) == 40 and len(lines[0].text) < 60
    assert not any(ln.expressive for ln in lines)


def test_hums_go_to_chatterbox_not_read_out():
    from kobox.engine import plan, settings
    s = settings({})
    lines = plan("Mmm hmmm, that sounds right. The draft is ready.", "neutral", s)
    assert lines[0].hum and lines[0].expressive and lines[0].cb_text == "Mm-hmm."   # not skipped by instant start
    assert lines[0].kk_text == ""                                                   # Kokoro never says "mmm hmmm"
    assert lines[1].text == "that sounds right." and not lines[1].expressive
    cue = plan("Okay. [hmm] Let me check that.", "neutral", s)
    assert [ln.cb_text for ln in cue if ln.hum] == ["Hmm."]
    assert not any(ln.hum for ln in plan("The gap is 5 mm wide.", "neutral", s))


def test_hum_only_line_passes_accuracy_check():
    from kobox.engine import judge, settings
    assert judge("Mm-hmm.", "Mm-hmm.", settings({}), set())[0]
    assert judge("Hmm.", "", settings({}), set())[0]


class FlakyEars(FakeEars):
    """Hears a garbled take first, then the right words."""
    def __init__(self, *heard):
        self.seq = list(heard)

    def verify(self, audio, wait=0.25):
        return self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]


def test_requested_expression_waits_for_the_real_take(isolated_home, monkeypatch):
    v, player = make_voice(monkeypatch, FakeCB(delay=1.2), FakeEars("The client changed the brief again."),
                           {"kobox": {"grace_max_s": 0.3}})
    assert asyncio.run(v.speak(TEXT)) is True
    assert abs(dominant_hz(player.clips[1]) - 330) < 10            # waited past the grace: Chatterbox performed it
    assert "1 of 1 expressions performed" in v.kobox.label()


def test_garbled_take_is_rerendered_once(isolated_home, monkeypatch):
    cb = FakeCB()
    v, player = make_voice(monkeypatch, cb, FlakyEars("The giant ranged the beef.", "The client changed the brief again."))
    assert asyncio.run(v.speak(TEXT)) is True
    assert len(cb.calls) == 2 and abs(dominant_hz(player.clips[1]) - 330) < 10


def test_long_expressive_sentence_only_performs_its_clause():
    from kobox.engine import plan, settings
    t = ("Okay. [laughs] Nothing is on fire, for once, and the rest of the list can comfortably wait "
         "until this afternoon when you are back.")
    lines = plan(t, "neutral", settings({}))
    exp = [ln for ln in lines if ln.expressive]
    assert len(exp) == 1 and exp[0].cb_text == "[laugh] Nothing is on fire,"
    assert lines[-1].text.startswith("for once") and not lines[-1].expressive


def test_prepared_hum_plays_instantly(isolated_home, monkeypatch):
    cb = FakeCB(delay=5.0)
    v, player = make_voice(monkeypatch, cb, FakeEars("x"))
    v.kobox.bank.add("Mm-hmm.", tone(0.5, -27, 330))
    t = time.time()
    assert asyncio.run(v.speak("Mm-hmm. That sounds right.")) is True
    assert time.time() - t < 1.0 and not cb.calls                       # no render, no wait
    assert abs(dominant_hz(player.clips[0]) - 330) < 10 and v.kobox.last["by_bank"] == 1


@pytest.mark.parametrize("text,cb", [
    ("Okay. [happy] We got the sign-off!", "[happy] We got the sign-off!"),
    ("Okay. [gasps] Wait, Friday?", "[gasp] Wait, Friday?"),
    ("Okay. [groans] The brief changed again.", "[groan] The brief changed again."),
    ("Okay. [sarcastically] Oh good, another change.", "[sarcastic] Oh good, another change."),
    ("Okay. [tearfully] I'm so sorry.", "[crying] I'm so sorry."),
    ("Okay. [clears throat] Right, the agenda.", "[clear throat] Right, the agenda."),
    ("Okay. [shh] Between us.", "[shush] Between us."),
    ("Okay. Ugh, not the printer again.", "Ugh, not the printer again."),
    ("Okay. Aww, that's lovely.", "Aww, that's lovely."),
])
def test_full_palette_reaches_chatterbox(text, cb):
    from kobox.engine import plan, settings
    exp = [ln for ln in plan(text, "neutral", settings({})) if ln.expressive]
    assert len(exp) == 1 and exp[0].cb_text == cb


def test_kokoro_never_reads_new_cues_aloud():
    from kobox.prosody import strip_cues
    assert strip_cues("[gasps] Wait. [sarcastically] Great. [clears throat] Right.") == "Wait. Great. Right."


def test_parts_split_sound_from_words():
    from kobox.expressions import parts
    assert parts("[laughs] Okay, fine.") == ("laugh", "Okay, fine.", None)
    assert parts("Ugh, not the printer again.") == ("ugh", "Not the printer again.", None)
    assert parts("That's it. [sighs]") == (None, "That's it.", "sigh")
    assert parts("[happy] [laughs] You did it!") == ("laugh", "[happy] You did it!", None)


def test_late_live_take_falls_back_to_real_bank_laugh(isolated_home, monkeypatch):
    v, player = make_voice(monkeypatch, FakeCB(delay=2.0), FakeEars("x"),
                           {"kobox": {"expression_wait_s": 0.3, "grace_max_s": 0.3}})
    v.kobox.bank.add("laugh", tone(0.4, -27, 500))
    assert asyncio.run(v.speak("Right. [laughs] Okay, fine. Moving on.")) is True
    clip = player.clips[1]
    assert abs(dominant_hz(clip[: int(SR * 0.3)]) - 500) < 15          # her real laugh first...
    assert abs(dominant_hz(clip[int(SR * 0.6):]) - KOKORO_HZ) < 15     # ...then Kokoro says the words
    assert v.kobox.last["by_bank"] == 1 and "expression bank" in v.kobox.label()
    assert v.kobox.last["takes"]


def test_interjection_only_line_is_instant(isolated_home, monkeypatch):
    cb = FakeCB(delay=5.0)
    v, player = make_voice(monkeypatch, cb, FakeEars("x"))
    v.kobox.bank.add("aww", tone(0.4, -27, 500))
    t = time.time()
    assert asyncio.run(v.speak("Aww. That's lovely of you.")) is True
    assert time.time() - t < 1.0 and abs(dominant_hz(player.clips[0]) - 500) < 15


def test_feedback_retires_disliked_and_varies_takes(isolated_home):
    from kobox.expressions import Bank
    b = Bank()
    t1, t2 = b.add("laugh", tone(0.3)), b.add("laugh", tone(0.3, f=300))
    picks = [b.pick("laugh")[0] for _ in range(6)]
    assert all(picks[i] != picks[i + 1] for i in range(5))                 # never the same take twice running
    assert b.feedback([t1], -0.6) == [t1] and b.active("laugh") == [t2]
    assert b.set_expressiveness(+5) == 4 and Bank().prefs["expressiveness"] == 4   # persisted


@pytest.mark.parametrize("heard,sound", [("ha ha ha", "laugh"), ("Hahaha.", "laugh"), ("*sigh*", "sigh"),
                                         ("Oh.", "ooh"), ("Mm-hmm.", "Mm-hmm."), ("Whoa!", "whoa"),
                                         ("I love you baby", None), ("on", None), ("", None)])
def test_only_pure_expressions_are_learned(heard, sound):
    from kobox.expressions import classify_heard
    assert classify_heard(heard) == sound



def test_bank_export_and_seed_round_trip(isolated_home, tmp_path):
    from kobox.expressions import Bank
    b = Bank()
    keep = b.add("laugh", tone(0.3), score=1.6)
    gone = b.add("sigh", tone(0.3))
    b.feedback([gone], -0.6)                                             # retired: not shipped
    b.add("laugh", tone(0.3), source="converted")                        # someone else's recording: never shipped
    assert b.export(tmp_path / "ship") == 1
    fresh = Bank(tmp_path / "home")                                     # a new install, empty...
    assert fresh.seed(tmp_path / "ship") == 1                           # ...gets her real take with its learned score
    assert fresh.index[keep]["score"] == 1.6 and fresh.pick("laugh")[0] == keep


@pytest.mark.parametrize("said,expect", [
    ("That laugh sounded off.", ("down", "laugh")), ("I loved that laugh!", ("up", "laugh")),
    ("Fewer expressions please", ("less", None)), ("be more expressive", ("more", None)),
    ("The laugh track in the video sounded off to everyone at the review", None), ("hello there", None),
])
def test_parse_feedback(said, expect):
    from kobox.engine import parse_feedback
    assert parse_feedback(said) == expect


def test_feedback_api_retires_only_the_named_sound(isolated_home, monkeypatch):
    v, _ = make_voice(monkeypatch, FakeCB(), FakeEars("x"))
    laugh, sigh = v.kobox.bank.add("laugh", tone(0.3)), v.kobox.bank.add("sigh", tone(0.3))
    v.kobox.last = {"takes": [laugh, sigh]}
    r = v.kobox.feedback("down", "laugh")
    assert r["retired"] == [laugh] and v.kobox.bank.active("sigh") == [sigh] and "dropped that laugh" in r["note"]
    assert v.kobox.feedback("less")["expressiveness"] == 1
    v.kobox.last = {"takes": []}
    assert v.kobox.feedback("down")["live"] is True


def test_prompt_guide_tracks_expressiveness():
    from kobox.engine import prompt_guide
    g = prompt_guide(0)
    assert "[laughs]" in g and "rarely" in g and "don't copy them back" in g
    assert "generously" in prompt_guide(4)


def test_hear_learns_from_the_listener(isolated_home, monkeypatch):
    from datetime import datetime
    v, _ = make_voice(monkeypatch, FakeCB(), FakeEars("x"))
    laugh = v.kobox.bank.add("laugh", tone(0.3))
    v.kobox.last = {"takes": [laugh], "at": datetime.now().isoformat(timespec="seconds")}
    assert v.kobox.hear("hahaha 😂")["kind"] == "laughed_back"
    assert v.kobox.bank.index[laugh]["score"] == 1.15                      # they laughed back: that take scores up
    assert v.kobox.hear("What's on my calendar tomorrow?") is None          # ordinary talk: left to the assistant
    r = v.kobox.hear("that laugh sounded off")
    assert r["kind"] == "feedback" and r["takes"] == 1 and "less" in r["note"]   # liked once, so one 👎 makes it rarer
    assert v.kobox.hear("that laugh sounded off")["retired"] == [laugh]          # a second 👎 retires it


def test_bank_update_merges_new_takes_and_keeps_local_scores(isolated_home, tmp_path):
    from kobox.expressions import Bank
    up = Bank(tmp_path / "upstream")
    a = up.add("laugh", tone(0.3))
    up.export(tmp_path / "ship1")
    mine = Bank(tmp_path / "mine")
    mine.seed(tmp_path / "ship1")
    mine.feedback([a], 0.3)                                                  # my own taste
    b = up.add("sigh", tone(0.3))                                            # upstream learned a new take
    up.export(tmp_path / "ship2")
    assert mine.seed(tmp_path / "ship2", merge=True) == 1
    assert mine.index[a]["score"] == 1.3 and b in mine.index
