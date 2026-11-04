# Changelog

All notable changes to Kobox. Versions follow [Semantic Versioning](https://semver.org).

## 0.2.1 (2026-11-04)

**Fixes stutters, breaks and slow starts in real use.**
- **Audio now plays from its own process** (`kobox/audio_out.py`). Before, it was fed from the host's Python process, so whenever the host was busy (UI messages, an LLM streaming, rendering), the speaker ran dry.
  - Measured under simulated load: **546 glitches in 7 replies before, 0 after** (worst lateness 195 ms before, 0.3 ms after).
  - One queue for all audio, so two replies can never overlap.
  - Line-start markers fire when a line is actually heard.
  - A watchdog restarts a stuck player instead of hanging the host.
  - The player's errors are logged to `logs/kobox-audio.log`.
- **Interrupted replies no longer delay the next one:** Kokoro stops rendering lines that will never be spoken.
- **A faster first reply:** Kokoro and the audio process are warmed up at start.
- **Every reply is measured:** glitches, worst lateness and per-step timings, in `last["audio"]` and `last["phases_ms"]`.
- **Fixed:** the Whisper check crashed when no vocabulary prompt was set.

## 0.2.0 (2026-11-03)

- **Ships her expression bank:** 60 checked takes across 20 sounds, with learned scores, so a new install expresses itself immediately. `update-bank` merges newer takes and keeps your own scores.
- **Learning loop in the library:**
  - `Kobox.hear(user_text)` learns from feedback phrases and laugh-backs;
  - `Kobox.feedback(up|down|less|more, sound=…)`;
  - `parse_feedback()`.
- **`prompt_guide()` / `kobox prompt`:** the expressions guide for any assistant's LLM (cues, frequency, respond rather than mirror).
- **`kobox serve`:** a local HTTP service (`/speak`, `/heard`, `/feedback`, `/stop`, `/prompt`, `/status`). It listens on 127.0.0.1 only, refuses browser requests and non-JSON bodies, and takes an optional token.
- **Optional settings file** (`~/.kobox/config/kobox.json`); the defaults are REHA's exact settings.
- **Docs:** INTEGRATION.md; a new "the loop" animation.

## 0.1.0 (2026-11-03)

First public release.

- **Engine:**
  - Kokoro and Chatterbox as one voice: sentence-level planning, an instant first line, and parallel rendering.
  - Every performed line is trimmed, tone-matched and level-matched, then played in one output stream with 20 ms equal-power crossfades.
- **Accuracy check:** Whisper large-v3-turbo with forgiving word error rate (≤ 0.25); numbers and names must match; one re-render before falling back.
- **Expressions:**
  - The full Chatterbox Turbo palette: laugh, chuckle, sigh, gasp, groan, sniff, clear throat, shush, hum; happy, surprised, sarcastic, dramatic, fear, angry, crying, whispering.
  - Performed interjections: ooh, aww, ugh, whoa, pfft, heh, uh-oh, yay, oops, phew, mm-hmm.
- **Expression bank:** checked takes for 20 sounds, instant and never repeated twice in a row; used as the fallback for late live takes; feedback scoring; weekly fresh takes.
- **Learning from recordings:** cut into moments, reject singing and words, re-voice into the reference voice, keep only pure expressions.
- **Never silent, never fake:** the fallback goes live take, then bank take plus words, then a beat. Cues are never read aloud.
- **CLI:** `kobox say | doctor | fill-bank | learn`.
