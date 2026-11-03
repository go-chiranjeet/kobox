# Changelog

All notable changes to Kobox. Versions follow [Semantic Versioning](https://semver.org).

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
