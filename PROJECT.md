# Kobox: project

The working document for Kobox: what it is, how it's put together, what was decided and why, what's next, and how to build and release a new version. For using Kobox, start with the [README](README.md).

## What and why

Kobox makes **one voice from two speech engines**. **Kokoro** speaks the everyday lines instantly; **Chatterbox** performs the laughs, sighs, gasps and moods for real; **Whisper** checks every performed line before it plays.

It was built for REHA, a local, private AI assistant. Her voice needed to start talking at once *and* express herself like a person, without the gaps, loudness jumps and garbled words that come from switching engines.

**Who it's for:** anyone building a local voice assistant, companion, game character or narrator who wants expressive speech without a cloud TTS.

## Status

| | |
|---|---|
| Version | **0.2.1** (see [CHANGELOG](CHANGELOG.md)) |
| Tested on | macOS, Apple Silicon (M4 Pro, 24 GB) |
| Expected to work | other Apple Silicon Macs. Linux works with Chatterbox on CPU (slower, so more lines come from the expression bank) |
| Tests | model-free unit tests run in CI on every push |

## Design principles

1. **Start at once.** The first sound comes from Kokoro in a fraction of a second; nothing waits for the slow engine.
2. **One voice.** Every performed line is trimmed, tone-matched and level-matched to Kokoro, and plays in one continuous stream with crossfades.
3. **Checked before heard.** A performed line that says the wrong words, numbers or names never plays.
4. **Never silent, never fake.** Every expression has a fallback: live take, then a real take from the bank plus words, then a short beat. A cue is never read aloud, and a laugh is never imitated by a speech engine saying "ha ha".
5. **Her voice never changes.** Learning from recordings borrows the *expression* only, re-voiced from her own reference. Sung notes and words are rejected.
6. **Everything local.** No audio or text leaves the machine.

## Architecture

| Module | Role |
|---|---|
| `kobox/engine.py` | The Kobox engine. Plans lines (plain vs expressive, mood lines, instant start, clause shortening). Renders Kokoro and Chatterbox in parallel. Matches tone and level, runs the accuracy check, applies deadlines and adaptive waits, and runs the fallback chain. Owns the single output stream (`StreamPlayer`) and the per-reply report and label. |
| `kobox/expressions.py` | The expression bank: rendering, checking, storing, picking (weighted, never repeating) and scoring takes. Weekly refresh, feedback, and expressiveness level. Stage 2 of learning from recordings. |
| `kobox/expression_convert.py` | Stage 1 of learning from recordings, run in the Chatterbox environment: cuts a recording into moments, rejects sung notes (pYIN pitch), voice-converts the rest into the reference voice. |
| `kobox/speech.py` | Voice wrapper: Kokoro rendering with prosody, the Chatterbox worker client, cue mapping to Chatterbox tokens, the engine fallback chain. |
| `kobox/prosody.py` | Cue vocabulary and parser (`[laughs]`, `*softly*`, `(sighs)`), delivery styles, clause-level tempo, pitch and pause planning for Kokoro. |
| `kobox/textproc.py` | Word error rate with forgiving normalisation (numbers, fillers, compounds), and the words that must be heard (numbers, names). |
| `kobox/hearing.py` | Local Whisper (whisper.cpp). `verify()` never queues behind live listening. |
| `kobox/audio_out.py` | The audio output process: one queue, crossfades, line-start markers, glitch measurement. A busy host can't starve it, and replies can't overlap. |
| `kobox/chatterbox_worker.py` | Long-lived Chatterbox Turbo process speaking JSON lines; runs in its own Python environment. |

`kobox/bank/` is her shipped expression bank (rendered takes with learned scores), exported from REHA on every sync. `kobox/__main__.py` holds the CLI and the local `serve` service. `Kobox.hear()`, `Kobox.feedback()`, `parse_feedback()` and `prompt_guide()` (in `engine.py`) are the learning loop and the LLM guide.

How the pieces move is animated in the README: [how it works](docs/animations/how-it-works.svg), [no gap](docs/animations/no-gap.svg), [one voice](docs/animations/one-voice.svg), [never silent](docs/animations/never-silent.svg), [learns](docs/animations/learns.svg), [the loop](docs/animations/the-loop.svg).

## Decisions and why

| Decision | Why |
|---|---|
| Two engines, not one | Kokoro alone can't laugh. Chatterbox alone takes 1-3 s per line and would delay every reply. |
| No voice conversion on spoken words | Tested in phase 0: converting Kokoro lines into the expressive voice blurred words ("fifteen" became "fifty"). It's used only for wordless expressions, where there's nothing to blur. |
| Whisper gate: WER ≤ 0.25, and every number and name must match | Chatterbox occasionally garbles a word. A wrong number is worse than a missing laugh. |
| One retry, then fall back | A second render usually fixes a garbled take. A third would make the reply wait too long. |
| Wait for requested expressions (up to 8 s safety cap), but only 1.2 s when the bank can cover, and 1.5 s at the start of a reply | People prefer the real expression to a substitute, but a long silence feels broken. The bank makes the short wait possible. |
| Chatterbox performs only the clause the laugh belongs to | Render time grows with text length. Kokoro finishes the sentence in the same voice. |
| Expression bank of real takes | Instant playback, variety, a better fallback than a pause, and something feedback can shape. |
| Plain text is planned sentence by sentence | Long replies start at once instead of after the whole text is rendered. |
| Hums and interjections split into their own lines | Kokoro reads "Mmm hmm" and "Pfft" as words. Chatterbox, or the bank, performs them. |
| Reject held musical pitch when learning | A song's sung "ooh" would otherwise teach her to sing. |
| Audio is played from a separate process | In a busy host, the in-process audio callback was starved: 546 glitches in 7 replies under load, 0 from a separate process. Found from a user report of stutters in long conversations. |
| Chatterbox's Perth watermark is kept | Responsible use of a voice-cloning model. |

## Roadmap

- [ ] **Evaluation:** a 30-line test set, a per-line loudness consistency metric, and a blind A/B listening page. Then publish the numbers.
- [ ] **Songs as a source:** vocal separation with Demucs before cutting moments.
- [ ] **Review UI for learned takes:** listen to, approve or remove takes learned from recordings.
- [ ] **Packaging:** publish to PyPI, plus a one-command installer for the Chatterbox environment.
- [ ] **Broader platform support:** Linux and Windows audio output, and CUDA for Chatterbox.
- [ ] **Monthly expressions report:** most-loved takes, what was retired, what was learned.

## Building a new version

**Set up for development:**
```bash
./scripts/setup.sh                           # environments + models
.venv/bin/python -m pytest -q tests          # unit tests (no models needed)
.venv/bin/python -m kobox say "Okay. [laughs] Testing the new build."   # real models
```

**Where to change things:**
- planning or timing: `engine.py` (`plan`, `DEFAULTS`);
- how a line is made to match: `engine.match`;
- the cue vocabulary: `prosody.py`, plus `speech.py` (`_CB_EVENT`, `CB_STYLE`);
- the bank: `expressions.py` (`SOUNDS`).

**Rules for every change:**
- Add a test.
- Keep tests free of models and audio devices.
- Measure voice changes with the real models before claiming they work.

**Releasing (SemVer):**
1. Bump `version` in `pyproject.toml` and `kobox/__init__.py`.
2. Add a section to `CHANGELOG.md`.
3. Commit, then tag: `git tag -s vX.Y.Z -m "vX.Y.Z" && git push --tags`. Commits and tags must be signed; `main` rejects unsigned commits.
4. Create a GitHub release from the tag, with the changelog section as notes.

Use major versions for breaking API or settings changes, minor for new capabilities (a new expression type, a new fallback), and patch for fixes and tuning.

**Contributing:**
- Issues and pull requests are welcome.
- The library is developed inside the app it was built for and exported here automatically. Accepted pull requests are applied there first, then return through the next sync.
- Please don't include personal data, names or recordings of real people in issues, tests or examples.

## Responsible use

- Every Chatterbox line carries Resemble AI's Perth watermark.
- Teach Kobox only from recordings you have the right to use. Never use it to imitate a real person without consent.
- The included reference voice is REHA's, for reproducible demos.
