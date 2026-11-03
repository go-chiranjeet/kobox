# Use Kobox in your own assistant

Kobox gives your assistant a voice that starts at once, performs real expressions, and learns what its listener likes. You need three things: the setup, a few lines of prompt for your LLM, and two calls (speak and hear).

## 1. Install (once)

```bash
git clone https://github.com/go-chiranjeet/kobox && cd kobox
./scripts/setup.sh            # Kobox + Chatterbox environments, ~700 MB of models
.venv/bin/python -m kobox doctor
.venv/bin/python -m kobox say "Okay. [laughs] It works."
```

**Her expression bank ships with the repo:** 60 real takes across 20 sounds, with the scores REHA's listener gave them. It's copied into `~/.kobox` on first use, so expressions work immediately, with no rendering needed. After a `git pull`, run `python -m kobox update-bank` to add her newer takes; your own scores are kept.

## 2. Teach your LLM the cues

Add the expressions guide to your assistant's system prompt:

```bash
.venv/bin/python -m kobox prompt > kobox-prompt.txt          # or GET /prompt, or engine.prompt_guide()
```

The guide covers:
- the cues Kobox performs: `[laughs]`, `[sighs]`, `[gasps]`, `[sarcastic]`, `Aww`, `Mm-hmm`…
- how often to use them;
- to **respond** to the user's emotions rather than mirror them (they laugh: enjoy it with them; they're frustrated: be on their side; they're angry: stay calm).

The frequency line follows the listener's "less/more" feedback. Regenerate the prompt now and then, or fetch `/prompt` each session.

## 3. Speak, and let Kobox hear the listener

Every reply goes to **speak**. Every user message goes to **hear** first. If `hear` returns something, Kobox has handled it (feedback) and you can say its `note`. Otherwise it's ordinary conversation for your assistant.

### From Python

```python
import asyncio
from kobox.speech import Voice
from kobox.hearing import Ears
from kobox.__main__ import VOICE, HEARING          # REHA's exact settings

voice = Voice(dict(VOICE))
voice.kobox.ears = Ears(dict(HEARING))

async def on_user_message(text: str) -> None:
    learned = voice.kobox.hear(text)              # feedback phrases and laugh-backs teach her bank
    if learned and learned.get("note"):
        await voice.speak(learned["note"])         # e.g. "Got it, I've dropped that laugh."
        return
    reply = await my_llm(text)                     # your assistant, with the expressions guide in its prompt
    await voice.speak(reply, mood="warm")
    print(voice.kobox.label())                     # "Kobox · 2 of 2 expressions performed · started 0.2 s"

async def main():
    await voice.warm()                             # loads models; renders any missing takes in the background
    await on_user_message("I finally fixed the bug haha")
asyncio.run(main())
```

### From any language: the local service

```bash
.venv/bin/python -m kobox serve --port 7766           # add --token SECRET (or KOBOX_TOKEN) to require a token
```

| Call | Body | Does |
|---|---|---|
| `POST /speak` | `{"text": "...", "mood": "warm"}` | speaks; returns `label` and the report once it has finished |
| `POST /heard` | `{"text": "<user message>"}` | learns from feedback phrases and laugh-backs; `{"kind":"conversation"}` otherwise |
| `POST /feedback` | `{"value": "up\|down\|less\|more", "sound": "laugh"}` or `{"text": "that laugh sounded off"}` | for your own 👍/👎 buttons |
| `POST /stop` | `{}` | stops speaking at once |
| `GET /prompt` | | the expressions guide for your LLM |
| `GET /status` | | engine, bank summary, last reply |

```bash
curl -s localhost:7766/speak -H 'Content-Type: application/json' -d '{"text":"Aww. [happy] That is lovely news!"}'
```

**Security:**
- The service listens on `127.0.0.1` only.
- Requests carrying a browser `Origin` or `Referer` header are refused, and so are non-JSON bodies, so a website can't make it speak.
- Bodies are capped at 64 KB.
- Use `--token` if other local users share the machine.

## 4. The learning loop

| Signal | Where it comes from | What Kobox does |
|---|---|---|
| 👍 / 👎 | your UI → `feedback("up"/"down")` | the takes used in that reply are used more often, or retired |
| "That laugh sounded off" / "I loved that laugh" | user message → `hear()` | the same, for the named sound only (within 10 minutes of the reply) |
| "Fewer expressions" / "Be more expressive" | user message → `hear()` or `feedback("less"/"more")` | expressiveness 0-4 changes how often the prompt guide asks for cues |
| The user laughs back ("haha", 😂) | user message → `hear()` | the takes in the last reply score up |
| Weekly refresh | automatic during `warm()` | a fresh take per sound; the weakest retires |
| Someone else's recording | `kobox learn file.wav` | their laughs and sighs, re-voiced into her voice; singing and words rejected |
| Her upstream learning | `git pull` + `kobox update-bank` | new takes from REHA's bank, with your scores kept |

Everything learned stays in `~/.kobox` on your machine.

## 5. Your own voice

The bank and reference are REHA's voice. For a different voice:
1. Point `KOBOX_VOICE_REF` at a clean 10-20 s clip **you have the right to use**.
2. Delete `~/.kobox/.runtime/state/expressions`.
3. Run `kobox fill-bank` to render a bank in the new voice.

Change Kokoro's voice blend in `~/.kobox/config/kobox.json`:

```json
{"voice": {"voice": "af_heart", "speed": 1.0, "warmth": 0.25, "kobox": {"expression_wait_s": 8, "bank_wait_s": 1.2}}}
```
