# Credits and references

Kobox is glue. It decides **who speaks which line, when, and how it is made to sound like one person**. The voices, the listening and the signal processing come from the projects below. Thank you to their authors.

Licences marked ✓ were read from the installed package's metadata. The others are the upstream project's stated licence: check the upstream page before you redistribute.

## Models

| Role in Kobox | Model | By | Licence | Link |
|---|---|---|---|---|
| Everyday speech: plain lines, instantly | **Kokoro-82M** (int8 ONNX build) | hexgrad | Apache-2.0 (model card) | [huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| Expressive speech: real laughs, sighs, gasps, moods | **Chatterbox Turbo** | Resemble AI | MIT ✓ | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) · [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) |
| Re-voicing someone else's expressions into her voice | **Chatterbox voice conversion** (`ChatterboxVC`, Turbo S3Gen) | Resemble AI | MIT ✓ | same as above |
| Accuracy check: did the performed line say the right words? | **Whisper large-v3-turbo**, as `ggml-large-v3-turbo-q5_0` | OpenAI; ggml conversion by the whisper.cpp project | MIT | [github.com/openai/whisper](https://github.com/openai/whisper) · [ggerganov/whisper.cpp on Hugging Face](https://huggingface.co/ggerganov/whisper.cpp) |

Kokoro builds on **StyleTTS 2** (Li et al.) and **ISTFTNet**. Chatterbox credits **CosyVoice**, **HiFT-GAN**, **Real-Time-Voice-Cloning** and **Llama**. Both upstream pages list their full lineage.

## Libraries

| Library | Used for | Licence |
|---|---|---|
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (thewh1teagle) | runs Kokoro on ONNX Runtime; provides the model files | see upstream (MIT) |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) via espeakng-loader | phonemes for Kokoro (pulled in by kokoro-onnx, not bundled here) | GPL-3.0 (upstream) |
| [ONNX Runtime](https://onnxruntime.ai) | Kokoro inference | MIT ✓ |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) + [pywhispercpp](https://github.com/absadiki/pywhispercpp) | fast local Whisper on Apple Silicon | MIT ✓ |
| [PyTorch](https://pytorch.org) | Chatterbox inference (Metal on Apple Silicon) | BSD-3-Clause ✓ |
| [Perth](https://github.com/resemble-ai/Perth) (Resemble AI) | imperceptible watermark that Chatterbox adds to every line it renders | MIT ✓ |
| [librosa](https://librosa.org) | pYIN pitch tracking for the "is this singing?" check (Mauch & Dixon, 2014) | ISC ✓ |
| [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) / PortAudio | one continuous output stream with crossfades | MIT ✓ |
| [python-soundfile](https://github.com/bastibe/python-soundfile) | reading and writing audio in the Chatterbox worker | BSD-3-Clause ✓ |
| [NumPy](https://numpy.org) | all the audio maths | BSD-3-Clause ✓ |

## Ideas that shaped it

- **Bracketed audio tags** such as `[laughs]`, `[whispers]` and `[sighs]`, popularised by ElevenLabs' expressive speech models. Kobox reads the same style of cue, but performs it with open models running locally.
- **Chatterbox Turbo's own paralinguistic tokens** (`[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[clear throat]`, `[shush]`, `[happy]`, `[surprised]`, `[sarcastic]`, `[dramatic]`, `[fear]`, `[angry]`, `[crying]`, `[whispering]`). Kobox maps its cue vocabulary onto these.
- **Word error rate** as a gate on generated speech, a standard ASR metric used here in reverse: to check a synthesiser rather than a recogniser.
- **Equal-power crossfades** from audio engineering, so joins between engines can't be heard.
- **[Demucs](https://github.com/facebookresearch/demucs)** (Meta, MIT) is planned for separating vocals when the source recording is a song.

## The voice

The reference clip in `kobox/voices/reha-ref.wav` is the voice of **REHA**, the assistant Kobox was built for. It is included so the demos are reproducible. Please don't use Kobox to imitate a real person without their consent.
