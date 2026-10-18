# poly-qwen-tts

A toolkit around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with a **web UI**, a **REST API** and a **CLI**:

| Feature | Web UI mode | API | CLI |
|---|---|---|---|
| Voice design from a text description | Voice Design | `POST /api/tts/design` | `poly-tts design` |
| Emotion & prosody control | Emotion & Prosody | `POST /api/tts/control` | `poly-tts control` |
| Zero-shot voice cloning (≥ 3 s clip) | Voice Clone | `POST /api/tts/clone` | `poly-tts clone` |
| Cross-lingual voice cloning | Cross-lingual | `POST /api/tts/cross-lingual` | `poly-tts cross-lingual` |
| 50-voice library (+ your saved clones) | Voice Library | `POST /api/tts/voice`, `GET/POST/DELETE /api/voices` | `poly-tts speak`, `poly-tts voices` |
| Preset voice + style instruction | Style | `POST /api/tts/style` | `poly-tts style` |
| Speech-to-text for reference clips (Whisper) | automatic in the clone modes | `POST /api/transcribe` | `poly-tts transcribe` |

Supported languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian (or `Auto`).

## Setup

Requires an NVIDIA GPU with about 12 GB of VRAM to keep all three 1.7B models loaded (fewer with `POLY_TTS_MAX_LOADED_MODELS`). Uses [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra qwen          # installs torch (CUDA 12.8 wheels) + qwen-tts
uv run poly-tts serve         # http://127.0.0.1:8000  (API docs at /docs)
```

The weights (~4.3 GB per model) are downloaded from Hugging Face the first time each model is used.

To try the UI, API and CLI without a GPU, use the mock backend, which outputs placeholder tones instead of speech: `uv sync && uv run poly-tts --backend mock serve`.

## Web UI

Each **tab is an independent workspace** with every control. Use `+` (or Alt+N) to open one and `×` (or middle-click, or Alt+W) to close it. Double-click a tab to rename it. Workspaces can generate in parallel, and their settings are remembered in the browser. In the clone modes you can upload a clip or record one from the microphone, and save it to the voice library.

## Automatic reference transcripts

Cloning is most faithful when the Base model gets the transcript of the reference clip. If you leave the transcript empty, it is generated with Whisper (`openai/whisper-large-v3-turbo`, ~1.6 GB, downloaded on first use):

* **Web UI**: uploading or recording a clip fills the empty *Reference transcript* box. Check it and fix it if needed. *Transcribe with Whisper* runs it again on demand.
* **API / CLI**: a clone, cross-lingual or save-voice request without `ref_text` is transcribed on the server. The transcript is returned as `ref_text_auto` in the result metadata. `POST /api/transcribe` and `poly-tts transcribe clip.wav` return a transcript on their own.
* Saved voices that have no transcript get one the first time they are used, and it is stored with the voice.

Pass `x_vector_only` (`--x-vector-only`) to skip the transcript, or set `POLY_TTS_AUTO_TRANSCRIBE=0` to turn auto-transcription off. Either way, cloning without a transcript uses the speaker embedding only. If transcription fails, the request still succeeds in embedding-only mode and returns a warning.

## Granular control

Structured controls are compiled into an instruction for the model:

| control | values |
|---|---|
| emotion | neutral, joy, excitement, tenderness, calm, sadness, anger, fear, panic, surprise, incredulity, disgust, sarcasm, nervousness, pride |
| intensity | slight, moderate, strong, extreme |
| delivery | normal, whisper, soft, breathy, loud, shouting, crying, laughing, trembling |
| pace | very_slow, slow, normal, fast, very_fast |
| pitch | very_low, low, normal, high, very_high |
| intonation | natural, flat, rising, falling, expressive |
| pauses | natural, none, few, frequent, dramatic |

Inline markup works in every mode:

```
I told you not to open that door! [pause 800ms] [style: whispering, terrified]It's right behind you.[/style]
```

* `[pause]` inserts 0.5 s of silence. `[pause 800ms]`, `[pause 1.5s]` and `[pause 2]` set an explicit length.
* `[style: …]…[/style]` adds an instruction for that span only. Cloned voices ignore it, because the Base model takes no instructions.

## Voice library

* **9 native speakers** (`vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`, `ono_anna`, `sohee`) from the CustomVoice model. They accept style instructions directly.
* **41 designed presets** spanning children to seniors, many character roles and all 10 languages (e.g. `narrator_deep`, `noir_detective`, `grandma_story`, `zh_storyteller`, `ja_anime_hero`, `fr_parisian`). The first time a preset is used, a reference clip is rendered with VoiceDesign and cached in `data/presets/`. Plain requests then clone that clip, so the timbre stays the same every time. Requests with a style instruction re-run VoiceDesign with the description plus the style.
* **Your clones**: save one with `poly-tts voices add`, `POST /api/voices` or the UI. They are stored in `data/voices/`.

`poly-tts voices list` lists every voice.

## CLI

```sh
poly-tts design "Welcome back, traveler." -d "Deep male voice with a slight rasp, calm authoritative narrator" -o narrator.wav
poly-tts control "No way!" --voice sohee --emotion panic --intensity extreme --delivery shouting
poly-tts clone "New words in my voice." --ref-audio me.wav --ref-text "What I said in the clip." --save-as "Me"
poly-tts cross-lingual "Bonjour à tous !" -t fr --voice me          # or --ref-audio clip.wav --ref-text ...
poly-tts speak "Ready for the big game?" --voice sports_commentator
poly-tts style "You actually won?" --voice serena -i "sound incredulous and excited"
echo "Text from a pipe" | poly-tts speak --voice aiden -o - > out.wav
poly-tts transcribe me.wav -l en                                  # Whisper transcript of a clip
```

Common options: `-l/--language`, `-o/--output` (`-` writes to stdout), `--json` (prints metadata), `--seed`, `--temperature`, `--top-p`, `--top-k`, `--max-new-tokens`, `--text-file`.

By default the CLI loads models in-process. For scripting many calls, run `poly-tts serve` once and add `--server http://127.0.0.1:8000` (or set `POLY_TTS_SERVER`) so the models stay loaded between calls.

## API

Synthesis endpoints return `audio/wav`. Metadata (duration, effective instruction, warnings, …) comes back in the `X-TTS-Metadata` header as URL-encoded JSON. JSON endpoints accept an optional `generation` object; multipart endpoints take it as a JSON string field.

```sh
curl -X POST localhost:8000/api/tts/style -H 'Content-Type: application/json' \
  -d '{"text":"You actually won?","voice_id":"serena","instruct":"sound incredulous"}' -o out.wav
curl -X POST localhost:8000/api/tts/clone -F text="Hello there" -F ref_audio=@me.wav -F ref_text="..." -o clone.wav
```

The full schema is at `/docs`.

## Configuration (environment variables)

| variable | default |
|---|---|
| `POLY_TTS_BACKEND` | `qwen` (`mock` for placeholder tones) |
| `POLY_TTS_DEVICE` | `auto` (`cuda:0`, `cpu`) |
| `POLY_TTS_DTYPE` | `bfloat16` |
| `POLY_TTS_ATTN` | `auto` (flash_attention_2 if installed, else sdpa) |
| `POLY_TTS_CUSTOM_VOICE_MODEL` / `_VOICE_DESIGN_MODEL` / `_BASE_MODEL` | the `Qwen/Qwen3-TTS-12Hz-1.7B-*` checkpoints |
| `POLY_TTS_MAX_LOADED_MODELS` | `3` (least recently used model is unloaded first) |
| `POLY_TTS_DATA_DIR` | `./data` |
| `POLY_TTS_MIN_REF_SECONDS` / `_MAX_REF_SECONDS` | `3` / `60` |
| `POLY_TTS_AUTO_TRANSCRIBE` | `1` (transcribe reference clips that have no transcript; `0` to disable) |
| `POLY_TTS_ASR_MODEL` | `openai/whisper-large-v3-turbo` |

The 0.6B CustomVoice checkpoint ignores instructions, so the defaults use 1.7B.

## Development

```sh
uv run pytest      # uses the mock backend; no GPU needed
```

Layout: `src/poly_tts/` contains `engine.py` (the feature logic), `backends/` (qwen, mock), `asr.py` (Whisper transcription), `controls.py` (controls and markup), `voices.py` and `library.py` (the voice library), `api/` (FastAPI), `web/` (UI) and `cli.py`.
