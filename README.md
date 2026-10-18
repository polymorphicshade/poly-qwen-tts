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
| Cloned voice with full emotion/style control | Styled Clone | `POST /api/tts/styled-clone` | `poly-tts styled-clone` |

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
* `[style: …]…[/style]` adds an instruction for that span only. Cloned voices ignore it, because the Base model takes no instructions. Use Styled Clone to style a cloned voice.

## Styled Clone

The Base model that clones voices takes no instructions, so plain clones can't follow emotion or style controls. Styled Clone works around this in two steps:

1. A **performer** speaks the text with every control (emotion, intensity, delivery, pace, pitch, intonation, pauses, free-text instruction and `[style:…]` markup). The performer is a native speaker or designed preset, a voice you describe, or by default a VoiceDesign voice picked to match the target's pitch range.
2. **Zero-shot voice conversion** gives that performance the cloned voice's timbre and keeps its timing and intonation. It uses [Seed-VC](https://github.com/Plachtaa/seed-vc) (best quality), or [OpenVoice v2](https://github.com/myshell-ai/OpenVoice) as a lighter, faster fallback.

The target can be an uploaded or recorded clip, a saved clone or a designed preset. A performer close to the target's gender and pitch converts best. Output is 22.05 kHz.

Setup: both converters run from a source checkout, because their PyPI packaging pins old numpy/protobuf versions that conflict with qwen-tts. Their weights download from Hugging Face on first use.

```sh
uv sync --extra qwen --extra vc
git clone https://github.com/Plachtaa/seed-vc data/vendor/seed-vc          # tested at 51383ef
git clone https://github.com/myshell-ai/OpenVoice data/vendor/OpenVoice    # optional fallback, tested at 74a1d14
```

```sh
poly-tts styled-clone "I can't believe you did this!" --ref-audio me.wav --emotion anger --intensity strong --delivery shouting
poly-tts styled-clone "Come closer..." --voice me --performer serena --delivery whisper --vc-steps 40
curl -X POST localhost:8000/api/tts/styled-clone -F text="No way!" -F ref_audio=@me.wav \
  -F controls='{"emotion":"surprise","intensity":"strong"}' -F conversion='{"steps":30}' -o styled.wav
```

Conversion options are `steps` (Seed-VC diffusion steps, default 25: 30–50 sound best, 4–10 are fast), `cfg_rate` (Seed-VC guidance, default 0.7) and `tau` (OpenVoice, default 0.3). `GET /api/info` reports which converters are installed under `voice_conversion`.

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

## Generation options

These sampling options tune how a take sounds. Leave them unset to use the model defaults. The web UI shows this advice next to each box, `poly-tts <command> --help` prints it, and `GET /api/info` returns it as `generation_options`.

| option | default | what it does |
|---|---|---|
| `temperature` | 0.9 | Lower (0.6–0.8) gives a steadier read. Higher (1.0–1.2) gives livelier intonation but can slur or glitch. |
| `top_p` | 1.0 | Lower (0.8–0.95) trims unlikely sounds, so speech is cleaner but flatter. |
| `top_k` | 50 | Smaller (10–30) is safer and more monotone. Larger allows more variety. |
| `repetition_penalty` | 1.05 | Raise it to 1.1–1.2 if the voice stutters or loops. Too high distorts the voice. |
| `max_new_tokens` | 8192 | Caps the audio length per segment (about 12 tokens per second). |
| `seed` | random | The same seed, text and settings give the same take. Fix it, then change one option at a time. |

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
| `POLY_TTS_VC` | `auto` (Styled Clone converter: Seed-VC if installed, else OpenVoice; or `seed-vc` / `openvoice`) |
| `POLY_TTS_SEED_VC_DIR` / `_OPENVOICE_DIR` | `<data dir>/vendor/seed-vc` / `<data dir>/vendor/OpenVoice` |

The 0.6B CustomVoice checkpoint ignores instructions, so the defaults use 1.7B.

## Development

```sh
uv run pytest      # uses the mock backend; no GPU needed
```

Layout: `src/poly_tts/` contains `engine.py` (the feature logic), `backends/` (qwen, mock), `asr.py` (Whisper transcription), `vc.py` (Styled Clone voice conversion), `controls.py` (controls and markup), `voices.py` and `library.py` (the voice library), `api/` (FastAPI), `web/` (UI) and `cli.py`.
