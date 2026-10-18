"""Command-line interface: ``poly-tts <command> ...`` (also ``python -m poly_tts``).

Every core feature has a subcommand. By default models run in-process; pass ``--server URL``
(or set ``POLY_TTS_SERVER``) to send requests to a running ``poly-tts serve`` instead, which
avoids reloading models on every call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from .backends.base import GENERATION_OPTIONS
from .controls import Controls, control_options


def _read_text(args) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text in (None, "-"):
        if sys.stdin.isatty():
            raise SystemExit("error: provide TEXT, --text-file, or pipe text on stdin")
        return sys.stdin.read()
    return args.text


def _read_ref_text(args) -> Optional[str]:
    if getattr(args, "ref_text_file", None):
        return Path(args.ref_text_file).read_text(encoding="utf-8").strip()
    return getattr(args, "ref_text", None)


def _generation(args) -> Optional[dict]:
    gen = {k: getattr(args, k) for k in ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "seed")}
    gen = {k: v for k, v in gen.items() if v is not None}
    return gen or None


def _controls(args) -> Optional[dict]:
    c = {k: getattr(args, k, None) for k in Controls.model_fields}
    c = {k: v for k, v in c.items() if v}
    return c or None


def _write_output(args, wav: bytes, meta: dict) -> None:
    if args.output == "-":
        sys.stdout.buffer.write(wav)
        sys.stdout.buffer.flush()
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(wav)
        meta = {**meta, "output": str(out)}
    if args.json:
        print(json.dumps(meta, ensure_ascii=False), file=sys.stderr if args.output == "-" else sys.stdout)
    else:
        where = "stdout" if args.output == "-" else args.output
        print(f"wrote {where} ({meta.get('duration')}s audio, generated in {meta.get('elapsed')}s)", file=sys.stderr)
        if meta.get("instruct"):
            print(f"instruction: {meta['instruct']}", file=sys.stderr)
        if meta.get("ref_text_auto"):
            print(f"reference transcript (Whisper): {meta['ref_text_auto']}", file=sys.stderr)
        if meta.get("converter"):
            print(f"performer: {meta['performer']}, converter: {meta['converter']}", file=sys.stderr)
        for w in meta.get("warnings", []):
            print(f"warning: {w}", file=sys.stderr)


# ---------------------------------------------------------------------------- parser

def _add_common(p: argparse.ArgumentParser, *, language: bool = True) -> None:
    p.add_argument("text", nargs="?", help="Text to speak ('-' or omitted: read stdin). Supports [pause 500ms] and [style: ...]...[/style].")
    p.add_argument("--text-file", help="Read the text from a file.")
    if language:
        p.add_argument("-l", "--language", default="Auto", help="Language name or code (default: Auto).")
    p.add_argument("-o", "--output", default="output.wav", help="Output WAV path, or '-' for stdout (default: output.wav).")
    p.add_argument("--json", action="store_true", help="Print result metadata as JSON.")
    g = p.add_argument_group("generation options", "Unset options use the model defaults.")
    for name, opt in GENERATION_OPTIONS.items():
        default = f" (default: {opt['default']})" if opt["default"] is not None else ""
        # argparse treats '%' as a format character in help strings.
        g.add_argument(f"--{name.replace('_', '-')}", type=float if isinstance(opt["step"], float) else int,
                       metavar=name.upper(), help=opt["description"].replace("%", "%%") + default)


def _add_controls(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("emotion & prosody controls")
    for name, choices in control_options().items():
        g.add_argument(f"--{name}", choices=choices, metavar=name.upper(), help=f"one of: {', '.join(choices)}")


def _add_reference(p: argparse.ArgumentParser, required: bool) -> None:
    p.add_argument("--ref-audio", type=Path, required=required, help="Reference clip, >= 3 s (WAV/FLAC/OGG/MP3).")
    p.add_argument("--ref-text", help="Transcript of the reference clip (if omitted, Whisper transcribes it).")
    p.add_argument("--ref-text-file", help="Read the reference transcript from a file.")
    p.add_argument("--x-vector-only", action="store_true", help="Speaker-embedding-only cloning (no transcript; less faithful).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poly-tts", description="Qwen3-TTS: voice design, control, cloning and a voice library.")
    parser.add_argument("--server", default=os.environ.get("POLY_TTS_SERVER"),
                        help="Use a running poly-tts server (e.g. http://127.0.0.1:8000) instead of loading models locally.")
    parser.add_argument("--backend", choices=["qwen", "mock"], help="Local backend (default: qwen; env POLY_TTS_BACKEND).")
    parser.add_argument("--device", help="Local device, e.g. cuda:0 or cpu (default: auto).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show model loading logs.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("design", help="Synthesize with a voice described in natural language.")
    _add_common(p)
    p.add_argument("-d", "--description", required=True, help="Voice description (timbre, age, accent, emotion, pace, style).")
    _add_controls(p)

    p = sub.add_parser("control", help="Emotion / prosody control with a library voice or a described voice.")
    _add_common(p)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--voice", help="Library voice id (see `poly-tts voices list`).")
    src.add_argument("-d", "--description", help="Voice description instead of a library voice.")
    p.add_argument("--instruct", help="Extra free-text instruction appended to the controls.")
    _add_controls(p)

    p = sub.add_parser("clone", help="Zero-shot voice cloning from a reference clip.")
    _add_common(p)
    _add_reference(p, required=True)
    p.add_argument("--save-as", metavar="NAME", help="Also save the cloned voice to the library under NAME.")

    p = sub.add_parser("cross-lingual", help="Clone a voice and make it speak another language.")
    _add_common(p, language=False)
    p.add_argument("-t", "--target-language", required=True, help="Language to speak (e.g. French, zh).")
    p.add_argument("--source-language", help="Language of the reference clip (informational).")
    _add_reference(p, required=False)
    p.add_argument("--voice", help="Use a library voice instead of --ref-audio.")

    p = sub.add_parser("styled-clone", help="Cloned voice with full emotion/style control (performance + voice conversion).",
                       description="A performer voice speaks the text with the emotion/prosody controls, then zero-shot "
                                   "voice conversion (Seed-VC, or OpenVoice v2) converts it to the cloned voice's timbre.")
    _add_common(p)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--ref-audio", type=Path, help="Clip of the voice to clone (>= 3 s).")
    target.add_argument("--voice", help="Saved cloned voice or designed preset to clone.")
    performer = p.add_mutually_exclusive_group()
    performer.add_argument("--performer", metavar="VOICE",
                           help="Native speaker or designed preset that performs the read (default: auto by pitch).")
    performer.add_argument("--performer-description", help="Describe the performer's voice instead.")
    p.add_argument("--instruct", help="Extra free-text style instruction.")
    _add_controls(p)
    g = p.add_argument_group("voice conversion")
    g.add_argument("--converter", choices=["auto", "seed-vc", "openvoice"], help="Default: auto (Seed-VC if installed).")
    g.add_argument("--vc-steps", type=int, help="Seed-VC diffusion steps (default 25; 30-50 = best quality, 4-10 = fast).")
    g.add_argument("--vc-cfg-rate", type=float, help="Seed-VC guidance strength 0-1 (default 0.7).")
    g.add_argument("--vc-tau", type=float, help="OpenVoice temperature (default 0.3).")

    p = sub.add_parser("speak", help="Speak with a voice from the library.")
    _add_common(p)
    p.add_argument("--voice", required=True, help="Library voice id.")

    p = sub.add_parser("style", help="Library voice + natural-language style instruction.")
    _add_common(p)
    p.add_argument("--voice", required=True, help="Library voice id (native or designed).")
    p.add_argument("-i", "--instruct", required=True, help='Style, e.g. "sound incredulous and excited".')

    p = sub.add_parser("voices", help="List or manage the voice library.")
    vs = p.add_subparsers(dest="voices_command", required=True, metavar="ACTION")
    pl = vs.add_parser("list", help="List voices.")
    pl.add_argument("--kind", choices=["native", "designed", "cloned"])
    pl.add_argument("--language")
    pl.add_argument("--gender", choices=["female", "male", "neutral", "unknown"])
    pl.add_argument("--json", action="store_true")
    pa = vs.add_parser("add", help="Save a cloned voice from a reference clip.")
    pa.add_argument("name")
    _add_reference(pa, required=True)
    pa.add_argument("-l", "--language", default="Auto")
    pa.add_argument("--description")
    pr = vs.add_parser("remove", help="Delete a saved cloned voice.")
    pr.add_argument("voice_id")

    p = sub.add_parser("transcribe", help="Transcribe a clip with Whisper (e.g. to get a reference transcript).")
    p.add_argument("audio", type=Path, help="Audio file (WAV/FLAC/OGG/MP3).")
    p.add_argument("-l", "--language", default="Auto", help="Spoken language (default: Auto-detect).")
    p.add_argument("-o", "--output", help="Also write the transcript to this text file.")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")

    p = sub.add_parser("info", help="Show backend, languages and control options.")

    p = sub.add_parser("serve", help="Run the web UI + REST API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--preload", action="store_true", help="Load all models at startup instead of on first use.")
    return parser


# ---------------------------------------------------------------------------- commands

def _client(args):
    if args.server:
        from .client import HttpClient

        return HttpClient(args.server)
    from .client import LocalClient

    return LocalClient()


def _apply_local_settings(args) -> None:
    if args.backend:
        os.environ["POLY_TTS_BACKEND"] = args.backend
    if args.device:
        os.environ["POLY_TTS_DEVICE"] = args.device


def run(args) -> int:
    if args.command == "serve":
        return _serve(args)

    client = _client(args)
    cmd = args.command
    if cmd == "info":
        print(json.dumps(client.info(), indent=2, ensure_ascii=False))
        return 0
    if cmd == "voices":
        return _voices(args, client)
    if cmd == "transcribe":
        result = client.transcribe(args.audio, args.language)
        if args.output:
            Path(args.output).write_text(result["text"] + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False) if args.json else result["text"])
        return 0

    text = _read_text(args)
    gen = _generation(args)
    if cmd == "design":
        wav, meta = client.design(text, args.description, args.language, _controls(args), gen)
    elif cmd == "control":
        wav, meta = client.control(text, args.language, args.voice, args.description, _controls(args), args.instruct, gen)
    elif cmd == "clone":
        ref_text = _read_ref_text(args)
        wav, meta = client.clone(text, args.ref_audio, ref_text, args.language, args.x_vector_only, gen)
        if args.save_as:
            voice = client.save_voice(args.save_as, args.ref_audio, ref_text, args.language)
            print(f"saved voice '{voice['id']}'", file=sys.stderr)
    elif cmd == "cross-lingual":
        if bool(args.ref_audio) == bool(args.voice):
            raise SystemExit("error: give exactly one of --ref-audio or --voice")
        wav, meta = client.cross_lingual(text, args.target_language, args.ref_audio, _read_ref_text(args), args.voice,
                                         args.source_language, args.x_vector_only, gen)
    elif cmd == "styled-clone":
        conversion = {k: v for k, v in (("steps", args.vc_steps), ("cfg_rate", args.vc_cfg_rate), ("tau", args.vc_tau))
                      if v is not None}
        wav, meta = client.styled_clone(text, args.language, args.ref_audio, args.voice, args.performer,
                                        args.performer_description, _controls(args), args.instruct, args.converter,
                                        conversion or None, gen)
    elif cmd == "speak":
        wav, meta = client.speak(text, args.voice, args.language, gen)
    elif cmd == "style":
        wav, meta = client.style(text, args.voice, args.instruct, args.language, gen)
    else:  # pragma: no cover - argparse enforces the choices
        raise SystemExit(f"unknown command {cmd}")
    _write_output(args, wav, meta)
    return 0


def _voices(args, client) -> int:
    if args.voices_command == "list":
        voices = client.voices()
        if args.kind:
            voices = [v for v in voices if v["kind"] == args.kind]
        if args.language:
            voices = [v for v in voices if v["language"].lower() == args.language.lower()]
        if args.gender:
            voices = [v for v in voices if v["gender"] == args.gender]
        if args.json:
            print(json.dumps(voices, indent=2, ensure_ascii=False))
        else:
            print(f"{'ID':<20} {'KIND':<9} {'LANGUAGE':<11} {'GENDER':<8} {'AGE':<12} NAME")
            for v in voices:
                print(f"{v['id']:<20} {v['kind']:<9} {v['language']:<11} {v['gender']:<8} {v['age']:<12} {v['name']}")
            print(f"\n{len(voices)} voices", file=sys.stderr)
    elif args.voices_command == "add":
        voice = client.save_voice(args.name, args.ref_audio, _read_ref_text(args), args.language, args.description)
        print(voice["id"])
    elif args.voices_command == "remove":
        client.remove_voice(args.voice_id)
        print(f"removed {args.voice_id}", file=sys.stderr)
    return 0


def _serve(args) -> int:
    import uvicorn

    from .api import create_app
    from .engine import get_engine

    engine = get_engine()
    if args.preload:
        from .backends.qwen import BASE, CUSTOM_VOICE, VOICE_DESIGN, QwenBackend

        if isinstance(engine.backend, QwenBackend):
            for kind in (CUSTOM_VOICE, VOICE_DESIGN, BASE):
                engine.backend._model(kind)
    print(f"Poly Qwen3-TTS on http://{args.host}:{args.port}  (API docs: /docs, backend: {engine.backend.name})", file=sys.stderr)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    # Voice names and instructions include CJK/Cyrillic text; legacy Windows consoles default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    _apply_local_settings(args)
    import logging

    logging.basicConfig(level=logging.INFO if args.verbose or args.command == "serve" else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return run(args)
    except (ValueError, LookupError) as e:
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    except Exception as e:
        from .backends import BackendError
        from .client import ApiError

        if isinstance(e, (BackendError, ApiError)):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
