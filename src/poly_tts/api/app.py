"""FastAPI application: REST API under ``/api`` and the web UI at ``/``.

Synthesis endpoints return ``audio/wav``. Generation metadata (duration, effective instruction,
warnings, …) is in the ``X-TTS-Metadata`` header as URL-encoded JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .. import __version__
from ..audio import Audio, load_audio
from ..backends import BackendError
from ..controls import control_options
from ..engine import Engine, SynthesisResult, get_engine
from ..voices import LANGUAGES
from .schemas import ControlRequest, DesignRequest, GenerationOptions, StyleRequest, VoiceRequest, params

STATIC_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
METADATA_HEADER = "X-TTS-Metadata"


def _audio_response(result: SynthesisResult) -> Response:
    meta = quote(json.dumps(result.metadata(), ensure_ascii=False))
    return Response(result.wav_bytes(), media_type="audio/wav", headers={METADATA_HEADER: meta})


async def _read_upload(upload: Optional[UploadFile]) -> Optional[Audio]:
    if upload is None:
        return None
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Reference audio is larger than 25 MB.")
    if not data:
        return None
    return load_audio(data)


def _generation_form(raw: Optional[str]) -> Optional[GenerationOptions]:
    if not raw:
        return None
    try:
        return GenerationOptions.model_validate_json(raw)
    except ValidationError as e:
        raise HTTPException(422, f"Invalid 'generation' JSON: {e.errors()}") from e


def create_app(engine: Optional[Engine] = None) -> FastAPI:
    app = FastAPI(
        title="Poly Qwen3-TTS",
        version=__version__,
        description="Voice design, granular emotion/prosody control, zero-shot and cross-lingual voice cloning, "
                    "a 50-voice library and style instructions, powered by Qwen3-TTS.",
    )
    app.state.engine = engine

    def eng() -> Engine:
        if app.state.engine is None:
            app.state.engine = get_engine()
        return app.state.engine

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(LookupError)
    async def _lookup_error(_: Request, exc: LookupError):
        return JSONResponse({"detail": str(exc.args[0] if exc.args else exc)}, status_code=404)

    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, exc: BackendError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    # ------------------------------------------------------------------ metadata

    @app.get("/api/info", tags=["meta"])
    def info():
        e = eng()
        voices = e.library.list()
        return {
            "version": __version__,
            "backend": e.info(),
            "languages": LANGUAGES,
            "controls": control_options(),
            "voice_counts": {k: sum(v.kind == k for v in voices) for k in ("native", "designed", "cloned")},
            "reference_audio": {"min_seconds": e.settings.min_ref_seconds, "max_seconds": e.settings.max_ref_seconds},
        }

    # ------------------------------------------------------------------ voice library

    @app.get("/api/voices", tags=["voices"])
    def list_voices(kind: Optional[str] = None, language: Optional[str] = None, gender: Optional[str] = None,
                    q: Optional[str] = None):
        voices = eng().library.list()
        if kind:
            voices = [v for v in voices if v.kind == kind]
        if language:
            voices = [v for v in voices if v.language.lower() == language.lower()]
        if gender:
            voices = [v for v in voices if v.gender == gender]
        if q:
            ql = q.lower()
            voices = [v for v in voices if ql in f"{v.id} {v.name} {v.description} {' '.join(v.tags)}".lower()]
        return [v.to_dict() for v in voices]

    @app.get("/api/voices/{voice_id}", tags=["voices"])
    def get_voice(voice_id: str):
        return eng().library.get(voice_id).to_dict()

    @app.post("/api/voices", tags=["voices"], status_code=201)
    async def save_voice(
        name: str = Form(...),
        ref_audio: UploadFile = File(...),
        ref_text: Optional[str] = Form(None),
        language: str = Form("Auto"),
        description: Optional[str] = Form(None),
    ):
        """Save a cloned voice to the library so it can be reused by id."""
        ref = await _read_upload(ref_audio)
        if ref is None:
            raise HTTPException(400, "ref_audio is empty.")
        voice = await run_in_threadpool(eng().save_voice, name, ref, ref_text, language, description)
        return voice.to_dict()

    @app.delete("/api/voices/{voice_id}", tags=["voices"], status_code=204)
    def delete_voice(voice_id: str):
        eng().remove_voice(voice_id)
        return Response(status_code=204)

    # ------------------------------------------------------------------ core features

    @app.post("/api/tts/design", tags=["tts"], response_class=Response,
              responses={200: {"content": {"audio/wav": {}}}})
    def design(req: DesignRequest):
        """Voice design: describe the voice in natural language."""
        return _audio_response(eng().voice_design(
            req.text, req.voice_description, req.language, req.controls, params(req.generation)))

    @app.post("/api/tts/control", tags=["tts"], response_class=Response,
              responses={200: {"content": {"audio/wav": {}}}})
    def control(req: ControlRequest):
        """Granular control: emotion, intensity, delivery, pace, pitch, intonation, pauses + inline markup."""
        return _audio_response(eng().control(
            req.text, req.language, req.voice_id, req.voice_description, req.controls, req.instruct,
            params(req.generation)))

    @app.post("/api/tts/clone", tags=["tts"], response_class=Response,
              responses={200: {"content": {"audio/wav": {}}}})
    async def clone(
        text: str = Form(...),
        ref_audio: UploadFile = File(..., description="Reference clip, 3 s or longer (WAV/FLAC/OGG/MP3)."),
        ref_text: Optional[str] = Form(None, description="Transcript of the reference clip (recommended)."),
        language: str = Form("Auto"),
        x_vector_only: bool = Form(False),
        generation: Optional[str] = Form(None, description="GenerationOptions as JSON."),
    ):
        """Zero-shot voice cloning."""
        ref = await _read_upload(ref_audio)
        if ref is None:
            raise HTTPException(400, "ref_audio is empty.")
        gen = _generation_form(generation)
        result = await run_in_threadpool(eng().clone, text, ref, ref_text, language, x_vector_only, params(gen))
        return _audio_response(result)

    @app.post("/api/tts/cross-lingual", tags=["tts"], response_class=Response,
              responses={200: {"content": {"audio/wav": {}}}})
    async def cross_lingual(
        text: str = Form(...),
        target_language: str = Form(...),
        ref_audio: Optional[UploadFile] = File(None),
        ref_text: Optional[str] = Form(None),
        voice_id: Optional[str] = Form(None, description="Use a library voice instead of uploading audio."),
        source_language: Optional[str] = Form(None),
        x_vector_only: bool = Form(False),
        generation: Optional[str] = Form(None),
    ):
        """Cross-lingual cloning: the cloned voice speaks ``target_language``."""
        ref = await _read_upload(ref_audio)
        gen = _generation_form(generation)
        result = await run_in_threadpool(
            eng().cross_lingual, text, target_language, ref, ref_text, voice_id or None, source_language,
            x_vector_only, params(gen))
        return _audio_response(result)

    @app.post("/api/tts/voice", tags=["tts"], response_class=Response,
              responses={200: {"content": {"audio/wav": {}}}})
    def voice(req: VoiceRequest):
        """Speak with a voice from the library (see GET /api/voices)."""
        return _audio_response(eng().speak(req.text, req.voice_id, req.language, params(req.generation)))

    @app.post("/api/tts/style", tags=["tts"], response_class=Response,
              responses={200: {"content": {"audio/wav": {}}}})
    def style(req: StyleRequest):
        """Preset voice + natural-language style instruction."""
        return _audio_response(eng().style(req.text, req.voice_id, req.instruct, req.language, params(req.generation)))

    # ------------------------------------------------------------------ web UI

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app
