"""Clients used by the CLI: run in-process (``LocalClient``) or against a server (``HttpClient``).

Both expose the same methods and return ``(wav_bytes, metadata)`` for synthesis calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

from .audio import load_audio

Result = tuple[bytes, dict[str, Any]]


class LocalClient:
    def __init__(self, engine=None):
        from .engine import get_engine

        self.engine = engine or get_engine()

    @staticmethod
    def _out(result) -> Result:
        return result.wav_bytes(), result.metadata()

    @staticmethod
    def _params(gen: Optional[dict]):
        from .backends import GenerationParams

        return GenerationParams(**(gen or {}))

    @staticmethod
    def _controls(controls: Optional[dict]):
        from .controls import Controls

        return Controls(**controls) if controls else None

    def design(self, text, voice_description, language="Auto", controls=None, generation=None) -> Result:
        return self._out(self.engine.voice_design(text, voice_description, language, self._controls(controls), self._params(generation)))

    def control(self, text, language="Auto", voice_id=None, voice_description=None, controls=None, instruct=None, generation=None) -> Result:
        return self._out(self.engine.control(text, language, voice_id, voice_description, self._controls(controls), instruct, self._params(generation)))

    def clone(self, text, ref_audio: Path, ref_text=None, language="Auto", x_vector_only=False, generation=None) -> Result:
        return self._out(self.engine.clone(text, load_audio(ref_audio), ref_text, language, x_vector_only, self._params(generation)))

    def cross_lingual(self, text, target_language, ref_audio: Optional[Path] = None, ref_text=None, voice_id=None,
                      source_language=None, x_vector_only=False, generation=None) -> Result:
        ref = load_audio(ref_audio) if ref_audio else None
        return self._out(self.engine.cross_lingual(text, target_language, ref, ref_text, voice_id, source_language, x_vector_only, self._params(generation)))

    def speak(self, text, voice_id, language="Auto", generation=None) -> Result:
        return self._out(self.engine.speak(text, voice_id, language, self._params(generation)))

    def style(self, text, voice_id, instruct, language="Auto", generation=None) -> Result:
        return self._out(self.engine.style(text, voice_id, instruct, language, self._params(generation)))

    def styled_clone(self, text, language="Auto", ref_audio: Optional[Path] = None, voice_id=None, performer_voice_id=None,
                     performer_description=None, controls=None, instruct=None, converter=None, conversion=None,
                     generation=None) -> Result:
        from .vc import ConversionParams

        ref = load_audio(ref_audio) if ref_audio else None
        return self._out(self.engine.styled_clone(
            text, language, ref, voice_id, performer_voice_id, performer_description, self._controls(controls), instruct,
            converter, ConversionParams(**(conversion or {})), self._params(generation)))

    def voices(self) -> list[dict]:
        return [v.to_dict() for v in self.engine.library.list()]

    def save_voice(self, name, ref_audio: Path, ref_text=None, language="Auto", description=None) -> dict:
        return self.engine.save_voice(name, load_audio(ref_audio), ref_text, language, description).to_dict()

    def remove_voice(self, voice_id) -> None:
        self.engine.remove_voice(voice_id)

    def info(self) -> dict:
        from .backends import GENERATION_OPTIONS
        from .controls import control_options
        from .voices import LANGUAGES

        return {"backend": self.engine.info(), "languages": LANGUAGES, "controls": control_options(),
                "generation_options": GENERATION_OPTIONS, "transcription": self.engine.transcription_info(),
                "voice_conversion": self.engine.voice_conversion_info()}

    def transcribe(self, audio: Path, language="Auto") -> dict:
        t = self.engine.transcribe(load_audio(audio), language)
        return {"text": t.text, "language": t.language or "Auto", "model": t.model}


class ApiError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, base_url: str, timeout: float = 900.0):
        import httpx

        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _check(self, res):
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail")
            except Exception:
                detail = res.text
            raise ApiError(f"HTTP {res.status_code}: {detail}")
        return res

    def _audio(self, res) -> Result:
        self._check(res)
        meta = json.loads(unquote(res.headers.get("X-TTS-Metadata", "{}")))
        return res.content, meta

    @staticmethod
    def _drop_none(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    def _post_json(self, path, body) -> Result:
        return self._audio(self.http.post(path, json=self._drop_none(body)))

    def _post_form(self, path, data: dict, ref_audio: Optional[Path]) -> Result:
        data = {k: (json.dumps(v) if isinstance(v, dict) else str(v).lower() if isinstance(v, bool) else v)
                for k, v in self._drop_none(data).items()}
        files = None
        if ref_audio:
            files = {"ref_audio": (Path(ref_audio).name, Path(ref_audio).read_bytes(), "application/octet-stream")}
        return self._audio(self.http.post(path, data=data, files=files))

    def design(self, text, voice_description, language="Auto", controls=None, generation=None) -> Result:
        return self._post_json("/api/tts/design", dict(text=text, voice_description=voice_description, language=language,
                                                        controls=controls, generation=generation))

    def control(self, text, language="Auto", voice_id=None, voice_description=None, controls=None, instruct=None, generation=None) -> Result:
        return self._post_json("/api/tts/control", dict(text=text, language=language, voice_id=voice_id, voice_description=voice_description,
                                                         controls=controls, instruct=instruct, generation=generation))

    def clone(self, text, ref_audio: Path, ref_text=None, language="Auto", x_vector_only=False, generation=None) -> Result:
        return self._post_form("/api/tts/clone", dict(text=text, ref_text=ref_text, language=language,
                                                       x_vector_only=x_vector_only, generation=generation), ref_audio)

    def cross_lingual(self, text, target_language, ref_audio: Optional[Path] = None, ref_text=None, voice_id=None,
                      source_language=None, x_vector_only=False, generation=None) -> Result:
        return self._post_form("/api/tts/cross-lingual", dict(
            text=text, target_language=target_language, ref_text=ref_text, voice_id=voice_id,
            source_language=source_language, x_vector_only=x_vector_only, generation=generation), ref_audio)

    def speak(self, text, voice_id, language="Auto", generation=None) -> Result:
        return self._post_json("/api/tts/voice", dict(text=text, voice_id=voice_id, language=language, generation=generation))

    def style(self, text, voice_id, instruct, language="Auto", generation=None) -> Result:
        return self._post_json("/api/tts/style", dict(text=text, voice_id=voice_id, instruct=instruct, language=language, generation=generation))

    def styled_clone(self, text, language="Auto", ref_audio: Optional[Path] = None, voice_id=None, performer_voice_id=None,
                     performer_description=None, controls=None, instruct=None, converter=None, conversion=None,
                     generation=None) -> Result:
        return self._post_form("/api/tts/styled-clone", dict(
            text=text, language=language, voice_id=voice_id, performer_voice_id=performer_voice_id,
            performer_description=performer_description, controls=controls, instruct=instruct, converter=converter,
            conversion=conversion, generation=generation), ref_audio)

    def voices(self) -> list[dict]:
        return self._check(self.http.get("/api/voices")).json()

    def save_voice(self, name, ref_audio: Path, ref_text=None, language="Auto", description=None) -> dict:
        data = self._drop_none(dict(name=name, ref_text=ref_text, language=language, description=description))
        files = {"ref_audio": (Path(ref_audio).name, Path(ref_audio).read_bytes(), "application/octet-stream")}
        return self._check(self.http.post("/api/voices", data=data, files=files)).json()

    def remove_voice(self, voice_id) -> None:
        self._check(self.http.delete(f"/api/voices/{voice_id}"))

    def info(self) -> dict:
        return self._check(self.http.get("/api/info")).json()

    def transcribe(self, audio: Path, language="Auto") -> dict:
        files = {"audio": (Path(audio).name, Path(audio).read_bytes(), "application/octet-stream")}
        return self._check(self.http.post("/api/transcribe", data={"language": language}, files=files)).json()
