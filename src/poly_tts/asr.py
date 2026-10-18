"""Speech recognition (Whisper) used to auto-transcribe reference clips for voice cloning.

A transcript lets the Base model build a full voice-clone prompt instead of falling back to the
less faithful speaker-embedding-only mode, so missing transcripts are filled in automatically.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .audio import Audio
from .backends import BackendError
from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: Optional[str] = None  # the language hint that was used (None: auto-detected)
    model: str = ""


class Transcriber(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, audio: Audio, language: Optional[str] = None) -> Transcript:
        """Transcribe mono audio. ``language`` is a language name (e.g. "English") or None to auto-detect."""


class WhisperTranscriber(Transcriber):
    """Whisper through the ``transformers`` ASR pipeline (installed with the ``qwen`` extra)."""

    name = "whisper"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.asr_model
        self._pipe: Any = None
        self._lock = threading.Lock()

    def _pipeline(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise BackendError(
                "Auto-transcription needs torch and transformers: `uv sync --extra qwen`."
            ) from e
        device = self.settings.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        log.info("Loading %s for transcription", self.model_id)
        t0 = time.perf_counter()
        try:
            self._pipe = pipeline("automatic-speech-recognition", model=self.model_id, dtype=dtype, device=device)
        except Exception as e:
            raise BackendError(f"Failed to load {self.model_id}: {e}") from e
        log.info("Loaded %s in %.1fs", self.model_id, time.perf_counter() - t0)
        return self._pipe

    def transcribe(self, audio: Audio, language: Optional[str] = None) -> Transcript:
        wav, sr = audio
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language.lower()
        with self._lock:
            pipe = self._pipeline()
            # chunk_length_s lets clips longer than Whisper's 30 s window be transcribed.
            out = pipe({"raw": np.asarray(wav, dtype=np.float32), "sampling_rate": sr},
                       chunk_length_s=30, generate_kwargs=generate_kwargs)
        return Transcript(text=" ".join(out["text"].split()), language=language, model=self.model_id)


class MockTranscriber(Transcriber):
    """Deterministic placeholder transcript, paired with the mock TTS backend."""

    name = "mock"

    def transcribe(self, audio: Audio, language: Optional[str] = None) -> Transcript:
        wav, sr = audio
        return Transcript(text=f"Mock transcript of {len(wav) / sr:.1f} seconds of audio.", language=language, model="mock")


def create_transcriber(settings: Settings) -> Transcriber:
    return MockTranscriber() if settings.backend == "mock" else WhisperTranscriber(settings)
