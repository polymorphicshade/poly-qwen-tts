"""Backend interface: the thin layer that actually runs a TTS model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from ..audio import Audio


@dataclass
class GenerationParams:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None
    seed: Optional[int] = None

    def model_kwargs(self) -> dict[str, Any]:
        """Kwargs forwarded to the model's generate call (unset values use the checkpoint defaults)."""
        return {k: v for k, v in asdict(self).items() if v is not None and k != "seed"}


class BackendError(RuntimeError):
    """The backend could not run (missing dependencies, model load failure, …)."""


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def custom_voice(self, texts: list[str], language: str, speaker: str, instructs: list[str],
                     params: GenerationParams) -> tuple[list[np.ndarray], int]:
        """Built-in CustomVoice speaker, one optional instruction per text."""

    @abstractmethod
    def voice_design(self, texts: list[str], language: str, instructs: list[str],
                     params: GenerationParams) -> tuple[list[np.ndarray], int]:
        """VoiceDesign: instruction describes the voice (and style) for each text."""

    @abstractmethod
    def create_clone_prompt(self, ref_audio: Audio, ref_text: Optional[str], x_vector_only: bool) -> Any:
        """Extract a reusable voice-clone prompt from reference audio (Base model)."""

    @abstractmethod
    def voice_clone(self, texts: list[str], language: str, prompt: Any,
                    params: GenerationParams) -> tuple[list[np.ndarray], int]:
        """Synthesize texts in the voice captured by ``prompt``."""

    def status(self) -> dict[str, Any]:
        return {"backend": self.name}
