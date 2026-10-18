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


# Labels, ranges and user-facing tuning advice for each GenerationParams field. This is the single
# source for the API schema descriptions, ``/api/info`` (rendered by the web UI) and the CLI help.
# Defaults are those of the Qwen3-TTS 1.7B checkpoints (generation_config.json).
GENERATION_OPTIONS: dict[str, dict[str, Any]] = {
    "temperature": {
        "label": "Temperature", "min": 0.05, "max": 2.0, "step": 0.01, "default": 0.9,
        "description": "How adventurous the delivery is. Lower (0.6–0.8) gives a steadier, more consistent read; "
                       "higher (1.0–1.2) gives livelier, more varied intonation but can slur words or add glitches.",
    },
    "top_p": {
        "label": "Top-p", "min": 0.05, "max": 1.0, "step": 0.01, "default": 1.0,
        "description": "Samples only from the most likely sounds whose probabilities add up to this share. "
                       "Lowering it (0.8–0.95) trims unlikely sounds for cleaner but flatter speech. 1.0 turns it off.",
    },
    "top_k": {
        "label": "Top-k", "min": 1, "max": 1000, "step": 1, "default": 50,
        "description": "Samples only from the k most likely sounds at each step. Smaller (10–30) is safer and more "
                       "monotone; larger allows more variety in tone and rhythm.",
    },
    "repetition_penalty": {
        "label": "Repetition penalty", "min": 1.0, "max": 2.0, "step": 0.01, "default": 1.05,
        "description": "Discourages repeating the same sounds. Raise it a little (1.1–1.2) if the voice stutters, "
                       "loops or drones on; too high can distort the voice. 1.0 turns it off.",
    },
    "max_new_tokens": {
        "label": "Max new tokens", "min": 16, "max": 8192, "step": 16, "default": 8192,
        "description": "Upper limit on generated audio per text segment (about 12 tokens per second). Lower it to stop "
                       "runaway output such as long trailing noise; raise it if long text gets cut off.",
    },
    "seed": {
        "label": "Seed", "min": 0, "max": None, "step": 1, "default": None,
        "description": "Makes a take repeatable: the same seed with the same text and settings gives the same audio. "
                       "Leave it empty for a fresh take each time. With a fixed seed, change one option at a time to "
                       "hear exactly what it does.",
    },
}


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
