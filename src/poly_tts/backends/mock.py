"""Mock backend: deterministic placeholder tones. No GPU, downloads or torch needed.

Useful for tests, UI development and trying the API surface. Each voice/instruction maps to a
different pitch and the output length scales with the text, so results are distinguishable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

from ..audio import Audio
from .base import Backend, GenerationParams

SAMPLE_RATE = 24000


def _hash01(*parts: object) -> float:
    h = hashlib.md5("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "little") / 2**32


def _tone(text: str, key: str, params: GenerationParams) -> np.ndarray:
    seconds = min(30.0, max(0.5, 0.065 * len(text)))
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    f0 = 110.0 + 220.0 * _hash01(key)
    jitter = 0.02 * _hash01(params.seed) if params.seed is not None else 0.0
    wave = sum(np.sin(2 * np.pi * f0 * (1 + jitter) * k * t) / k for k in (1, 2, 3))
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2  # ~4 "syllables" per second
    return (0.25 * wave * syllables).astype(np.float32)


class MockBackend(Backend):
    name = "mock"

    def __init__(self, settings: Any = None):
        self.calls: list[tuple[str, dict]] = []  # inspected by tests

    def custom_voice(self, texts, language, speaker, instructs, params):
        self.calls.append(("custom_voice", dict(texts=texts, language=language, speaker=speaker, instructs=instructs)))
        return [_tone(t, f"{speaker}|{i}", params) for t, i in zip(texts, instructs)], SAMPLE_RATE

    def voice_design(self, texts, language, instructs, params):
        self.calls.append(("voice_design", dict(texts=texts, language=language, instructs=instructs)))
        return [_tone(t, i, params) for t, i in zip(texts, instructs)], SAMPLE_RATE

    def create_clone_prompt(self, ref_audio: Audio, ref_text: Optional[str], x_vector_only: bool):
        wav, sr = ref_audio
        self.calls.append(("create_clone_prompt", dict(ref_text=ref_text, x_vector_only=x_vector_only)))
        return {"key": hashlib.md5(wav.tobytes()).hexdigest(), "x_vector_only": x_vector_only}

    def voice_clone(self, texts, language, prompt, params):
        self.calls.append(("voice_clone", dict(texts=texts, language=language, prompt=prompt)))
        return [_tone(t, prompt["key"], params) for t in texts], SAMPLE_RATE

    def status(self):
        return {"backend": self.name, "loaded": [], "note": "placeholder tones, not speech"}
