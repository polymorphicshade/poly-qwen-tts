from __future__ import annotations

from ..config import Settings
from .base import GENERATION_OPTIONS, Backend, BackendError, GenerationParams


def create_backend(settings: Settings) -> Backend:
    if settings.backend == "mock":
        from .mock import MockBackend

        return MockBackend(settings)
    if settings.backend == "qwen":
        from .qwen import QwenBackend

        return QwenBackend(settings)
    raise BackendError(f"Unknown backend {settings.backend!r} (expected 'qwen' or 'mock').")


__all__ = ["GENERATION_OPTIONS", "Backend", "BackendError", "GenerationParams", "create_backend"]
