"""Runtime settings, read from ``POLY_TTS_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"POLY_TTS_{name}", default)


@dataclass
class Settings:
    # "qwen" runs the real Qwen3-TTS models; "mock" synthesizes placeholder tones (no GPU / downloads).
    backend: str = field(default_factory=lambda: _env("BACKEND", "qwen"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))  # auto | cuda:0 | cpu
    dtype: str = field(default_factory=lambda: _env("DTYPE", "bfloat16"))  # bfloat16 | float16 | float32
    attn_implementation: str = field(default_factory=lambda: _env("ATTN", "auto"))  # auto | sdpa | flash_attention_2 | eager

    custom_voice_model: str = field(default_factory=lambda: _env("CUSTOM_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    voice_design_model: str = field(default_factory=lambda: _env("VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
    base_model: str = field(default_factory=lambda: _env("BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    # How many of the three model variants may be resident at once (LRU eviction beyond this).
    max_loaded_models: int = field(default_factory=lambda: int(_env("MAX_LOADED_MODELS", "3")))

    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(Path.cwd() / "data"))))
    min_ref_seconds: float = field(default_factory=lambda: float(_env("MIN_REF_SECONDS", "3.0")))
    max_ref_seconds: float = field(default_factory=lambda: float(_env("MAX_REF_SECONDS", "60.0")))

    # Transcribe reference clips with Whisper when no transcript is given (instead of x-vector-only cloning).
    auto_transcribe: bool = field(default_factory=lambda: _env("AUTO_TRANSCRIBE", "1").lower() not in ("0", "false", "no", "off"))
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "openai/whisper-large-v3-turbo"))
    # Styled Clone voice conversion: "auto" (Seed-VC, else OpenVoice), "seed-vc" or "openvoice".
    voice_converter: str = field(default_factory=lambda: _env("VC", "auto"))
    # Source checkouts of the converters (default: <data_dir>/vendor/seed-vc and <data_dir>/vendor/OpenVoice).
    seed_vc_path: str = field(default_factory=lambda: _env("SEED_VC_DIR", ""))
    openvoice_path: str = field(default_factory=lambda: _env("OPENVOICE_DIR", ""))

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def presets_dir(self) -> Path:
        return self.data_dir / "presets"

    @property
    def seed_vc_dir(self) -> Path:
        return Path(self.seed_vc_path) if self.seed_vc_path else self.data_dir / "vendor" / "seed-vc"

    @property
    def openvoice_dir(self) -> Path:
        return Path(self.openvoice_path) if self.openvoice_path else self.data_dir / "vendor" / "OpenVoice"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings
