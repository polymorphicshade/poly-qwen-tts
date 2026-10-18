from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..backends import GENERATION_OPTIONS, GenerationParams
from ..controls import Controls
from ..vc import ConversionParams


def _describe(name: str) -> str:
    opt = GENERATION_OPTIONS[name]
    default = f" Default: {opt['default']}." if opt["default"] is not None else ""
    return opt["description"] + default


class GenerationOptions(BaseModel):
    """Sampling options. Unset fields use the checkpoint's defaults."""

    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = Field(None, gt=0, le=2, description=_describe("temperature"))
    top_p: Optional[float] = Field(None, gt=0, le=1, description=_describe("top_p"))
    top_k: Optional[int] = Field(None, ge=1, le=1000, description=_describe("top_k"))
    repetition_penalty: Optional[float] = Field(None, ge=1, le=2, description=_describe("repetition_penalty"))
    max_new_tokens: Optional[int] = Field(None, ge=16, le=8192, description=_describe("max_new_tokens"))
    seed: Optional[int] = Field(None, ge=0, description=_describe("seed"))

    def to_params(self) -> GenerationParams:
        return GenerationParams(**self.model_dump())


def params(opts: Optional[GenerationOptions]) -> GenerationParams:
    return opts.to_params() if opts else GenerationParams()


TEXT = Field(..., min_length=1, max_length=5000, description="Text to speak. Supports [pause 500ms] and [style: ...]...[/style] markup.")
LANGUAGE = Field("Auto", description="Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian (or a code like 'en').")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation: Optional[GenerationOptions] = None


class DesignRequest(_Base):
    text: str = TEXT
    language: str = LANGUAGE
    voice_description: str = Field(..., min_length=1, max_length=2000, examples=[
        "Deep male voice with a slight rasp, speaking in a calm, authoritative narrator tone"])
    controls: Optional[Controls] = None


class ControlRequest(_Base):
    text: str = TEXT
    language: str = LANGUAGE
    voice_id: Optional[str] = Field(None, description="A library voice. Give this or voice_description.")
    voice_description: Optional[str] = Field(None, max_length=2000, description="A designed voice. Give this or voice_id.")
    controls: Optional[Controls] = None
    instruct: Optional[str] = Field(None, max_length=2000, description="Extra free-text instruction appended to the controls.")


class VoiceRequest(_Base):
    text: str = TEXT
    language: str = LANGUAGE
    voice_id: str


class StyleRequest(_Base):
    text: str = TEXT
    language: str = LANGUAGE
    voice_id: str
    instruct: str = Field(..., min_length=1, max_length=2000, examples=["Sound incredulous, as if you can't believe what you just heard"])


class ConversionOptions(BaseModel):
    """Voice-conversion options for Styled Clone. Unset fields use the converter's defaults."""

    model_config = ConfigDict(extra="forbid")

    steps: Optional[int] = Field(None, ge=1, le=200, description=(
        "Seed-VC diffusion steps (default 25). More steps (30-50) give cleaner, more faithful timbre; fewer (4-10) are faster."))
    cfg_rate: Optional[float] = Field(None, ge=0, le=1, description=(
        "Seed-VC guidance strength (default 0.7). Higher pushes harder toward the cloned timbre; lower keeps more of the performer."))
    tau: Optional[float] = Field(None, gt=0, le=1, description=(
        "OpenVoice sampling temperature (default 0.3). Lower is more stable; higher can sound more natural but less consistent."))

    def to_params(self) -> ConversionParams:
        return ConversionParams(**self.model_dump())
