from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..backends import GenerationParams
from ..controls import Controls


class GenerationOptions(BaseModel):
    """Sampling options. Unset fields use the checkpoint's defaults."""

    model_config = ConfigDict(extra="forbid")

    temperature: Optional[float] = Field(None, gt=0, le=2)
    top_p: Optional[float] = Field(None, gt=0, le=1)
    top_k: Optional[int] = Field(None, ge=1, le=1000)
    repetition_penalty: Optional[float] = Field(None, ge=1, le=2)
    max_new_tokens: Optional[int] = Field(None, ge=16, le=8192)
    seed: Optional[int] = Field(None, ge=0)

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
