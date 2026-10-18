"""Granular control: structured emotion/prosody settings and inline text markup.

Structured controls are compiled into the natural-language instruction that Qwen3-TTS understands.
The ``instruct`` channel is only honoured by the CustomVoice and VoiceDesign models; cloned voices
(Base model) support inline pauses only.

Inline markup
-------------
``[pause]``                      insert 0.5 s of silence
``[pause 800ms]`` / ``[pause 1.5s]`` / ``[pause 2]``   explicit length (bare numbers are seconds)
``[style: whispering, terrified]text[/style]``          extra instruction for that span only

Anything else in square brackets is left as text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

Emotion = Literal[
    "neutral", "joy", "excitement", "tenderness", "calm", "sadness", "anger", "fear", "panic",
    "surprise", "incredulity", "disgust", "sarcasm", "nervousness", "pride",
]
Intensity = Literal["slight", "moderate", "strong", "extreme"]
Delivery = Literal["normal", "whisper", "soft", "breathy", "loud", "shouting", "crying", "laughing", "trembling"]
Pace = Literal["very_slow", "slow", "normal", "fast", "very_fast"]
Pitch = Literal["very_low", "low", "normal", "high", "very_high"]
Intonation = Literal["natural", "flat", "rising", "falling", "expressive"]
Pauses = Literal["natural", "none", "few", "frequent", "dramatic"]

EMOTION_PHRASES: dict[str, str] = {
    "neutral": "neutral",
    "joy": "joyful, happy",
    "excitement": "excited, enthusiastic",
    "tenderness": "tender, affectionate",
    "calm": "calm, relaxed",
    "sadness": "sad, sorrowful",
    "anger": "angry",
    "fear": "fearful, scared",
    "panic": "panicked, frantic",
    "surprise": "surprised",
    "incredulity": "incredulous, disbelieving",
    "disgust": "disgusted",
    "sarcasm": "sarcastic",
    "nervousness": "nervous, anxious",
    "pride": "proud, triumphant",
}
INTENSITY_ADVERBS = {"slight": "slightly", "moderate": "", "strong": "very", "extreme": "extremely"}
DELIVERY_PHRASES = {
    "whisper": "whispering",
    "soft": "speaking softly and quietly",
    "breathy": "with a breathy voice",
    "loud": "speaking loudly",
    "shouting": "shouting",
    "crying": "while crying, voice breaking",
    "laughing": "while laughing",
    "trembling": "with a trembling, shaky voice",
}
PACE_PHRASES = {
    "very_slow": "at a very slow pace",
    "slow": "at a slow pace",
    "fast": "at a fast pace",
    "very_fast": "very fast, rushing the words",
}
PITCH_PHRASES = {
    "very_low": "with a much lower pitch than usual",
    "low": "with a lower pitch",
    "high": "with a higher pitch",
    "very_high": "with a much higher pitch than usual",
}
INTONATION_PHRASES = {
    "flat": "with flat, monotone intonation",
    "rising": "with rising, questioning intonation",
    "falling": "with falling, conclusive intonation",
    "expressive": "with highly expressive, dynamic intonation",
}
PAUSE_PHRASES = {
    "none": "without pausing",
    "few": "with few pauses",
    "frequent": "with frequent short pauses and hesitations",
    "dramatic": "with long dramatic pauses between phrases",
}


class Controls(BaseModel):
    """Structured emotion/prosody controls. Every field is optional; unset fields add nothing."""

    model_config = ConfigDict(extra="forbid")

    emotion: Optional[Emotion] = None
    intensity: Optional[Intensity] = None
    delivery: Optional[Delivery] = None
    pace: Optional[Pace] = None
    pitch: Optional[Pitch] = None
    intonation: Optional[Intonation] = None
    pauses: Optional[Pauses] = None

    def is_empty(self) -> bool:
        return self.to_instruct() == ""

    def to_instruct(self) -> str:
        """Compile to an English instruction, e.g. ``"Speak in a very angry tone, whispering, at a slow pace."``"""
        parts: list[str] = []
        if self.emotion and self.emotion != "neutral":
            adverb = INTENSITY_ADVERBS.get(self.intensity or "moderate", "")
            words = f"{adverb} {EMOTION_PHRASES[self.emotion]}".strip()
            article = "an" if words[0] in "aeiou" else "a"
            parts.append(f"in {article} {words} tone")
        elif self.emotion == "neutral":
            parts.append("in a neutral, even tone")
        for value, table in (
            (self.delivery, DELIVERY_PHRASES),
            (self.pace, PACE_PHRASES),
            (self.pitch, PITCH_PHRASES),
            (self.intonation, INTONATION_PHRASES),
            (self.pauses, PAUSE_PHRASES),
        ):
            if value and value in table:
                parts.append(table[value])
        if not parts:
            return ""
        return "Speak " + ", ".join(parts) + "."


def control_options() -> dict[str, list[str]]:
    """Allowed values for each control field (used by the UI and ``/api/info``)."""
    return {name: list(field.annotation.__args__[0].__args__) for name, field in Controls.model_fields.items()}


def join_instructions(*parts: Optional[str]) -> str:
    """Join non-empty instruction fragments into one instruction string."""
    cleaned = []
    for p in parts:
        if p and p.strip():
            p = p.strip()
            cleaned.append(p if p[-1] in ".!?。！？" else p + ".")
    return " ".join(cleaned)


# ---------------------------------------------------------------------------- inline markup

@dataclass
class Segment:
    text: str = ""
    style: Optional[str] = None  # extra instruction for this span
    pause: float = 0.0  # seconds; a pause segment has empty text

    @property
    def is_pause(self) -> bool:
        return self.pause > 0 and not self.text


DEFAULT_PAUSE_SECONDS = 0.5
MAX_PAUSE_SECONDS = 10.0

_TAG_RE = re.compile(
    r"\[(?:"
    r"pause(?:[\s=:]+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)?)?"
    r"|style\s*[:=]\s*(?P<style>[^\]]+)"
    r"|(?P<close>/style)"
    r")\s*\]",
    re.IGNORECASE,
)


def parse_markup(text: str) -> list[Segment]:
    """Split text into speech and pause segments according to the inline markup."""
    segments: list[Segment] = []
    style: Optional[str] = None
    pos = 0

    def add_text(chunk: str) -> None:
        chunk = chunk.strip()
        if chunk:
            segments.append(Segment(text=chunk, style=style))

    for m in _TAG_RE.finditer(text):
        add_text(text[pos:m.start()])
        pos = m.end()
        if m.group("close"):
            style = None
        elif m.group("style") is not None:
            if style is not None:
                raise ValueError("Nested [style:...] tags are not supported; close the previous one with [/style] first.")
            style = m.group("style").strip()
        else:
            seconds = DEFAULT_PAUSE_SECONDS
            if m.group("num"):
                seconds = float(m.group("num"))
                if (m.group("unit") or "s").lower() == "ms":
                    seconds /= 1000.0
            seconds = min(seconds, MAX_PAUSE_SECONDS)
            if seconds > 0:
                segments.append(Segment(pause=seconds))
    add_text(text[pos:])

    if not any(s.text for s in segments):
        raise ValueError("Text is empty (after removing markup).")
    return segments


def has_styles(segments: list[Segment]) -> bool:
    return any(s.style for s in segments)
