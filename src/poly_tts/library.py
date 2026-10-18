"""Voice library: built-in voices plus user-saved cloned voices persisted on disk.

Layout: ``<data_dir>/voices/<id>/reference.wav`` and ``meta.json``.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from .audio import Audio, load_audio, save_wav
from .voices import BUILTIN_VOICES, Voice, get_builtin_voice


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s[:40] or "voice"


class VoiceLibrary:
    def __init__(self, voices_dir: Path):
        self.dir = Path(voices_dir)
        self._lock = threading.Lock()

    def _meta(self, voice_id: str) -> Optional[dict]:
        path = self.dir / voice_id / "meta.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _voice_from_meta(meta: dict) -> Voice:
        return Voice(
            id=meta["id"], name=meta["name"], kind="cloned", gender=meta.get("gender", "unknown"),
            age=meta.get("age", "unknown"), language=meta.get("language", "Auto"),
            description=meta.get("description") or "Cloned from reference audio.",
            tags=tuple(meta.get("tags", ())), supports_style=False,
        )

    def cloned(self) -> list[Voice]:
        if not self.dir.is_dir():
            return []
        voices = []
        for d in sorted(self.dir.iterdir()):
            meta = self._meta(d.name) if d.is_dir() else None
            if meta:
                voices.append(self._voice_from_meta(meta))
        return voices

    def list(self) -> list[Voice]:
        return BUILTIN_VOICES + self.cloned()

    def get(self, voice_id: str) -> Voice:
        voice = get_builtin_voice(voice_id)
        if voice:
            return voice
        meta = self._meta(_slug(voice_id)) if voice_id else None
        if meta:
            return self._voice_from_meta(meta)
        raise LookupError(f"Unknown voice {voice_id!r}. List voices with `poly-tts voices list` or GET /api/voices.")

    def reference(self, voice_id: str) -> tuple[Audio, Optional[str]]:
        """Reference audio and transcript of a cloned voice."""
        meta = self._meta(_slug(voice_id))
        if not meta:
            raise LookupError(f"{voice_id!r} is not a saved cloned voice.")
        return load_audio(self.dir / meta["id"] / "reference.wav"), meta.get("ref_text") or None

    def add(self, name: str, ref_audio: Audio, ref_text: Optional[str], language: str = "Auto",
            description: Optional[str] = None, gender: str = "unknown") -> Voice:
        name = name.strip()
        if not name:
            raise ValueError("Voice name is required.")
        with self._lock:
            base = _slug(name)
            voice_id, n = base, 2
            while get_builtin_voice(voice_id) or (self.dir / voice_id).exists():
                voice_id, n = f"{base}_{n}", n + 1
            folder = self.dir / voice_id
            folder.mkdir(parents=True)
            save_wav(folder / "reference.wav", *ref_audio)
            meta = {
                "id": voice_id, "name": name, "language": language, "description": description or "",
                "ref_text": ref_text or "", "gender": gender, "created": time.time(),
            }
            (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._voice_from_meta(meta)

    def set_ref_text(self, voice_id: str, ref_text: str) -> None:
        """Store a transcript for a saved voice (e.g. one produced by auto-transcription)."""
        with self._lock:
            meta = self._meta(_slug(voice_id))
            if not meta:
                raise LookupError(f"{voice_id!r} is not a saved cloned voice.")
            meta["ref_text"] = ref_text
            (self.dir / meta["id"] / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def remove(self, voice_id: str) -> None:
        if get_builtin_voice(voice_id):
            raise ValueError("Built-in voices cannot be removed.")
        folder = self.dir / _slug(voice_id)
        if not (folder / "meta.json").is_file():
            raise LookupError(f"Unknown cloned voice {voice_id!r}.")
        shutil.rmtree(folder)
