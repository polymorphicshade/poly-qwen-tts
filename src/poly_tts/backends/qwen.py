"""Real backend running the Qwen3-TTS checkpoints through the ``qwen-tts`` package."""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from ..audio import Audio
from ..config import Settings
from .base import Backend, BackendError, GenerationParams

log = logging.getLogger(__name__)

CUSTOM_VOICE, VOICE_DESIGN, BASE = "custom_voice", "voice_design", "base"


class QwenBackend(Backend):
    name = "qwen"

    def __init__(self, settings: Settings):
        try:
            import torch  # noqa: F401
            from qwen_tts import Qwen3TTSModel  # noqa: F401
        except ImportError as e:
            raise BackendError(
                "The Qwen backend needs the 'qwen' extra: `uv sync --extra qwen` "
                "(or `pip install poly-qwen-tts[qwen]`). Use --backend mock to try the app without it."
            ) from e
        self.settings = settings
        self._models: OrderedDict[str, Any] = OrderedDict()
        # One lock for loading and generation: a single GPU runs one job at a time.
        self._lock = threading.RLock()
        self._model_ids = {
            CUSTOM_VOICE: settings.custom_voice_model,
            VOICE_DESIGN: settings.voice_design_model,
            BASE: settings.base_model,
        }

    # ------------------------------------------------------------------ model management

    def _load_kwargs(self) -> dict[str, Any]:
        import torch

        device = self.settings.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.settings.dtype]
        if device == "cpu" and dtype != torch.float32:
            dtype = torch.float32
        attn = self.settings.attn_implementation
        if attn == "auto":
            has_fa2 = importlib.util.find_spec("flash_attn") is not None
            attn = "flash_attention_2" if has_fa2 and device.startswith("cuda") and dtype != torch.float32 else "sdpa"
        return {"device_map": device, "dtype": dtype, "attn_implementation": attn}

    def _model(self, kind: str):
        from qwen_tts import Qwen3TTSModel

        with self._lock:
            if kind in self._models:
                self._models.move_to_end(kind)
                return self._models[kind]
            while len(self._models) >= max(1, self.settings.max_loaded_models):
                evicted, _ = self._models.popitem(last=False)
                log.info("Unloading %s model", evicted)
                self._free_memory()
            model_id = self._model_ids[kind]
            log.info("Loading %s (%s)", kind, model_id)
            t0 = time.perf_counter()
            try:
                model = Qwen3TTSModel.from_pretrained(model_id, **self._load_kwargs())
            except Exception as e:
                raise BackendError(f"Failed to load {model_id}: {e}") from e
            log.info("Loaded %s in %.1fs", model_id, time.perf_counter() - t0)
            self._models[kind] = model
            return model

    @staticmethod
    def _free_memory() -> None:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _seed(params: GenerationParams) -> None:
        if params.seed is None:
            return
        import torch

        torch.manual_seed(params.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(params.seed)

    @staticmethod
    def _to_float(wavs) -> list[np.ndarray]:
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]

    # ------------------------------------------------------------------ generation

    def custom_voice(self, texts, language, speaker, instructs, params):
        with self._lock:
            model = self._model(CUSTOM_VOICE)
            self._seed(params)
            wavs, sr = model.generate_custom_voice(
                text=texts, language=language, speaker=speaker,
                instruct=[i or "" for i in instructs], **params.model_kwargs(),
            )
        return self._to_float(wavs), int(sr)

    def voice_design(self, texts, language, instructs, params):
        with self._lock:
            model = self._model(VOICE_DESIGN)
            self._seed(params)
            wavs, sr = model.generate_voice_design(
                text=texts, language=language, instruct=list(instructs), **params.model_kwargs(),
            )
        return self._to_float(wavs), int(sr)

    def create_clone_prompt(self, ref_audio: Audio, ref_text: Optional[str], x_vector_only: bool):
        with self._lock:
            model = self._model(BASE)
            return model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text or None, x_vector_only_mode=x_vector_only,
            )

    def voice_clone(self, texts, language, prompt, params):
        with self._lock:
            model = self._model(BASE)
            self._seed(params)
            wavs, sr = model.generate_voice_clone(
                text=texts, language=language, voice_clone_prompt=prompt, **params.model_kwargs(),
            )
        return self._to_float(wavs), int(sr)

    def status(self) -> dict[str, Any]:
        import torch

        info: dict[str, Any] = {
            "backend": self.name,
            "models": self._model_ids,
            "loaded": list(self._models),
            "cuda": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        return info
