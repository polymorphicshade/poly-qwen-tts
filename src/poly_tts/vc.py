"""Zero-shot voice conversion, used by Styled Clone to give a styled performance a cloned voice's timbre.

Two converters are supported. Both run from a source checkout (their PyPI packaging pins old numpy /
protobuf versions that conflict with qwen-tts), with weights downloaded from Hugging Face on first use:

* **Seed-VC** (default, best quality): diffusion-transformer VC, 22.05 kHz speech model.
  ``git clone https://github.com/Plachtaa/seed-vc data/vendor/seed-vc``
* **OpenVoice v2** (lighter, faster fallback): tone-colour converter only.
  ``git clone https://github.com/myshell-ai/OpenVoice data/vendor/OpenVoice``

Both need the ``vc`` extra (``uv sync --extra qwen --extra vc``).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import time
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .audio import Audio
from .backends import BackendError
from .config import Settings

log = logging.getLogger(__name__)

CONVERTERS = ("seed-vc", "openvoice")


@dataclass
class ConversionParams:
    steps: Optional[int] = None  # Seed-VC diffusion steps (default 25)
    cfg_rate: Optional[float] = None  # Seed-VC classifier-free guidance strength (default 0.7)
    tau: Optional[float] = None  # OpenVoice sampling temperature (default 0.3)


class VoiceConverter(ABC):
    name: str = "base"

    @abstractmethod
    def convert(self, source: Audio, reference: Audio, params: ConversionParams) -> Audio:
        """Re-voice ``source`` with the timbre of ``reference``, keeping its words, timing and intonation."""

    def available(self) -> tuple[bool, str]:
        """Whether this converter can run, and why not if it can't."""
        return True, ""

    def load(self) -> None:
        """Load the model now (raises BackendError if it can't be loaded)."""


def _device(settings: Settings) -> str:
    import torch

    return ("cuda:0" if torch.cuda.is_available() else "cpu") if settings.device == "auto" else settings.device


def _resample(wav: np.ndarray, sr: int, target: int):
    import torch
    import torchaudio

    t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    return t if sr == target else torchaudio.functional.resample(t, sr, target)


class _CheckoutConverter(VoiceConverter):
    """A converter that imports its model code from a source checkout."""

    repo_url = ""
    marker = ""  # a file that identifies a valid checkout

    def __init__(self, settings: Settings, repo_dir: Path):
        self.settings = settings
        self.repo_dir = Path(repo_dir)
        self._lock = threading.Lock()
        self._loaded = False

    def available(self) -> tuple[bool, str]:
        if not (self.repo_dir / self.marker).is_file():
            return False, (f"{self.name} is not installed: run `git clone {self.repo_url} {self.repo_dir}` "
                           f"and `uv sync --extra qwen --extra vc`.")
        missing = [m for m in self.modules if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"{self.name} needs {', '.join(missing)}: run `uv sync --extra qwen --extra vc`."
        return True, ""

    modules: tuple[str, ...] = ("torch", "torchaudio", "librosa")

    def load(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        ok, why = self.available()
        if not ok:
            raise BackendError(why)
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        log.info("Loading %s", self.name)
        t0 = time.perf_counter()
        try:
            self._load()
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"Failed to load {self.name}: {e}") from e
        log.info("Loaded %s in %.1fs", self.name, time.perf_counter() - t0)
        self._loaded = True

    @abstractmethod
    def _load(self) -> None: ...


class SeedVCConverter(_CheckoutConverter):
    """Seed-VC's 22.05 kHz speech model (no F0 conditioning), run in-process without temp files.

    Mirrors ``SeedVCWrapper.convert_voice`` (``f0_condition=False``) from the Seed-VC repository.
    """

    name = "seed-vc"
    repo_url = "https://github.com/Plachtaa/seed-vc"
    marker = "modules/commons.py"
    modules = ("torch", "torchaudio", "librosa", "munch", "einops", "yaml", "scipy", "matplotlib")
    HF_REPO = "Plachta/Seed-VC"
    CHECKPOINT = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
    CONFIG = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    MAX_REF_SECONDS = 25  # Seed-VC uses at most 25 s of reference
    OVERLAP_FRAMES = 16

    @staticmethod
    def _stub_dac() -> None:
        # Seed-VC imports dac's VectorQuantize at module level, but the speech model never builds it
        # (vector_quantize: false). descript-audio-codec would pin protobuf 3.19, so stand in for it.
        if importlib.util.find_spec("dac") is not None or "dac" in sys.modules:
            return

        class VectorQuantize:  # noqa: D401 - placeholder
            def __init__(self, *args, **kwargs):
                raise BackendError("This Seed-VC model needs descript-audio-codec (pip install descript-audio-codec).")

        for name in ("dac", "dac.nn", "dac.nn.quantize"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["dac.nn.quantize"].VectorQuantize = VectorQuantize

    def _load(self) -> None:
        import torch
        import yaml
        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, WhisperModel

        self._stub_dac()
        from modules.audio import mel_spectrogram
        from modules.bigvgan import bigvgan
        from modules.campplus.DTDNN import CAMPPlus
        from modules.commons import build_model, load_checkpoint, recursive_munch

        self.device = torch.device(_device(self.settings))
        ckpt = hf_hub_download(self.HF_REPO, self.CHECKPOINT)
        config = yaml.safe_load(Path(hf_hub_download(self.HF_REPO, self.CONFIG)).read_text(encoding="utf-8"))
        model_params = recursive_munch(config["model_params"])
        model = build_model(model_params, stage="DiT")
        model, *_ = load_checkpoint(model, None, ckpt, load_only_params=True, ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.model = model

        spect = config["preprocess_params"]["spect_params"]
        self.sr = config["preprocess_params"]["sr"]
        self.hop = spect["hop_length"]
        mel_args = dict(n_fft=spect["n_fft"], win_size=spect["win_length"], hop_size=self.hop,
                        num_mels=spect["n_mels"], sampling_rate=self.sr, fmin=0, fmax=None, center=False)
        self.to_mel = lambda x: mel_spectrogram(x, **mel_args)

        whisper_name = getattr(model_params.speech_tokenizer, "name", "openai/whisper-small")
        self.whisper = WhisperModel.from_pretrained(whisper_name, dtype=torch.float16).to(self.device)
        del self.whisper.decoder
        self.whisper_fe = AutoFeatureExtractor.from_pretrained(whisper_name)

        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus.load_state_dict(torch.load(hf_hub_download("funasr/campplus", "campplus_cn_common.bin"), map_location="cpu"))
        self.campplus.eval().to(self.device)

        self.vocoder = bigvgan.BigVGAN.from_pretrained(model_params.vocoder.name, use_cuda_kernel=False)
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)

    def _whisper_features(self, wav_16k):
        """Whisper encoder features (50 per second), in 30 s windows with 5 s of overlap for long audio."""
        import torch

        def encode(chunk):
            inputs = self.whisper_fe([chunk.squeeze(0).cpu().numpy()], return_tensors="pt",
                                     return_attention_mask=True, sampling_rate=16000)
            feats = self.whisper._mask_input_features(inputs.input_features, attention_mask=inputs.attention_mask)
            out = self.whisper.encoder(feats.to(self.device, self.whisper.encoder.dtype), return_dict=True)
            return out.last_hidden_state.to(torch.float32)[:, :chunk.size(-1) // 320 + 1]

        window, overlap = 16000 * 30, 16000 * 5
        if wav_16k.size(-1) <= window:
            return encode(wav_16k)
        parts, pos, buffer = [], 0, None
        while pos < wav_16k.size(-1):
            if buffer is None:
                chunk = wav_16k[:, :window]
                pos = chunk.size(-1)
                parts.append(encode(chunk))
            else:
                chunk = torch.cat([buffer, wav_16k[:, pos:pos + window - overlap]], dim=-1)
                pos += chunk.size(-1) - overlap
                parts.append(encode(chunk)[:, 50 * 5:])
            buffer = chunk[:, -overlap:]
        return torch.cat(parts, dim=1)

    def convert(self, source: Audio, reference: Audio, params: ConversionParams) -> Audio:
        import torch
        import torchaudio

        with self._lock:
            self._ensure_loaded()
            steps = params.steps or 25
            cfg_rate = 0.7 if params.cfg_rate is None else params.cfg_rate
            sr, dev = self.sr, self.device
            with torch.inference_mode():
                src = _resample(np.clip(source[0], -1, 1), source[1], sr).to(dev)
                ref = _resample(np.clip(reference[0], -1, 1), reference[1], sr)[:, :sr * self.MAX_REF_SECONDS].to(dev)
                src_16k = torchaudio.functional.resample(src, sr, 16000)
                ref_16k = torchaudio.functional.resample(ref, sr, 16000)

                content_src = self._whisper_features(src_16k)
                content_ref = self._whisper_features(ref_16k)
                mel_src, mel_ref = self.to_mel(src.float()), self.to_mel(ref.float())

                fbank = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
                style = self.campplus((fbank - fbank.mean(dim=0, keepdim=True)).unsqueeze(0))

                regulate = self.model.length_regulator
                cond = regulate(content_src, ylens=torch.LongTensor([mel_src.size(2)]).to(dev), n_quantizers=3, f0=None)[0]
                prompt = regulate(content_ref, ylens=torch.LongTensor([mel_ref.size(2)]).to(dev), n_quantizers=3, f0=None)[0]

                # The DiT attends over reference + source frames; long sources are converted in windows
                # that overlap by a few frames and are crossfaded.
                max_window = sr // self.hop * 30 - mel_ref.size(2)
                overlap_wave = self.OVERLAP_FRAMES * self.hop
                fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap_wave)) ** 2
                pieces: list[np.ndarray] = []
                tail: Optional[np.ndarray] = None
                done = 0
                while done < cond.size(1):
                    chunk = cond[:, done:done + max_window]
                    last = done + max_window >= cond.size(1)
                    cat = torch.cat([prompt, chunk], dim=1)
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                        mel = self.model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(dev), mel_ref, style,
                                                       None, steps, inference_cfg_rate=cfg_rate)
                    mel = mel[:, :, mel_ref.size(-1):]
                    wave = self.vocoder(mel.float())[0, 0].float().cpu().numpy()
                    if tail is not None:
                        n = min(overlap_wave, len(wave))
                        wave[:n] = wave[:n] * fade_in[:n] + tail[:n] * fade_in[::-1][:n]
                    if last:
                        pieces.append(wave)
                        break
                    pieces.append(wave[:-overlap_wave])
                    tail = wave[-overlap_wave:]
                    done += mel.size(2) - self.OVERLAP_FRAMES
        return np.concatenate(pieces).astype(np.float32), sr


class OpenVoiceConverter(_CheckoutConverter):
    """OpenVoice v2 tone-colour converter (only its VITS-style converter; no MeloTTS or watermarking)."""

    name = "openvoice"
    repo_url = "https://github.com/myshell-ai/OpenVoice"
    marker = "openvoice/models.py"
    HF_REPO = "myshell-ai/OpenVoiceV2"

    def _load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from openvoice import utils
        from openvoice.mel_processing import spectrogram_torch
        from openvoice.models import SynthesizerTrn

        self.device = _device(self.settings)
        hps = utils.get_hparams_from_file(hf_hub_download(self.HF_REPO, "converter/config.json"))
        model = SynthesizerTrn(len(getattr(hps, "symbols", [])), hps.data.filter_length // 2 + 1,
                               n_speakers=hps.data.n_speakers, **hps.model).to(self.device)
        state = torch.load(hf_hub_download(self.HF_REPO, "converter/checkpoint.pth"), map_location=self.device)
        model.load_state_dict(state["model"], strict=False)
        self.model, self.hps = model.eval(), hps
        d = hps.data
        self.spectrogram = lambda y: spectrogram_torch(y, d.filter_length, d.sampling_rate, d.hop_length,
                                                       d.win_length, center=False)

    def _embedding(self, wav):
        return self.model.ref_enc(self.spectrogram(wav).transpose(1, 2)).unsqueeze(-1)

    def convert(self, source: Audio, reference: Audio, params: ConversionParams) -> Audio:
        import torch

        with self._lock:
            self._ensure_loaded()
            sr = self.hps.data.sampling_rate
            tau = 0.3 if params.tau is None else params.tau
            with torch.no_grad():
                src = _resample(source[0], source[1], sr).to(self.device)
                ref = _resample(reference[0], reference[1], sr).to(self.device)
                spec = self.spectrogram(src)
                lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
                out = self.model.voice_conversion(spec, lengths, sid_src=self._embedding(src),
                                                  sid_tgt=self._embedding(ref), tau=tau)[0]
        return out[0, 0].float().cpu().numpy(), sr


class MockConverter(VoiceConverter):
    """Pairs with the mock TTS backend: returns the performance unchanged and records the call."""

    name = "mock"

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def convert(self, source: Audio, reference: Audio, params: ConversionParams) -> Audio:
        self.calls.append({"source_seconds": len(source[0]) / source[1],
                           "reference_seconds": len(reference[0]) / reference[1], "params": params})
        return source


def create_converters(settings: Settings) -> dict[str, VoiceConverter]:
    if settings.backend == "mock":
        mock = MockConverter()
        return {name: mock for name in CONVERTERS}
    return {"seed-vc": SeedVCConverter(settings, settings.seed_vc_dir),
            "openvoice": OpenVoiceConverter(settings, settings.openvoice_dir)}
