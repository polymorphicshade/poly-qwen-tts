# poly-qwen-tts: web UI + REST API on an NVIDIA GPU.
#
# The PyTorch CUDA wheels bundle the CUDA runtime, so a slim Python base is enough. The host
# only needs the NVIDIA driver and the NVIDIA Container Toolkit. Model weights are not baked in:
# they download on first use into /data/cache (mount a volume there). The Styled Clone voice
# converters (Seed-VC, OpenVoice v2) are checked out at their tested commits into /opt/vendor.
#
#   docker build -t poly-qwen-tts .
#   docker run --gpus all -p 8000:8000 -v tts-cache:/data/cache -v tts-library:/data/library poly-qwen-tts

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------- build
FROM python:${PYTHON_VERSION}-slim-bookworm AS build

# git: the voice converters run from source checkouts (see scripts/setup_vc.py).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv

# Install into /opt/venv with the image's Python; torch comes from the CUDA 12.8 index pinned in
# uv.lock (driver >= 570, every GPU from Turing to Blackwell).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /src
# Dependencies first, in their own layer: torch is by far the largest and changes least.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra qwen --extra vc --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra qwen --extra vc --no-editable

# Check out each converter at the commit poly_tts.vc was tested with.
RUN /opt/venv/bin/python -c "from poly_tts.vc import SeedVCConverter as S, OpenVoiceConverter as O; print(S.repo_url, S.commit, 'seed-vc'); print(O.repo_url, O.commit, 'OpenVoice')" \
    | while read -r url commit dir; do \
        git init -q "/opt/vendor/$dir" \
        && git -C "/opt/vendor/$dir" fetch -q --depth 1 "$url" "$commit" \
        && git -C "/opt/vendor/$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD \
        && rm -rf "/opt/vendor/$dir/.git" || exit 1; \
      done \
    && test -d /opt/vendor/seed-vc && test -d /opt/vendor/OpenVoice \
    && find /opt/venv /opt/vendor -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm

# ffmpeg: decodes compressed reference clips (MP3 etc.) for Whisper and the converters.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data/cache /data/library \
    && chown -R app:app /data

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/vendor /opt/vendor

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    POLY_TTS_DATA_DIR=/data/library \
    POLY_TTS_SEED_VC_DIR=/opt/vendor/seed-vc \
    POLY_TTS_OPENVOICE_DIR=/opt/vendor/OpenVoice \
    POLY_TTS_SERVE_ARGS=""

USER app
WORKDIR /data
VOLUME ["/data/cache", "/data/library"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# Extra `serve` flags (e.g. "--preload") go in POLY_TTS_SERVE_ARGS; everything else is a
# POLY_TTS_* variable (see README.md).
CMD ["sh", "-c", "exec poly-tts serve --host 0.0.0.0 --port 8000 $POLY_TTS_SERVE_ARGS"]
