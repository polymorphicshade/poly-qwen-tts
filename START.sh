#!/usr/bin/env bash
# Launches the poly-qwen-tts server. Extra arguments are passed to "poly-tts serve".
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found at .venv. See README.md for setup."
    exit 1
fi

source ".venv/bin/activate"
exec poly-tts serve --port 8000 "$@"
