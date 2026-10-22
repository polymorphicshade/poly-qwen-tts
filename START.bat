@echo off
rem Launches the poly-qwen-tts server. Extra arguments are passed to "poly-tts serve".
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found at .venv. See README.md for setup.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
poly-tts serve --port 8000 %*
pause
