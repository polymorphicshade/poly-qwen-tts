# Overview
This repo is meant to be a custom implementation of Wan Video: https://github.com/QwenLM/Qwen3-TTS

Use Python, Torch, FastAPI, etc to complete the implementation.

# Requirements
1. Core Functionality
   1. Natural Language Voice Design (VoiceDesign): Synthesize custom voices using text prompts describing timbre, emotional tone, pace, and style (e.g., "Deep male voice with a slight rasp, speaking in a calm, authoritative narrator tone").
   2. Granular Control: Express emotions (joy, anger, sadness, panic) and prosodic inflections (whispering, pauses, pitch shifts).
   3. Zero-Shot Voice Cloning: Clone target voices using as little as 3 seconds of audio input.
   4. Cross-Lingual Voice Cloning: Clone a voice in one language (e.g., English) and make it fluently speak another language (e.g., Mandarin or French) while preserving original timbre and accent characteristics.
   5. Diverse Voice Library: Out-of-the-box access to 49+ high-quality voice timbres spanning different age groups, genders, and expressive roles.
   6. Style Instructions: Combine pre-set voices with natural language style instructions (e.g., choose a default speaker and instruct them to sound incredulous or excited).
2. Web UI
   1. a simple python web UI that allows the user to easily-use this implementation
   2. should allow the user to create/close tabs
      - each tab represents a workspace with all the controls available to the user (similar to if they were to open the web UI in multiple browser tabs)
3. Web API
   1. each core functionality should have an API endpoint
   2. the web UI should implement this API
4. Terminal CLI
   1. the app should support command-line invocation (i.e. for scripting)
   2. each core functionality should be accessible through a CLI