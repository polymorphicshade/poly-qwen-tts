# Instructions
For each item, create a side branch and implement it. Do not merge anything.
Name each branch in the form `branch-title` (do not include the TODO number in the branch name).

Any new features added should have an accompanying API access point (if applicable).

Skip anything mared as "[DONE]".

# TODO
1. When the user uploads a voice to clone from, if the Reference transcript field is empty, use Whipser to auto-transcribe the audio.
2. Add user-friendly descriptions near wach of the generation options so the user knows how to use them to tune the voice.
3. Add a "Styled Clone" mode that gives cloned voices full inflection control: Qwen CustomVoice/VoiceDesign first generates the speech using the existing emotion, pace, pitch and style controls, then a zero-shot voice-conversion model (Seed-VC, or OpenVoice v2 as a lighter fallback) converts that audio to the cloned voice's timbre while keeping its prosody. Expose it in the web UI, the CLI and a `POST /api/tts/styled-clone` endpoint.