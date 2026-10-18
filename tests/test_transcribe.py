import json
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from poly_tts.api import create_app
from poly_tts.audio import load_audio
from poly_tts.cli import main

from .conftest import make_wav


def prompt_calls(engine):
    return [c[1] for c in engine.backend.calls if c[0] == "create_clone_prompt"]


def test_clone_without_transcript_is_auto_transcribed(engine):
    r = engine.clone("hello", load_audio(make_wav(4)))
    call = prompt_calls(engine)[-1]
    assert call["x_vector_only"] is False
    assert call["ref_text"] == r.ref_text_auto and r.ref_text_auto.startswith("Mock transcript")
    assert r.metadata()["ref_text_auto"] == r.ref_text_auto
    assert not r.warnings


def test_given_transcript_and_x_vector_only_skip_transcription(engine):
    r = engine.clone("hello", load_audio(make_wav(4)), "what I said")
    assert r.ref_text_auto is None and "ref_text_auto" not in r.metadata()
    r = engine.clone("hello", load_audio(make_wav(4)), x_vector_only=True)
    assert r.ref_text_auto is None and prompt_calls(engine)[-1]["x_vector_only"] is True


def test_auto_transcribe_disabled_falls_back_to_xvector(engine):
    engine.settings.auto_transcribe = False
    r = engine.clone("hello", load_audio(make_wav(4)))
    assert prompt_calls(engine)[-1]["x_vector_only"] is True
    assert any("transcript" in w for w in r.warnings)


def test_failed_transcription_falls_back_to_xvector(engine, monkeypatch):
    def boom(audio, language=None):
        raise ValueError("no speech")

    monkeypatch.setattr(engine.transcriber, "transcribe", boom)
    r = engine.clone("hello", load_audio(make_wav(4)))
    assert prompt_calls(engine)[-1]["x_vector_only"] is True
    assert any("auto-transcription failed" in w for w in r.warnings)


def test_cross_lingual_transcribes_in_source_language(engine, monkeypatch):
    seen = []
    original = engine.transcriber.transcribe
    monkeypatch.setattr(engine.transcriber, "transcribe", lambda a, language=None: seen.append(language) or original(a, language))
    r = engine.cross_lingual("Bonjour", "fr", load_audio(make_wav(4)), source_language="en")
    assert seen == ["English"] and r.ref_text_auto


def test_transcripts_are_cached(engine, monkeypatch):
    calls = []
    original = engine.transcriber.transcribe
    monkeypatch.setattr(engine.transcriber, "transcribe", lambda a, language=None: calls.append(1) or original(a, language))
    ref = load_audio(make_wav(4))
    engine.clone("a", ref)
    engine.clone("b", ref)
    assert len(calls) == 1


def test_transcribe_rejects_silence(engine):
    import numpy as np

    with pytest.raises(ValueError, match="silent"):
        engine.transcribe((np.zeros(16000 * 4, dtype=np.float32), 16000))


def test_saved_voice_gets_transcript(engine):
    voice = engine.save_voice("No Text", load_audio(make_wav(5)))
    _, ref_text = engine.library.reference(voice.id)
    assert ref_text and ref_text.startswith("Mock transcript")


def test_saved_voice_without_transcript_is_transcribed_on_use(engine):
    engine.settings.auto_transcribe = False
    voice = engine.save_voice("Old Voice", load_audio(make_wav(5)))
    assert engine.library.reference(voice.id)[1] is None
    engine.settings.auto_transcribe = True
    engine.speak("hi", voice.id)
    assert prompt_calls(engine)[-1]["x_vector_only"] is False
    assert engine.library.reference(voice.id)[1].startswith("Mock transcript")


def test_api_transcribe_and_clone_metadata(engine):
    client = TestClient(create_app(engine))
    res = client.post("/api/transcribe", data={"language": "en"}, files={"audio": ("a.wav", make_wav(4), "audio/wav")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["text"].startswith("Mock transcript") and body["language"] == "English" and body["duration"] == 4.0
    assert client.post("/api/transcribe", data={"language": "Klingon"},
                       files={"audio": ("a.wav", make_wav(4), "audio/wav")}).status_code == 400
    res = client.post("/api/tts/clone", data={"text": "Hi"}, files={"ref_audio": ("r.wav", make_wav(4), "audio/wav")})
    assert res.status_code == 200
    assert json.loads(unquote(res.headers["X-TTS-Metadata"]))["ref_text_auto"].startswith("Mock transcript")
    assert client.get("/api/info").json()["transcription"] == {"auto_transcribe": True, "model": "mock"}


def test_cli_transcribe(engine, ref_wav, tmp_path, capsys):
    out = tmp_path / "t.txt"
    code = main(["--backend", "mock", "transcribe", str(ref_wav), "-o", str(out), "--json"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["text"].startswith("Mock transcript") and out.read_text(encoding="utf-8").strip() == result["text"]
