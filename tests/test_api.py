import io
import json
from urllib.parse import unquote

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from poly_tts.api import create_app

from .conftest import make_wav


@pytest.fixture
def client(engine):
    return TestClient(create_app(engine))


def meta(res):
    return json.loads(unquote(res.headers["X-TTS-Metadata"]))


def assert_wav(res):
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(res.content))
    assert sr == 24000 and len(data) > 0


def test_info_and_voices(client):
    info = client.get("/api/info").json()
    assert "English" in info["languages"] and "emotion" in info["controls"]
    assert info["voice_counts"]["native"] == 9
    voices = client.get("/api/voices").json()
    assert len(voices) >= 49
    assert all(v["language"] == "Japanese" for v in client.get("/api/voices", params={"language": "japanese"}).json())
    assert client.get("/api/voices/aiden").json()["name"] == "Aiden"
    assert client.get("/api/voices/nope").status_code == 404


def test_design(client):
    res = client.post("/api/tts/design", json={
        "text": "Hello", "voice_description": "Deep male narrator", "controls": {"emotion": "calm"},
        "generation": {"seed": 1, "temperature": 0.8},
    })
    assert_wav(res)
    assert meta(res)["instruct"] == "Deep male narrator. Speak in a calm, relaxed tone."


def test_control(client):
    res = client.post("/api/tts/control", json={
        "text": "No way! [pause 300ms] Really?", "voice_id": "sohee",
        "controls": {"emotion": "panic", "intensity": "extreme", "delivery": "shouting"},
    })
    assert_wav(res)
    assert meta(res)["segments"] == 2


def test_validation_errors(client):
    assert client.post("/api/tts/control", json={"text": "x", "controls": {"emotion": "meh"}, "voice_id": "aiden"}).status_code == 422
    assert client.post("/api/tts/control", json={"text": "x"}).status_code == 400
    assert client.post("/api/tts/voice", json={"text": "x", "voice_id": "ghost"}).status_code == 404
    assert client.post("/api/tts/voice", json={"text": "x", "voice_id": "aiden", "language": "Klingon"}).status_code == 400


def test_voice_and_style(client):
    assert_wav(client.post("/api/tts/voice", json={"text": "Hi", "voice_id": "british_butler", "language": "en"}))
    res = client.post("/api/tts/style", json={"text": "Hi", "voice_id": "serena", "instruct": "sound incredulous"})
    assert_wav(res)
    assert meta(res)["instruct"] == "sound incredulous"


def test_clone_and_cross_lingual(client):
    files = {"ref_audio": ("ref.wav", make_wav(4), "audio/wav")}
    res = client.post("/api/tts/clone", data={"text": "Cloned", "ref_text": "reference words",
                                              "generation": json.dumps({"seed": 3})}, files=files)
    assert_wav(res)
    res = client.post("/api/tts/cross-lingual", data={"text": "Hola", "target_language": "es", "ref_text": "hi"}, files=files)
    assert_wav(res)
    assert meta(res)["language"] == "Spanish"
    res = client.post("/api/tts/cross-lingual", data={"text": "Hola", "target_language": "es", "voice_id": "aiden"})
    assert_wav(res)


def test_clone_rejects_short_and_bad_audio(client):
    short = client.post("/api/tts/clone", data={"text": "x"}, files={"ref_audio": ("r.wav", make_wav(1), "audio/wav")})
    assert short.status_code == 400 and "at least" in short.json()["detail"]
    junk = client.post("/api/tts/clone", data={"text": "x"}, files={"ref_audio": ("r.wav", b"not audio", "audio/wav")})
    assert junk.status_code == 400
    bad_gen = client.post("/api/tts/clone", data={"text": "x", "generation": "{\"top_p\": 5}"},
                          files={"ref_audio": ("r.wav", make_wav(4), "audio/wav")})
    assert bad_gen.status_code == 422


def test_save_use_delete_voice(client):
    res = client.post("/api/voices", data={"name": "Studio Me", "ref_text": "words", "language": "English"},
                      files={"ref_audio": ("me.wav", make_wav(6), "audio/wav")})
    assert res.status_code == 201
    vid = res.json()["id"]
    assert vid in {v["id"] for v in client.get("/api/voices", params={"kind": "cloned"}).json()}
    assert_wav(client.post("/api/tts/voice", json={"text": "Mine", "voice_id": vid}))
    assert client.post("/api/tts/style", json={"text": "x", "voice_id": vid, "instruct": "angry"}).status_code == 400
    assert client.delete(f"/api/voices/{vid}").status_code == 204
    assert client.delete(f"/api/voices/{vid}").status_code == 404


def test_web_ui_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "workspace-template" in res.text
    assert client.get("/static/app.js").status_code == 200
