import json
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from poly_tts.api import create_app
from poly_tts.audio import load_audio
from poly_tts.backends import BackendError
from poly_tts.cli import main
from poly_tts.controls import Controls
from poly_tts.engine import DEFAULT_PERFORMERS
from poly_tts.vc import ConversionParams, OpenVoiceConverter, SeedVCConverter

from .conftest import make_wav


def last_call(engine, name):
    return [c for c in engine.backend.calls if c[0] == name][-1][1]


def conversions(engine):
    return engine.converters["seed-vc"].calls


def test_described_performer_then_conversion(engine):
    r = engine.styled_clone("I can't believe it!", "en", ref_audio=load_audio(make_wav(4)),
                            performer_description="Young woman, bright voice", controls=Controls(emotion="incredulity"),
                            conversion=ConversionParams(steps=30))
    assert last_call(engine, "voice_design")["instructs"] == [
        "Young woman, bright voice. Speak in an incredulous, disbelieving tone."]
    call = conversions(engine)[-1]
    assert call["reference_seconds"] == pytest.approx(4.0, abs=0.2) and call["params"].steps == 30
    meta = r.metadata()
    assert meta["mode"] == "styled_clone" and meta["performer"] == "description" and meta["converter"] == "mock"
    assert not r.warnings


def test_native_performer_takes_controls_and_markup(engine):
    engine.styled_clone("Calm. [style: whispering]secret[/style]", ref_audio=load_audio(make_wav(4)),
                        performer_voice_id="ryan", controls=Controls(emotion="calm"))
    assert last_call(engine, "custom_voice")["instructs"] == ["Speak in a calm, relaxed tone.",
                                                              "Speak in a calm, relaxed tone. whispering."]


@pytest.mark.parametrize("freq,gender", [(110.0, "male"), (230.0, "female")])
def test_default_performer_matches_reference_pitch(engine, freq, gender):
    r = engine.styled_clone("Hi!", ref_audio=load_audio(make_wav(4, freq=freq)), controls=Controls(emotion="joy"))
    assert last_call(engine, "voice_design")["instructs"][0].startswith(DEFAULT_PERFORMERS[gender])
    assert r.metadata()["performer"] == f"auto ({gender})"


def test_library_targets(engine):
    voice = engine.save_voice("Target", load_audio(make_wav(6)), "words")
    r = engine.styled_clone("Hi", voice_id=voice.id, instruct="sound excited")
    assert r.voice_id == voice.id and conversions(engine)[-1]["reference_seconds"] == pytest.approx(6.0, abs=0.2)
    engine.styled_clone("Hi", voice_id="noir_detective", instruct="sound excited")  # designed preset: its anchor clip
    with pytest.raises(ValueError, match="native speaker"):
        engine.styled_clone("Hi", voice_id="aiden", instruct="angry")


def test_validation(engine):
    ref = load_audio(make_wav(4))
    with pytest.raises(ValueError, match="exactly one"):
        engine.styled_clone("Hi")
    with pytest.raises(ValueError, match="exactly one"):
        engine.styled_clone("Hi", ref_audio=ref, voice_id="noir_detective")
    with pytest.raises(ValueError, match="at most one"):
        engine.styled_clone("Hi", ref_audio=ref, performer_voice_id="ryan", performer_description="deep")
    with pytest.raises(ValueError, match="Unknown voice converter"):
        engine.styled_clone("Hi", ref_audio=ref, converter="rvc")
    with pytest.raises(ValueError, match="at least 3"):
        engine.styled_clone("Hi", ref_audio=load_audio(make_wav(1)))
    cloned = engine.save_voice("Perf", ref, "words")
    with pytest.raises(ValueError, match="can't perform styles"):
        engine.styled_clone("Hi", ref_audio=ref, performer_voice_id=cloned.id)


def test_unstyled_request_warns(engine):
    r = engine.styled_clone("Hi", ref_audio=load_audio(make_wav(4)))
    assert any("Voice Clone is usually more faithful" in w for w in r.warnings)


def test_missing_converter_checkouts_explain_setup(engine, tmp_path):
    for cls in (SeedVCConverter, OpenVoiceConverter):
        conv = cls(engine.settings, tmp_path / "nope")
        ok, why = conv.available()
        assert not ok and "setup_vc.py" in why and "git clone" in why
        with pytest.raises(BackendError, match="git clone"):
            conv.load()


def test_api_styled_clone(engine):
    client = TestClient(create_app(engine))
    files = {"ref_audio": ("r.wav", make_wav(4), "audio/wav")}
    res = client.post("/api/tts/styled-clone", files=files, data={
        "text": "No way! [pause 300ms] Really?", "performer_voice_id": "sohee",
        "controls": json.dumps({"emotion": "panic", "intensity": "extreme"}), "converter": "seed-vc",
        "conversion": json.dumps({"steps": 10}), "generation": json.dumps({"seed": 1}),
    })
    assert res.status_code == 200, res.text
    meta = json.loads(unquote(res.headers["X-TTS-Metadata"]))
    assert meta["segments"] == 2 and meta["performer"] == "sohee" and meta["instruct"] == "Speak in an extremely panicked, frantic tone."
    assert client.post("/api/tts/styled-clone", files=files,
                       data={"text": "x", "controls": '{"emotion": "meh"}'}).status_code == 422
    assert client.post("/api/tts/styled-clone", files=files,
                       data={"text": "x", "conversion": '{"steps": 0}'}).status_code == 422
    assert client.post("/api/tts/styled-clone", data={"text": "x"}).status_code == 400
    info = client.get("/api/info").json()["voice_conversion"]
    assert info["default"] == "auto" and info["converters"]["seed-vc"]["available"]


def test_cli_styled_clone(engine, ref_wav, tmp_path, capsys):
    out = tmp_path / "s.wav"
    code = main(["--backend", "mock", "styled-clone", "Get out!", "--ref-audio", str(ref_wav), "--emotion", "anger",
                 "--performer", "ryan", "--vc-steps", "12", "-o", str(out), "--json"])
    assert code == 0, capsys.readouterr().err
    meta = json.loads(capsys.readouterr().out)
    assert meta["mode"] == "styled_clone" and meta["performer"] == "ryan" and out.is_file()
    assert conversions(engine)[-1]["params"].steps == 12
