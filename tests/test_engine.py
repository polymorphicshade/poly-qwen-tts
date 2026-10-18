import numpy as np
import pytest

from poly_tts.audio import load_audio
from poly_tts.controls import Controls

from .conftest import make_wav


def last_call(engine, name):
    return [c for c in engine.backend.calls if c[0] == name][-1][1]


def test_voice_design_combines_description_and_controls(engine):
    r = engine.voice_design("Hello there", "A deep narrator voice", "en", Controls(emotion="joy"))
    assert r.sample_rate == 24000 and r.duration > 0
    assert last_call(engine, "voice_design")["instructs"] == ["A deep narrator voice. Speak in a joyful, happy tone."]
    assert r.language == "English"


def test_pauses_insert_silence(engine):
    plain = engine.speak("one two", "aiden")
    paused = engine.speak("one [pause 1s] two", "aiden")
    assert paused.duration > plain.duration + 0.5
    assert paused.segments == 2


def test_control_native_voice_per_segment_styles(engine):
    engine.control("Loud. [style: whispering]quiet[/style]", voice_id="ryan", controls=Controls(emotion="anger"))
    call = last_call(engine, "custom_voice")
    assert call["speaker"] == "Ryan"
    assert call["instructs"] == ["Speak in an angry tone.", "Speak in an angry tone. whispering."]


def test_control_requires_exactly_one_voice_source(engine):
    with pytest.raises(ValueError):
        engine.control("hi")
    with pytest.raises(ValueError):
        engine.control("hi", voice_id="ryan", voice_description="deep")


def test_designed_preset_anchors_and_reuses_prompt(engine, settings):
    engine.speak("first", "noir_detective")
    engine.speak("second", "noir_detective")
    assert sum(c[0] == "create_clone_prompt" for c in engine.backend.calls) == 1
    assert (settings.presets_dir / "mock" / "noir_detective.wav").is_file()
    assert last_call(engine, "voice_clone")["texts"] == ["second"]


def test_designed_preset_with_style_uses_voice_design(engine):
    engine.style("hi", "noir_detective", "sound excited")
    instr = last_call(engine, "voice_design")["instructs"][0]
    assert "film-noir" in instr and instr.endswith("sound excited.")


def test_clone_requires_three_seconds(engine):
    with pytest.raises(ValueError, match="at least 3"):
        engine.clone("hi", load_audio(make_wav(1.5)), "ref")


def test_clone_without_transcript_falls_back_to_xvector(engine):
    r = engine.clone("hello", load_audio(make_wav(4)))
    assert last_call(engine, "create_clone_prompt")["x_vector_only"] is True
    assert any("transcript" in w for w in r.warnings)


def test_clone_prompt_cached_for_same_reference(engine):
    ref = load_audio(make_wav(4))
    engine.clone("a", ref, "the transcript")
    engine.clone("b", ref, "the transcript")
    assert sum(c[0] == "create_clone_prompt" for c in engine.backend.calls) == 1


def test_cross_lingual(engine):
    r = engine.cross_lingual("Bonjour", "fr", load_audio(make_wav(4)), "hello world", source_language="en")
    assert r.language == "French" and r.mode == "cross_lingual"
    assert last_call(engine, "voice_clone")["language"] == "French"
    with pytest.raises(ValueError):
        engine.cross_lingual("Bonjour", "Auto", load_audio(make_wav(4)))
    with pytest.raises(ValueError):
        engine.cross_lingual("Bonjour", "French")


def test_saved_voice_roundtrip(engine):
    voice = engine.save_voice("My Narrator", load_audio(make_wav(5)), "some words", "English")
    assert voice.id == "my_narrator" and voice.kind == "cloned"
    assert engine.library.get("my_narrator").name == "My Narrator"
    r = engine.cross_lingual("你好", "zh", voice_id="my_narrator")
    assert r.voice_id == "my_narrator"
    with pytest.raises(ValueError, match="cloned voice"):
        engine.style("hi", "my_narrator", "angry")
    engine.remove_voice("my_narrator")
    with pytest.raises(LookupError):
        engine.library.get("my_narrator")


def test_unknown_voice(engine):
    with pytest.raises(LookupError):
        engine.speak("hi", "nobody")


def test_builtin_voices_cannot_be_removed(engine):
    with pytest.raises(ValueError):
        engine.remove_voice("aiden")


def test_seed_is_deterministic(engine):
    from poly_tts.backends import GenerationParams

    a = engine.speak("same", "aiden", params=GenerationParams(seed=7))
    b = engine.speak("same", "aiden", params=GenerationParams(seed=7))
    assert np.array_equal(a.audio, b.audio)
