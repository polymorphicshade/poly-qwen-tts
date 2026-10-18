import pytest

from poly_tts.controls import Controls, join_instructions, parse_markup
from poly_tts.voices import BUILTIN_VOICES, DESIGNED_VOICES, NATIVE_VOICES, normalize_language


def test_controls_compile_to_instruction():
    c = Controls(emotion="anger", intensity="strong", delivery="whisper", pace="slow", pitch="high")
    assert c.to_instruct() == "Speak in a very angry tone, whispering, at a slow pace, with a higher pitch."


def test_empty_controls():
    assert Controls().is_empty()
    assert Controls(pace="normal").to_instruct() == ""


def test_controls_reject_unknown_values():
    with pytest.raises(ValueError):
        Controls(emotion="bored-ish")


def test_join_instructions():
    assert join_instructions("Deep voice", None, " ", "Speak slowly.") == "Deep voice. Speak slowly."


def test_parse_pauses_and_styles():
    segs = parse_markup("Hi. [pause 800ms] [style: whispering]Come here[pause 1.5s]now[/style] bye [1]")
    assert [(s.text, s.style, s.pause) for s in segs] == [
        ("Hi.", None, 0.0),
        ("", None, 0.8),
        ("Come here", "whispering", 0.0),
        ("", None, 1.5),
        ("now", "whispering", 0.0),
        ("bye [1]", None, 0.0),
    ]


def test_parse_default_pause_and_cap():
    segs = parse_markup("a [pause] b [pause 99]")
    assert segs[1].pause == 0.5
    assert segs[3].pause == 10.0


def test_parse_errors():
    with pytest.raises(ValueError):
        parse_markup("[pause 1s]")
    with pytest.raises(ValueError):
        parse_markup("[style: a] x [style: b] y")


def test_voice_library_size_and_ids():
    assert len(NATIVE_VOICES) == 9
    assert len(BUILTIN_VOICES) >= 49
    ids = [v.id for v in BUILTIN_VOICES]
    assert len(ids) == len(set(ids))
    assert {v.gender for v in DESIGNED_VOICES} >= {"female", "male"}
    assert {v.age for v in BUILTIN_VOICES} >= {"child", "young", "adult", "senior"}


def test_normalize_language():
    assert normalize_language("en") == "English"
    assert normalize_language("FRENCH") == "French"
    assert normalize_language(None) == "Auto"
    with pytest.raises(ValueError):
        normalize_language("Klingon")
