import json

import soundfile as sf

from poly_tts.cli import main


def run(capsys, *argv):
    code = main(["--backend", "mock", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_design(engine, tmp_path, capsys):
    out = tmp_path / "d.wav"
    code, stdout, _ = run(capsys, "design", "Hello", "-d", "Raspy narrator", "--emotion", "sadness", "-o", str(out), "--json")
    assert code == 0
    meta = json.loads(stdout)
    assert meta["instruct"] == "Raspy narrator. Speak in a sad, sorrowful tone."
    assert sf.info(str(out)).samplerate == 24000


def test_control_speak_style(engine, tmp_path, capsys):
    for args in (
        ["control", "Hi [pause 1s] there", "--voice", "ryan", "--delivery", "whisper", "--pace", "slow"],
        ["speak", "Hi", "--voice", "grandma_story"],
        ["style", "Hi", "--voice", "vivian", "-i", "sound excited"],
    ):
        out = tmp_path / f"{args[0]}.wav"
        code, _, err = run(capsys, *args, "-o", str(out))
        assert code == 0, err
        assert out.is_file()


def test_clone_cross_lingual_and_library(engine, tmp_path, ref_wav, capsys):
    out = tmp_path / "c.wav"
    code, _, err = run(capsys, "clone", "Cloned words", "--ref-audio", str(ref_wav), "--ref-text", "hi", "-o", str(out),
                       "--save-as", "CLI Voice")
    assert code == 0 and "saved voice 'cli_voice'" in err
    code, _, _ = run(capsys, "cross-lingual", "Guten Tag", "-t", "de", "--voice", "cli_voice", "-o", str(out), "--seed", "4")
    assert code == 0
    code, stdout, _ = run(capsys, "voices", "list", "--kind", "cloned", "--json")
    assert [v["id"] for v in json.loads(stdout)] == ["cli_voice"]
    code, _, _ = run(capsys, "voices", "remove", "cli_voice")
    assert code == 0


def test_stdin_and_errors(engine, tmp_path, capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    out = tmp_path / "s.wav"
    code, _, _ = run(capsys, "speak", "--voice", "aiden", "-o", str(out))
    assert code == 0 and out.is_file()
    code, _, err = run(capsys, "speak", "Hi", "--voice", "nobody", "-o", str(out))
    assert code == 2 and "Unknown voice" in err


def test_voices_list_table(engine, capsys):
    code, stdout, _ = run(capsys, "voices", "list")
    assert code == 0 and "noir_detective" in stdout and "uncle_fu" in stdout
