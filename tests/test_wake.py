"""Wake word gate: 'hey rex' before commands, optional detector, whisper or Vosk."""
from __future__ import annotations

import argparse
import sys
import types

import audio
import config as C
import main
import wake
from support import ROOT


def _on(monkeypatch, window=8):
    monkeypatch.setattr(C, "WAKE_WORD", True)
    monkeypatch.setattr(C, "WAKE_WINDOW", window)
    monkeypatch.setattr(C, "WAKE_THRESHOLD", 0.5)
    monkeypatch.setattr(C, "WAKE_MODEL", "hey_rex")
    wake.reset_state()
    audio.command_filter = wake.accept_command


def test_wake_word_defaults_off_and_commands_still_parse():
    assert C.WAKE_WORD is False
    assert C.STT_ENGINE == "vosk"
    assert C.WHISPER_MODEL == "tiny"
    assert C.WHISPER_COMPUTE == "int8"
    assert C.WAKE_WINDOW == 8
    audio.command_filter = None
    assert audio.apply_heard("patrol") == ("patrol", "patrol")
    assert "REX_WAKE_WORD=" in (ROOT / ".env.example").read_text()
    assert "REX_WAKE_WORD" not in (ROOT / "rex.service").read_text()


def test_phrase_gate_and_window(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(wake.time, "monotonic", lambda: clock["t"])
    _on(monkeypatch, window=8)
    assert audio.apply_heard("patrol") is None
    assert audio.apply_heard("please walk") is None
    assert wake.accept_command("hey rex") == ""
    assert audio.apply_heard("hey rex")[0] == ""
    assert audio.apply_heard("patrol") == ("patrol", "patrol")
    assert audio.apply_heard("Hey, Rex stop") == ("stop", "stop")
    clock["t"] = 109
    assert audio.apply_heard("sit") is None
    assert audio.apply_heard("hey rex patrol") == ("patrol", "patrol")
    monkeypatch.setattr(C, "WAKE_WORD", False)
    assert wake.accept_command("patrol") == "patrol"


def test_openwakeword_hit_arms_the_window(monkeypatch):
    _on(monkeypatch)

    class _Det:
        def __init__(self):
            self.score = 0.1

        def predict(self, samples):
            return {"hey_rex": [self.score]}

    det = _Det()
    monkeypatch.setattr(wake, "_load_detector", lambda: det)
    assert wake.audio_has_wake(b"\x00\x01") is False
    assert wake.accept_command("patrol", samples=b"\x00\x01") is None
    det.score = 0.95
    assert wake.audio_has_wake(b"\x00\x01") is True
    assert wake.accept_command("patrol", samples=b"\x00\x01") == "patrol"
    assert audio.apply_heard("sit") == ("rest", "sit")


def test_missing_detector_keeps_the_phrase_gate(monkeypatch, capsys):
    _on(monkeypatch)
    bad = types.ModuleType("openwakeword")
    model = types.ModuleType("openwakeword.model")

    def _model(*_a, **_k):
        raise RuntimeError("no hey_rex model")

    model.Model = _model
    bad.model = model
    monkeypatch.setitem(sys.modules, "openwakeword", bad)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model)
    wake.reset_state()
    monkeypatch.setattr(C, "WAKE_WORD", True)
    assert wake._load_detector() is None
    assert "phrase_gate" in capsys.readouterr().out
    assert wake.accept_command("patrol") is None
    assert wake.accept_command("hey rex sit") == "sit"


def test_whisper_settings_and_vosk_fallback(monkeypatch, capsys):
    wake.reset_state()
    monkeypatch.setattr(C, "STT_ENGINE", "whisper")
    monkeypatch.setattr(C, "WHISPER_MODEL", "base")
    monkeypatch.setattr(C, "WHISPER_COMPUTE", "float16")
    built = []

    class _Model:
        def __init__(self, size, device, compute_type):
            built.append((size, device, compute_type))

        def transcribe(self, path, language="en", beam_size=1):
            return [argparse.Namespace(text=" hey rex sit ")], None

    monkeypatch.setattr(wake, "_whisper_class", lambda: _Model)
    assert wake.transcribe_wav("clip.wav", lambda _path: "from-vosk") == "hey rex sit"
    assert built == [("base", "cpu", "int8")]
    assert "whisper_compute_rejected" in capsys.readouterr().out

    wake.reset_state()
    monkeypatch.setattr(C, "WHISPER_MODEL", "large")
    assert wake.transcribe_wav("clip.wav", lambda _path: "from-vosk") == "hey rex sit"
    assert built[-1][0] == "tiny"
    assert "whisper_model_rejected" in capsys.readouterr().out

    def _missing():
        raise ImportError("faster_whisper")

    monkeypatch.setattr(wake, "_whisper_class", _missing)
    assert wake.transcribe_wav("clip.wav", lambda _path: "from-vosk") == "from-vosk"
    assert "fallback_vosk" in capsys.readouterr().out

    monkeypatch.setattr(C, "STT_ENGINE", "vosk")
    assert wake.transcribe_wav("clip.wav", lambda _path: "from-vosk") == "from-vosk"


def test_run_enables_the_gate_only_when_asked(monkeypatch, capsys):
    monkeypatch.setattr(C, "WAKE_WORD", True)
    monkeypatch.setattr(main._shutdown, "wait", lambda timeout=None: None)
    import vision_grok
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    args = argparse.Namespace(
        patrol=False, sim=True, allow_voice_patrol=False, seconds=0.05,
        allow_no_imu=False, no_tts=True, no_listen=True, force_first_grok=False,
        follow=False,
    )
    assert main.run(args) == 0
    out = capsys.readouterr().out
    assert "wake_word_enabled phrase='hey rex'" in out
    assert audio.command_filter is wake.accept_command
    assert audio.apply_heard("patrol") is None
