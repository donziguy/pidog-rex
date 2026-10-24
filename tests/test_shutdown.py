"""Ctrl+C / SIGTERM must stop the legs and abandon Grok and TTS."""
from __future__ import annotations

import io
import signal
import threading
import time

import abort
import audio
import main
from locomotion import Locomotion
from safety import SafetyMonitor


class _Hang:
    def __init__(self):
        self.stdin = io.StringIO()
        self.killed = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class _Stdin:
    def __init__(self):
        self.text = ""
        self.closed = False

    def write(self, data):
        self.text += data
        return len(data)

    def close(self):
        self.closed = True


class _Done:
    def __init__(self):
        self.stdin = _Stdin()
        self.returncode = 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_first_signal_stops_legs_second_forces_shutdown(make_hw, monkeypatch, capsys):
    hw = make_hw()
    loco = Locomotion(hw, SafetyMonitor(hw))
    assert loco.enable("test")
    main._bits.update(hw=hw, loco=loco, in_shutdown=False)
    main._sig_count = 0
    exits = []
    monkeypatch.setattr(main.os, "_exit", lambda code: exits.append(code))

    main._on_signal(signal.SIGINT, None)
    first = capsys.readouterr().out
    assert "signal_received=2" in first
    assert "signal_force_shutdown" not in first
    assert main._shutdown.is_set()
    assert loco.enabled is False
    assert hw.dog.legs_stop_calls >= 1
    assert exits == []

    main._on_signal(signal.SIGINT, None)
    second = capsys.readouterr().out
    assert "signal_force_shutdown" in second
    assert "hardware_shutdown lie_down + close" in second
    assert "hardware_shutdown_done" in second
    assert hw.dog.lie_calls >= 1
    assert exits == [130]


def test_second_sigterm_exits_143(make_hw, monkeypatch):
    hw = make_hw()
    main._bits.update(hw=hw, loco=None, in_shutdown=False)
    main._sig_count = 1
    exits = []
    monkeypatch.setattr(main.os, "_exit", lambda code: exits.append(code))
    main._on_signal(signal.SIGTERM, None)
    assert exits == [143]
    assert hw.dog.lie_calls >= 1


def test_second_signal_during_shutdown_does_not_reenter(make_hw, monkeypatch, capsys):
    hw = make_hw()
    main._bits.update(hw=hw, loco=None, in_shutdown=True)
    main._sig_count = 1
    exits = []
    monkeypatch.setattr(main.os, "_exit", lambda code: exits.append(code))
    before = hw.dog.lie_calls
    main._on_signal(signal.SIGINT, None)
    out = capsys.readouterr().out
    assert exits == [130]
    assert hw.dog.lie_calls == before
    assert "hardware_shutdown" not in out
    assert "signal_force_shutdown" in out


def test_reinstall_puts_the_abort_handler_back():
    abort.install(signal.SIGINT, main._on_signal)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    abort.reinstall()
    assert signal.getsignal(signal.SIGINT) is main._on_signal


def test_inflight_tts_is_abandoned(monkeypatch, capsys):
    hung = _Hang()
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **k: hung)

    def trip():
        time.sleep(0.12)
        abort.event.set()
        abort.kill_tts()

    threading.Thread(target=trip, daemon=True).start()
    started = time.monotonic()
    audio.speak("get off the rug", enabled=True)
    assert time.monotonic() - started < 2
    assert hung.killed
    assert "tts_abandoned" in capsys.readouterr().out


def test_piper_gets_the_phrase_on_stdin(monkeypatch, tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    piper = tmp_path / "piper"
    piper.write_bytes(b"x")
    monkeypatch.setenv("PIDOG_PIPER_MODEL", str(model))
    monkeypatch.setattr(audio.shutil, "which", lambda name: str(piper) if name == "piper" else None)
    calls = []

    def popen(cmd, **kwargs):
        proc = _Done()
        calls.append((list(cmd), kwargs.get("stdin"), proc))
        return proc

    monkeypatch.setattr(audio.subprocess, "Popen", popen)
    audio.speak("good boy", enabled=True)
    piper_calls = [item for item in calls if item[0] and item[0][0] == str(piper)]
    assert len(piper_calls) == 1
    assert piper_calls[0][1] is subprocess_pipe()
    deadline = time.monotonic() + 1
    phrase = piper_calls[0][2].stdin
    while phrase.text != "good boy\n" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert phrase.text == "good boy\n"
    assert phrase.closed


def subprocess_pipe():
    import subprocess
    return subprocess.PIPE


def test_pause_stops_when_shutdown_is_requested():
    abort.event.set()
    assert abort.pause(5, sleep=lambda _s: None) is False
