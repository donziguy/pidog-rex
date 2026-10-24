"""Isolate Rex tests from the developer's Pi, API keys, and home directory.

Rex binds log/memory paths at import from Path.home(). That import sees a
scratch HOME so it cannot read ~/.config/pidog/security_dog.env. HOME is then
restored: a child process with a temp HOME still needs the real user site
(requests, Pillow), which Python derives from HOME.
"""
from __future__ import annotations

import os
import signal
import site
import threading
from pathlib import Path

_REAL_HOME = os.environ.get("HOME", "")
_USER_SITE = site.getusersitepackages()
_HOME = Path("/tmp/rex-pytest-home")
(_HOME / ".config" / "pidog").mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_HOME)
os.environ["REX_SIM"] = "1"
os.environ.pop("XAI_API_KEY", None)
os.environ.pop("XAI_KEY", None)
os.environ.pop("PIDOG_PRIMARY_HUMAN", None)
# Feature flags and tokens must not leak in from the developer environment.
# REX_SIM stays on so every test uses the software dog.
for _key in list(os.environ):
    if _key.startswith("REX_") and _key != "REX_SIM":
        os.environ.pop(_key, None)

import pytest
import requests

import audio
import brain
import config as C
import main
import sim
import vision_grok

if _REAL_HOME:
    os.environ["HOME"] = _REAL_HOME
_py_path = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
if _USER_SITE and _USER_SITE not in _py_path:
    _py_path.insert(0, _USER_SITE)
os.environ["PYTHONPATH"] = os.pathsep.join(_py_path)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MEMORY_PATH", tmp_path / "rex_memory.json")
    monkeypatch.setattr(C, "LOG_PATH", tmp_path / "rex.log")
    monkeypatch.setattr(C, "FACE_DIR", tmp_path / "faces")
    try:
        import faces
        faces.reset_state()
    except Exception:
        pass
    brain._mem = None
    brain.state = "booting"
    brain._last_mutter = 0.0
    brain._last_roast = 0.0
    vision_grok._last_thumb = None
    vision_grok._pending_thumb = None
    vision_grok._last_call = 0.0
    vision_grok._fail_streak = 0
    sim.reset()
    import dashboard
    dashboard.extensions.clear()
    audio.speaking.clear()
    audio.muted = False
    main._shutdown.clear()
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGINT, prev_int)
    signal.signal(signal.SIGTERM, prev_term)
    main._shutdown.clear()
    for thread in threading.enumerate():
        if thread.name in {"safety", "listener", "dashboard"} and thread is not threading.current_thread():
            stopper = getattr(thread, "stop", None)
            if stopper:
                stopper()
            thread.join(timeout=1)
    brain._mem = None
    sim.reset()


@pytest.fixture(autouse=True)
def blocked_network(monkeypatch):
    """Grok and Ollama must not reach the network. Tests that need a response replace this."""
    calls = []

    def blocked(url="blocked", *args, **kwargs):
        calls.append(url)
        raise requests.ConnectionError("network disabled in tests")

    monkeypatch.setattr(requests, "post", blocked)
    monkeypatch.setattr(requests, "get", blocked)
    return calls


@pytest.fixture(autouse=True)
def blocked_audio(monkeypatch):
    """TTS/STT subprocesses (espeak, aplay, arecord, piper, vosk) never spawn.

    ``audio`` does ``import subprocess``, so patching ``audio.subprocess.run``
    replaces ``subprocess.run`` for this process. Python interpreters are passed
    through so the suite can still launch Rex in a subprocess.
    """
    import subprocess as sp

    real_run = sp.run
    real_popen = sp.Popen
    calls = []

    def _argv(cmd):
        return list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        head = Path(str(_argv(cmd)[0])).name if _argv(cmd) else ""
        if head.startswith("python"):
            return real_run(cmd, *args, **kwargs)
        return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_popen(cmd, *args, **kwargs):
        # subprocess.run() reaches Popen. Only the Vosk worker is stubbed.
        blob = " ".join(str(part) for part in _argv(cmd))
        if "KaldiRecognizer" in blob or "vosk" in blob:
            calls.append(cmd)
            raise RuntimeError("STT subprocess disabled in tests")
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    monkeypatch.setattr(audio.subprocess, "Popen", fake_popen)
    return calls


@pytest.fixture
def make_hw():
    from hardware import Hardware

    made = []

    def factory():
        hw = Hardware()
        made.append(hw)
        return hw

    yield factory
    for hw in made:
        try:
            hw.shutdown()
        except Exception:
            pass
