"""Shared helpers for the Rex suite. No hardware, no network."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def assert_halted(mon, dog) -> None:
    """A trip stops the legs, lies the dog down, and locks movement."""
    assert mon.locked.is_set()
    assert mon.reason
    assert not mon.allow_move()
    assert dog.legs_stop_calls >= 1
    assert dog.lie_calls >= 1
    kinds = [event[0] for event in dog.events]
    assert kinds.index("legs_stop") < kinds.index("lie")


def run_py(script: str, *, extra=None, drop_sim: bool = False, timeout: int = 30):
    env = os.environ.copy()
    env.pop("XAI_API_KEY", None)
    env.pop("XAI_KEY", None)
    if drop_sim:
        env.pop("REX_SIM", None)
    if extra:
        env.update(extra)
        env.pop("XAI_API_KEY", None)
        env.pop("XAI_KEY", None)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc


def run_main(args, *, home, drop_sim: bool = False, timeout: int = 30):
    env = os.environ.copy()
    env.pop("XAI_API_KEY", None)
    env.pop("XAI_KEY", None)
    if drop_sim:
        env.pop("REX_SIM", None)
    else:
        env["REX_SIM"] = "1"
    env["HOME"] = str(home)
    env["PIDOG_OLLAMA"] = "0"
    env["PIDOG_SFX"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
