"""Cooperative shutdown. SIGINT/SIGTERM set the event; blocking calls poll it.

Grok and Ollama run the HTTP call on a daemon thread so the caller can return
when the event is set instead of waiting out the socket timeout. TTS keeps the
Popen handle here so the signal handler can kill it.
"""
from __future__ import annotations

import threading
import time

import requests

event = threading.Event()
_tts_lock = threading.Lock()
_tts_proc = None
_handlers: dict = {}


class Aborted(Exception):
    """A blocking call stopped because shutdown was requested."""


def requested() -> bool:
    return event.is_set()


def install(signum, handler) -> None:
    """Remember a handler so hardware shutdown can put it back.

    Pidog.close() replaces SIGINT with its own "please wait" handler. A second
    Ctrl+C has to reach ours, which forces lie-down, close, and exit.
    """
    _handlers[signum] = handler
    import signal
    signal.signal(signum, handler)


def reinstall() -> None:
    import signal
    for signum, handler in list(_handlers.items()):
        try:
            signal.signal(signum, handler)
        except Exception:
            pass


def track_tts(proc) -> None:
    global _tts_proc
    with _tts_lock:
        _tts_proc = proc


def kill_tts() -> None:
    with _tts_lock:
        proc = _tts_proc
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass


def post(url, timeout, **kwargs):
    """``requests.post`` that raises ``Aborted`` once shutdown is requested.

    The socket thread is a daemon. On abort it is left to finish or die with
    the process; the caller does not wait for it.
    """
    if requested():
        raise Aborted()
    box: dict = {}

    def run() -> None:
        try:
            box["resp"] = requests.post(url, timeout=timeout, **kwargs)
        except Exception as exc:
            box["exc"] = exc

    thread = threading.Thread(target=run, name="http", daemon=True)
    thread.start()
    while thread.is_alive():
        if requested():
            raise Aborted()
        thread.join(0.2)
    if "exc" in box:
        raise box["exc"]
    return box["resp"]


def pause(seconds: float, sleep=time.sleep) -> bool:
    """Wait up to ``seconds``. Return False if shutdown was requested.

    ``sleep`` is injected so tests can no-op it. The wait is sliced so a real
    sleep still notices shutdown within a tenth of a second, and a no-op sleep
    still finishes without spinning on the wall clock.
    """
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        if requested():
            return False
        step = min(0.1, remaining)
        sleep(step)
        remaining -= step
    return not requested()
