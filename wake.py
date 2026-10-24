"""Offline wake word. Off unless REX_WAKE_WORD=1.

With the gate on, Rex ignores a command unless the transcript contains "hey rex"
or an optional openWakeWord model scores a hit, and then for REX_WAKE_WINDOW
seconds after that. Speech-to-text stays on the existing Vosk worker unless
REX_STT_ENGINE=whisper, which uses faster-whisper (tiny or base, CPU, int8).
If faster-whisper is not installed, Rex logs that and uses Vosk.
"""
from __future__ import annotations

import time

import config as C
from config import log

_gate_until = 0.0
_detector = None
_detector_failed = False
_whisper_model = None


def reset_state() -> None:
    global _gate_until, _detector, _detector_failed, _whisper_model
    _gate_until = 0.0
    _detector = None
    _detector_failed = False
    _whisper_model = None


def _norm(text: str) -> str:
    cleaned = (text or "").lower().replace(",", " ").replace("!", " ").replace(".", " ")
    return " ".join(cleaned.split())


def arm() -> None:
    global _gate_until
    _gate_until = time.monotonic() + max(0.5, float(C.WAKE_WINDOW))


def armed() -> bool:
    return time.monotonic() < _gate_until


def feed_text(text: str):
    """Return the command text, "" when only the wake word was heard, or None to drop it."""
    if not C.WAKE_WORD:
        return text
    low = _norm(text)
    if "hey rex" in low:
        arm()
        rest = " ".join(low.replace("hey rex", " ", 1).split())
        return rest
    if armed():
        return low
    return None


def accept_command(text: str, samples=None, wav_path=None):
    """audio.command_filter. Samples or a wav path can arm the window via openWakeWord."""
    if not C.WAKE_WORD:
        return text
    if samples is not None and audio_has_wake(samples):
        arm()
    elif wav_path and audio_has_wake_file(wav_path):
        arm()
    return feed_text(text)


def _load_detector():
    global _detector, _detector_failed
    if _detector_failed:
        return None
    if _detector is not None:
        return _detector
    try:
        from openwakeword.model import Model
        _detector = Model(wakeword_models=[C.WAKE_MODEL])
    except Exception:
        _detector_failed = True
        log("wake_detector_unavailable phrase_gate")
        return None
    return _detector


def _score(scores) -> float:
    if isinstance(scores, dict):
        best = 0.0
        want = (C.WAKE_MODEL or "hey_rex").lower()
        for key, val in scores.items():
            name = str(key).lower()
            if want not in name and not ("hey" in name and "rex" in name):
                continue
            try:
                number = float(val if isinstance(val, (int, float)) else val[-1])
            except Exception:
                continue
            best = max(best, number)
        return best
    try:
        return float(scores)
    except (TypeError, ValueError):
        return 0.0


def audio_has_wake(samples) -> bool:
    if not C.WAKE_WORD or samples is None:
        return False
    det = _load_detector()
    if det is None:
        return False
    try:
        scores = det.predict(samples)
    except Exception as exc:
        log(f"wake_predict_error={type(exc).__name__}")
        return False
    return _score(scores) >= float(C.WAKE_THRESHOLD)


def audio_has_wake_file(path) -> bool:
    det = _load_detector()
    if det is None:
        return False
    try:
        import wave
        with wave.open(str(path), "rb") as handle:
            samples = handle.readframes(handle.getnframes())
    except Exception:
        return False
    return audio_has_wake(samples)


def _whisper_class():
    from faster_whisper import WhisperModel
    return WhisperModel


def _whisper_settings() -> tuple[str, str]:
    size = (C.WHISPER_MODEL or "tiny").lower()
    if size not in {"tiny", "base"}:
        log(f"whisper_model_rejected {size} -> tiny")
        size = "tiny"
    compute = (C.WHISPER_COMPUTE or "int8").lower()
    if compute != "int8":
        log(f"whisper_compute_rejected {compute} -> int8")
        compute = "int8"
    return size, compute


def transcribe_whisper(path) -> str | None:
    """Return text, or None when faster-whisper is not installed. Never raises."""
    global _whisper_model
    try:
        whisper_model = _whisper_class()
    except Exception:
        return None
    size, compute = _whisper_settings()
    try:
        if _whisper_model is None:
            _whisper_model = whisper_model(size, device="cpu", compute_type=compute)
        segments, _info = _whisper_model.transcribe(str(path), language="en", beam_size=1)
        return " ".join(getattr(seg, "text", "").strip() for seg in segments).strip()
    except Exception as exc:
        log(f"whisper_error={type(exc).__name__}")
        return None


def transcribe_wav(path, vosk_text) -> str:
    """Whisper when REX_STT_ENGINE=whisper and the package imports; otherwise Vosk."""
    if (C.STT_ENGINE or "vosk").lower() == "whisper":
        text = transcribe_whisper(path)
        if text is not None:
            return text
        log("whisper_unavailable fallback_vosk")
    return vosk_text(path)
