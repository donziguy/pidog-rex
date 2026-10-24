"""TTS, sound effects, and a background speech listener (arecord + local Vosk)."""
from __future__ import annotations
import math, queue, shutil, struct, subprocess, sys, threading, time, wave
from pathlib import Path
import config as C
from config import log
import brain

speaking = threading.Event()
_last_tts_end = 0.0


def speak(text: str, enabled: bool = True) -> None:
    global _last_tts_end
    text = (text or "").strip()
    if not text:
        return
    log(f"TTS: {text}")
    if not enabled:
        return
    speaking.set()
    try:
        model = Path(__import__("os").environ.get("PIDOG_PIPER_MODEL", "") or "/nonexistent")
        piper = shutil.which("piper") or str(Path.home() / ".local/bin/piper")
        if model.exists() and Path(piper).exists():
            wav = "/tmp/rex_voice.wav"
            subprocess.run([piper, "--model", str(model), "--output_file", wav], input=text, text=True,
                           check=False, timeout=30, capture_output=True)
            subprocess.run(["aplay", "-q", wav], check=False, timeout=30)
        else:
            subprocess.run(["espeak", "-s", "142", "-p", "32", text], check=False, timeout=25, capture_output=True)
    except Exception as exc:
        log(f"tts_error={type(exc).__name__}")
    finally:
        _last_tts_end = time.time()
        speaking.clear()


_PALETTE = {"attention": [(780, .08), (980, .08)], "alert": [(520, .1), (760, .1), (1040, .12)],
            "growl": [(130, .18), (105, .18), (145, .16)], "bark": [(900, .06), (420, .08), (900, .05)],
            "sigh": [(260, .18), (210, .22), (170, .24)], "huff": [(180, .08), (140, .1)],
            "happy": [(650, .07), (850, .07), (1050, .09)], "annoyed": [(220, .09), (170, .09), (220, .09)]}


def sfx(effect: str, enabled: bool = True) -> None:
    if not (enabled and C.SFX_ENABLED):
        return
    path = Path(f"/tmp/rex_sfx_{effect}.wav")
    try:
        if not path.exists():
            sr, amp, out = 16000, int(14000 * C.SFX_VOLUME / 100), []
            for f, dur in _PALETTE.get(effect, _PALETTE["attention"]):
                n = int(sr * dur)
                for i in range(n):
                    t = i / sr
                    wob = 1 + (0.08 * math.sin(2 * math.pi * 22 * t) if f < 250 else 0)
                    env = min(1.0, i / (sr * .015), (n - i) / (sr * .02))
                    out.append(int(amp * env * math.sin(2 * math.pi * f * wob * t)))
                out += [0] * int(sr * .025)
            with wave.open(str(path), "w") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes(b"".join(struct.pack("<h", max(-32767, min(32767, x))) for x in out))
        subprocess.run(["aplay", "-q", str(path)], check=False, timeout=8)
        brain.bump("sfx_played")
    except Exception as exc:
        log(f"sfx_error={exc}")


# Persistent Vosk worker: loads the model once and transcribes WAV paths sent on stdin.
_VOSK_WORKER = r"""
import json, sys, wave
from vosk import Model, KaldiRecognizer, SetLogLevel
SetLogLevel(-1)
m = Model(sys.argv[1])
print("READY", flush=True)
for line in sys.stdin:
    try:
        wf = wave.open(line.strip(), 'rb'); rec = KaldiRecognizer(m, wf.getframerate()); parts = []
        while True:
            d = wf.readframes(4000)
            if not d: break
            if rec.AcceptWaveform(d): parts.append(json.loads(rec.Result()).get('text', ''))
        parts.append(json.loads(rec.FinalResult()).get('text', ''))
        print(' '.join(p for p in parts if p).strip(), flush=True)
    except Exception:
        print('', flush=True)
"""

COMMANDS = [  # (keywords, command) - checked in order; "stop" first
    (("stop", "freeze", "halt", "stay", "no rex", "bad dog"), "stop"),
    (("patrol", "go walk", "walk around", "go explore"), "patrol"),
    (("sit", "rest", "lie down", "shelf"), "rest"),
    (("good boy", "good dog", "hello", "hi rex"), "greet"),
]


def parse_command(text: str) -> str | None:
    t = f" {text.lower()} "
    for words, cmd in COMMANDS:
        if any(f" {w} " in t for w in words):
            return cmd
    return None


class Listener(threading.Thread):
    """Records short clips back-to-back and pushes (command|None, text) onto self.events."""

    def __init__(self):
        super().__init__(name="listener", daemon=True)
        self.events: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._proc = None
        self.ready = False
        self.on_stop = None  # called immediately (from this thread) on a 'stop' command

    def stop(self):
        self._stop.set()
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass

    def _start_worker(self) -> bool:
        py = C.STT_PYTHON if Path(C.STT_PYTHON).exists() else sys.executable
        if not C.STT_MODEL_PATH.exists():
            log(f"stt_model_missing {C.STT_MODEL_PATH}")
            return False
        self._proc = subprocess.Popen([py, "-c", _VOSK_WORKER, str(C.STT_MODEL_PATH)], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        ok = (self._proc.stdout.readline().strip() == "READY")
        log(f"stt_worker_ready={ok}")
        return ok

    def _peak(self, path: Path) -> int:
        try:
            with wave.open(str(path)) as w:
                data = w.readframes(w.getnframes())
            vals = struct.unpack(f"<{len(data)//2}h", data)
            return max((abs(v) for v in vals), default=0)
        except Exception:
            return 0

    def run(self):
        if not C.STT_ENABLED or not shutil.which("arecord"):
            log("listener_disabled")
            return
        try:
            self.ready = self._start_worker()
        except Exception as exc:
            log(f"stt_worker_error={exc}")
        if not self.ready:
            return
        wav = Path("/tmp/rex_listen.wav")
        secs = max(2, int(round(C.STT_SECONDS)))
        while not self._stop.is_set():
            if speaking.is_set() or time.time() - _last_tts_end < 1.0:
                time.sleep(0.2); continue
            r = subprocess.run(["arecord", "-q", "-D", C.STT_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1",
                                "-d", str(secs), str(wav)], capture_output=True, text=True, timeout=secs + 6)
            if self._stop.is_set():
                break
            if r.returncode != 0:
                log(f"arecord_rc={r.returncode}"); time.sleep(1); continue
            if self._peak(wav) < C.STT_MIN_PEAK:
                continue
            try:
                self._proc.stdin.write(str(wav) + "\n"); self._proc.stdin.flush()
                text = self._proc.stdout.readline().strip()
            except Exception as exc:
                log(f"stt_worker_died={exc}"); self.ready = False; return
            if text:
                cmd = parse_command(text)
                log(f"heard text_len={len(text)} cmd={cmd}")
                brain.bump("stt_transcripts")
                if cmd == "stop" and self.on_stop:
                    self.on_stop()
                self.events.put((cmd, text))
