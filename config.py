"""Rex config + logging. All tunables come from ~/.config/pidog/security_dog.env (PIDOG_*)."""
from __future__ import annotations
import os, time, threading
from pathlib import Path

ENV_PATH = Path.home() / ".config" / "pidog" / "security_dog.env"
LOG_PATH = Path.home() / "pidog-rex" / "rex.log"
_SECRET_NAMES = ("XAI_API_KEY", "XAI_KEY")
_log_lock = threading.Lock()


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and (k.startswith(("PIDOG_", "XAI_")) or k not in os.environ):
            os.environ[k] = v


load_env_file()


def _redact(msg: str) -> str:
    for name in _SECRET_NAMES:
        val = os.environ.get(name)
        if val and len(val) > 6 and val in msg:
            msg = msg.replace(val, "***")
    return msg


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{threading.current_thread().name}] {_redact(str(msg))}"
    with _log_lock:
        print(line, flush=True)
        if os.environ.get("INVOCATION_ID"):
            return  # systemd captures stdout to the log file
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _b(name, default):
    return os.environ.get(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _i(name, default):
    return int(_f(name, default))


def xai_key() -> str | None:
    return os.environ.get("XAI_API_KEY") or os.environ.get("XAI_KEY")


# Identity / memory
PRIMARY_HUMAN = os.environ.get("PIDOG_PRIMARY_HUMAN", "Boss")
MEMORY_PATH = Path(os.environ.get("PIDOG_MEMORY_PATH", str(Path.home() / ".config" / "pidog" / "rex_memory.json")))

# Grok vision
GROK_URL = "https://api.x.ai/v1/chat/completions"
GROK_MODEL = os.environ.get("PIDOG_GROK_MODEL", "grok-4.5")
GROK_MIN_INTERVAL = max(60.0, _f("PIDOG_GROK_MIN_INTERVAL", 60))   # hard floor 60 s
GROK_IMAGE_WIDTH = _i("PIDOG_GROK_IMAGE_WIDTH", 320)
SCENE_DIFF_THRESHOLD = _f("PIDOG_SCENE_DIFF", 12.0)  # mean abs gray diff (0-255) on 32x24 thumbnail
GROK_RETRIES = _i("PIDOG_GROK_RETRIES", 3)
GROK_TIMEOUT = _f("PIDOG_GROK_TIMEOUT", 25)

# Ollama fallback / mutters
OLLAMA_ENABLED = _b("PIDOG_OLLAMA", True)
OLLAMA_URL = os.environ.get("PIDOG_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("PIDOG_OLLAMA_MODEL", "qwen2.5:3b")
IDLE_MUTTER_COOLDOWN = _f("PIDOG_IDLE_MUTTER_COOLDOWN", 75)
ROAST_COOLDOWN = _f("PIDOG_ROAST_COOLDOWN", 35)

# Head / sound
HEAD_PITCH_DEFAULT = _i("PIDOG_HEAD_PITCH", 18)
HEAD_PITCH_COMP = _i("PIDOG_HEAD_PITCH_COMP", -15)
SOUND_TRACKING = _b("PIDOG_SOUND_TRACKING", True)
SOUND_YAW_LIMIT = _i("PIDOG_SOUND_YAW_LIMIT", 55)
SOUND_FRONT_OFFSET = _i("PIDOG_SOUND_FRONT_OFFSET", 0)
SOUND_YAW_INVERT = _b("PIDOG_SOUND_YAW_INVERT", False)
SOUND_PAN_AXIS = os.environ.get("PIDOG_SOUND_PAN_AXIS", "roll").lower()
SOUND_COOLDOWN = _f("PIDOG_SOUND_COOLDOWN", 1.5)

# Audio
TTS_ENABLED = True
SFX_ENABLED = _b("PIDOG_SFX", True)
SFX_VOLUME = _i("PIDOG_SFX_VOLUME", 70)
STT_ENABLED = _b("PIDOG_STT", True)
STT_DEVICE = os.environ.get("PIDOG_STT_DEVICE", "plughw:CARD=Device,DEV=0")
STT_SECONDS = _f("PIDOG_STT_RECORD_SECONDS", 3.0)
STT_MODEL_PATH = Path(os.environ.get("PIDOG_STT_MODEL", str(Path.home() / ".local/share/vosk/vosk-model-small-en-us-0.15")))
STT_PYTHON = os.environ.get("PIDOG_STT_PYTHON", str(Path.home() / ".local/stt-venv/bin/python"))
STT_MIN_PEAK = _i("PIDOG_STT_MIN_PEAK", 1200)

# Locomotion (patrol) - conservative defaults; old PIDOG_PATROL_STEP_* knobs are ignored on purpose
WALK_SPEED = min(80, _i("PIDOG_REX_WALK_SPEED", 65))
WALK_MAX_STEPS = max(1, min(2, _i("PIDOG_REX_WALK_STEPS", 2)))
PATROL_MAX_SECONDS = _f("PIDOG_REX_PATROL_SECONDS", 120)
PATROL_REST_SECONDS = _f("PIDOG_REX_REST_SECONDS", 90)
OBSTACLE_CLEAR_CM = _f("PIDOG_REX_CLEAR_CM", 60)
OBSTACLE_TURN_CM = _f("PIDOG_REX_TURN_CM", 40)

# Safety
SAFETY_POLL = 0.1
SAFETY_STOP_CM = _f("PIDOG_REX_STOP_CM", 25)
SAFETY_TILT_DEG = _f("PIDOG_REX_TILT_DEG", 30)
SAFETY_BATTERY_LOW_V = _f("PIDOG_REX_BATTERY_LOW_V", 6.6)
SAFETY_EMPTY_READS = _i("PIDOG_REX_EMPTY_READS", 3)  # consecutive empty ultrasonic reads while walking
