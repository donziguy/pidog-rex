"""Software dog for off-Pi development and tests.

Imported only when ``REX_SIM`` is set (or ``main.py --sim``). The real
``pidog`` / ``picamera2`` path never imports this module.

Script a sensor over time with ``script([(seconds_from_now, value), ...])``.
The value holds until the next timestamp. ``set`` pins a constant, ``queue``
returns one value per read (the last value sticks).

    dog.distance.script([(0, 100), (1.5, 15)])   # clear, then an obstacle
    dog.battery.script([(0, 7.4), (3.0, 6.2)])   # pack sags
    dog.imu_pitch.set(0)
    dog.touch.set("N")
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

# Applied to every SimDog constructed after configure(), until reset().
_pending: dict = {}
dogs: list = []


def configure(**kwargs) -> None:
    """Overrides for the next dogs: ``imu`` (bool), ``clock``, ``step_seconds``."""
    _pending.update(kwargs)


def reset() -> None:
    _pending.clear()
    dogs.clear()


class ManualClock:
    """Deterministic clock for tests. ``advance`` moves scripted sensors and anything
    patched onto ``time.time`` together."""

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def monotonic(self) -> float:
        return self.t

    def time(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class _WallClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()


class Sensor:
    """One scriptable reading. Thread-safe: the safety monitor and the walk loop share these."""

    def __init__(self, value, clock):
        self._clock = clock
        self._value = value
        self._script = None  # list[(absolute_time, value)] or None
        self._queue = None  # list or None; last entry repeats
        self._fn = None
        self._lock = threading.Lock()
        self.history = []

    def set(self, value):
        with self._lock:
            self._value = value
            self._script = None
            self._queue = None
            self._fn = None
        return self

    def set_fn(self, fn):
        with self._lock:
            self._fn = fn
            self._script = None
            self._queue = None
        return self

    def queue(self, values):
        """Each ``read`` pops the next value. The last value repeats."""
        vals = list(values)
        if not vals:
            raise ValueError("queue needs at least one value")
        with self._lock:
            self._queue = vals
            self._script = None
            self._fn = None
        return self

    def script(self, events):
        """``events`` is ``[(seconds_from_now, value), ...]``. Each value holds until the next time."""
        t0 = self._clock.monotonic()
        schedule = sorted((t0 + float(dt), val) for dt, val in events)
        with self._lock:
            self._script = schedule
            self._queue = None
            self._fn = None
        return self

    def push(self, value) -> None:
        """Append one queued value (used for camera frames)."""
        with self._lock:
            if self._queue is None:
                self._queue = [value]
                self._script = None
                self._fn = None
            else:
                self._queue.append(value)

    def peek(self):
        with self._lock:
            return self._current(consume=False)

    def read(self):
        with self._lock:
            value = self._current(consume=True)
            self.history.append(value)
            if len(self.history) > 200:
                del self.history[:-200]
            return value

    def _current(self, consume: bool):
        if self._fn is not None:
            return self._fn()
        if self._queue is not None:
            if consume and len(self._queue) > 1:
                return self._queue.pop(0)
            return self._queue[0]
        if self._script:
            now = self._clock.monotonic()
            value = self._value
            for t, val in self._script:
                if now >= t:
                    value = val
                else:
                    break
            return value
        return self._value


class _LegBuffer(list):
    def __init__(self, dog):
        super().__init__()
        self._dog = dog

    def clear(self) -> None:
        super().clear()
        self._dog._halt_legs()


class _Ears:
    def __init__(self, dog):
        self._dog = dog

    def isdetected(self) -> bool:
        return self._dog.sound.peek() is not None

    def read(self) -> int:
        return int(self._dog.sound.read())


class _Touch:
    def __init__(self, dog):
        self._dog = dog

    def read(self) -> str:
        return self._dog.touch.read()


class _Strip:
    def __init__(self, dog):
        self._dog = dog

    def set_mode(self, mode, color, bps=1, brightness=1) -> None:
        self._dog._record(("chest", mode, color, bps, brightness))


def _default_frame():
    from PIL import Image
    # Noise so the JPEG is comfortably over hardware.photo's 1000-byte floor.
    return Image.effect_noise((160, 120), 40).convert("RGB")


def _ensure_min_size(path: Path, minimum: int = 1500) -> None:
    """Keep pixels intact but satisfy hardware.photo's size check for tiny frames."""
    data = path.read_bytes()
    if len(data) >= minimum:
        return
    if data[:2] != b"\xff\xd8":
        path.write_bytes(data + b"\0" * (minimum - len(data)))
        return
    payload = b"rex-sim" + b"\0" * max(0, minimum - len(data))
    length = len(payload) + 2
    segment = b"\xff\xfe" + length.to_bytes(2, "big") + payload
    path.write_bytes(data[:2] + segment + data[2:])


class SimCamera:
    """picamera2 stand-in. Frames come from ``dog.frames``."""

    simulated = True

    def __init__(self, dog):
        self.dog = dog
        self.started = False
        self.configuration = None

    def create_still_configuration(self, main=None):
        return {"main": main}

    def configure(self, config) -> None:
        self.configuration = config

    def start(self) -> None:
        self.started = True

    def capture_file(self, path: str) -> None:
        from PIL import Image
        frame = self.dog.frames.read()
        if not isinstance(frame, Image.Image):
            frame = _default_frame()
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.save(dest, "JPEG", quality=90)
        _ensure_min_size(dest)

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.stop()


class SimDog:
    """Pidog-shaped fake. Records every leg, head, and chest call for tests."""

    def __init__(self):
        cfg = dict(_pending)
        self.clock = cfg.get("clock") or _WallClock()
        self.step_seconds = float(cfg.get("step_seconds", 0.0))
        imu = cfg.get("imu", True)
        self.thread_list = ["sensory", "legs", "head", "tail"]
        if imu:
            self.thread_list.append("imu")
        self.distance = Sensor(200.0, self.clock)
        self.battery = Sensor(7.6, self.clock)
        self.touch = Sensor("N", self.clock)
        self.sound = Sensor(None, self.clock)
        self.imu_pitch = Sensor(0.0, self.clock)
        self.imu_roll = Sensor(0.0, self.clock)
        self.frames = Sensor(None, self.clock)
        self.ears = _Ears(self)
        self.dual_touch = _Touch(self)
        self.rgb_strip = _Strip(self)
        self.legs_thread_lock = threading.Lock()
        self.legs_action_buffer = _LegBuffer(self)
        self.head_angles = [0, 0, 0]
        self.head_commands = []
        self.action_log = []
        self.events = []
        self.legs_stop_calls = 0
        self.lie_calls = 0
        self.sensory_process = None
        self.closed = False
        self.hold_legs = None  # threading.Event; while clear, the current action does not finish
        self.action_started = threading.Event()
        self._action_active = False
        self._cancelled = False
        self._action_done_at = 0.0
        self._lock = threading.Lock()
        dogs.append(self)

    # ---- sensors (attributes Pidog exposes directly) ----------------------
    @property
    def pitch(self) -> float:
        return float(self.imu_pitch.read())

    @pitch.setter
    def pitch(self, value) -> None:
        self.imu_pitch.set(float(value))

    @property
    def roll(self) -> float:
        return float(self.imu_roll.read())

    @roll.setter
    def roll(self, value) -> None:
        self.imu_roll.set(float(value))

    def read_distance(self):
        return self.distance.read()

    def get_battery_voltage(self):
        return self.battery.read()

    def distance_by_yaw(self, yaw_to_cm: dict, default=None) -> None:
        """Distance follows the head yaw. Used by the patrol scanner."""
        def fn():
            yaw = int(self.head_angles[0])
            return yaw_to_cm.get(yaw, default)
        self.distance.set_fn(fn)

    def pause_legs(self) -> None:
        """Hold the current (and next) leg action until ``release_legs`` or a stop."""
        self.hold_legs = threading.Event()

    def release_legs(self) -> None:
        if self.hold_legs is not None:
            self.hold_legs.set()

    # ---- motions -----------------------------------------------------------
    def head_move(self, frames, pitch_comp=0, speed=30) -> None:
        self.head_commands.append({"frames": [list(f) for f in frames], "pitch_comp": pitch_comp, "speed": speed})
        if frames:
            self.head_angles = [int(v) for v in frames[-1]]
        self._record(("head", list(self.head_angles), speed))

    def wait_head_done(self) -> None:
        return None

    def wait_tail_done(self) -> None:
        return None

    def do_action(self, action, step_count=1, speed=50) -> None:
        steps = int(step_count)
        speed = int(speed)
        self.action_log.append({"action": action, "steps": steps, "speed": speed})
        self._record(("action", action, steps, speed))
        self._action_active = True
        self._cancelled = False
        self._action_done_at = self.clock.monotonic() + self.step_seconds * max(1, steps)
        self.legs_action_buffer.append(action)
        self.action_started.clear()
        self.action_started.set()

    def is_legs_done(self) -> bool:
        if not self._action_active or self._cancelled:
            return True
        if self.hold_legs is not None and not self.hold_legs.is_set():
            return False
        return self.clock.monotonic() >= self._action_done_at

    def stop_and_lie(self, speed=50) -> None:
        self.lie_calls += 1
        self._record(("lie", speed))
        self.legs_action_buffer.clear()

    def _halt_legs(self) -> None:
        self.legs_stop_calls += 1
        self._cancelled = True
        self._action_active = False
        self._record(("legs_stop",))
        if self.hold_legs is not None:
            self.hold_legs.set()

    def _record(self, event) -> None:
        with self._lock:
            self.events.append(event)

    def close(self) -> None:
        # Pidog.close() always raises SystemExit(0). Match that so shutdown's handler runs.
        if self.closed:
            return
        self.closed = True
        raise SystemExit(0)
