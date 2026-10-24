"""Thin, exception-safe wrapper over SunFounder Pidog + picamera2.

Note: Pidog() itself forks one 'sensory_process' (multiprocessing) for the
ultrasonic sensor; that child shows up as a second python process. It is
expected and is terminated by Pidog.close().
"""
from __future__ import annotations
import math, os, threading, time
from pathlib import Path
import config as C
from config import log

PHOTO_PATH = Path("/tmp/rex_scan.jpg")
CHEST_COLORS = {"listening": (0, 0, 255), "analyzing": (255, 255, 0), "talking": (0, 255, 0),
                "alert": (255, 0, 0), "suspicious": (255, 128, 0), "idle": (0, 255, 255), "locked": (255, 0, 255)}


def _sim_enabled() -> bool:
    return os.environ.get("REX_SIM", "").strip().lower() in {"1", "true", "yes", "on"}


# pidog's _imu_thread forces the startup pose to ax=-16384, ay=0, az=0. That is only
# right when gravity already lies on the accelerometer X axis. This dog's SH3001 at
# rest is mostly Y/Z, so the offset leaves a large constant on X and compresses
# pitch/roll. Rex measures tilt itself, from raw counts, against a baseline taken
# while the dog is still.
_ACCEL_MIN_COUNTS = 50.0
_I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v) -> float:
    return math.sqrt(_dot(v, v))


def _unit(v):
    n = _norm(v)
    if n <= 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _skew(v):
    x, y, z = v
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def _matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _madd(a, b):
    return tuple(tuple(a[i][j] + b[i][j] for j in range(3)) for i in range(3))


def _mscale(a, s):
    return tuple(tuple(x * s for x in row) for row in a)


def _mv(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def _transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def rotation_align(src, dst=(-1.0, 0.0, 0.0)):
    """Rotation that maps ``src`` onto ``dst``. Identity when they already agree."""
    a, b = _unit(src), _unit(dst)
    if _norm(a) == 0 or _norm(b) == 0:
        return _I
    v = _cross(a, b)
    c = max(-1.0, min(1.0, _dot(a, b)))
    if c > 1.0 - 1e-12:
        return _I
    if c < -1.0 + 1e-12:
        axis = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = _unit(_cross(a, axis))
        k = _skew(axis)
        return _madd(_I, _mscale(_matmul(k, k), 2.0))
    k = _skew(v)
    return _madd(_madd(_I, k), _mscale(_matmul(k, k), (1.0 - c) / _dot(v, v)))


def angle_between(a, b) -> float:
    """Angle in degrees between two accel vectors. 0 when either is unusable."""
    if _norm(a) < _ACCEL_MIN_COUNTS or _norm(b) < _ACCEL_MIN_COUNTS:
        return 0.0
    c = max(-1.0, min(1.0, _dot(_unit(a), _unit(b))))
    return math.degrees(math.acos(c))


def library_pitch_roll(acc) -> tuple[float, float]:
    """pidog's pitch/roll, valid once gravity has been rotated onto -X."""
    ax, ay, az = acc
    ay, az = -ay, -az
    horiz_p = math.sqrt(ax * ax + az * az)
    horiz_r = math.sqrt(ax * ax + ay * ay)
    if horiz_p < 1e-9 or horiz_r < 1e-9:
        return 0.0, 0.0
    pitch = math.degrees(math.atan(ay / horiz_p))
    roll = math.degrees(math.atan(az / horiz_r))
    return pitch, roll


def library_vector(pitch_deg: float, roll_deg: float) -> tuple[float, float, float]:
    """Accel direction whose :func:`library_pitch_roll` is ``(pitch_deg, roll_deg)``.

    Rest (0, 0) is ``(-1, 0, 0)``, matching the pose pidog's calibration assumes.
    """
    tp = math.tan(math.radians(pitch_deg))
    tr = math.tan(math.radians(roll_deg))
    tp2, tr2 = tp * tp, tr * tr
    denom = 1.0 - tp2 * tr2
    if abs(denom) < 1e-8:
        denom = 1e-8 if denom >= 0 else -1e-8
    b = tp2 * (1.0 + tr2) / denom
    c = tr2 * (1.0 + b)
    ax = -1.0
    ay_f = 0.0 if abs(pitch_deg) < 1e-12 or b <= 0 else math.copysign(math.sqrt(b), pitch_deg)
    az_f = 0.0 if abs(roll_deg) < 1e-12 or c <= 0 else math.copysign(math.sqrt(c), roll_deg)
    return (ax, -ay_f, -az_f)


def raw_accel_for_attitude(mount, pitch_deg: float, roll_deg: float) -> tuple[float, float, float]:
    """Raw counts for a pitch/roll attitude, given the at-rest raw vector ``mount``."""
    mount = (float(mount[0]), float(mount[1]), float(mount[2]))
    direction = library_vector(pitch_deg, roll_deg)
    align = rotation_align(mount, (-1.0, 0.0, 0.0))
    raw_hat = _mv(_transpose(align), _unit(direction))
    scale = _norm(mount) or 16384.0
    return tuple(x * scale for x in raw_hat)


def _accel_tuple(acc):
    try:
        vec = (float(acc[0]), float(acc[1]), float(acc[2]))
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(x) for x in vec):
        return None
    return vec


def read_raw_accel(dog):
    """Raw SH3001 counts, or None.

    ``accData`` is already ``raw + imu_acc_offset``. Prefer
    ``imu._sh3001_getimudata()`` (pre-offset). Otherwise undo the offset.
    """
    imu = getattr(dog, "imu", None)
    getter = getattr(imu, "_sh3001_getimudata", None) if imu is not None else None
    if getter is None:
        getter = getattr(dog, "_sh3001_getimudata", None)
    if callable(getter):
        try:
            data = getter()
        except Exception:
            return None
        if data is False or data is None:
            return None
        acc = data[0] if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)) else data
        return _accel_tuple(acc)
    acc = getattr(dog, "accData", None)
    off = getattr(dog, "imu_acc_offset", None)
    if acc is None or off is None:
        return None
    try:
        return _accel_tuple(tuple(float(acc[i]) - float(off[i]) for i in range(3)))
    except (TypeError, ValueError, IndexError):
        return None


class TiltTracker:
    """Tilt of the current gravity vector relative to a startup baseline.

    ``tilt_deg`` is the angle between the two vectors. ``pitch`` and ``roll`` are
    the signed components after the baseline is rotated onto -X, for logs.
    A failed read keeps the last good sample and increments ``fail_streak``.
    """

    def __init__(self):
        self.baseline = None
        self.align = _I
        self.sample_count = 0
        self.tilt_deg = 0.0
        self.fail_streak = 0
        self.last = (0.0, 0.0)

    def capture(self, read, samples: int = 10, gap: float = 0.0) -> bool:
        vecs = []
        for _ in range(samples):
            vec = read()
            if vec is not None and _norm(vec) >= _ACCEL_MIN_COUNTS:
                vecs.append(vec)
            if gap:
                time.sleep(gap)
        if not vecs:
            return False
        self.sample_count = len(vecs)
        self.baseline = tuple(sum(v[i] for v in vecs) / len(vecs) for i in range(3))
        self.align = rotation_align(self.baseline, (-1.0, 0.0, 0.0))
        self.fail_streak = 0
        self.tilt_deg = 0.0
        self.last = (0.0, 0.0)
        return True

    def update(self, raw) -> tuple[float, float]:
        if raw is None or self.baseline is None or _norm(raw) < _ACCEL_MIN_COUNTS:
            self.fail_streak += 1
            return self.last
        self.fail_streak = 0
        self.tilt_deg = angle_between(raw, self.baseline)
        pitch, roll = library_pitch_roll(_mv(self.align, raw))
        self.last = (pitch, roll)
        return self.last


class Hardware:
    def __init__(self):
        # Sim is imported only in this branch, so a Pi that never sets REX_SIM
        # does not import it (and does not import anything new).
        self.simulated = _sim_enabled()
        if self.simulated:
            from sim import SimDog
            log("sim_init (no physical pidog)")
            self.dog = SimDog()
        else:
            from pidog import Pidog
            log("pidog_init (legs go to the 'lie' pose they boot in)")
            self.dog = Pidog()
            time.sleep(0.8)
        self.head = {"yaw": 0, "roll": 0, "pitch": C.HEAD_PITCH_DEFAULT}
        self._cam = None
        self._cam_lock = threading.Lock()
        self.imu_ok = "imu" in getattr(self.dog, "thread_list", [])
        self._tilt = TiltTracker()
        self.tilt_deg = 0.0
        self.imu_fail_streak = 0
        self.tilt_baseline = None
        if self.imu_ok:
            # The dog is still in the boot lie pose. Average a handful of raw
            # samples so one noisy read does not become the "level" reference.
            gap = 0.0 if self.simulated else 0.02
            if self._tilt.capture(lambda: read_raw_accel(self.dog), samples=10, gap=gap):
                self.tilt_baseline = self._tilt.baseline
                b = self.tilt_baseline
                log(f"tilt_baseline samples={self._tilt.sample_count} ax={b[0]:.0f} ay={b[1]:.0f} az={b[2]:.0f}")
            else:
                log("tilt_baseline_failed (IMU reads unavailable; repeated failures trip safety)")
        log(f"pidog_init_ok imu_ok={self.imu_ok}")

    # ---- sensors -------------------------------------------------------
    def distance(self) -> float | None:
        """cm, or None when the ultrasonic gives no valid echo."""
        try:
            d = float(self.dog.read_distance())
            return d if 2.0 < d < 400.0 else None
        except Exception:
            return None

    def tilt(self) -> tuple[float, float]:
        """``(pitch, roll)`` in degrees relative to the startup gravity baseline.

        The magnitude is ``tilt_deg`` (angle between the current raw accel and
        the baseline), which is what the safety monitor compares to
        ``SAFETY_TILT_DEG``. Pitch and roll are the signed components, for logs.
        A failed IMU read returns the last good sample and bumps
        ``imu_fail_streak``. With no IMU at all, both stay 0 and the streak
        does not climb (patrol is already refused).
        """
        if not self.imu_ok:
            self.tilt_deg = 0.0
            self.imu_fail_streak = 0
            return 0.0, 0.0
        try:
            pitch, roll = self._tilt.update(read_raw_accel(self.dog))
        except Exception:
            pitch, roll = self._tilt.update(None)
        self.tilt_deg = self._tilt.tilt_deg
        self.imu_fail_streak = self._tilt.fail_streak
        return pitch, roll

    def battery(self) -> float | None:
        try:
            return float(self.dog.get_battery_voltage())
        except Exception:
            return None

    def touch(self) -> str:
        try:
            return self.dog.dual_touch.read()
        except Exception:
            return "N"

    def sound_direction(self) -> int | None:
        try:
            ears = self.dog.ears
            if ears.isdetected():
                d = int(ears.read())
                return d if 0 <= d <= 359 else None
        except Exception as exc:
            log(f"sound_read_error={exc}")
        return None

    # ---- head / tail / chest --------------------------------------------
    def move_head(self, yaw=0, roll=0, pitch=None, speed=30, wait=True) -> None:
        pitch = C.HEAD_PITCH_DEFAULT if pitch is None else pitch
        yaw, roll, pitch = (int(max(-60, min(60, v))) for v in (yaw, roll, pitch))
        self.head = {"yaw": yaw, "roll": roll, "pitch": pitch}
        try:
            self.dog.head_move([[yaw, roll, pitch]], pitch_comp=C.HEAD_PITCH_COMP, speed=speed)
            if wait:
                self.dog.wait_head_done()
        except Exception as exc:
            log(f"head_move_error={exc}")

    def head_pan(self, pan: int, speed=45) -> None:
        pan = int(max(-C.SOUND_YAW_LIMIT, min(C.SOUND_YAW_LIMIT, pan)))
        axis = C.SOUND_PAN_AXIS
        if axis == "yaw":
            self.move_head(pan, 0, None, speed)
        elif axis == "pitch":
            self.move_head(0, 0, pan, speed)
        else:
            self.move_head(0, pan, None, speed)

    def sound_to_pan(self, direction: int) -> int:
        adj = (direction - C.SOUND_FRONT_OFFSET) % 360
        centered = ((adj + 180) % 360) - 180
        pan = int(max(-C.SOUND_YAW_LIMIT, min(C.SOUND_YAW_LIMIT, centered)))
        return -pan if C.SOUND_YAW_INVERT else pan

    def head_gesture(self, frames: list, speed=28) -> None:
        y, r = self.head["yaw"], self.head["roll"]
        try:
            self.dog.head_move([[y + dy, r, p] for dy, p in frames], pitch_comp=C.HEAD_PITCH_COMP, speed=speed)
            self.dog.wait_head_done()
        except Exception as exc:
            log(f"head_gesture_error={exc}")

    def wag(self, steps=3, speed=45) -> None:
        try:
            self.dog.do_action("wag_tail", step_count=steps, speed=speed)
            self.dog.wait_tail_done()
        except Exception as exc:
            log(f"wag_error={exc}")

    def chest(self, state: str) -> None:
        try:
            rgb = getattr(self.dog, "rgb_strip", None)
            color = CHEST_COLORS.get(state)
            if rgb is not None and color is not None:
                rgb.set_mode("boom" if state == "locked" else "breath", "#%02x%02x%02x" % color,
                             bps=1, brightness=0.6)
        except Exception:
            pass

    # ---- legs (only Locomotion and SafetyMonitor call these) -------------
    def legs_stop(self) -> None:
        try:
            with self.dog.legs_thread_lock:
                self.dog.legs_action_buffer.clear()
        except Exception as exc:
            log(f"legs_stop_error={exc}")

    def lie_down(self, speed=60) -> None:
        try:
            self.dog.stop_and_lie(speed=speed)
        except Exception as exc:
            log(f"lie_down_error={exc}")

    # ---- camera -----------------------------------------------------------
    def photo(self) -> Path | None:
        with self._cam_lock:
            try:
                if self._cam is None:
                    if self.simulated:
                        from sim import SimCamera
                        self._cam = SimCamera(self.dog)
                    else:
                        from picamera2 import Picamera2
                        self._cam = Picamera2()
                    self._cam.configure(self._cam.create_still_configuration(main={"size": (640, 480), "format": "RGB888"}))
                    self._cam.start()
                    if not self.simulated:
                        time.sleep(1.0)
                self._cam.capture_file(str(PHOTO_PATH))
                return PHOTO_PATH if PHOTO_PATH.exists() and PHOTO_PATH.stat().st_size > 1000 else None
            except Exception as exc:
                log(f"photo_error={exc}")
                self.close_camera()
                return None

    def close_camera(self) -> None:
        try:
            if self._cam is not None:
                self._cam.stop(); self._cam.close()
        except Exception:
            pass
        self._cam = None

    def shutdown(self) -> None:
        """Lie down, then close everything. Must run on the main thread (Pidog.close uses signals)."""
        log("hardware_shutdown lie_down + close")
        self.close_camera()
        self.legs_stop()
        self.lie_down(speed=50)
        done = threading.Event()

        def _close():
            try:
                self.dog.close()  # note: Pidog.close() always raises SystemExit(0); swallow it so our rc survives
            except SystemExit:
                pass
            except Exception as exc:
                log(f"dog_close_error={exc}")
            done.set()
        if threading.current_thread() is threading.main_thread():
            _close()
        else:
            t = threading.Thread(target=_close, daemon=True); t.start(); done.wait(6)
        sp = getattr(self.dog, "sensory_process", None)
        try:
            if sp is not None and sp.is_alive():
                sp.kill(); sp.join(2)
        except Exception:
            pass
        log("hardware_shutdown_done")
