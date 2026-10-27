"""SafetyMonitor: independent 10 Hz watchdog that can stop the legs and lock out movement.

Triggers (any one):
  * ultrasonic under the stop distance while walking (instant, on any valid
    reading). The base is SAFETY_STOP_CM. PIDOG_REX_STOP_CM wins over the
    legacy PIDOG_OBSTACLE_STOP_CM; see config.resolve_base_stop. When the
    current move is faster than SAFETY_FAST_SPEED (the default 95 and the
    cap of 98 both are), the stop is SAFETY_FAST_STOP_CM and never closer
    than the base. Tilt does not change with speed. Patrol logs the
    effective distance for the speed it is actually using.
  * no valid ultrasonic reading for SAFETY_EMPTY_SECONDS while walking
    (about 1 s; PIDOG_REX_EMPTY_SECONDS or REX_EMPTY_SECONDS). That pauses
    the step and holds in place. It does not lock patrol. A reading before
    SAFETY_DISTANCE_HOLD_SECONDS (default 5 s) resumes the walk. Silence
    through that window trips, and the trip line includes the empty count.
    A brief gap from gait vibration does not even pause.
  * body tilt > SAFETY_TILT_DEG while walking or standing. Tilt is the angle
    between the current raw acceleration and the vector captured at startup,
    not pidog's pitch/roll (those assume gravity is on the accelerometer X
    axis). A commanded pose is different: sit holds the torso near 30 deg,
    so that posture, and the stand-up out of it, is not a tip. A further
    tip off the pose, or any tip once the body is upright, still trips.
    Past SAFETY_POSTURE_TILT_DEG the body has fallen even mid-pose.
  * N consecutive IMU read failures (any time), including a frozen chip whose
    raw accel repeats (imu_stale). A single miss keeps the last good sample;
    the streak matches SAFETY_EMPTY_READS. The trip reason stays imu_read_fail,
    so patrol and follow stop the same way they do for a dead read.
    ``--allow-no-imu`` is the supervised exception: a stale chip does not trip
    and does not refuse patrol/follow. Tilt stop is blind in that case, same
    as a missing IMU. Distance, battery, and touch still trip.
  * battery below SAFETY_BATTERY_LOW_V for ~3 s of plausible samples. 0.0 V
    and anything outside a 2S pack is dropped, not logged as the voltage,
    and does not clear a low streak that is already building.
  * head touch while walking (in shelf mode touch is just a pet)
  * external trip() e.g. the "stop" voice command
On trip: clear leg buffer, lie down, set locked=True. Movement stays locked until
an explicit reset() (voice "rex patrol" or restart).
"""
from __future__ import annotations
import math
import threading, time
import config as C
from config import log
import brain
from hardware import PITCHED_POSES, angle_between

# A 2S pack. A fresh charge is about 8.4 V. 0.0 V is a dropped ADC read.
BATTERY_MIN_V = 3.0  # exclusive, the floor the low-battery check already used
BATTERY_MAX_V = 9.2
BATTERY_SMOOTH = 3


def stop_cm_for_speed(speed) -> float:
    """Obstacle stop for the move that is running now.

    Base is SAFETY_STOP_CM. Speeds above SAFETY_FAST_SPEED use
    SAFETY_FAST_STOP_CM, and never a value closer than the base, so a low
    PIDOG_REX_FAST_STOP_CM cannot weaken the 25 cm stop. An unknown speed
    (shelf, or a walk that has not published one) keeps the base.
    """
    base = float(C.SAFETY_STOP_CM)
    try:
        fast = float(speed) > float(C.SAFETY_FAST_SPEED)
    except (TypeError, ValueError):
        fast = False
    if not fast:
        return base
    return max(base, float(C.SAFETY_FAST_STOP_CM))


def _cm(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return "?"


def describe_stop(speed=None) -> str:
    """Which obstacle setting won, and the stop distance at ``speed``.

    The base is ``PIDOG_REX_STOP_CM`` when set, else the legacy
    ``PIDOG_OBSTACLE_STOP_CM``, else 25 cm. Speeds above SAFETY_FAST_SPEED,
    including 95 and 98, use the fast stop, never closer than that base.
    """
    if speed is None:
        speed = C.WALK_SPEED
    effect = stop_cm_for_speed(speed)
    ignored = ""
    if C.SAFETY_STOP_IGNORED is not None:
        ignored = f" ignored_legacy_cm={_cm(C.SAFETY_STOP_IGNORED)}"
    return (
        f"base_cm={_cm(C.SAFETY_STOP_CM)} source={C.SAFETY_STOP_SOURCE}{ignored} "
        f"fast_cm={_cm(C.SAFETY_FAST_STOP_CM)} above_speed={_cm(C.SAFETY_FAST_SPEED)} "
        f"speed={_cm(speed)} effective_stop_cm={_cm(effect)}"
    )


def plausible_voltage(raw) -> float | None:
    """A pack voltage worth keeping, or None for a dropped / impossible read."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value <= BATTERY_MIN_V or value > BATTERY_MAX_V:
        return None
    return value


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class BatteryFilter:
    """Drop impossible pack readings and smooth the ones that remain.

    0.0 V and other values outside (``BATTERY_MIN_V``, ``BATTERY_MAX_V``] are
    ignored. They do not replace the reported voltage and they do not move
    the low-battery streak. ``value`` is the median of the last few plausible
    samples, so one noisy dip does not become the number on the dashboard.
    ``latest`` is the last plausible sample. Three of those under the low
    line still count as a dying pack, including when a 0.0 sits between them.
    """

    def __init__(self, window: int = BATTERY_SMOOTH):
        self.window = max(1, int(window))
        self.samples: list[float] = []
        self.value: float | None = None
        self.latest: float | None = None

    def push(self, raw) -> bool:
        """Fold one reading. Return True when it was a plausible sample."""
        value = plausible_voltage(raw)
        if value is None:
            return False
        self.latest = value
        self.samples.append(value)
        extra = len(self.samples) - self.window
        if extra > 0:
            del self.samples[:extra]
        self.value = _median(self.samples)
        return True


def pose_residual(hw) -> float | None:
    """Angle between the live accel and the settled pose, or None if unknown."""
    ref = getattr(hw, "pose_reference", None)
    raw = getattr(hw, "last_raw_accel", None)
    if ref is None or raw is None:
        return None
    return angle_between(raw, ref)


class SafetyMonitor(threading.Thread):
    def __init__(self, hw, allow_no_imu: bool = False):
        super().__init__(name="safety", daemon=True)
        self.hw = hw
        self.allow_no_imu = bool(allow_no_imu)
        self.walking = threading.Event()      # set by Locomotion while legs may move
        self.walk_speed = None                 # speed of that move; None keeps the base stop
        self.locked = threading.Event()
        self.reason = ""
        self._halt = threading.Event()  # not named _stop: that shadows Thread._stop and breaks join()
        self._empty = 0
        self._empty_since = None
        self.distance_hold = False  # ranger went quiet; standing still, not locked
        self._low_batt = 0
        self._last_batt_check = 0.0
        self._battery = BatteryFilter()
        self._pose_note = None
        self.last = {"distance": None, "pitch": 0.0, "roll": 0.0, "tilt": 0.0, "battery": None, "touch": "N"}
        self.ticks = 0
        self._trip_lock = threading.Lock()
        self.on_trip = None  # optional callback(reason)

    def allow_move(self) -> bool:
        return not self.locked.is_set()

    def trip(self, reason: str) -> None:
        with self._trip_lock:
            first = not self.locked.is_set()
            self.locked.set()
            self.reason = reason
        if not first:
            return
        log(f"SAFETY_TRIP reason={reason} walking={self.walking.is_set()} sensors={self.last}")
        # Always, not only while `walking` is set. Tilt, low battery, and voice stop
        # can fire during a backward step or a sit, and those moves leave the walking
        # flag clear on purpose (distance stop stays off while backing up).
        self.hw.legs_stop()
        self.hw.lie_down(speed=70)
        self.walking.clear()
        self.hw.chest("locked")
        brain.bump("safety_trips"); brain.adjust_mood(event="safety"); brain.remember("safety_trip", reason)
        if self.on_trip:
            try:
                self.on_trip(reason)
            except Exception as exc:
                log(f"on_trip_error={exc}")

    def reset(self, why: str) -> None:
        if self.locked.is_set():
            log(f"safety_reset why={why} previous_reason={self.reason}")
        self.locked.clear(); self.reason = ""; self._empty = 0; self._empty_since = None
        self.distance_hold = False
        self._pose_note = None

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        log(
            f"safety_monitor_start poll={C.SAFETY_POLL}s {describe_stop(C.WALK_SPEED)} "
            f"tilt={C.SAFETY_TILT_DEG:g} batt_low={C.SAFETY_BATTERY_LOW_V:g} "
            f"distance_hold={C.SAFETY_DISTANCE_HOLD_SECONDS:g}s"
        )
        if C.SAFETY_STOP_IGNORED is not None:
            log(
                f"obstacle_stop PIDOG_REX_STOP_CM={_cm(C.SAFETY_STOP_CM)} "
                f"wins over PIDOG_OBSTACLE_STOP_CM={_cm(C.SAFETY_STOP_IGNORED)}"
            )
        while not self._halt.is_set():
            try:
                self._tick()
            except Exception as exc:
                log(f"safety_tick_error={exc}")
            self.ticks += 1
            self._halt.wait(C.SAFETY_POLL)

    def _tick(self) -> None:
        hw, walking = self.hw, self.walking.is_set()
        d = hw.distance(); pitch, roll = hw.tilt(); t = hw.touch()
        tilt_deg = float(getattr(hw, "tilt_deg", max(abs(pitch), abs(roll))))
        self.last.update(distance=d, pitch=round(pitch, 1), roll=round(roll, 1), tilt=round(tilt_deg, 1), touch=t)
        stale = bool(getattr(hw, "imu_stale", False))
        # --allow-no-imu: a frozen chip is the same as no IMU. The angle is
        # stuck, so tilt stop is blind, and the stale streak must not lock
        # movement. A live chip (not stale) still trips on tilt and bad reads.
        if self.allow_no_imu and stale:
            note = getattr(hw, "note_stale_override", None)
            if note:
                note()
        else:
            reason = self._tilt_trip_reason(tilt_deg, pitch, roll)
            if reason:
                # The angle is the trip, not max(|pitch|, |roll|). On a tilted mount a
                # 35 deg rotation can split into two components that are each under 30.
                self.trip(reason)
            elif int(getattr(hw, "imu_fail_streak", 0) or 0) >= C.SAFETY_EMPTY_READS:
                self.trip("imu_read_fail")
        now = time.time()
        self._watch_distance(d, walking)
        if walking and t != "N":
            self.trip(f"head_touch {t}")
        if now - self._last_batt_check >= 1.0:
            self._last_batt_check = now
            accepted = self._battery.push(hw.battery())
            # Impossible reads (0.0 V, open ADC) keep the last real voltage.
            self.last["battery"] = self._battery.value
            if accepted and self._battery.latest is not None and self._battery.latest < C.SAFETY_BATTERY_LOW_V:
                self._low_batt += 1
            elif accepted:
                self._low_batt = 0
            if self._low_batt >= 3 and self._battery.latest is not None:
                self.trip(f"battery_low {self._battery.latest}V")

    def _tilt_trip_reason(self, tilt_deg: float, pitch: float, roll: float) -> str | None:
        """Why this angle is a fall, or None when the pose explains it.

        Walking and standing still use the boot baseline. A settled sit uses
        the angle off that pose, so the sit itself (about 30 deg) does not
        trip and a further tip does. Standing up stays quiet until the body
        is near level again, unless the tilt rises well past the pose or
        past SAFETY_POSTURE_TILT_DEG.
        """
        hw = self.hw
        pose = getattr(hw, "body_pose", "boot")
        motion = bool(getattr(hw, "pose_motion", False))
        ceiling = float(C.SAFETY_POSTURE_TILT_DEG)
        detail = f"pitch={pitch:.0f} roll={roll:.0f}"
        if motion or pose == "recovering":
            if pose == "recovering" and not motion:
                origin = getattr(hw, "recover_from", None)
                if origin is not None and tilt_deg > float(origin) + float(C.SAFETY_RECOVER_MARGIN):
                    return f"tilt {tilt_deg:.0f} {detail}"
                if tilt_deg <= float(C.SAFETY_RECOVER_CLEAR_DEG):
                    hw.body_pose = "upright"
                    hw.pose_reference = None
                    hw.recover_from = None
                    log(f"tilt_recover_clear tilt={tilt_deg:.1f}")
                    return None
            if tilt_deg > ceiling:
                return f"tilt {tilt_deg:.0f} {detail}"
            self._note_pose_tilt(str(pose), tilt_deg, pitch)
            return None
        if pose in PITCHED_POSES:
            residual = pose_residual(hw)
            if residual is None:
                if tilt_deg > ceiling:
                    return f"tilt {tilt_deg:.0f} {detail}"
                self._note_pose_tilt(str(pose), tilt_deg, pitch)
                return None
            if residual > float(C.SAFETY_TILT_DEG):
                return f"tilt {residual:.0f} {detail}"
            self._note_pose_tilt(str(pose), tilt_deg, pitch)
            return None
        if tilt_deg > float(C.SAFETY_TILT_DEG):
            return f"tilt {tilt_deg:.0f} {detail}"
        return None

    def _note_pose_tilt(self, pose: str, tilt_deg: float, pitch: float) -> None:
        over = tilt_deg > float(C.SAFETY_TILT_DEG)
        key = (pose, over)
        if key == self._pose_note:
            return
        self._pose_note = key
        if over:
            log(
                f"tilt_pose_held pose={pose} tilt={tilt_deg:.1f} "
                f"pitch={pitch:.1f} limit={_cm(C.SAFETY_TILT_DEG)}"
            )

    def begin_distance_hold(self, empty: int | None = None, where: str = "walk") -> None:
        """Stop the legs and wait for the ranger. Does not lock movement."""
        now = time.time()
        if self._empty_since is None:
            self._empty_since = now
        if empty:
            self._empty = max(self._empty, int(empty))
        if self.distance_hold:
            return
        self.distance_hold = True
        self.walking.clear()
        self.walk_speed = None
        log(
            f"distance_hold where={where} empty={self._empty} "
            f"pause={_cm(C.SAFETY_EMPTY_SECONDS)}s limit={_cm(C.SAFETY_DISTANCE_HOLD_SECONDS)}s"
        )
        self.hw.legs_stop()

    def end_distance_hold(self, reading: float) -> None:
        held = self.distance_hold
        self.distance_hold = False
        self._empty = 0
        self._empty_since = None
        if held:
            log(f"distance_hold_clear cm={float(reading):g}")

    def distance_hold_expired(self, now: float | None = None) -> bool:
        if self._empty_since is None:
            return False
        if now is None:
            now = time.time()
        return (now - self._empty_since) >= float(C.SAFETY_DISTANCE_HOLD_SECONDS)

    def _watch_distance(self, distance, walking: bool) -> None:
        now = time.time()
        if distance is None:
            # Shelf mode with a quiet ranger is not a dropout. Only a step
            # that is underway, or a hold we already started, keeps the clock.
            if not walking and not self.distance_hold:
                return
            self._empty += 1
            if self._empty_since is None:
                self._empty_since = now
            elapsed = now - self._empty_since
            if elapsed >= float(C.SAFETY_DISTANCE_HOLD_SECONDS):
                self.trip(f"distance_empty empty={self._empty}")
                return
            if walking and elapsed >= float(C.SAFETY_EMPTY_SECONDS):
                self.begin_distance_hold(where="walking")
            return
        if self.distance_hold:
            self.end_distance_hold(distance)
        else:
            self._empty = 0
            self._empty_since = None
        if walking:
            stop = stop_cm_for_speed(self.walk_speed)
            if distance < stop:
                self.trip(f"obstacle {distance:.0f}cm")
