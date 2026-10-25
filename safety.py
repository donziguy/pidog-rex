"""SafetyMonitor: independent 10 Hz watchdog that can stop the legs and lock out movement.

Triggers (any one):
  * ultrasonic under the stop distance while walking (instant, on any valid
    reading). The base is SAFETY_STOP_CM. When the current move is faster than
    SAFETY_FAST_SPEED, the stop is SAFETY_FAST_STOP_CM and never closer than
    the base. Tilt does not change with speed.
  * no valid ultrasonic reading for SAFETY_EMPTY_SECONDS while walking
    (about 1 s; PIDOG_REX_EMPTY_SECONDS or REX_EMPTY_SECONDS). A brief gap
    from gait vibration does not trip. The trip line includes the empty count.
  * body tilt > SAFETY_TILT_DEG (any time). Tilt is the angle between the
    current raw acceleration and the vector captured at startup, not pidog's
    pitch/roll (those assume gravity is on the accelerometer X axis).
  * N consecutive IMU read failures (any time), including a frozen chip whose
    raw accel repeats (imu_stale). A single miss keeps the last good sample;
    the streak matches SAFETY_EMPTY_READS. The trip reason stays imu_read_fail,
    so patrol and follow stop the same way they do for a dead read.
    ``--allow-no-imu`` is the supervised exception: a stale chip does not trip
    and does not refuse patrol/follow. Tilt stop is blind in that case, same
    as a missing IMU. Distance, battery, and touch still trip.
  * battery below SAFETY_BATTERY_LOW_V for ~3 s (any time)
  * head touch while walking (in shelf mode touch is just a pet)
  * external trip() e.g. the "stop" voice command
On trip: clear leg buffer, lie down, set locked=True. Movement stays locked until
an explicit reset() (voice "rex patrol" or restart).
"""
from __future__ import annotations
import threading, time
import config as C
from config import log
import brain


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
        self._low_batt = 0
        self._last_batt_check = 0.0
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

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        log(
            f"safety_monitor_start poll={C.SAFETY_POLL}s stop_cm={C.SAFETY_STOP_CM} "
            f"fast_stop_cm={C.SAFETY_FAST_STOP_CM} above_speed={C.SAFETY_FAST_SPEED} "
            f"tilt={C.SAFETY_TILT_DEG} batt_low={C.SAFETY_BATTERY_LOW_V}"
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
        elif tilt_deg > C.SAFETY_TILT_DEG:
            # The angle is the trip, not max(|pitch|, |roll|). On a tilted mount a
            # 35 deg rotation can split into two components that are each under 30.
            self.trip(f"tilt {tilt_deg:.0f} pitch={pitch:.0f} roll={roll:.0f}")
        elif int(getattr(hw, "imu_fail_streak", 0) or 0) >= C.SAFETY_EMPTY_READS:
            self.trip("imu_read_fail")
        now = time.time()
        if walking:
            if d is None:
                self._empty += 1
                if self._empty_since is None:
                    self._empty_since = now
                elif now - self._empty_since >= C.SAFETY_EMPTY_SECONDS:
                    self.trip(f"distance_empty empty={self._empty}")
            else:
                self._empty = 0
                self._empty_since = None
                stop = stop_cm_for_speed(self.walk_speed)
                if d < stop:
                    self.trip(f"obstacle {d:.0f}cm")
            if t != "N":
                self.trip(f"head_touch {t}")
        if now - self._last_batt_check >= 1.0:
            self._last_batt_check = now
            v = hw.battery(); self.last["battery"] = v
            self._low_batt = self._low_batt + 1 if (v is not None and 3.0 < v < C.SAFETY_BATTERY_LOW_V) else 0
            if self._low_batt >= 3:
                self.trip(f"battery_low {v}V")
