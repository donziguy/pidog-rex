"""SafetyMonitor: independent 10 Hz watchdog that can stop the legs and lock out movement.

Triggers (any one):
  * ultrasonic < SAFETY_STOP_CM while walking
  * N consecutive empty ultrasonic reads while walking
  * body tilt > SAFETY_TILT_DEG (any time)
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


class SafetyMonitor(threading.Thread):
    def __init__(self, hw):
        super().__init__(name="safety", daemon=True)
        self.hw = hw
        self.walking = threading.Event()      # set by Locomotion while legs may move
        self.locked = threading.Event()
        self.reason = ""
        self._halt = threading.Event()  # not named _stop: that shadows Thread._stop and breaks join()
        self._empty = 0
        self._low_batt = 0
        self._last_batt_check = 0.0
        self.last = {"distance": None, "pitch": 0.0, "roll": 0.0, "battery": None, "touch": "N"}
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
        self.locked.clear(); self.reason = ""; self._empty = 0

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        log(f"safety_monitor_start poll={C.SAFETY_POLL}s stop_cm={C.SAFETY_STOP_CM} tilt={C.SAFETY_TILT_DEG} batt_low={C.SAFETY_BATTERY_LOW_V}")
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
        self.last.update(distance=d, pitch=round(pitch, 1), roll=round(roll, 1), touch=t)
        if abs(pitch) > C.SAFETY_TILT_DEG or abs(roll) > C.SAFETY_TILT_DEG:
            self.trip(f"tilt pitch={pitch:.0f} roll={roll:.0f}")
        now = time.time()
        if now - self._last_batt_check >= 1.0:
            self._last_batt_check = now
            v = hw.battery(); self.last["battery"] = v
            self._low_batt = self._low_batt + 1 if (v is not None and 3.0 < v < C.SAFETY_BATTERY_LOW_V) else 0
            if self._low_batt >= 3:
                self.trip(f"battery_low {v}V")
        if walking:
            if d is None:
                self._empty += 1
                if self._empty >= C.SAFETY_EMPTY_READS:
                    self.trip("distance_empty")
            else:
                self._empty = 0
                if d < C.SAFETY_STOP_CM:
                    self.trip(f"obstacle {d:.0f}cm")
            if t != "N":
                self.trip(f"head_touch {t}")
