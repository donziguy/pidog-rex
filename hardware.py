"""Thin, exception-safe wrapper over SunFounder Pidog + picamera2.

Note: Pidog() itself forks one 'sensory_process' (multiprocessing) for the
ultrasonic sensor; that child shows up as a second python process. It is
expected and is terminated by Pidog.close().
"""
from __future__ import annotations
import math, threading, time
from pathlib import Path
import config as C
from config import log

PHOTO_PATH = Path("/tmp/rex_scan.jpg")
CHEST_COLORS = {"listening": (0, 0, 255), "analyzing": (255, 255, 0), "talking": (0, 255, 0),
                "alert": (255, 0, 0), "suspicious": (255, 128, 0), "idle": (0, 255, 255), "locked": (255, 0, 255)}


class Hardware:
    def __init__(self):
        from pidog import Pidog
        log("pidog_init (legs go to the 'lie' pose they boot in)")
        self.dog = Pidog()
        time.sleep(0.8)
        self.head = {"yaw": 0, "roll": 0, "pitch": C.HEAD_PITCH_DEFAULT}
        self._cam = None
        self._cam_lock = threading.Lock()
        self.imu_ok = "imu" in getattr(self.dog, "thread_list", [])
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
        try:
            return float(getattr(self.dog, "pitch", 0.0) or 0.0), float(getattr(self.dog, "roll", 0.0) or 0.0)
        except Exception:
            return 0.0, 0.0

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
                    from picamera2 import Picamera2
                    self._cam = Picamera2()
                    self._cam.configure(self._cam.create_still_configuration(main={"size": (640, 480), "format": "RGB888"}))
                    self._cam.start()
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
