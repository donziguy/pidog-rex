"""Patrol walking. Off unless Rex is started with --patrol or told "rex patrol".

Every move: safety.allow_move() -> plan from a fresh distance read -> 1-2 steps at
WALK_SPEED with safety.walking set (so the 10 Hz monitor enforces stop-distance,
empty reads and head touch) -> poll until the legs finish or the monitor trips.
"""
from __future__ import annotations
import time
import config as C
from config import log
import brain


class Locomotion:
    def __init__(self, hw, safety):
        self.hw, self.safety = hw, safety
        self.enabled = False
        self.session_start = 0.0
        self.rest_until = 0.0

    # ---- mode control -----------------------------------------------------
    allow_no_imu = False

    def enable(self, why: str) -> bool:
        if not self.hw.imu_ok and not self.allow_no_imu:
            log("patrol_refused imu_unavailable (tilt safety would be blind); use --allow-no-imu to override")
            return False
        self.safety.reset(why)
        self.enabled = True
        self.session_start = time.time()
        self.rest_until = 0.0
        log(f"patrol_enabled why={why} speed={C.WALK_SPEED} max_steps={C.WALK_MAX_STEPS} cap={C.PATROL_MAX_SECONDS}s imu_ok={self.hw.imu_ok}")
        return True

    def disable(self, why: str) -> None:
        if self.enabled:
            log(f"patrol_disabled why={why}")
        self.enabled = False
        self.safety.walking.clear()

    def resting(self) -> bool:
        return time.time() < self.rest_until

    # ---- one tick -----------------------------------------------------------
    def tick(self) -> str:
        if not self.enabled:
            return "off"
        if not self.safety.allow_move():
            self.disable(f"safety_locked:{self.safety.reason}")
            return "locked"
        if self.resting():
            return "resting"
        if self.session_start and time.time() - self.session_start > C.PATROL_MAX_SECONDS:
            log(f"patrol_session_cap {C.PATROL_MAX_SECONDS}s -> rest {C.PATROL_REST_SECONDS}s")
            self._do("sit", 1, 40, walking=False)
            self.rest_until = time.time() + C.PATROL_REST_SECONDS
            self.session_start = self.rest_until  # next session starts after rest
            return "rest"
        move, steps = self.plan()
        if move == "halt":
            return "halt"
        ok = self._do(move, steps, C.WALK_SPEED, walking=True)
        return move if ok else "aborted"

    def plan(self) -> tuple[str, int]:
        self.hw.move_head(0, 0, None, speed=40)
        reads = [d for d in (self.hw.distance() for _ in range(3)) if d is not None]
        if not reads:
            log("plan halt: no distance reading")
            self.safety.trip("distance_empty_before_move")
            return "halt", 0
        c = min(reads)
        if c >= C.OBSTACLE_CLEAR_CM:
            return "forward", C.WALK_MAX_STEPS
        if c >= C.OBSTACLE_TURN_CM:
            return "forward", 1
        if c < C.SAFETY_STOP_CM + 5:
            log(f"plan too_close C={c:.0f} -> backward")
            return "backward", 1
        left, right = self._scan()
        side = "turn_left" if (left or 0) >= (right or 0) else "turn_right"
        best = max(left or 0, right or 0)
        if best < C.OBSTACLE_TURN_CM:
            log(f"plan boxed C={c:.0f} L={left} R={right} -> backward")
            return "backward", 1
        log(f"plan turn C={c:.0f} L={left} R={right} -> {side}")
        return side, 2

    def _scan(self):
        out = []
        for yaw in (35, -35):
            self.hw.move_head(yaw, 0, None, speed=50)
            time.sleep(0.25)
            vals = [d for d in (self.hw.distance() for _ in range(3)) if d is not None]
            out.append(min(vals) if vals else None)
        self.hw.move_head(0, 0, None, speed=50)
        return out[0], out[1]

    def _do(self, action: str, steps: int, speed: int, walking: bool) -> bool:
        if not self.safety.allow_move():
            return False
        dog = self.hw.dog
        log(f"walk action={action} steps={steps} speed={speed}")
        if walking and action != "backward":
            self.safety.walking.set()   # monitor enforces stop distance (not while backing away)
        try:
            dog.do_action(action, step_count=steps, speed=speed)
            deadline = time.time() + 8 + 3 * steps
            while not dog.is_legs_done():
                if not self.safety.allow_move():
                    log("walk interrupted by safety")
                    self.hw.legs_stop()
                    return False
                if time.time() > deadline:
                    log("walk timeout -> legs_stop")
                    self.hw.legs_stop()
                    return False
                time.sleep(0.02)
            # A trip clears the leg buffer, which makes is_legs_done() true. That is
            # an abort, not a finished step — don't count it.
            if not self.safety.allow_move():
                log("walk interrupted by safety")
                self.hw.legs_stop()
                return False
            brain.bump("floor_moves")
            return True
        except Exception as exc:
            log(f"walk_error={exc}")
            self.hw.legs_stop()
            return False
        finally:
            self.safety.walking.clear()
