"""Patrol walking. Off unless Rex is started with --patrol or told "rex patrol".

Every move: safety.allow_move() -> plan from a fresh distance read -> 1-2 steps at
WALK_SPEED with safety.walking set (so the 10 Hz monitor enforces stop-distance,
empty reads and head touch) -> poll until the legs finish or the monitor trips.
"""
from __future__ import annotations
import time
import config as C
from config import log
import abort
import brain

# Before a step, take several echoes. Gait vibration and a nose-down head
# return None for a sample or two; one miss must not cancel the walk.
DISTANCE_SAMPLES = 5
DISTANCE_SAMPLE_GAP = 0.06

# pidog Walk: 8 sections * 6 steps, plus the closing pose that returns to the
# origin. do_action queues that many angle groups per step. robot_hat 2.3.5
# Robot.servo_move spends (-9.9 * speed + 1000) ms on each group (at least
# 10 ms) before pidog pops it. At speed 65 that is ~0.36 s * 49 = ~17 s per
# step, ~35 s for two. The old deadline of 8 + 3*steps (11 s / 14 s) always
# fired "walk timeout -> legs_stop" before the buffer could empty.
WALK_CYCLE_FRAMES = 49
_WALK_ACTIONS = {"forward", "backward", "turn_left", "turn_right", "trot"}


def servo_frame_seconds(speed: float) -> float:
    """Nominal seconds for one leg angle-group at ``speed`` (robot_hat 2.3.5)."""
    speed = max(0.0, min(100.0, float(speed)))
    return max(0.010, (-9.9 * speed + 1000.0) / 1000.0)


def expected_leg_frames(action: str, steps: int) -> int:
    per = WALK_CYCLE_FRAMES if action in _WALK_ACTIONS else 1
    return per * max(1, int(steps))


def leg_wait_budget(speed: float, steps: int, queued: int) -> float:
    """Wall-clock seconds to allow a queued leg action to finish.

    4x the nominal frame time covers I2C overrun in servo_move (each 10 ms
    step writes every leg servo). The ``8 + 3*steps`` floor keeps a one-pose
    action, and the simulator's single-token step timer, at least as long as
    the previous deadline.
    """
    frames = max(1, int(queued))
    nominal = frames * servo_frame_seconds(speed) * 4.0 + 1.0
    return max(8.0 + 3.0 * max(1, int(steps)), nominal)


def sample_distance(hw, samples: int = DISTANCE_SAMPLES, gap: float = DISTANCE_SAMPLE_GAP):
    """Median of up to ``samples`` echoes, ``gap`` seconds apart.

    Returns ``(median_or_None, empty_count)``. Invalid reads (None, outside
    2-400 cm) count as empty and are left out of the median.
    """
    valid = []
    empty = 0
    for i in range(max(1, int(samples))):
        if abort.requested():
            break
        reading = hw.distance()
        if reading is None:
            empty += 1
        else:
            valid.append(float(reading))
        if i + 1 < samples:
            time.sleep(gap)
    if not valid:
        return None, empty
    valid.sort()
    mid = len(valid) // 2
    if len(valid) % 2:
        median = valid[mid]
    else:
        median = (valid[mid - 1] + valid[mid]) / 2.0
    return median, empty


def leg_stall_seconds(speed: float) -> float:
    """How long the leg buffer may sit still before the servo thread is stuck.

    pidog pops a group only after ``servo_move`` returns, so a live walk
    shrinks the buffer about once per frame. Several times that, with a 3 s
    floor, is a stall. A one-group action (sit) uses the overall budget instead.
    """
    return max(3.0, servo_frame_seconds(speed) * 8.0)


class Locomotion:
    def __init__(self, hw, safety):
        self.hw, self.safety = hw, safety
        self.enabled = False
        self.session_start = 0.0
        self.rest_until = 0.0

    # ---- mode control -----------------------------------------------------
    allow_no_imu = False

    def enable(self, why: str) -> bool:
        stale = bool(getattr(self.hw, "imu_stale", False))
        if stale and not self.allow_no_imu:
            log("patrol_refused imu_stale (frozen IMU; tilt safety would be blind); use --allow-no-imu to override")
            return False
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
        # Level the ultrasonic before trusting a range. The shelf pitch aims
        # slightly down once the head yaws, and the walk then rocks that aim
        # into the floor.
        self.hw.level_head(0, 0, speed=40)
        c, empty = sample_distance(self.hw)
        if c is None:
            log(f"plan halt: no distance reading empty={empty}")
            self.safety.trip(f"distance_empty_before_move empty={empty}")
            return "halt", 0
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
            self.hw.level_head(yaw, 0, speed=50)
            time.sleep(0.25)
            reading, _empty = sample_distance(self.hw)
            out.append(reading)
        self.hw.level_head(0, 0, speed=50)
        return out[0], out[1]

    def _do(self, action: str, steps: int, speed: int, walking: bool) -> bool:
        if not self.safety.allow_move():
            return False
        dog = self.hw.dog
        log(f"walk action={action} steps={steps} speed={speed}")
        if walking:
            # Hold the ranger level for the step. Follow calls _do directly
            # and never goes through plan().
            self.hw.level_head(0, 0, speed=40)
        if walking and action != "backward":
            self.safety.walking.set()   # monitor enforces stop distance (not while backing away)
        try:
            before = _buffer_len(dog)
            dog.do_action(action, step_count=steps, speed=speed)
            # Real pidog and the realistic sim: the buffer is what's left to
            # run. The plain sim appends a token and never drops old ones, so
            # only the growth from this call is this step's queue.
            if getattr(dog, "realistic_legs", False):
                queued = _buffer_len(dog)
            else:
                queued = _buffer_len(dog) - before
            if queued <= 0 and not dog.is_legs_done():
                queued = expected_leg_frames(action, steps)
            # A multi-group queue that stops shrinking is a stuck servo thread.
            # One group (sit, or the sim's single token) is watched by the budget.
            stall_s = leg_stall_seconds(speed) if queued > 1 else None
            deadline = time.time() + leg_wait_budget(speed, steps, max(queued, 1))
            last_len = _buffer_len(dog)
            last_change = time.time()
            while not dog.is_legs_done():
                if abort.requested() or not self.safety.allow_move():
                    log("walk interrupted by shutdown" if abort.requested() else "walk interrupted by safety")
                    self.hw.legs_stop()
                    return False
                now = time.time()
                n = _buffer_len(dog)
                if n < last_len:
                    last_len = n
                    last_change = now
                # `stall_s` is set only for a multi-group queue, so a buffer that
                # never shrinks (thread dead before the first pop) times out too.
                if (stall_s is not None and (now - last_change) > stall_s) or now > deadline:
                    log("walk timeout -> legs_stop")
                    self.hw.legs_stop()
                    return False
                time.sleep(0.02)
            # A trip clears the leg buffer, which makes is_legs_done() true. That is
            # an abort, not a finished step — don't count it.
            if abort.requested() or not self.safety.allow_move():
                log("walk interrupted by shutdown" if abort.requested() else "walk interrupted by safety")
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


def _buffer_len(dog) -> int:
    try:
        return len(dog.legs_action_buffer)
    except Exception:
        return 0
