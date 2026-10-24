"""Patrol planner, speed cap, session cap, and the IMU gate."""
from __future__ import annotations

import pytest

import brain
import config as C
import sim
from locomotion import Locomotion
from safety import SafetyMonitor
from support import assert_halted, run_py


@pytest.fixture
def rig(make_hw, monkeypatch):
    monkeypatch.setattr("locomotion.time.sleep", lambda *_a, **_k: None)
    hw = make_hw()
    mon = SafetyMonitor(hw)
    loco = Locomotion(hw, mon)
    assert loco.enable("test")
    return hw, mon, loco


@pytest.mark.parametrize(
    "cm, move, steps",
    [
        (80, "forward", 2),
        (60, "forward", 2),
        (45, "forward", 1),
        (40, "forward", 1),
        (20, "backward", 1),
    ],
)
def test_plan_step_choice(rig, cm, move, steps):
    hw, _mon, loco = rig
    hw.dog.distance.set(cm)
    assert loco.plan() == (move, steps)
    yaws = [cmd["frames"][-1][0] for cmd in hw.dog.head_commands]
    assert 35 not in yaws and -35 not in yaws  # no scan when the choice is already clear


def test_plan_turns_toward_the_open_side(rig):
    hw, _mon, loco = rig
    hw.dog.distance_by_yaw({0: 35, 35: 90, -35: 20})
    assert loco.plan() == ("turn_left", 2)
    hw.dog.head_commands.clear()
    hw.dog.distance_by_yaw({0: 35, 35: 15, -35: 80})
    assert loco.plan() == ("turn_right", 2)


def test_plan_backs_up_when_both_sides_are_blocked(rig):
    hw, _mon, loco = rig
    hw.dog.distance_by_yaw({0: 35, 35: 10, -35: 15})
    assert loco.plan() == ("backward", 1)


def test_plan_uses_a_single_good_reading(rig):
    hw, _mon, loco = rig
    hw.dog.distance.queue([None, 90, None])
    assert loco.plan() == ("forward", 2)


def test_plan_halts_and_trips_when_distance_is_missing(rig):
    hw, mon, loco = rig
    hw.dog.distance.set(None)
    assert loco.plan() == ("halt", 0)
    assert_halted(mon, hw.dog)
    assert mon.reason == "distance_empty_before_move empty=5"


def test_sample_distance_uses_the_median_of_valid_reads(rig):
    from locomotion import sample_distance
    hw, _mon, _loco = rig
    hw.dog.distance.queue([None, None, 40, None, 100])
    median, empty = sample_distance(hw)
    assert median == 70
    assert empty == 3
    # out of the 2-400 cm window counts as empty; the last good echo repeats
    hw.dog.distance.queue([1, 500, 40])
    median, empty = sample_distance(hw)
    assert median == 40
    assert empty == 2
    hw.dog.distance.set(None)
    median, empty = sample_distance(hw)
    assert median is None
    assert empty == 5


def test_plan_levels_the_ultrasonic(rig):
    """Shelf pitch 18 with pitch_comp -15 is +3° at yaw 0 and nose-down at ±35.

    The walk pose commands the pitch that cancels pitch_comp, so the servo is level.
    """
    hw, _mon, loco = rig
    hw.dog.distance.set(100)
    assert loco.plan() == ("forward", 2)
    assert hw.head["yaw"] == 0
    assert hw.head["pitch"] == hw.level_pitch(0) == -C.HEAD_PITCH_COMP
    assert hw.head["pitch"] != C.HEAD_PITCH_DEFAULT
    assert _servo_pitch(0, 0, hw.head["pitch"]) == pytest.approx(0)


def test_scan_levels_the_head_at_each_yaw(rig):
    hw, _mon, loco = rig
    hw.dog.distance_by_yaw({0: 35, 35: 90, -35: 20})
    assert loco.plan() == ("turn_left", 2)
    yaws = []
    for cmd in hw.dog.head_commands:
        yaw, _roll, pitch = cmd["frames"][-1]
        yaws.append(yaw)
        assert pitch == hw.level_pitch(yaw)
        assert _servo_pitch(yaw, 0, pitch) == pytest.approx(0, abs=0.6)
    assert 35 in yaws and -35 in yaws


def test_walk_holds_a_level_head(rig):
    hw, _mon, loco = rig
    hw.move_head(0, 0, C.HEAD_PITCH_DEFAULT)
    hw.dog.distance.set(100)
    assert loco._do("forward", 1, 65, walking=True) is True
    assert hw.head["pitch"] == hw.level_pitch(0, 0)


def test_walk_aborts_when_shutdown_is_requested(rig, monkeypatch, capsys):
    import abort
    hw, _mon, loco = rig
    hw.dog.distance.set(100)
    before = brain.memory()["stats"]["floor_moves"]
    sleeps = {"n": 0}

    def sleep(_dt):
        sleeps["n"] += 1
        if sleeps["n"] >= 4:
            abort.event.set()

    monkeypatch.setattr("locomotion.time.sleep", sleep)
    assert loco.tick() == "aborted"
    assert brain.memory()["stats"]["floor_moves"] == before
    assert "walk interrupted by shutdown" in capsys.readouterr().out
    assert hw.dog.legs_stop_calls >= 1


def _servo_pitch(yaw, roll, pitch) -> float:
    """pidog head mix: pitch_servo = roll*|yaw|/90 + pitch*(1-|yaw|/90) + pitch_comp."""
    ratio = min(1.0, abs(float(yaw)) / 90.0)
    return float(roll) * ratio + float(pitch) * (1.0 - ratio) + C.HEAD_PITCH_COMP


def test_walk_uses_speed_65_and_respects_a_configured_80(rig, monkeypatch):
    hw, _mon, loco = rig
    assert C.WALK_SPEED == 65
    hw.dog.distance.set(100)
    assert loco.tick() == "forward"
    assert hw.dog.action_log[-1]["speed"] == 65
    assert hw.dog.action_log[-1]["steps"] == C.WALK_MAX_STEPS == 2
    monkeypatch.setattr(C, "WALK_SPEED", 80)
    assert loco.tick() == "forward"
    assert hw.dog.action_log[-1]["speed"] == 80


def test_speed_hard_cap_and_grok_interval_floor(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    proc = run_py(
        f"""
        import os, importlib
        os.environ["HOME"] = {str(home)!r}
        os.environ["PIDOG_REX_WALK_SPEED"] = "100"
        import config
        assert config.WALK_SPEED == 80, config.WALK_SPEED
        os.environ["PIDOG_REX_WALK_SPEED"] = "80"
        importlib.reload(config)
        assert config.WALK_SPEED == 80
        os.environ["PIDOG_REX_WALK_SPEED"] = "40"
        importlib.reload(config)
        assert config.WALK_SPEED == 40
        os.environ.pop("PIDOG_REX_WALK_SPEED")
        importlib.reload(config)
        assert config.WALK_SPEED == 65, config.WALK_SPEED
        os.environ["PIDOG_GROK_MIN_INTERVAL"] = "5"
        importlib.reload(config)
        assert config.GROK_MIN_INTERVAL == 60
        print("caps-ok")
        """,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "caps-ok" in proc.stdout


class _Clock:
    def __init__(self, t):
        self.t = t

    def time(self):
        return self.t


def test_session_cap_then_rest(make_hw, monkeypatch):
    assert C.PATROL_MAX_SECONDS == 120
    assert C.PATROL_REST_SECONDS == 90
    clock = _Clock(10_000.0)
    monkeypatch.setattr("locomotion.time.time", clock.time)
    hw = make_hw()
    loco = Locomotion(hw, SafetyMonitor(hw))
    hw.dog.distance.set(100)
    assert loco.enable("test")
    clock.t = 10_000.0 + 120  # strictly greater than the cap, so 120 itself still walks
    assert loco.tick() == "forward"
    clock.t = 10_000.0 + 120 + 0.5
    assert loco.tick() == "rest"
    assert hw.dog.action_log[-1] == {"action": "sit", "steps": 1, "speed": 40}
    assert loco.rest_until - clock.t == pytest.approx(90)
    forwards = sum(1 for action in hw.dog.action_log if action["action"] == "forward")
    assert loco.tick() == "resting"
    assert sum(1 for action in hw.dog.action_log if action["action"] == "forward") == forwards
    clock.t = loco.rest_until - 0.1
    assert loco.tick() == "resting"
    clock.t = loco.rest_until  # rest is over; the next session just started
    assert loco.tick() == "forward"


def test_speed_65_walk_outlasts_the_old_fixed_deadline():
    """robot_hat spends (-9.9*speed+1000) ms on every angle group.

    A walk cycle is 49 groups. At speed 65 that is ~17 s for one step and ~35 s
    for two, past the old ``8 + 3*steps`` deadline that logged walk timeout.
    """
    from locomotion import WALK_CYCLE_FRAMES, expected_leg_frames, leg_wait_budget, servo_frame_seconds
    frame = servo_frame_seconds(65)
    assert frame == pytest.approx((-9.9 * 65 + 1000) / 1000)
    one = WALK_CYCLE_FRAMES * frame
    two = expected_leg_frames("forward", 2) * frame
    assert one > 8 + 3 * 1
    assert two > 8 + 3 * 2
    assert leg_wait_budget(65, 1, WALK_CYCLE_FRAMES) > one
    assert leg_wait_budget(65, 2, expected_leg_frames("forward", 2)) > two


@pytest.mark.parametrize("cm, steps", [(100, 2), (45, 1)])
def test_walk_completes_when_frames_take_longer_than_the_old_deadline(make_hw, monkeypatch, capsys, cm, steps):
    from locomotion import expected_leg_frames, servo_frame_seconds
    clock = sim.ManualClock(5_000.0)
    sim.configure(clock=clock, realistic_legs=True)
    monkeypatch.setattr("locomotion.time.time", clock.time)
    monkeypatch.setattr("locomotion.time.sleep", lambda dt: clock.advance(dt))
    hw = make_hw()
    loco = Locomotion(hw, SafetyMonitor(hw))
    hw.dog.distance.set(cm)
    assert loco.enable("test")
    before = brain.memory()["stats"]["floor_moves"]
    started = clock.t
    assert loco.tick() == "forward"
    elapsed = clock.t - started
    frames = expected_leg_frames("forward", steps)
    assert hw.dog.action_log[-1] == {"action": "forward", "steps": steps, "speed": 65}
    assert len(hw.dog.legs_action_buffer) == 0
    assert elapsed > 8 + 3 * steps
    # plan() samples the ultrasonic (5 reads, 4 gaps) before the leg frames run.
    from locomotion import DISTANCE_SAMPLES, DISTANCE_SAMPLE_GAP
    sampling = (DISTANCE_SAMPLES - 1) * DISTANCE_SAMPLE_GAP
    assert elapsed == pytest.approx(frames * servo_frame_seconds(65) + sampling, abs=0.05)
    assert hw.dog.legs_stop_calls == 0
    assert brain.memory()["stats"]["floor_moves"] == before + 1
    assert "walk timeout" not in capsys.readouterr().out


def test_stalled_leg_buffer_still_times_out(make_hw, monkeypatch, capsys):
    from locomotion import expected_leg_frames, leg_stall_seconds, leg_wait_budget, servo_frame_seconds
    clock = sim.ManualClock(0.0)
    sim.configure(clock=clock, realistic_legs=True)
    monkeypatch.setattr("locomotion.time.time", clock.time)
    monkeypatch.setattr("locomotion.time.sleep", lambda dt: clock.advance(dt))
    hw = make_hw()
    hw.dog.pause_legs()  # servo thread never pops, so the buffer never shrinks
    loco = Locomotion(hw, SafetyMonitor(hw))
    hw.dog.distance.set(100)
    assert loco.enable("test")
    before = brain.memory()["stats"]["floor_moves"]
    assert loco.tick() == "aborted"
    frames = expected_leg_frames("forward", 2)
    full = frames * servo_frame_seconds(65)
    assert clock.t > leg_stall_seconds(65)
    assert clock.t < full
    assert clock.t < leg_wait_budget(65, 2, frames)
    assert hw.dog.legs_stop_calls >= 1
    assert brain.memory()["stats"]["floor_moves"] == before
    assert "walk timeout -> legs_stop" in capsys.readouterr().out


def test_patrol_refused_without_imu_unless_allowed(make_hw):
    sim.configure(imu=False)
    hw = make_hw()
    assert hw.imu_ok is False
    assert "imu" not in hw.dog.thread_list
    loco = Locomotion(hw, SafetyMonitor(hw))
    assert loco.enable("--patrol flag") is False
    assert loco.enabled is False
    loco.allow_no_imu = True
    assert loco.enable("--allow-no-imu") is True
    assert loco.enabled is True
