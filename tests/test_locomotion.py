"""Patrol planner, speed cap, session cap, and the IMU gate."""
from __future__ import annotations

import pytest

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
    assert mon.reason == "distance_empty_before_move"


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
