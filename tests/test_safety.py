"""SafetyMonitor: the trips that must stop the dog."""
from __future__ import annotations

import threading

import pytest

import audio
import brain
import config as C
import sim
from safety import SafetyMonitor
from locomotion import Locomotion
from support import assert_halted, run_py


def test_documented_safety_defaults():
    assert C.SAFETY_STOP_CM == 25
    assert C.SAFETY_TILT_DEG == 30
    assert C.SAFETY_BATTERY_LOW_V == 6.6
    assert C.SAFETY_EMPTY_READS == 3
    assert C.SAFETY_EMPTY_SECONDS == pytest.approx(1.0)
    assert C.SAFETY_POLL == pytest.approx(0.1)


def _monitor(hw):
    return SafetyMonitor(hw)


def test_obstacle_under_25cm_while_walking(make_hw):
    hw = make_hw()
    mon = _monitor(hw)
    mon.walking.set()
    hw.dog.distance.set(25)  # not under the limit
    mon._tick()
    assert not mon.locked.is_set()
    hw.dog.distance.set(15)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason == "obstacle 15cm"
    chest = [event for event in hw.dog.events if event[0] == "chest"][-1]
    assert chest[1:] == ("boom", "#ff00ff", 1, 0.6)


def test_obstacle_while_idle_does_not_trip(make_hw):
    hw = make_hw()
    mon = _monitor(hw)
    hw.dog.distance.set(10)
    mon._tick()
    assert not mon.locked.is_set()
    assert hw.dog.legs_stop_calls == 0


def test_empty_distance_needs_about_a_second_while_walking(make_hw, monkeypatch):
    """A burst of None at the same instant does not trip. A full second of them does."""
    clock = sim.ManualClock(100.0)
    monkeypatch.setattr("safety.time.time", clock.time)
    hw = make_hw()
    mon = _monitor(hw)
    mon.walking.set()
    hw.dog.distance.set(None)
    for _ in range(8):
        mon._tick()
    assert not mon.locked.is_set()
    assert mon._empty == 8
    clock.advance(0.9)
    mon._tick()
    assert not mon.locked.is_set()
    clock.advance(0.1)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason.startswith("distance_empty")
    assert "empty=" in mon.reason
    # a real echo, after a reset, clears the window
    mon.reset("test")
    mon.walking.set()
    hw.dog.distance.set(None)
    mon._tick()
    clock.advance(0.4)
    hw.dog.distance.set(80)
    mon._tick()
    assert mon._empty == 0
    assert mon._empty_since is None
    assert not mon.locked.is_set()


def test_obstacle_is_instant_during_an_empty_window(make_hw, monkeypatch):
    clock = sim.ManualClock(50.0)
    monkeypatch.setattr("safety.time.time", clock.time)
    hw = make_hw()
    mon = _monitor(hw)
    mon.walking.set()
    hw.dog.distance.set(None)
    mon._tick()
    clock.advance(0.2)
    hw.dog.distance.set(15)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason == "obstacle 15cm"


def test_empty_window_follows_the_configured_seconds(make_hw, monkeypatch):
    clock = sim.ManualClock(0.0)
    monkeypatch.setattr("safety.time.time", clock.time)
    monkeypatch.setattr(C, "SAFETY_EMPTY_SECONDS", 0.4)
    hw = make_hw()
    mon = _monitor(hw)
    mon.walking.set()
    hw.dog.distance.set(None)
    mon._tick()
    clock.advance(0.39)
    mon._tick()
    assert not mon.locked.is_set()
    clock.advance(0.01)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason.startswith("distance_empty empty=")


def test_empty_seconds_env(tmp_path):
    proc = run_py(
        """
        import os, importlib
        os.environ.pop("PIDOG_REX_EMPTY_SECONDS", None)
        os.environ.pop("REX_EMPTY_SECONDS", None)
        import config
        importlib.reload(config)
        assert config.SAFETY_EMPTY_SECONDS == 1.0, config.SAFETY_EMPTY_SECONDS
        os.environ["REX_EMPTY_SECONDS"] = "2.5"
        importlib.reload(config)
        assert config.SAFETY_EMPTY_SECONDS == 2.5
        os.environ["PIDOG_REX_EMPTY_SECONDS"] = ""
        importlib.reload(config)
        assert config.SAFETY_EMPTY_SECONDS == 2.5  # blank PIDOG_ falls through to REX_
        os.environ["PIDOG_REX_EMPTY_SECONDS"] = "1.4"
        importlib.reload(config)
        assert config.SAFETY_EMPTY_SECONDS == 1.4
        os.environ["PIDOG_REX_EMPTY_SECONDS"] = "0.01"
        importlib.reload(config)
        assert config.SAFETY_EMPTY_SECONDS == 0.2
        os.environ["PIDOG_REX_EMPTY_SECONDS"] = "nope"
        importlib.reload(config)
        assert config.SAFETY_EMPTY_SECONDS == 1.0
        print("empty-seconds-ok")
        """,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "empty-seconds-ok" in proc.stdout


def test_empty_distance_while_idle_does_not_count(make_hw):
    hw = make_hw()
    mon = _monitor(hw)
    hw.dog.distance.set(None)
    for _ in range(5):
        mon._tick()
    assert not mon.locked.is_set()
    assert mon._empty == 0


def test_tilt_over_30_degrees_stops_even_when_idle(make_hw):
    hw = make_hw()
    mon = _monitor(hw)
    hw.dog.pitch = 30  # not over
    hw.dog.roll = 0
    mon._tick()
    assert not mon.locked.is_set()
    hw.dog.pitch = 31
    mon._tick()
    assert_halted(mon, hw.dog)
    assert "tilt" in mon.reason and "pitch=31" in mon.reason


def test_roll_over_30_degrees(make_hw):
    hw = make_hw()
    mon = _monitor(hw)
    hw.dog.roll = -31
    mon._tick()
    assert_halted(mon, hw.dog)
    assert "roll=-31" in mon.reason


def test_scripted_tilt_crosses_the_limit(make_hw, monkeypatch):
    clock = sim.ManualClock(10.0)
    sim.configure(clock=clock)
    hw = make_hw()
    monkeypatch.setattr("safety.time.time", clock.time)
    mon = _monitor(hw)
    hw.dog.imu_pitch.script([(0.0, 0.0), (1.0, 40.0)])
    mon._tick()
    assert not mon.locked.is_set()
    clock.advance(1.0)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert "tilt" in mon.reason


def test_low_battery_held_for_three_samples(make_hw, monkeypatch):
    """Battery is sampled once a second and trips on the third consecutive low reading.

    That is the monitor's "~3 s" window. A single sag does not trip. The dog is not
    walking here; the trip still stops the legs and lies down.
    """
    clock = sim.ManualClock(500.0)
    sim.configure(clock=clock)
    hw = make_hw()
    monkeypatch.setattr("safety.time.time", clock.time)
    mon = _monitor(hw)
    hw.dog.battery.script([(0.0, 7.4), (1.0, 6.2)])
    mon._tick()  # t=500, still 7.4 V
    assert mon._low_batt == 0 and not mon.locked.is_set()
    clock.advance(1.0)  # pack sags to 6.2
    mon._tick()
    assert mon._low_batt == 1 and not mon.locked.is_set()
    clock.advance(1.0)
    mon._tick()
    assert mon._low_batt == 2 and not mon.locked.is_set()
    clock.advance(0.2)  # inside the 1 s gate: not a new sample
    mon._tick()
    assert mon._low_batt == 2 and not mon.locked.is_set()
    clock.advance(0.8)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason.startswith("battery_low")
    assert "6.2" in mon.reason


def test_bogus_battery_readings_are_ignored(make_hw, monkeypatch):
    clock = sim.ManualClock(800.0)
    sim.configure(clock=clock)
    hw = make_hw()
    monkeypatch.setattr("safety.time.time", clock.time)
    mon = _monitor(hw)

    def samples(n):
        for _ in range(n):
            clock.advance(1.0)
            mon._tick()

    hw.dog.battery.set(6.6)  # not below the limit
    samples(3)
    assert mon._low_batt == 0

    hw.dog.battery.set(6.2)
    samples(2)
    assert mon._low_batt == 2
    hw.dog.battery.set(2.5)  # under 3 V: glitch, not a dying pack
    samples(1)
    assert mon._low_batt == 0

    hw.dog.battery.set(6.2)
    samples(2)
    hw.dog.battery.set(3.0)  # the code treats 3.0 V as bogus too (strictly greater than 3)
    samples(1)
    assert mon._low_batt == 0

    hw.dog.battery.set(6.2)
    samples(2)
    hw.dog.battery.set(None)  # sensor failed open
    samples(1)
    assert mon._low_batt == 0
    assert not mon.locked.is_set()
    assert hw.dog.lie_calls == 0


def test_head_touch_while_walking(make_hw):
    hw = make_hw()
    mon = _monitor(hw)
    hw.dog.touch.set("L")
    mon._tick()  # shelf mode: a pet, not a trip
    assert not mon.locked.is_set()
    mon.walking.set()
    hw.dog.touch.set("N")
    mon._tick()
    assert not mon.locked.is_set()
    hw.dog.touch.set("L")
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason == "head_touch L"


def test_voice_stop_locks_and_lies_down(make_hw):
    hw = make_hw()
    mon = _monitor(hw)
    heard = []
    mon.on_trip = heard.append
    listener = audio.Listener()
    listener.on_stop = lambda: mon.trip("voice_stop")
    assert audio.parse_command("Rex, stop right now") == "stop"
    assert audio.parse_command("freeze") == "stop"
    assert audio.parse_command("stop patrol") == "stop"  # stop wins
    assert audio.parse_command("good boy") == "greet"
    listener.on_stop()
    listener.on_stop()  # latched: no second lie-down
    assert heard == ["voice_stop"]
    assert_halted(mon, hw.dog)
    assert mon.reason == "voice_stop"
    assert hw.dog.lie_calls == 1


def test_trip_locks_movement_until_reset(make_hw, monkeypatch):
    hw = make_hw()
    monkeypatch.setattr("locomotion.time.sleep", lambda *_a, **_k: None)
    mon = _monitor(hw)
    loco = Locomotion(hw, mon)
    mon.on_trip = lambda reason: loco.disable(f"trip:{reason}")
    assert loco.enable("test")
    hw.dog.distance.set(100)
    mon.trip("voice_stop")
    assert_halted(mon, hw.dog)
    assert not loco.enabled
    assert loco.tick() == "off"
    # flipping the flag without reset() does not unlock the legs
    loco.enabled = True
    before = len(hw.dog.action_log)
    assert loco.tick() == "locked"
    assert len(hw.dog.action_log) == before
    assert loco.enable("rex patrol") is True  # reset is how patrol starts again
    assert mon.allow_move()
    assert loco.tick() == "forward"


def test_distance_drop_mid_walk_aborts_the_step(make_hw):
    hw = make_hw()
    hw.dog.pause_legs()
    mon = _monitor(hw)
    loco = Locomotion(hw, mon)
    mon.on_trip = lambda reason: loco.disable(f"trip:{reason}")
    assert loco.enable("test")
    before = brain.memory()["stats"]["floor_moves"]
    result = {}

    def go():
        result["value"] = loco.tick()

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    assert hw.dog.action_started.wait(2)
    assert hw.dog.action_log[-1]["action"] == "forward"
    assert mon.walking.is_set()
    hw.dog.distance.set(15)  # clear path, then an obstacle mid-step
    mon._tick()
    thread.join(2)
    assert not thread.is_alive()
    assert result["value"] == "aborted"
    assert_halted(mon, hw.dog)
    assert mon.reason == "obstacle 15cm"
    assert brain.memory()["stats"]["floor_moves"] == before
    assert not loco.enabled


def test_backward_ignores_distance_but_tilt_stops_the_legs(make_hw):
    hw = make_hw()
    hw.dog.distance.set(20)  # close enough that the planner backs up
    hw.dog.pause_legs()
    mon = _monitor(hw)
    loco = Locomotion(hw, mon)
    assert loco.enable("test")
    result = {}

    def go():
        result["value"] = loco.tick()

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    assert hw.dog.action_started.wait(2)
    assert hw.dog.action_log[-1]["action"] == "backward"
    assert not mon.walking.is_set()
    mon._tick()  # 20 cm, but distance stop stays off while backing up
    assert not mon.locked.is_set()
    hw.dog.pitch = 45
    mon._tick()
    thread.join(2)
    assert not thread.is_alive()
    assert result["value"] == "aborted"
    assert_halted(mon, hw.dog)
    assert "tilt" in mon.reason
