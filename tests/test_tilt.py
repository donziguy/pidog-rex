"""Tilt from raw acceleration, not pidog's X-axis calibration."""
from __future__ import annotations

import math

import pytest

import config as C
import sim
from hardware import IMU_STALE_READS, TiltTracker, angle_between, read_raw_accel, raw_accel_for_attitude
from safety import SafetyMonitor
from support import assert_halted


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _rotate(v, axis, deg):
    k = _unit(axis)
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    cr = _cross(k, v)
    return tuple(v[i] * c + cr[i] * s + k[i] * _dot(k, v) * (1 - c) for i in range(3))


def _axes(mount):
    g = _unit(mount)
    a1 = _unit(_cross(g, (1.0, 0.0, 0.0) if abs(g[0]) < 0.9 else (0.0, 1.0, 0.0)))
    a2 = _unit(_cross(g, a1))
    out = []
    for i in range(0, 180, 3):
        th = math.radians(i)
        out.append(tuple(math.cos(th) * a1[j] + math.sin(th) * a2[j] for j in range(3)))
    return out


def _library_pitch_roll(raw, rest):
    """What pidog reports after forcing the boot pose to ax=-16384, ay=0, az=0."""
    offset = (-16384.0 - rest[0], -rest[1], -rest[2])
    ax, ay, az = (raw[i] + offset[i] for i in range(3))
    ay, az = -ay, -az
    pitch = math.degrees(math.atan(ay / math.sqrt(ax * ax + az * az)))
    roll = math.degrees(math.atan(az / math.sqrt(ax * ax + ay * ay)))
    return pitch, roll


# Real dog at rest, plus the pure axes the library's calibration gets wrong.
_MOUNTS = {
    "x": (-16384.0, 0.0, 0.0),
    "y": (0.0, -16384.0, 0.0),
    "z": (0.0, 0.0, -16384.0),
    "yz": (247.0, -10461.0, -18627.0),
}


def _rig(make_hw, mount):
    sim.configure(imu_mount=mount)
    hw = make_hw()
    return hw, SafetyMonitor(hw)


@pytest.mark.parametrize("name", list(_MOUNTS))
def test_baseline_on_each_axis_and_hand_tilt(make_hw, name):
    mount = _MOUNTS[name]
    hw, mon = _rig(make_hw, mount)
    assert hw.tilt_baseline == pytest.approx(mount, abs=1.0)
    mon._tick()
    assert hw.tilt_deg == pytest.approx(0.0, abs=0.05)
    assert not mon.locked.is_set()

    axis = _axes(mount)[0]
    hw.dog.imu_raw.set(_rotate(mount, axis, 10))
    mon._tick()
    assert hw.tilt_deg == pytest.approx(10.0, abs=0.2)
    assert not mon.locked.is_set()

    # A 35 deg rotation whose pidog pitch/roll both stay under 30. The old
    # check (abs(pitch) or abs(roll) > 30, using the library numbers) does not
    # fire. The angle against the baseline does.
    chosen = None
    for axis in _axes(mount):
        raw = _rotate(mount, axis, 35)
        pitch, roll = _library_pitch_roll(raw, mount)
        if max(abs(pitch), abs(roll)) < C.SAFETY_TILT_DEG:
            chosen = raw
            break
    assert chosen is not None
    hw.dog.imu_raw.set(chosen)
    mon._tick()
    assert hw.tilt_deg == pytest.approx(35.0, abs=0.2)
    assert hw.tilt_deg > C.SAFETY_TILT_DEG
    assert_halted(mon, hw.dog)
    assert mon.reason.startswith("tilt ")
    assert "35" in mon.reason


def test_attitude_pitch_round_trips_on_the_yz_mount(make_hw):
    hw, mon = _rig(make_hw, _MOUNTS["yz"])
    hw.dog.pitch = 10
    mon._tick()
    assert hw.tilt() == pytest.approx((10.0, 0.0), abs=0.05)
    assert hw.tilt_deg == pytest.approx(10.0, abs=0.05)
    assert not mon.locked.is_set()
    hw.dog.pitch = 35
    mon._tick()
    assert_halted(mon, hw.dog)
    assert "pitch=35" in mon.reason


def test_split_tilt_under_30_per_axis_still_trips(make_hw):
    """35 deg can land as ~24 pitch and ~24 roll. Either component alone is safe."""
    mount = _MOUNTS["yz"]
    hw, mon = _rig(make_hw, mount)
    raw = None
    for axis in _axes(mount):
        candidate = _rotate(mount, axis, 35)
        hw.dog.imu_raw.set(candidate)
        pitch, roll = hw.tilt()
        if max(abs(pitch), abs(roll)) < C.SAFETY_TILT_DEG and hw.tilt_deg > C.SAFETY_TILT_DEG:
            raw = candidate
            break
    assert raw is not None
    # tilt() above did not go through the monitor. One tick must trip.
    hw.dog.imu_raw.set(raw)
    mon._tick()
    assert max(abs(mon.last["pitch"]), abs(mon.last["roll"])) < C.SAFETY_TILT_DEG
    assert mon.last["tilt"] == pytest.approx(35.0, abs=0.5)
    assert_halted(mon, hw.dog)


def test_imu_read_failure_returns_last_good_then_trips(make_hw):
    hw = make_hw()
    mon = SafetyMonitor(hw)
    hw.dog.pitch = 12
    mon._tick()
    assert hw.tilt_deg == pytest.approx(12.0, abs=0.05)
    assert not mon.locked.is_set()

    hw.dog.imu_read_ok = False
    for _ in range(C.SAFETY_EMPTY_READS - 1):
        mon._tick()
    assert not mon.locked.is_set()
    assert hw.imu_fail_streak == C.SAFETY_EMPTY_READS - 1
    assert mon.last["pitch"] == pytest.approx(12.0, abs=0.2)
    assert mon.last["tilt"] == pytest.approx(12.0, abs=0.2)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason == "imu_read_fail"


def test_imu_failure_streak_resets_on_a_good_read(make_hw):
    hw = make_hw()
    mon = SafetyMonitor(hw)
    hw.dog.imu_read_ok = False
    mon._tick()
    mon._tick()
    assert hw.imu_fail_streak == 2
    hw.dog.imu_read_ok = True
    mon._tick()
    assert hw.imu_fail_streak == 0
    assert not mon.locked.is_set()
    hw.dog.imu_read_ok = False
    for _ in range(C.SAFETY_EMPTY_READS - 1):
        mon._tick()
    assert not mon.locked.is_set()
    mon._tick()
    assert mon.reason == "imu_read_fail"


def test_missing_imu_does_not_count_as_read_failures(make_hw):
    sim.configure(imu=False)
    hw = make_hw()
    mon = SafetyMonitor(hw)
    for _ in range(5):
        mon._tick()
    assert hw.tilt() == (0.0, 0.0)
    assert hw.imu_fail_streak == 0
    assert not mon.locked.is_set()


def test_raw_read_prefers_the_chip_and_can_undo_the_offset():
    raw = (247.0, -10461.0, -18627.0)
    offset = (-16631.0, 10461.0, 18627.0)

    class _Chip:
        def __init__(self):
            self.calls = 0

        def _sh3001_getimudata(self):
            self.calls += 1
            return (list(raw), [0, 0, 0])

    class _Both:
        def __init__(self):
            self.imu = _Chip()
            # Post-offset garbage. Must be ignored while the raw read works.
            self.accData = [0.0, 0.0, 0.0]
            self.imu_acc_offset = [0.0, 0.0, 0.0]

    dog = _Both()
    assert read_raw_accel(dog) == pytest.approx(raw)
    assert dog.imu.calls == 1

    class _OffsetOnly:
        accData = [raw[i] + offset[i] for i in range(3)]
        imu_acc_offset = list(offset)

    assert read_raw_accel(_OffsetOnly()) == pytest.approx(raw, abs=0.1)
    # At rest the library's adjusted sample is on -X, which is what hid the tilt.
    adjusted = _OffsetOnly.accData
    assert adjusted[0] == pytest.approx(-16384, abs=1)
    assert adjusted[1] == pytest.approx(0, abs=1)
    assert adjusted[2] == pytest.approx(0, abs=1)

    class _Dead:
        def _sh3001_getimudata(self):
            return False

        imu = None

    dead = _Dead()
    dead.imu = dead
    assert read_raw_accel(dead) is None


def test_tracker_averages_the_baseline_and_keeps_the_last_good_sample():
    rest = _MOUNTS["yz"]
    seen = {"n": 0}

    def read():
        seen["n"] += 1
        # A zero and a failure in the window must not move the average.
        if seen["n"] in (3, 7):
            return None
        if seen["n"] == 4:
            return (0.0, 0.0, 0.0)
        return rest

    tracker = TiltTracker()
    assert tracker.capture(read, samples=10, gap=0.0)
    assert tracker.sample_count == 7
    assert tracker.baseline == pytest.approx(rest)
    pitch, roll = tracker.update(raw_accel_for_attitude(rest, 10, -4))
    assert (pitch, roll) == pytest.approx((10.0, -4.0), abs=0.05)
    assert tracker.tilt_deg == pytest.approx(angle_between(raw_accel_for_attitude(rest, 10, -4), rest), abs=0.05)
    again = tracker.update(None)
    assert again == pytest.approx((pitch, roll))
    assert tracker.fail_streak == 1
    assert tracker.tilt_deg == pytest.approx(angle_between(raw_accel_for_attitude(rest, 10, -4), rest), abs=0.05)


def test_sensorcheck_offers_imu_mode(capsys):
    import sensorcheck
    with pytest.raises(SystemExit) as exc:
        sensorcheck.main(["--help"])
    assert exc.value.code == 0
    assert "--imu" in capsys.readouterr().out


def _logged(text: str, msg: str) -> list[str]:
    suffix = f"] {msg}"
    return [line for line in text.splitlines() if line.endswith(suffix)]


def _scripted_dog(make_hw, script, mount=_MOUNTS["x"]):
    """Sim dog with the frozen-IMU check on and a raw-accel script."""
    sim.configure(imu_frozen_check=True, imu_mount=mount, imu_script=script)
    return make_hw()


def test_identical_accel_stream_trips_stale_and_safety(make_hw, capsys):
    """A jittery boot, then one accel frame repeated, is a frozen chip.

    The safety trip is imu_read_fail, which is what already stops patrol and follow.
    """
    mount = _MOUNTS["x"]
    frozen = (mount[0] + 50.0, mount[1], mount[2])
    seen = {"n": 0}

    def script():
        seen["n"] += 1
        if seen["n"] <= 10:
            return (mount[0] + seen["n"], mount[1], mount[2])
        return frozen

    hw = _scripted_dog(make_hw, script, mount)
    mon = SafetyMonitor(hw)
    assert hw.imu_stale is False
    capsys.readouterr()
    for _ in range(IMU_STALE_READS - 1):
        hw.tilt()
        assert hw.imu_stale is False
        assert hw.imu_fail_streak == 0
    hw.tilt()
    assert hw.imu_stale is True
    assert hw.imu_fail_streak == 1
    text = capsys.readouterr().out
    assert len(_logged(text, "imu_stale")) == 1
    assert _logged(text, "imu_stale_at_boot") == []
    while hw.imu_fail_streak < C.SAFETY_EMPTY_READS - 1:
        hw.tilt()
        assert not mon.locked.is_set()
    assert _logged(capsys.readouterr().out, "imu_stale") == []
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason == "imu_read_fail"
    assert not mon.allow_move()


def test_jittering_accel_does_not_go_stale(make_hw, capsys):
    mount = _MOUNTS["yz"]
    seen = {"n": 0}

    def script():
        seen["n"] += 1
        return (mount[0] + (seen["n"] % 5) - 2, mount[1], mount[2])

    hw = _scripted_dog(make_hw, script, mount)
    mon = SafetyMonitor(hw)
    for _ in range(IMU_STALE_READS + C.SAFETY_EMPTY_READS + 5):
        mon._tick()
        assert hw.imu_stale is False
        assert hw.imu_fail_streak == 0
        assert not mon.locked.is_set()
    assert "imu_stale" not in capsys.readouterr().out


def test_stale_clears_when_samples_vary(make_hw, capsys):
    mount = _MOUNTS["x"]
    frozen = (mount[0] + 50.0, mount[1], mount[2])
    seen = {"n": 0}

    def script():
        seen["n"] += 1
        if seen["n"] <= 10:
            return (mount[0] + seen["n"], mount[1], mount[2])
        return frozen

    hw = _scripted_dog(make_hw, script, mount)
    for _ in range(IMU_STALE_READS):
        hw.tilt()
    assert hw.imu_stale is True
    assert hw.imu_fail_streak == 1
    hw.dog.imu_raw.set((frozen[0] + 200.0, frozen[1], frozen[2]))
    hw.tilt()
    assert hw.imu_stale is False
    assert hw.imu_fail_streak == 0
    hw.dog.imu_raw.set((frozen[0] - 200.0, frozen[1], frozen[2]))
    hw.tilt()
    assert hw.imu_stale is False
    assert len(_logged(capsys.readouterr().out, "imu_stale")) == 1


def test_identical_baseline_is_stale_at_boot(make_hw, capsys):
    sim.configure(imu_frozen_check=True)
    hw = make_hw()
    text = capsys.readouterr().out
    assert hw.imu_stale is True
    assert len(_logged(text, "imu_stale_at_boot")) == 1
    assert _logged(text, "imu_stale") == []
    mon = SafetyMonitor(hw)
    for _ in range(C.SAFETY_EMPTY_READS - 1):
        mon._tick()
        assert not mon.locked.is_set()
        assert hw.imu_stale is True
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason == "imu_read_fail"


def test_sensorcheck_imu_prints_stale_flag(monkeypatch, capsys):
    import sensorcheck

    def _run(expect: str) -> None:
        sleeps = {"n": 0}

        def sleep(_dt):
            sleeps["n"] += 1
            raise KeyboardInterrupt

        monkeypatch.setattr(sensorcheck.time, "sleep", sleep)
        monkeypatch.setattr(sensorcheck.os, "_exit", lambda _code: None)
        sensorcheck.main(["--imu"])
        out = capsys.readouterr().out
        assert f"stale={expect}" in out

    _run("False")
    sim.configure(imu_frozen_check=True)
    _run("True")


def _walk_session(hw, mon, allow):
    import argparse

    import main
    from locomotion import Locomotion

    loco = Locomotion(hw, mon)
    loco.allow_no_imu = allow
    args = argparse.Namespace(allow_no_imu=allow, patrol=False, follow=False, allow_voice_patrol=False)
    return main.Session(hw, mon, loco, listener=None, tts=False, args=args), loco


def test_stale_without_override_trips_and_refuses_patrol_and_follow(make_hw, capsys):
    import follow as follow_mod

    sim.configure(imu_frozen_check=True)
    hw = make_hw()
    mon = SafetyMonitor(hw, allow_no_imu=False)
    session, loco = _walk_session(hw, mon, False)
    assert loco.enable("patrol") is False
    assert loco.enabled is False
    assert follow_mod.Follow(session).start() is False
    for _ in range(C.SAFETY_EMPTY_READS):
        mon._tick()
    text = capsys.readouterr().out
    assert "patrol_refused imu_stale" in text
    assert "follow_refused imu_stale" in text
    assert "imu_stale_overridden" not in text
    assert_halted(mon, hw.dog)
    assert mon.reason == "imu_read_fail"


def test_allow_no_imu_lets_a_frozen_chip_walk(make_hw, capsys):
    import follow as follow_mod

    sim.configure(imu_frozen_check=True)
    hw = make_hw()
    assert hw.imu_stale is True
    hw.note_stale_override()
    mon = SafetyMonitor(hw, allow_no_imu=True)
    session, loco = _walk_session(hw, mon, True)
    assert loco.enable("override") is True
    assert follow_mod.Follow(session).start() is True
    for _ in range(C.SAFETY_EMPTY_READS + 5):
        mon._tick()
        assert not mon.locked.is_set()
    text = capsys.readouterr().out
    assert len(_logged(text, "imu_stale_overridden")) == 1
    mon._tick()
    assert _logged(capsys.readouterr().out, "imu_stale_overridden") == []
    mon.walking.set()
    hw.dog.distance.set(10)
    mon._tick()
    assert mon.reason.startswith("obstacle")


def test_allow_no_imu_logs_once_when_the_chip_freezes_later(make_hw, capsys):
    mount = _MOUNTS["x"]
    frozen = (mount[0] + 50.0, mount[1], mount[2])
    seen = {"n": 0}

    def script():
        seen["n"] += 1
        if seen["n"] <= 10:
            return (mount[0] + seen["n"], mount[1], mount[2])
        return frozen

    hw = _scripted_dog(make_hw, script, mount)
    assert hw.imu_stale is False
    hw.note_stale_override()
    capsys.readouterr()
    mon = SafetyMonitor(hw, allow_no_imu=True)
    for _ in range(IMU_STALE_READS - 1):
        mon._tick()
        assert not mon.locked.is_set()
    assert hw.imu_stale is False
    mon._tick()
    assert hw.imu_stale is True
    assert not mon.locked.is_set()
    assert mon.reason == ""
    assert len(_logged(capsys.readouterr().out, "imu_stale_overridden")) == 1
    for _ in range(5):
        mon._tick()
    assert not mon.locked.is_set()
    assert _logged(capsys.readouterr().out, "imu_stale_overridden") == []


def test_allow_no_imu_still_trips_a_live_tilt(make_hw, capsys):
    hw = make_hw()
    hw.dog.pitch = 35
    mon = SafetyMonitor(hw, allow_no_imu=True)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason.startswith("tilt ")
    assert "imu_stale_overridden" not in capsys.readouterr().out


def test_sit_posture_does_not_trip_and_a_real_tip_still_does(make_hw, capsys):
    """Sit is about 30 deg of pitch on Rex's mount. That pose is not a fall.

    A 33 deg tip while standing still still trips. So does a further rotation
    off the sit attitude, and a collapse past the posture ceiling mid-move.
    """
    sim.configure(imu_mount=_MOUNTS["yz"])
    hw = make_hw()
    mon = SafetyMonitor(hw)
    hw.dog.roll = 0

    hw.dog.pitch = 33
    hw.begin_posture("sit")
    mon._tick()
    assert hw.tilt_deg == pytest.approx(33.0, abs=0.2)
    assert hw.tilt_deg > C.SAFETY_TILT_DEG
    assert not mon.locked.is_set()
    hw.dog.pitch = 30.5
    hw.end_posture("sit")
    mon._tick()
    assert hw.body_pose == "sit"
    assert hw.tilt_deg == pytest.approx(30.5, abs=0.2)
    assert hw.tilt_deg > C.SAFETY_TILT_DEG
    assert not mon.locked.is_set()
    assert "tilt_pose_held" in capsys.readouterr().out

    ref = hw.pose_reference
    assert ref is not None
    tipped = _rotate(ref, _axes(ref)[3], 33)
    hw.dog.imu_raw.set(tipped)
    mon._tick()
    assert_halted(mon, hw.dog)
    assert mon.reason.startswith("tilt ")

    mon.reset("test")
    hw.dog.imu_raw.set(None)
    hw.dog.pitch = 30.5
    hw.dog.roll = 0
    hw.begin_posture("sit")
    hw.end_posture("sit")
    hw.begin_gait()
    assert hw.body_pose == "recovering"
    mon._tick()
    assert not mon.locked.is_set()
    hw.dog.pitch = 10  # stood up, inside the clear line
    mon._tick()
    assert hw.body_pose == "upright"
    assert not mon.locked.is_set()
    hw.dog.pitch = 33
    mon._tick()
    assert_halted(mon, hw.dog)
    assert "pitch=33" in mon.reason


def test_posture_motion_still_trips_a_collapse(make_hw):
    sim.configure(imu_mount=_MOUNTS["yz"])
    hw = make_hw()
    mon = SafetyMonitor(hw)
    hw.dog.pitch = 56
    hw.begin_posture("stand")
    mon._tick()
    assert hw.tilt_deg == pytest.approx(56.0, abs=0.3)
    assert hw.tilt_deg > C.SAFETY_POSTURE_TILT_DEG
    assert_halted(mon, hw.dog)
    assert "pitch=56" in mon.reason


def test_allow_no_imu_help_mentions_a_frozen_imu():
    import main
    text = main.build_parser().format_help()
    assert "--allow-no-imu" in text
    assert "frozen" in text
