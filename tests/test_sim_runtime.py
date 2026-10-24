"""Sim selection, boot, and the pieces of the dog the safety tests don't touch."""
from __future__ import annotations

import argparse

import pytest
from PIL import Image

import brain
import config as C
import main
import sim
import vision_grok
from support import ROOT, run_main, run_py

_WALK = {"forward", "backward", "turn_left", "turn_right"}


def _args(**overrides):
    base = dict(
        patrol=False,
        sim=True,
        allow_voice_patrol=False,
        seconds=0.05,
        allow_no_imu=False,
        no_tts=True,
        no_listen=True,
        force_first_grok=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _fast(monkeypatch):
    """Skip the main loop's half-second idle wait; the deadline still ends the run."""
    monkeypatch.setattr(main._shutdown, "wait", lambda timeout=None: None)
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)


def test_sensor_scripts_follow_the_clock(make_hw):
    clock = sim.ManualClock(0.0)
    sim.configure(clock=clock)
    hw = make_hw()
    hw.dog.distance.script([(0, 100), (1.5, 15)])
    hw.dog.battery.script([(0, 7.4), (2.0, 6.2)])
    assert hw.distance() == 100
    assert hw.battery() == pytest.approx(7.4)
    clock.advance(1.5)
    assert hw.distance() == 15
    assert hw.battery() == pytest.approx(7.4)
    clock.advance(0.5)
    assert hw.battery() == pytest.approx(6.2)
    assert hw.simulated is True
    assert hw.imu_ok is True


def test_touch_sound_imu_chest_and_camera(make_hw):
    hw = make_hw()
    assert hw.touch() == "N"
    hw.dog.touch.set("R")
    assert hw.touch() == "R"
    assert hw.sound_direction() is None
    hw.dog.sound.set(12)
    assert hw.sound_direction() == 12
    hw.dog.sound.set(400)
    assert hw.sound_direction() is None
    assert hw.tilt() == (0.0, 0.0)
    hw.dog.pitch = 4
    hw.dog.roll = -2
    assert hw.tilt() == pytest.approx((4.0, -2.0))
    hw.chest("listening")
    hw.chest("locked")
    chests = [event for event in hw.dog.events if event[0] == "chest"]
    assert chests[0][1:] == ("breath", "#0000ff", 1, 0.6)
    assert chests[1][1:] == ("boom", "#ff00ff", 1, 0.6)
    hw.dog.frames.push(Image.new("RGB", (32, 24), (0, 0, 0)))
    path = hw.photo()
    assert path is not None and path.stat().st_size > 1000
    with Image.open(path) as im:
        assert im.getpixel((0, 0))[:3] == (0, 0, 0)


def test_sound_pan_stays_on_roll(make_hw):
    assert C.SOUND_PAN_AXIS == "roll"
    hw = make_hw()
    hw.head_pan(40)
    assert hw.head == {"yaw": 0, "roll": 40, "pitch": C.HEAD_PITCH_DEFAULT}
    hw.head_pan(-80)
    assert hw.head["yaw"] == 0
    assert hw.head["roll"] == -C.SOUND_YAW_LIMIT
    assert hw.head["pitch"] == C.HEAD_PITCH_DEFAULT


def test_flags_default_off_and_shelf_mode_does_not_walk(make_hw, monkeypatch, capsys):
    args = main.build_parser().parse_args([])
    assert args.patrol is False and args.sim is False and args.allow_no_imu is False
    _fast(monkeypatch)
    assert main.run(_args(patrol=False)) == 0
    dog = sim.dogs[-1]
    assert dog.closed
    assert not any(action["action"] in _WALK for action in dog.action_log)
    out = capsys.readouterr().out
    assert "sim_init" in out
    assert "FATAL" not in out


def test_allow_no_imu_is_wired_through_run(monkeypatch, capsys):
    _fast(monkeypatch)
    sim.configure(imu=False)
    assert main.run(_args(patrol=True, allow_no_imu=False)) == 0
    refused = capsys.readouterr().out
    assert "patrol_refused" in refused and "imu_unavailable" in refused
    assert not any(action["action"] in _WALK for action in sim.dogs[-1].action_log)

    main._shutdown.clear()
    sim.reset()
    sim.configure(imu=False)
    assert main.run(_args(patrol=True, allow_no_imu=True)) == 0
    assert any(action["action"] == "forward" for action in sim.dogs[-1].action_log)
    assert "patrol_enabled" in capsys.readouterr().out


def test_main_loop_survives_api_errors(monkeypatch, capsys):
    _fast(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "test-not-a-real-key")
    assert main.run(_args()) == 0
    out = capsys.readouterr().out
    assert "FATAL" not in out
    assert "grok_exception" in out
    assert "vision_fallback" in out
    assert brain.state == "online" or "main_loop_exit" in out


def test_service_never_enables_patrol_or_sim():
    text = (ROOT / "rex.service").read_text()
    assert "REX_SIM" not in text
    exec_lines = []
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        if "ExecStart" in code:
            exec_lines.append(code)
    assert exec_lines
    for line in exec_lines:
        assert "--patrol" not in line
        assert "--sim" not in line


def test_env_example_has_no_secret_values():
    for line in (ROOT / ".env.example").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        _key, value = stripped.split("=", 1)
        assert value.strip().strip('"').strip("'") == ""


def test_real_hardware_path_imports_pidog_and_not_sim(tmp_path):
    proc = run_py(
        f"""
        import os, sys
        os.environ["HOME"] = {str(tmp_path)!r}
        os.environ.pop("REX_SIM", None)
        import hardware
        assert "sim" not in sys.modules
        try:
            hardware.Hardware()
        except ModuleNotFoundError as exc:
            assert exc.name == "pidog"
        else:
            raise SystemExit("Hardware() succeeded without pidog")
        assert "sim" not in sys.modules
        assert "picamera2" not in sys.modules
        print("real-path-ok")
        """,
        drop_sim=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "real-path-ok" in proc.stdout


def test_sim_path_does_not_import_pi_libraries(tmp_path):
    proc = run_py(
        f"""
        import os, sys
        os.environ["HOME"] = {str(tmp_path)!r}
        os.environ["REX_SIM"] = "1"
        import hardware
        hw = hardware.Hardware()
        assert hw.simulated
        assert hw.distance() == 200
        assert hw.photo() is not None
        hw.move_head(0, 10, None)
        assert hw.head["roll"] == 10
        hw.shutdown()
        for banned in ("pidog", "robot_hat", "vilib", "picamera2"):
            assert banned not in sys.modules, banned
        print("sim-path-ok")
        """,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "sim-path-ok" in proc.stdout


def test_process_boots_with_sim_flag_and_env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    flagged = run_main(
        ["--sim", "--seconds", "0.4", "--no-tts", "--no-listen"],
        home=home,
        drop_sim=True,
    )
    assert flagged.returncode == 0, flagged.stdout + flagged.stderr
    assert "sim_init" in flagged.stdout
    assert "FATAL" not in flagged.stdout
    assert "patrol_enabled" not in flagged.stdout

    home2 = tmp_path / "home2"
    home2.mkdir()
    via_env = run_main(
        ["--seconds", "0.4", "--no-tts", "--no-listen"],
        home=home2,
    )
    assert via_env.returncode == 0, via_env.stdout + via_env.stderr
    assert "sim_init" in via_env.stdout
    assert "pidog" not in via_env.stderr
