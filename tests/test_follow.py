"""Supervised follow: --follow only, bbox steering, caps, safety thread, no systemd flag."""
from __future__ import annotations

import argparse

import config as C
import faces
import follow as follow_mod
import main
import sim
from support import ROOT

_WALK = {"forward", "backward", "turn_left", "turn_right"}


def _session(hw):
    from locomotion import Locomotion
    from safety import SafetyMonitor

    safety = SafetyMonitor(hw)
    loco = Locomotion(hw, safety)
    safety.on_trip = lambda reason: loco.disable(f"trip:{reason}")
    args = argparse.Namespace(allow_no_imu=False, patrol=False, follow=False, allow_voice_patrol=False)
    session = main.Session(hw, safety, loco, listener=None, tts=False, args=args)
    session.guard = None
    session.follow = None
    return session


def _hit(name, bbox, stranger=False):
    return faces.FaceHit(name, stranger, 0.0, bbox, "sim")


def _fast(monkeypatch):
    monkeypatch.setattr(main._shutdown, "wait", lambda timeout=None: None)
    import vision_grok
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)


def _run_args(**overrides):
    base = dict(
        patrol=False, sim=True, allow_voice_patrol=False, seconds=0.05,
        allow_no_imu=False, no_tts=True, no_listen=True, force_first_grok=False,
        follow=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_follow_is_opt_in_and_absent_from_the_service():
    parser = main.build_parser()
    assert parser.parse_args([]).follow is False
    assert parser.parse_args(["--follow"]).follow is True
    assert C.FOLLOW_MAX_STEPS == 8
    assert C.FOLLOW_MAX_SECONDS == 60
    assert C.FOLLOW_SPEED <= C.WALK_SPEED
    text = (ROOT / "rex.service").read_text()
    execs = [line.split("#", 1)[0] for line in text.splitlines() if "ExecStart" in line.split("#", 1)[0]]
    assert execs and all("--follow" not in line and "--patrol" not in line for line in execs)
    assert "REX_SIM" not in text


def test_bbox_steers_toward_cory_and_ignores_strangers(make_hw, monkeypatch):
    monkeypatch.setattr(C, "FOLLOW_MAX_STEPS", 8)
    monkeypatch.setattr(C, "PRIMARY_HUMAN", "Cory")
    hw = make_hw()
    session = _session(hw)
    follower = follow_mod.Follow(session)
    assert follower.start() is True
    assert follower.tick([_hit("Cory", (0.0, 0.2, 0.2, 0.4))]) == "turn_left"
    assert follower.tick([_hit("Cory", (0.4, 0.2, 0.2, 0.4))]) == "forward"
    assert follower.tick([_hit("Cory", (0.75, 0.1, 0.2, 0.4))]) == "turn_right"
    other = _hit("Sam", (0.0, 0.2, 0.2, 0.4))
    cory = _hit("Cory", (0.75, 0.1, 0.2, 0.4))
    assert follower.tick([other, cory]) == "turn_right"
    before = follower.steps
    assert follower.tick([_hit("", (0.4, 0.2, 0.2, 0.4), stranger=True)]) == "lost"
    assert follower.active is True and follower.steps == before
    actions = [a["action"] for a in hw.dog.action_log]
    assert actions[:4] == ["turn_left", "forward", "turn_right", "turn_right"]
    assert all(a["speed"] == C.FOLLOW_SPEED for a in hw.dog.action_log if a["action"] in _WALK)


def test_lost_person_and_caps_stop_the_session(make_hw, monkeypatch):
    monkeypatch.setattr(C, "FOLLOW_MAX_STEPS", 1)
    monkeypatch.setattr(C, "FOLLOW_LOST_TICKS", 3)
    hw = make_hw()
    follower = follow_mod.Follow(_session(hw))
    assert follower.start()
    hit = _hit("Cory", (0.45, 0.2, 0.1, 0.4))
    assert follower.tick([hit]) == "forward"
    assert follower.tick([hit]) == "cap"
    assert follower.active is False
    assert any(a["action"] == "sit" for a in hw.dog.action_log)

    monkeypatch.setattr(C, "FOLLOW_MAX_STEPS", 8)
    follower = follow_mod.Follow(_session(make_hw()))
    follower.start()
    follower.started = 0
    assert follower.tick([hit]) == "cap"

    monkeypatch.setattr(C, "FOLLOW_MAX_SECONDS", 60)
    follower = follow_mod.Follow(_session(make_hw()))
    follower.start()
    for _ in range(2):
        assert follower.tick([]) == "lost"
        assert follower.active is True
    assert follower.tick([]) == "lost"
    assert follower.active is False


def test_follow_holds_a_distance_dropout_and_can_resume(make_hw, monkeypatch):
    monkeypatch.setattr("locomotion.time.sleep", lambda *_a, **_k: None)
    clock = sim.ManualClock(0.0)
    monkeypatch.setattr("safety.time.time", clock.time)
    hw = make_hw()
    session = _session(hw)
    follower = follow_mod.Follow(session)
    assert follower.start()
    hit = _hit("Cory", (0.45, 0.2, 0.1, 0.4))
    assert follower.tick([hit]) == "forward"
    hw.dog.distance.set(None)
    session.safety.begin_distance_hold(3, where="test")
    assert follower.tick([hit]) == "hold"
    assert follower.active is True
    assert [a["action"] for a in hw.dog.action_log] == ["forward"]
    hw.dog.distance.set(90)
    assert follower.tick([hit]) == "forward"
    assert follower.active is True
    hw.dog.distance.set(None)
    session.safety.begin_distance_hold(4, where="test")
    clock.advance(C.SAFETY_DISTANCE_HOLD_SECONDS)
    assert follower.tick([hit]) == "halt"
    assert follower.active is False
    assert session.safety.locked.is_set()
    assert session.safety.reason.startswith("distance_empty")


def test_guard_and_dead_imu_refuse_follow(make_hw, monkeypatch, capsys):
    import guard
    monkeypatch.setattr(C, "GUARD_ENABLED", True)
    monkeypatch.setattr(C, "GUARD_ARMED", True)
    hw = make_hw()
    session = _session(hw)
    session.guard = guard.Guard(session, notifiers=[])
    assert session.guard.armed is True
    assert follow_mod.Follow(session).start() is False
    assert "follow_refused guard_armed" in capsys.readouterr().out

    hw.imu_ok = False
    session.guard.armed = False
    assert follow_mod.Follow(session).start() is False
    assert "follow_refused imu_unavailable" in capsys.readouterr().out
    session.args.allow_no_imu = True
    assert follow_mod.Follow(session).start() is True


def test_safety_thread_aborts_a_follow_step(make_hw, monkeypatch):
    monkeypatch.setattr(C, "FOLLOW_MAX_STEPS", 8)
    hw = make_hw()
    session = _session(hw)
    hw.dog.distance.set(10)
    hw.dog.pause_legs()
    session.safety.start()
    follower = follow_mod.Follow(session)
    assert follower.start()
    try:
        assert follower.tick([_hit("Cory", (0.45, 0.2, 0.1, 0.4))]) == "aborted"
    finally:
        session.safety.stop()
        session.safety.join(timeout=1)
    assert session.safety.locked.is_set()
    assert session.safety.reason.startswith("obstacle")
    assert follower.steps == 0
    assert follower.active is False
    assert hw.dog.legs_stop_calls >= 1


def test_run_follows_only_with_the_flag(monkeypatch, capsys):
    _fast(monkeypatch)
    monkeypatch.setattr(C, "FACE_ENABLED", True)
    monkeypatch.setattr(C, "FOLLOW_MAX_STEPS", 4)
    sim.configure(people=[{"name": "Cory", "known": True, "bbox": [0.4, 0.2, 0.2, 0.4]}])
    assert main.run(_run_args(follow=False)) == 0
    assert "follow_enabled" not in capsys.readouterr().out
    assert not any(a["action"] in _WALK for a in sim.dogs[-1].action_log)

    main._shutdown.clear()
    sim.reset()
    sim.configure(people=[{"name": "Cory", "known": True, "bbox": [0.4, 0.2, 0.2, 0.4]}])
    assert main.run(_run_args(follow=True, patrol=True)) == 0
    out = capsys.readouterr().out
    assert "follow_enabled" in out
    assert "patrol_enabled" not in out
    actions = [a["action"] for a in sim.dogs[-1].action_log]
    assert "forward" in actions
    assert actions.count("forward") <= 4


def test_armed_guard_blocks_follow_in_run(monkeypatch, capsys):
    _fast(monkeypatch)
    monkeypatch.setattr(C, "GUARD_ENABLED", True)
    monkeypatch.setattr(C, "GUARD_ARMED", True)
    monkeypatch.setattr(C, "FACE_ENABLED", True)
    sim.configure(people=[{"name": "Cory", "known": True, "bbox": [0.4, 0.2, 0.2, 0.4]}])
    assert main.run(_run_args(follow=True)) == 0
    out = capsys.readouterr().out
    assert "follow_refused guard_armed" in out
    assert "I am not following" in out
    assert not any(a["action"] in _WALK for a in sim.dogs[-1].action_log)
