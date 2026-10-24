"""Shelf guard: stranger or motion alerts, cooldown, no walking, pluggable notifiers."""
from __future__ import annotations

import argparse

from PIL import Image

import config as C
import faces
import guard
import main
import sim
from support import ROOT

_WALK = {"forward", "backward", "turn_left", "turn_right"}


def _session(hw):
    from locomotion import Locomotion
    from safety import SafetyMonitor

    safety = SafetyMonitor(hw)
    loco = Locomotion(hw, safety)
    args = argparse.Namespace(allow_no_imu=False, patrol=False, allow_voice_patrol=False)
    return main.Session(hw, safety, loco, listener=None, tts=False, args=args)


class _Recorder:
    name = "recorder"

    def __init__(self):
        self.calls = []

    def send(self, title, message, image):
        self.calls.append({"title": title, "message": message, "image": image})


def test_defaults_are_off_and_secrets_are_redacted(monkeypatch, capsys):
    assert C.GUARD_ENABLED is False
    assert C.GUARD_ARMED is False
    assert C.GUARD_COOLDOWN == 60
    assert C.NTFY_TOPIC == ""
    assert C.ALERT_WEBHOOK_URL == ""
    assert C.NTFY_SERVER == "https://ntfy.sh"
    assert guard.build_notifiers() == []
    token = "ntfy-token-secret"
    hook = "https://example.test/hook/super-secret-path"
    monkeypatch.setenv("REX_NTFY_TOKEN", token)
    monkeypatch.setenv("REX_ALERT_WEBHOOK_URL", hook)
    C.log(f"leak {token} {hook}")
    out = capsys.readouterr().out
    assert token not in out and hook not in out and "***" in out
    example = (ROOT / ".env.example").read_text()
    assert "REX_NTFY_TOPIC=" in example and "REX_ALERT_WEBHOOK_URL=" in example
    assert "super-secret" not in example
    assert "REX_NTFY" not in (ROOT / "rex.service").read_text()


def test_unknown_person_and_motion_alert_once_then_cool_down(make_hw, monkeypatch):
    monkeypatch.setattr(C, "GUARD_ENABLED", True)
    monkeypatch.setattr(C, "GUARD_ARMED", False)
    monkeypatch.setattr(C, "GUARD_COOLDOWN", 60)
    monkeypatch.setattr(C, "GUARD_MOTION", 18)
    hw = make_hw()
    session = _session(hw)
    rec = _Recorder()
    g = guard.Guard(session, notifiers=[rec])
    assert g.armed is False
    black = Image.new("RGB", (32, 24), (0, 0, 0))
    white = Image.new("RGB", (32, 24), (255, 255, 255))
    stranger = faces.FaceHit("", True, 1.0, (0.1, 0.1, 0.2, 0.2), "sim")
    assert g.tick(black, hits=[stranger]) is None  # disarmed
    assert rec.calls == []
    assert g.set_armed(True) is True
    assert session.loco.enabled is False
    assert g.tick(black, hits=[stranger]) == "unknown_person"
    assert rec.calls[0]["title"].startswith("Rex alert")
    assert rec.calls[0]["image"][:2] == b"\xff\xd8"
    assert g.tick(black, hits=[stranger]) == "cooldown"
    assert len(rec.calls) == 1
    g.last_alert = 0
    g.prev = black
    assert guard.motion_score(black, black) == 0
    assert g.tick(black, hits=[]) is None
    assert g.tick(white, hits=[]) == "motion"
    assert len(rec.calls) == 2
    assert "motion" in rec.calls[1]["message"]
    assert not any(a["action"] in _WALK for a in hw.dog.action_log)


def test_arming_refuses_patrol_until_disarmed(make_hw, monkeypatch, capsys):
    monkeypatch.setattr(C, "GUARD_ENABLED", True)
    hw = make_hw()
    session = _session(hw)
    assert session.loco.enable("test") is True
    g = guard.Guard(session, notifiers=[])
    assert g.set_armed(True) is True
    assert session.loco.enabled is False
    assert session.loco.enable("dashboard") is False
    assert "patrol_refused guard_armed" in capsys.readouterr().out
    assert g.set_armed(False) is False  # disarmed, the method returns the armed flag
    assert session.loco.enable("again") is True


def test_disabled_guard_cannot_be_armed(make_hw, monkeypatch, capsys):
    monkeypatch.setattr(C, "GUARD_ENABLED", False)
    g = guard.Guard(_session(make_hw()), notifiers=[])
    assert g.set_armed(True) is False
    assert g.armed is False
    assert "guard_ignored" in capsys.readouterr().out


def test_ntfy_and_webhook_payloads_keep_secrets_out_of_the_log(monkeypatch, capsys):
    import requests

    token = "ntfy-token-secret"
    hook = "https://example.test/hook/super-secret-path"
    monkeypatch.setattr(C, "NTFY_TOPIC", "rex-test-topic")
    monkeypatch.setattr(C, "NTFY_SERVER", "https://ntfy.sh")
    monkeypatch.setattr(C, "NTFY_TOKEN", token)
    monkeypatch.setattr(C, "ALERT_WEBHOOK_URL", "")
    calls = []

    def post(url, data=None, json=None, headers=None, timeout=None, **_k):
        calls.append({"url": url, "data": data, "json": json, "headers": headers})
        return None

    monkeypatch.setattr(requests, "post", post)
    notes = guard.build_notifiers()
    assert [n.name for n in notes] == ["ntfy"]
    notes[0].send("Rex alert: motion", "Rex guard saw motion.", b"\xff\xd8\xff")
    assert calls[0]["url"] == "https://ntfy.sh/rex-test-topic"
    assert calls[0]["data"] == b"\xff\xd8\xff"
    assert calls[0]["headers"]["Authorization"] == f"Bearer {token}"
    assert calls[0]["headers"]["Filename"] == "rex.jpg"

    monkeypatch.setattr(C, "NTFY_TOPIC", "")
    monkeypatch.setattr(C, "ALERT_WEBHOOK_URL", hook)
    calls.clear()
    notes = guard.build_notifiers()
    assert [n.name for n in notes] == ["webhook"]
    notes[0].send("Rex alert: motion", "Rex guard saw motion.", b"\xff\xd8abc")
    body = calls[0]["json"]
    assert calls[0]["url"] == hook
    assert body["source"] == "rex" and body["message"].startswith("Rex guard")
    assert body["image_base64"]
    out = capsys.readouterr().out
    assert token not in out and hook not in out


def test_dashboard_arm_blocks_patrol(make_hw, monkeypatch, capsys):
    monkeypatch.setattr(C, "GUARD_ENABLED", True)
    monkeypatch.setattr(C, "DASH_ALLOW_PATROL", True)
    monkeypatch.setattr(C, "DASH_TOKEN", "test-token-not-a-secret")
    guard.ensure_registered()
    from dashboard import Dashboard
    hw = make_hw()
    session = _session(hw)
    session.guard = guard.Guard(session, notifiers=[])
    dash = Dashboard(session)
    dash.build_app()
    client = dash.app.test_client()
    headers = {"X-Rex-Token": "test-token-not-a-secret"}
    assert "Arm guard" in client.get("/").get_data(as_text=True)
    assert client.post("/api/guard", json={"armed": True}, headers=headers).status_code == 202
    dash.drain()
    assert session.guard.armed is True
    assert client.get("/api/status", headers=headers).get_json()["guard_armed"] is True
    assert client.post("/api/mode", json={"mode": "patrol"}, headers=headers).status_code == 202
    dash.drain()
    assert session.loco.enabled is False
    assert "shelf_only" in capsys.readouterr().out


def test_armed_run_does_not_walk_and_alerts_on_a_stranger(monkeypatch, capsys):
    monkeypatch.setattr(C, "GUARD_ENABLED", True)
    monkeypatch.setattr(C, "GUARD_ARMED", True)
    monkeypatch.setattr(C, "FACE_ENABLED", True)
    monkeypatch.setattr(C, "GUARD_COOLDOWN", 60)
    monkeypatch.setattr(main._shutdown, "wait", lambda timeout=None: None)
    import vision_grok
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    sim.configure(people=[{"name": "", "stranger": True, "bbox": [0.0, 0.0, 0.2, 0.2]}])
    args = argparse.Namespace(
        patrol=True, sim=True, allow_voice_patrol=True, seconds=0.05,
        allow_no_imu=False, no_tts=True, no_listen=True, force_first_grok=False,
    )
    assert main.run(args) == 0
    out = capsys.readouterr().out
    assert "Guard is armed. I am not walking." in out
    assert "guard_alert reason=unknown_person" in out
    assert out.count("guard_alert ") == 1
    assert "patrol_enabled" not in out
    dog = sim.dogs[-1]
    assert not any(action["action"] in _WALK for action in dog.action_log)
