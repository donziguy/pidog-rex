"""Local dashboard: token, status, snapshot, log, and controls. No real bind in the assertions
that only need the Flask test client.
"""
from __future__ import annotations

import argparse

from PIL import Image

import brain
import config as C
import main
from dashboard import Dashboard, token_ok
from support import ROOT


TOKEN = "test-token-not-a-secret"


def _session(hw):
    from locomotion import Locomotion
    from safety import SafetyMonitor

    safety = SafetyMonitor(hw)
    loco = Locomotion(hw, safety)
    safety.on_trip = lambda reason: loco.disable(f"trip:{reason}")
    args = argparse.Namespace(allow_no_imu=False, patrol=False)
    return main.Session(hw, safety, loco, listener=None, tts=False, args=args)


def _dash(hw, monkeypatch, token=TOKEN, allow_patrol=False):
    monkeypatch.setattr(C, "DASH_TOKEN", token)
    monkeypatch.setattr(C, "DASH_ALLOW_PATROL", allow_patrol)
    dash = Dashboard(_session(hw))
    dash.build_app()
    return dash


def _auth(token=TOKEN):
    return {"X-Rex-Token": token}


def test_defaults_are_off_and_bind_is_open_with_required_token():
    assert C.DASH_ENABLED is False
    assert C.DASH_HOST == "0.0.0.0"
    assert C.DASH_PORT == 8080
    assert C.DASH_TOKEN == ""
    assert C.DASH_ALLOW_PATROL is False
    assert token_ok("") is False
    assert token_ok(TOKEN) is False  # nothing configured


def test_page_has_stop_and_does_not_embed_a_token(make_hw, monkeypatch):
    dash = _dash(make_hw(), monkeypatch)
    res = dash.app.test_client().get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "STOP" in body
    assert TOKEN not in body
    assert "REX_DASH_TOKEN" not in body


def test_apis_require_a_token(make_hw, monkeypatch):
    dash = _dash(make_hw(), monkeypatch)
    client = dash.app.test_client()
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/snapshot").status_code == 401
    assert client.get("/api/log").status_code == 401
    assert client.post("/api/stop").status_code == 401
    assert client.get("/api/status", headers={"Authorization": "Bearer no-such-token"}).status_code == 401
    ok = client.get("/api/status", headers=_auth())
    assert ok.status_code == 200
    bearer = client.get("/api/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert bearer.status_code == 200
    query = client.get(f"/api/snapshot?token={TOKEN}")
    assert query.status_code == 200
    assert query.data[:2] == b"\xff\xd8"


def test_status_snapshot_and_log(make_hw, monkeypatch):
    hw = make_hw()
    hw.dog.frames.push(Image.new("RGB", (80, 60), (0, 0, 255)))
    dash = _dash(hw, monkeypatch)
    dash.safety.last.update(distance=42.0, tilt=7.5, pitch=3.0, roll=-1.0, battery=7.4, touch="N")
    brain.memory()["last_roast"] = "the shelf is a disgrace"
    C.LOG_PATH.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    client = dash.app.test_client()
    status = client.get("/api/status", headers=_auth()).get_json()
    assert status["battery"] == 7.4
    assert status["tilt"] == 7.5
    assert status["distance"] == 42.0
    assert status["mode"] == "shelf"
    assert status["last_grok"] == "the shelf is a disgrace"
    assert status["allow_patrol"] is False
    assert status["imu_ok"] is True
    assert status["imu_stale"] is False
    hw.imu_stale = True
    assert client.get("/api/status", headers=_auth()).get_json()["imu_stale"] is True
    snap = client.get("/api/snapshot", headers=_auth())
    assert snap.mimetype == "image/jpeg"
    with Image.open(__import__("io").BytesIO(snap.data)) as im:
        pixel = im.getpixel((0, 0))[:3]
        assert pixel[2] > 200 and pixel[0] < 40
    lines = client.get("/api/log", headers=_auth()).get_json()["lines"]
    assert lines[-3:] == ["alpha", "beta", "gamma"]


def test_speak_and_head_run_on_the_main_thread_drain(make_hw, monkeypatch, capsys):
    hw = make_hw()
    dash = _dash(hw, monkeypatch)
    client = dash.app.test_client()
    assert client.post("/api/speak", json={"text": "sit still"}, headers=_auth()).status_code == 202
    assert client.post("/api/head", json={"yaw": 15, "roll": -5, "pitch": 10}, headers=_auth()).status_code == 202
    assert hw.head["yaw"] == 0  # queued, not applied on the request thread
    assert "TTS:" not in capsys.readouterr().out
    assert dash.drain() == 2
    assert "sit still" in capsys.readouterr().out
    assert hw.head == {"yaw": 15, "roll": -5, "pitch": 10}
    assert client.post("/api/speak", json={"text": "  "}, headers=_auth()).status_code == 400


def test_stop_trips_safety_immediately(make_hw, monkeypatch):
    hw = make_hw()
    dash = _dash(hw, monkeypatch)
    res = dash.app.test_client().post("/api/stop", headers=_auth())
    assert res.status_code == 200
    assert res.get_json()["stopped"] is True
    assert dash.safety.locked.is_set()
    assert dash.safety.reason == "dashboard_stop"
    assert hw.dog.legs_stop_calls >= 1
    assert hw.dog.lie_calls >= 1


def test_patrol_requires_the_flag_and_still_obeys_safety(make_hw, monkeypatch, capsys):
    hw = make_hw()
    dash = _dash(hw, monkeypatch, allow_patrol=False)
    client = dash.app.test_client()
    refused = client.post("/api/mode", json={"mode": "patrol"}, headers=_auth())
    assert refused.status_code == 403
    assert dash.drain() == 0
    assert dash.loco.enabled is False

    monkeypatch.setattr(C, "DASH_ALLOW_PATROL", True)
    dash.build_app()
    client = dash.app.test_client()
    queued = client.post("/api/mode", json={"mode": "patrol"}, headers=_auth())
    assert queued.status_code == 202
    assert dash.loco.enabled is False
    dash.drain()
    assert dash.loco.enabled is True
    assert dash.last_result["ok"] is True

    hw.dog.distance.set(10)
    dash.safety.walking.set()
    dash.safety._tick()
    assert dash.safety.locked.is_set()
    assert dash.loco.enabled is False
    assert dash.safety.reason.startswith("obstacle")

    shelf = client.post("/api/mode", json={"mode": "shelf"}, headers=_auth())
    assert shelf.status_code == 202
    dash.drain()
    assert dash.loco.enabled is False


def test_patrol_still_refuses_a_dead_imu(make_hw, monkeypatch, capsys):
    monkeypatch.setattr(C, "DASH_ALLOW_PATROL", True)
    hw = make_hw()
    hw.imu_ok = False
    dash = _dash(hw, monkeypatch, allow_patrol=True)
    dash.submit("mode", mode="patrol")
    dash.drain()
    assert dash.loco.enabled is False
    assert dash.last_result["error"] == "patrol_refused"
    assert "patrol_refused" in capsys.readouterr().out
    assert not any(a["action"] == "forward" for a in hw.dog.action_log)


def test_start_refuses_without_a_token_and_binds_when_configured(make_hw, monkeypatch, capsys):
    hw = make_hw()
    monkeypatch.setattr(C, "DASH_ENABLED", True)
    monkeypatch.setattr(C, "DASH_TOKEN", "")
    dash = Dashboard(_session(hw))
    assert dash.start() is False
    assert dash.thread is None
    assert "dashboard_refused missing REX_DASH_TOKEN" in capsys.readouterr().out

    monkeypatch.setattr(C, "DASH_TOKEN", TOKEN)
    monkeypatch.setattr(C, "DASH_HOST", "0.0.0.0")
    monkeypatch.setattr(C, "DASH_PORT", 8080)
    bound = {}

    def fake_make_server(host, port, app, threaded=True):
        bound["host"] = host
        bound["port"] = port
        assert TOKEN not in repr(app)

        class _Server:
            def serve_forever(self):
                return None

            def shutdown(self):
                bound["down"] = True

        return _Server()

    import werkzeug.serving
    monkeypatch.setattr(werkzeug.serving, "make_server", fake_make_server)
    dash = Dashboard(_session(hw))
    assert dash.start() is True
    dash.thread.join(timeout=2)
    assert bound["host"] == "0.0.0.0"
    assert bound["port"] == 8080
    out = capsys.readouterr().out
    assert "dashboard_listen host=0.0.0.0 port=8080" in out
    assert TOKEN not in out
    dash.stop()


def test_token_is_redacted_from_logs(monkeypatch, capsys):
    secret = "super-secret-token"
    monkeypatch.setenv("REX_DASH_TOKEN", secret)
    C.log(f"leaked {secret} in a line")
    out = capsys.readouterr().out
    assert secret not in out
    assert "***" in out


def test_run_skips_the_dashboard_unless_enabled(monkeypatch, capsys):
    monkeypatch.setattr(main._shutdown, "wait", lambda timeout=None: None)
    import vision_grok
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    args = argparse.Namespace(
        patrol=False, sim=True, allow_voice_patrol=False, seconds=0.05,
        allow_no_imu=False, no_tts=True, no_listen=True, force_first_grok=False,
    )
    assert main.run(args) == 0
    out = capsys.readouterr().out
    assert "dashboard_disabled" in out
    assert "FATAL" not in out


def test_service_and_example_do_not_carry_a_token():
    text = (ROOT / "rex.service").read_text()
    assert "REX_DASH" not in text
    example = (ROOT / ".env.example").read_text()
    assert "REX_DASH_TOKEN=" in example
    for line in example.splitlines():
        if line.startswith("REX_DASH"):
            assert line.split("=", 1)[1].strip() == ""
