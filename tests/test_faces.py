"""Local faces: Pillow match, stranger vs Cory, sim sightings, CLI, dashboard enroll."""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

import brain
import config as C
import faces
import main
import sim
from support import ROOT


def _portrait(left, right) -> Image.Image:
    im = Image.new("RGB", (32, 32), left)
    for x in range(16, 32):
        for y in range(32):
            im.putpixel((x, y), right)
    return im


def _session(hw, tts=False):
    from locomotion import Locomotion
    from safety import SafetyMonitor

    safety = SafetyMonitor(hw)
    loco = Locomotion(hw, safety)
    args = argparse.Namespace(allow_no_imu=False, patrol=False)
    return main.Session(hw, safety, loco, listener=None, tts=tts, args=args)


def test_feature_is_off_and_primary_human_is_cory():
    assert C.FACE_ENABLED is False
    assert C.PRIMARY_HUMAN == "Cory"
    assert brain.PRIMARY_HUMAN == "Cory"
    assert C.FACE_BACKEND == "auto"
    default = Path.home() / ".config" / "pidog" / "faces"
    assert ROOT.resolve() not in default.resolve().parents
    ignored = (ROOT / ".gitignore").read_text()
    assert "faces/" in ignored and "*.npy" in ignored


def test_pillow_matches_the_same_face_and_ignores_a_different_one(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "FACE_BACKEND", "pillow")
    monkeypatch.setattr(C, "FACE_THRESHOLD", 0.35)
    store = faces.FaceStore(tmp_path / "faces")
    cory = _portrait((0, 0, 0), (220, 30, 30))
    other = _portrait((240, 240, 240), (20, 20, 220))
    assert faces.cosine_distance([1, 0], [1, 0]) == 0
    assert faces.cosine_distance([0, 0], [1, 1]) == 1
    assert faces.enroll_image(cory, "Cory", store=store, backend=faces.PillowBackend())
    hits = faces.recognize_image(cory, store=store, backend=faces.PillowBackend())
    assert len(hits) == 1
    assert hits[0].name == "Cory" and hits[0].stranger is False
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    missed = faces.recognize_image(other, store=store, backend=faces.PillowBackend())
    assert missed == []  # pillow does not call an unknown frame a stranger
    assert store.names() == ["Cory"]
    assert store.path.parent == tmp_path / "faces"
    assert not str(store.path.resolve()).startswith(str(ROOT.resolve()))
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert faces.enroll_image(cory, "nope!", store=store) is False


def test_detector_marks_unknowns_as_strangers_and_greets_cory(make_hw, monkeypatch, capsys):
    faces.reset_state()
    # A fresh boot has a small monotonic clock. The cooldown must not treat 0 as "just greeted".
    monkeypatch.setattr(faces.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(C, "FACE_ENABLED", True)
    monkeypatch.setattr(C, "FACE_GREET_COOLDOWN", 1000)
    monkeypatch.setattr(C, "FACE_STRANGER_COOLDOWN", 1000)
    monkeypatch.setattr(C, "PRIMARY_HUMAN", "Cory")

    class Detector:
        name = "fake"
        detects_strangers = True

        def __init__(self):
            self.vec = [1.0, 0.0, 0.0]

        def detect(self, image):
            return [((0.1, 0.2, 0.3, 0.4), list(self.vec))]

    detector = Detector()
    monkeypatch.setattr(faces, "get_backend", lambda: detector)
    store = faces.FaceStore()
    hw = make_hw()
    session = _session(hw)
    blank = Image.new("RGB", (8, 8), (1, 2, 3))
    hits = faces.recognize_image(blank, store=store, backend=detector)
    assert hits[0].stranger is True and hits[0].name == ""
    assert hits[0].bbox == pytest.approx((0.1, 0.2, 0.3, 0.4))
    before = brain.memory()["mood"]["suspicion"]
    faces.observe(session, photo=blank, store=store)
    assert session.person_seen == "stranger"
    out = capsys.readouterr().out
    assert "I don't know you" in out
    assert "face_stranger" in out
    assert store.names() == []
    assert brain.memory()["mood"]["suspicion"] > before
    assert any(ev["kind"] == "stranger" for ev in brain.memory()["recent_events"])

    store.enroll("Cory", [1.0, 0.0, 0.0], "fake")
    faces.observe(session, photo=blank, store=store)
    assert session.person_seen == "Cory"
    assert "Hello Cory" in capsys.readouterr().out
    faces.observe(session, photo=blank, store=store)
    second = capsys.readouterr().out
    assert second.count("Hello Cory") == 0
    assert store.names() == ["Cory"]


def test_sim_people_supply_bbox_without_a_model(make_hw):
    hw = make_hw()
    hw.dog.people.set([
        {"name": "Cory", "bbox": [0.1, 0.2, 0.3, 0.4], "known": True},
        {"name": "shadow", "bbox": [0.0, 0.0, 0.2, 0.2]},
    ])
    hits = faces.recognize_hw(hw, store=faces.FaceStore())
    assert hits[0].name == "Cory" and hits[0].stranger is False
    assert hits[0].bbox == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert hits[0].source == "sim"
    assert hits[1].stranger is True and hits[1].name == ""


def test_cli_enroll_list_recognize_does_not_open_a_dog(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "FACE_BACKEND", "pillow")
    monkeypatch.setattr(C, "FACE_DIR", tmp_path / "faces")
    import hardware

    def _no_second_dog(*_a, **_k):
        raise AssertionError("second Pidog")

    monkeypatch.setattr(hardware, "Hardware", _no_second_dog)
    path = tmp_path / "cory.jpg"
    _portrait((0, 0, 0), (200, 40, 40)).save(path)
    assert faces.main(["enroll", "Cory", "--image", str(path)]) == 0
    assert faces.main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "Cory" in listed and "pillow" in listed
    assert faces.main(["recognize", "--image", str(path)]) == 0
    assert "Cory" in capsys.readouterr().out


def test_dashboard_enroll_button_is_gated(make_hw, monkeypatch):
    monkeypatch.setattr(C, "FACE_BACKEND", "pillow")
    monkeypatch.setattr(C, "FACE_ENABLED", False)
    monkeypatch.setattr(C, "DASH_TOKEN", "test-token-not-a-secret")
    faces.ensure_registered()
    from dashboard import Dashboard
    hw = make_hw()
    hw.dog.frames.push(_portrait((0, 0, 0), (180, 20, 20)))
    dash = Dashboard(_session(hw))
    dash.build_app()
    client = dash.app.test_client()
    headers = {"X-Rex-Token": "test-token-not-a-secret"}
    assert "Enroll from camera" in client.get("/").get_data(as_text=True)
    blocked = client.post("/api/enroll", json={"name": "Cory"}, headers=headers)
    assert blocked.status_code == 403
    assert faces.FaceStore().names() == []
    monkeypatch.setattr(C, "FACE_ENABLED", True)
    dash.build_app()
    client = dash.app.test_client()
    queued = client.post("/api/enroll", json={"name": "Cory"}, headers=headers)
    assert queued.status_code == 202
    assert faces.FaceStore().names() == []
    dash.drain()
    assert faces.FaceStore().names() == ["Cory"]
    status = client.get("/api/status", headers=headers).get_json()
    assert status["known_faces"] == ["Cory"]
    assert status["face_enabled"] is True


def test_auto_prefers_face_recognition_when_it_imports(monkeypatch):
    fr = types.ModuleType("face_recognition")
    monkeypatch.setitem(sys.modules, "face_recognition", fr)
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    assert faces._face_stack_available() is True


def test_dlib_backend_reports_normalized_boxes(monkeypatch):
    monkeypatch.setattr(C, "FACE_BACKEND", "auto")
    monkeypatch.setattr(faces, "_face_stack_available", lambda: True)
    fr = types.ModuleType("face_recognition")
    fr.face_locations = lambda arr: [(1, 9, 8, 2)]
    fr.face_encodings = lambda arr, locs: [[1.0, 0.0]]
    monkeypatch.setitem(sys.modules, "face_recognition", fr)

    class _Arr:
        shape = (10, 10, 3)

    monkeypatch.setattr(faces, "_rgb_array", lambda image: _Arr())
    backend = faces.get_backend()
    assert backend.name == "face_recognition"
    bbox, emb = backend.detect(Image.new("RGB", (10, 10)))[0]
    assert emb == [1.0, 0.0]
    assert bbox == (0.2, 0.1, 0.7, 0.7)


def test_run_greets_cory_from_a_sim_sighting_and_stays_put(monkeypatch, capsys):
    faces.reset_state()
    monkeypatch.setattr(faces.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(C, "FACE_ENABLED", True)
    monkeypatch.setattr(C, "FACE_GREET_COOLDOWN", 1000)
    monkeypatch.setattr(C, "PRIMARY_HUMAN", "Cory")
    monkeypatch.setattr(main._shutdown, "wait", lambda timeout=None: None)
    import vision_grok
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    sim.configure(people=[{"name": "Cory", "known": True, "bbox": [0.4, 0.2, 0.2, 0.5]}])
    args = argparse.Namespace(
        patrol=False, sim=True, allow_voice_patrol=False, seconds=0.05,
        allow_no_imu=False, no_tts=True, no_listen=True, force_first_grok=False,
    )
    assert main.run(args) == 0
    out = capsys.readouterr().out
    assert "Hello Cory" in out
    assert "face_greet name=Cory" in out
    dog = sim.dogs[-1]
    assert not any(action["action"] in {"forward", "backward", "turn_left", "turn_right"} for action in dog.action_log)
    text = (ROOT / "rex.service").read_text()
    assert "faces.py" not in text
    assert "--patrol" not in [line.split("#", 1)[0] for line in text.splitlines() if "ExecStart" in line][0]
