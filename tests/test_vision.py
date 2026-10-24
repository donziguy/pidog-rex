"""Grok throttle, scene diff, and the Ollama fallback. No network."""
from __future__ import annotations

import json

import pytest
from PIL import Image

import brain
import config as C
import vision_grok


def _solid(path, value):
    Image.new("L", (32, 24), value).save(path)
    return path


def _resp(status, payload):
    class Resp:
        def __init__(self):
            self.status_code = status
            self.ok = 200 <= status < 300
            self._payload = payload

        def json(self):
            return self._payload

    return Resp()


_GROK = {
    "has_person": False,
    "has_dog": False,
    "has_animal": False,
    "has_suspicious_object": False,
    "threat_level": "none",
    "scene_summary": "empty shelf",
    "roast": "Nothing here but bad cable management.",
}


class _Clock:
    def __init__(self, t):
        self.t = float(t)

    def time(self):
        return self.t


def test_scene_diff_threshold(tmp_path):
    assert C.SCENE_DIFF_THRESHOLD == 12
    dark = _solid(tmp_path / "dark.png", 0)
    mild = _solid(tmp_path / "mild.png", 11)
    over = _solid(tmp_path / "over.png", 12)
    vision_grok._last_thumb = None
    changed, diff = vision_grok.scene_changed(dark)
    assert changed is True and diff == 255.0
    assert vision_grok._last_thumb is None  # measuring does not move the baseline
    vision_grok._commit_scene()
    changed, diff = vision_grok.scene_changed(mild)
    assert changed is False and diff == pytest.approx(11)
    changed, diff = vision_grok.scene_changed(over)
    assert changed is True and diff == pytest.approx(12)
    # still the dark baseline: a sub-threshold drift is not remembered
    assert vision_grok._last_thumb == vision_grok._thumb(dark)


def test_grok_throttle_keeps_a_change_that_arrives_early(tmp_path, monkeypatch):
    assert C.GROK_MIN_INTERVAL == 60
    clock = _Clock(1_000_000)
    monkeypatch.setattr(vision_grok.time, "time", clock.time)
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("XAI_API_KEY", "test-not-a-real-key")
    calls = []
    body = json.dumps(_GROK)

    def post(url, json=None, timeout=None, headers=None, **kwargs):
        calls.append(url)
        assert headers["Authorization"] == "Bearer test-not-a-real-key"
        return _resp(200, {"choices": [{"message": {"content": body}}]})

    monkeypatch.setattr("requests.post", post)
    first = _solid(tmp_path / "a.png", 0)
    second = _solid(tmp_path / "b.png", 200)
    opened = vision_grok.analyze(first)
    assert opened["_source"] == "grok" and not opened.get("skipped")
    assert len(calls) == 1
    baseline = list(vision_grok._last_thumb)

    clock.t += 10
    held = vision_grok.analyze(second)
    assert held["skipped"] is True and held["reason"] == "throttled"
    assert held["_changed"] is True
    assert len(calls) == 1
    assert vision_grok._last_thumb == baseline  # cooldown must not swallow the change

    clock.t += 49  # 59 s from the call: still inside the hard 60 s floor
    assert vision_grok.analyze(second)["reason"] == "throttled"
    assert len(calls) == 1

    clock.t += 1  # 60 s
    sent = vision_grok.analyze(second)
    assert sent["_source"] == "grok"
    assert len(calls) == 2

    clock.t += 120  # same scene, interval elapsed: nothing to say
    quiet = vision_grok.analyze(second)
    assert quiet["skipped"] is True and quiet["_changed"] is False
    assert len(calls) == 2

    forced = vision_grok.analyze(second, force=True)
    assert forced["_source"] == "grok"
    assert len(calls) == 3


def test_ollama_fallback_when_grok_returns_500(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("XAI_API_KEY", "test-not-a-real-key")
    calls = []
    roast = "The cloud brain tripped over its own paws."

    def post(url, json=None, timeout=None, headers=None, **kwargs):
        calls.append(url)
        if "api.x.ai" in url:
            return _resp(500, {})
        assert "/api/generate" in url
        return _resp(200, {"response": roast})

    monkeypatch.setattr("requests.post", post)
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 30), force=True)
    assert [url for url in calls if "api.x.ai" in url].__len__() == C.GROK_RETRIES == 3
    assert any("/api/generate" in url for url in calls)
    assert result["skipped"] is False
    assert result["_source"] == "ollama"
    assert result["roast"] == roast


def test_grok_auth_failure_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("XAI_API_KEY", "test-not-a-real-key")
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        if "api.x.ai" in url:
            return _resp(401, {})
        return _resp(200, {"response": "Still judging."})

    monkeypatch.setattr("requests.post", post)
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 30), force=True)
    assert len([url for url in calls if "api.x.ai" in url]) == 1
    assert result["_source"] == "ollama"


def test_missing_key_does_not_call_grok(tmp_path, monkeypatch, blocked_network):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_KEY", raising=False)
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 0), force=True)
    assert result["_source"] == "ollama"
    assert result["skipped"] is False
    assert result["roast"]
    assert all("x.ai" not in str(url) for url in blocked_network)


def test_api_errors_do_not_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("XAI_API_KEY", "test-not-a-real-key")

    def post(url, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr("requests.post", post)
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 80), force=True)
    assert result["skipped"] is False
    assert result["_source"] == "ollama"
    assert result["roast"] in {
        "I heard that, meatbag. Try being less suspicious.",
        "Another elite shift guarding bad cable management.",
        "If boredom was a felony, this room gets life.",
        "I am surrounded by glowing rectangles and poor decisions.",
    }


def test_grok_returns_when_shutdown_is_requested(tmp_path, monkeypatch, capsys):
    """An in-flight Grok call must not sit on the socket timeout after Ctrl+C."""
    import threading
    import time

    import abort

    monkeypatch.setenv("XAI_API_KEY", "test-not-a-real-key")
    started = threading.Event()

    def post(url, **kwargs):
        started.set()
        # Stay blocked. The caller has to abandon this thread, not wait for it.
        time.sleep(30)
        return _resp(200, {"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr("requests.post", post)

    def trip():
        assert started.wait(2)
        abort.event.set()

    threading.Thread(target=trip, daemon=True).start()
    t0 = time.monotonic()
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 40), force=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 2
    assert result["skipped"] is True
    assert result["reason"] == "shutdown"
    assert vision_grok._fail_streak == 0
    assert "grok_abandoned" in capsys.readouterr().out


def test_personality_text_is_unchanged():
    assert brain.PERSONALITY.startswith("You are Rex, a grumpy 15-year veteran security dog.")
    assert "Do not mention being an AI." in brain.PERSONALITY
