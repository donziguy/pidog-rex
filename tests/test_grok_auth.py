"""Subscription OAuth: preferred over an API key, refresh once, no secrets in logs."""
from __future__ import annotations

import json
import os
import stat

import config as C
import grok_oauth
import vision_grok
from test_vision import _GROK, _solid


def _resp(status, payload, text=None):
    class Resp:
        def __init__(self):
            self.status_code = status
            self.ok = 200 <= status < 300
            self.text = text if text is not None else json.dumps(payload)
            self._payload = payload

        def json(self):
            return self._payload

    return Resp()


def _sub_body(payload=_GROK):
    return {
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": json.dumps(payload)}],
        }]
    }


def test_poll_outcomes_match_the_device_grant():
    assert grok_oauth.parse_poll('{"error":"authorization_pending"}')[0] == "pending"
    assert grok_oauth.parse_poll('{"error":"slow_down"}')[0] == "slow_down"
    kind, access, refresh = grok_oauth.parse_poll(
        '{"access_token":"at-1","refresh_token":"rt-1","expires_in":3600}'
    )
    assert kind == "token" and access == "at-1" and refresh == "rt-1"
    assert grok_oauth.parse_token_grant('{"access_token":"only-at"}') == ("only-at", None)
    assert grok_oauth.CLIENT_ID == "b1a00492-073a-47ea-816f-4c329264a828"
    assert grok_oauth.SCOPE == "openid profile email offline_access grok-cli:access api:access"
    assert grok_oauth.DEVICE_ENDPOINT == "https://auth.x.ai/oauth2/device/code"
    assert grok_oauth.TOKEN_ENDPOINT == "https://auth.x.ai/oauth2/token"
    assert grok_oauth.SUB_MODEL == "grok-4.7"
    assert grok_oauth.RESPONSES_URL == "https://api.x.ai/v1/responses"
    assert grok_oauth.MODELS_URL == "https://api.x.ai/v1/models"


def test_responses_sse_completed_event_is_the_answer():
    body = "\n".join([
        'data: {"type":"response.output_text.delta","delta":"nope"}',
        'data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"{\\"ok\\":true}"}]}]}}',
    ])
    parsed = grok_oauth.parse_responses_body(body)
    assert grok_oauth.extract_output_text(parsed) == '{"ok":true}'


def test_oauth_tokens_are_redacted(monkeypatch, capsys):
    access = "oauth-access-token-value"
    refresh = "oauth-refresh-token-value"
    monkeypatch.setenv("GROK_OAUTH_TOKEN", access)
    monkeypatch.setenv("GROK_OAUTH_REFRESH_TOKEN", refresh)
    C.log(f"leak {access} and {refresh}")
    out = capsys.readouterr().out
    assert access not in out and refresh not in out
    assert "***" in out


def test_subscription_is_preferred_for_vision(tmp_path, monkeypatch, capsys):
    access = "sub-access-token-value"
    api = "api-key-secret-value"
    monkeypatch.setenv("GROK_OAUTH_TOKEN", access)
    monkeypatch.setenv("GROK_OAUTH_REFRESH_TOKEN", "sub-refresh-token-value")
    monkeypatch.setenv("XAI_API_KEY", api)
    monkeypatch.delenv("REX_GROK_AUTH", raising=False)
    calls = []

    def post(url, json=None, data=None, timeout=None, headers=None, **kwargs):
        calls.append((url, json, data, headers))
        assert headers["Authorization"] == f"Bearer {access}"
        return _resp(200, _sub_body())

    monkeypatch.setattr("requests.post", post)
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 40), force=True)
    assert result["_source"] == "grok"
    assert result["roast"] == _GROK["roast"]
    assert len(calls) == 1
    url, body, data, _headers = calls[0]
    assert url == grok_oauth.RESPONSES_URL
    assert data is None
    assert body["model"] == "grok-4.7"
    assert body["stream"] is True and body["store"] is False
    content = body["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    out = capsys.readouterr().out
    assert "grok_auth=sub" in out
    assert "grok_auth=api" not in out
    assert access not in out and api not in out


def test_refresh_on_401_persists_and_retries(tmp_path, monkeypatch, capsys):
    old_access = "old-access-token-value"
    old_refresh = "old-refresh-token-value"
    new_access = "new-access-token-value"
    new_refresh = "new-refresh-token-value"
    api = "api-key-secret-value"
    env = tmp_path / "security_dog.env"
    env.write_text(
        "\n".join([
            "# keep",
            f"XAI_API_KEY={api}",
            f"GROK_OAUTH_TOKEN={old_access}",
            "PIDOG_OLLAMA=0",
            f"GROK_OAUTH_REFRESH_TOKEN={old_refresh}",
            "",
        ])
    )
    os.chmod(env, 0o644)
    monkeypatch.setattr(C, "ENV_PATH", env)
    monkeypatch.setenv("GROK_OAUTH_TOKEN", old_access)
    monkeypatch.setenv("GROK_OAUTH_REFRESH_TOKEN", old_refresh)
    monkeypatch.setenv("XAI_API_KEY", api)
    monkeypatch.delenv("REX_GROK_AUTH", raising=False)
    calls = []

    def post(url, json=None, data=None, timeout=None, headers=None, **kwargs):
        calls.append((url, json, data, (headers or {}).get("Authorization")))
        if url == grok_oauth.TOKEN_ENDPOINT:
            assert data["grant_type"] == "refresh_token"
            assert data["client_id"] == grok_oauth.CLIENT_ID
            assert data["refresh_token"] == old_refresh
            return _resp(200, {"access_token": new_access, "refresh_token": new_refresh})
        assert url == grok_oauth.RESPONSES_URL
        if len([item for item in calls if item[0] == url]) == 1:
            return _resp(401, {"error": "unauthorized"})
        assert headers["Authorization"] == f"Bearer {new_access}"
        return _resp(200, _sub_body())

    monkeypatch.setattr("requests.post", post)
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 12), force=True)
    assert result["_source"] == "grok"
    assert [item[0] for item in calls] == [
        grok_oauth.RESPONSES_URL,
        grok_oauth.TOKEN_ENDPOINT,
        grok_oauth.RESPONSES_URL,
    ]
    text = env.read_text()
    assert f"GROK_OAUTH_TOKEN={new_access}" in text
    assert f"GROK_OAUTH_REFRESH_TOKEN={new_refresh}" in text
    assert f"XAI_API_KEY={api}" in text
    assert "PIDOG_OLLAMA=0" in text
    assert text.splitlines()[0] == "# keep"
    assert old_access not in text and old_refresh not in text
    backup = tmp_path / "security_dog.env.bak"
    assert old_access in backup.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert not (tmp_path / "security_dog.env.tmp").exists()
    out = capsys.readouterr().out
    assert "grok_auth=sub" in out
    assert "grok_auth_expired" not in out
    for secret in (old_access, old_refresh, new_access, new_refresh, api):
        assert secret not in out


def test_refresh_failure_falls_back_to_the_api_key(tmp_path, monkeypatch, capsys):
    access = "sub-access-token-value"
    refresh = "sub-refresh-token-value"
    api = "api-key-secret-value"
    leaked = "should-not-leak-token-value"
    monkeypatch.setenv("GROK_OAUTH_TOKEN", access)
    monkeypatch.setenv("GROK_OAUTH_REFRESH_TOKEN", refresh)
    monkeypatch.setenv("XAI_API_KEY", api)
    monkeypatch.setenv("REX_GROK_AUTH", "sub")
    calls = []

    def post(url, data=None, timeout=None, headers=None, **kwargs):
        calls.append((url, (headers or {}).get("Authorization"), data))
        if url == grok_oauth.TOKEN_ENDPOINT:
            return _resp(400, {"error": "invalid_grant", "access_token": leaked})
        if url == grok_oauth.RESPONSES_URL:
            return _resp(401, {"error": "unauthorized"})
        assert url == C.GROK_URL
        assert headers["Authorization"] == f"Bearer {api}"
        return _resp(200, {"choices": [{"message": {"content": json.dumps(_GROK)}}]})

    monkeypatch.setattr("requests.post", post)
    result = vision_grok.analyze(_solid(tmp_path / "frame.png", 80), force=True)
    assert result["_source"] == "grok"
    assert [item[0] for item in calls] == [
        grok_oauth.RESPONSES_URL,
        grok_oauth.TOKEN_ENDPOINT,
        C.GROK_URL,
    ]
    out = capsys.readouterr().out
    assert "grok_auth_expired" in out
    assert "grok_auth=sub" in out
    assert "grok_auth=api" in out
    for secret in (access, refresh, api, leaked):
        assert secret not in out


def test_api_mode_ignores_the_subscription_token(tmp_path, monkeypatch, capsys):
    access = "sub-access-token-value"
    api = "api-key-secret-value"
    monkeypatch.setenv("GROK_OAUTH_TOKEN", access)
    monkeypatch.setenv("XAI_API_KEY", api)
    monkeypatch.setenv("REX_GROK_AUTH", "api")
    calls = []

    def post(url, timeout=None, headers=None, **kwargs):
        calls.append(url)
        assert headers["Authorization"] == f"Bearer {api}"
        assert kwargs["json"]["model"] == C.GROK_MODEL
        return _resp(200, {"choices": [{"message": {"content": json.dumps(_GROK)}}]})

    monkeypatch.setattr("requests.post", post)
    assert vision_grok.analyze(_solid(tmp_path / "frame.png", 3), force=True)["_source"] == "grok"
    assert calls == [C.GROK_URL]
    out = capsys.readouterr().out
    assert "grok_auth=api" in out
    assert access not in out and api not in out


def test_device_login_prints_the_code_and_replaces_keys(tmp_path, monkeypatch, capsys):
    env = tmp_path / "security_dog.env"
    old_access = "old-access-token-value"
    env.write_text(f"# keep\nXAI_API_KEY=api-key-secret-value\nGROK_OAUTH_TOKEN={old_access}\n")
    monkeypatch.setattr(C, "ENV_PATH", env)
    device = "dc-secret-value"
    access = "new-access-token-value"
    refresh = "new-refresh-token-value"
    seen = []

    def post(url, data=None, timeout=None, **kwargs):
        seen.append((url, dict(data or {})))
        if url == grok_oauth.DEVICE_ENDPOINT:
            body = {
                "device_code": device,
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://accounts.x.ai/oauth2/device",
                "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH",
                "expires_in": 60,
                "interval": 1,
            }
            return _resp(200, body)
        assert data["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
        assert data["client_id"] == grok_oauth.CLIENT_ID
        assert data["device_code"] == device
        return _resp(200, {"access_token": access, "refresh_token": refresh})

    def get(url, headers=None, timeout=None, **kwargs):
        assert url == grok_oauth.MODELS_URL
        assert headers["Authorization"] == f"Bearer {access}"
        return _resp(200, {"data": []})

    assert grok_oauth.run_device_login(post=post, get=get, sleep=lambda _s: None) == 0
    assert seen[0][0] == grok_oauth.DEVICE_ENDPOINT
    assert seen[0][1]["client_id"] == grok_oauth.CLIENT_ID
    assert seen[0][1]["scope"] == grok_oauth.SCOPE
    out = capsys.readouterr().out
    assert "https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH" in out
    assert "ABCD-EFGH" in out
    assert "saved" in out
    assert "models_probe status=200" in out
    for secret in (device, access, refresh, "api-key-secret-value"):
        assert secret not in out
    text = env.read_text()
    assert text.splitlines()[0] == "# keep"
    assert "XAI_API_KEY=api-key-secret-value" in text
    assert f"GROK_OAUTH_TOKEN={access}" in text
    assert f"GROK_OAUTH_REFRESH_TOKEN={refresh}" in text
    assert old_access not in text
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert old_access in (tmp_path / "security_dog.env.bak").read_text()
