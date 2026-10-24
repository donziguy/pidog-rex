"""Grok subscription OAuth, ported from NoirOS noir-kernel.

The device grant, client id, scope, and token endpoint are the public Grok
CLI client in ``oauth_device.rs``. grok-sub calls the xAI API with that
access token as a Bearer. Inference for the subscription core is
``POST https://api.x.ai/v1/responses`` with model ``grok-4.7``. A credential
check is ``GET https://api.x.ai/v1/models``. Refresh is one form POST to
``https://auth.x.ai/oauth2/token`` and is not retried.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import requests

import abort
import config as C
from config import log

# Public client used by the official Grok CLI and by NoirOS. Not a secret.
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
DEVICE_ENDPOINT = "https://auth.x.ai/oauth2/device/code"
TOKEN_ENDPOINT = "https://auth.x.ai/oauth2/token"
MODELS_URL = "https://api.x.ai/v1/models"
RESPONSES_URL = "https://api.x.ai/v1/responses"
# NoirOS pins grok-sub to this id. It has no separate vision model.
SUB_MODEL = "grok-4.7"

ACCESS_ENV = "GROK_OAUTH_TOKEN"
REFRESH_ENV = "GROK_OAUTH_REFRESH_TOKEN"


class LoginError(Exception):
    """A device login stopped. The message is safe to print."""


def public_detail(body: str) -> str:
    """Drop token-shaped bodies. Status text only, never a credential."""
    text = " ".join((body or "").split())
    lower = text.lower()
    if any(word in lower for word in ("access_token", "refresh_token", "device_code", "bearer ")):
        return "login failed"
    return text[:160] or "login failed"


def credentials() -> tuple[str, str] | None:
    """``(sub|api, bearer)``. ``auto`` prefers a subscription token."""
    mode = (os.environ.get("REX_GROK_AUTH") or "auto").strip().lower()
    if mode not in {"sub", "api", "auto"}:
        mode = "auto"
    oauth = (os.environ.get(ACCESS_ENV) or "").strip()
    api = (C.xai_key() or "").strip()
    if mode == "api":
        return ("api", api) if api else None
    if mode == "sub":
        return ("sub", oauth) if oauth else None
    if oauth:
        return ("sub", oauth)
    if api:
        return ("api", api)
    return None


def parse_device_auth(body: str) -> dict:
    import json
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LoginError("login failed") from exc
    missing = [name for name in ("device_code", "user_code", "verification_uri") if not value.get(name)]
    if missing:
        raise LoginError("login failed")
    return {
        "device_code": str(value["device_code"]),
        "user_code": str(value["user_code"]),
        "verification_uri": str(value["verification_uri"]),
        "verification_uri_complete": value.get("verification_uri_complete") or None,
        "expires_in": int(value.get("expires_in") or 900),
        "interval": int(value.get("interval") or 5),
    }


def parse_poll(body: str) -> tuple[str, str | None, str | None]:
    """Return ``(pending|slow_down|token|failed, access, refresh)``."""
    import json
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return "failed", None, None
    access = value.get("access_token")
    if isinstance(access, str) and access:
        refresh = value.get("refresh_token")
        refresh = refresh if isinstance(refresh, str) and refresh else None
        return "token", access, refresh
    error = value.get("error")
    if error == "authorization_pending":
        return "pending", None, None
    if error == "slow_down":
        return "slow_down", None, None
    return "failed", None, None


def parse_token_grant(body: str) -> tuple[str, str | None]:
    kind, access, refresh = parse_poll(body)
    if kind != "token" or not access:
        raise LoginError("login failed")
    return access, refresh


def request_device_code(post=None) -> dict:
    post = post or requests.post
    try:
        response = post(
            DEVICE_ENDPOINT,
            data={"client_id": CLIENT_ID, "scope": SCOPE},
            timeout=15,
        )
    except Exception as exc:
        raise LoginError("login failed") from exc
    text = getattr(response, "text", "") or ""
    if getattr(response, "status_code", 0) != 200:
        raise LoginError(public_detail(text) if text else "login failed")
    return parse_device_auth(text if text else _json_text(response))


def poll_once(device_code: str, post=None) -> tuple[str, str | None, str | None]:
    post = post or requests.post
    try:
        response = post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": CLIENT_ID,
            },
            timeout=15,
        )
    except Exception:
        return "pending", None, None
    text = getattr(response, "text", "") or _json_text(response)
    return parse_poll(text)


def refresh_access_token(refresh_token: str) -> tuple[str, str | None]:
    """One refresh grant. Raises ``abort.Aborted`` or ``LoginError``."""
    response = abort.post(
        TOKEN_ENDPOINT,
        15,
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        },
    )
    if getattr(response, "status_code", 0) == 429 or not getattr(response, "ok", False):
        raise LoginError("login failed")
    text = getattr(response, "text", "") or _json_text(response)
    return parse_token_grant(text)


def refresh_saved_token() -> str | None:
    """Persist a rotated access token. ``None`` when refresh cannot be used."""
    refresh = (os.environ.get(REFRESH_ENV) or "").strip()
    if not refresh:
        return None
    try:
        access, rotated = refresh_access_token(refresh)
    except abort.Aborted:
        raise
    except Exception:
        return None
    updates = {ACCESS_ENV: access}
    if rotated:
        updates[REFRESH_ENV] = rotated
    try:
        save_env_values(updates)
    except Exception:
        return None
    return access


def save_env_values(updates: dict[str, str], path: Path | None = None) -> None:
    """Replace keys in the env file. Back up first. Mode 600. No logging."""
    path = Path(path or C.ENV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        os.chmod(backup, 0o600)
        existing = path.read_text(encoding="utf-8")
    lines = existing.splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = ""
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    body = "\n".join(out)
    if body and not body.endswith("\n"):
        body += "\n"
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, body.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    for key, value in updates.items():
        os.environ[key] = value


def parse_responses_body(body: str) -> dict | None:
    """Final Responses object from JSON or an SSE ``response.completed`` event."""
    import json
    trimmed = (body or "").lstrip()
    if trimmed.startswith("{"):
        try:
            value = json.loads(trimmed)
        except json.JSONDecodeError:
            return None
        response = value.get("response")
        if isinstance(response, dict):
            return response
        return value if isinstance(value, dict) else None
    found = None
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("type") in {"response.completed", "response.done"}:
            response = event.get("response")
            if isinstance(response, dict):
                found = response
    return found


def extract_output_text(response: dict | None) -> str:
    if not isinstance(response, dict):
        return ""
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "refusal" and isinstance(item.get("refusal"), str):
            parts.append(item["refusal"])
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif part.get("type") == "refusal":
                text = part.get("refusal") if isinstance(part.get("refusal"), str) else part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def probe_models(token: str, get=None) -> int | None:
    """GET the models list. Return the status code, or None if it never answered."""
    get = get or requests.get
    try:
        response = get(MODELS_URL, headers={"Authorization": f"Bearer {token}"}, timeout=15)
    except Exception:
        return None
    return int(getattr(response, "status_code", 0) or 0)


def _json_text(response) -> str:
    import json
    try:
        return json.dumps(response.json())
    except Exception:
        return ""


def run_device_login(post=None, get=None, sleep=None, echo=None) -> int:
    """Device login. ``echo`` receives the verification URL and the user code only."""
    import time
    post = post or requests.post
    sleep = sleep or time.sleep
    echo = echo or (lambda line: print(line, flush=True))
    try:
        auth = request_device_code(post)
    except LoginError as exc:
        echo(str(exc) or "login failed")
        return 1
    # The complete URI is the one-tap link (user code filled in). The code is
    # still printed on its own. The device code and the tokens are not.
    echo(auth["verification_uri_complete"] or auth["verification_uri"])
    echo(auth["user_code"])
    interval = max(1, int(auth["interval"]))
    deadline = time.monotonic() + max(30, min(900, int(auth["expires_in"])))
    while time.monotonic() < deadline:
        sleep(interval)
        kind, access, refresh = poll_once(auth["device_code"], post)
        if kind == "pending":
            continue
        if kind == "slow_down":
            interval = min(30, interval + 5)
            continue
        if kind == "failed" or not access:
            echo("login failed")
            return 1
        updates = {ACCESS_ENV: access}
        if refresh:
            updates[REFRESH_ENV] = refresh
        try:
            save_env_values(updates)
        except Exception:
            echo("login failed")
            return 1
        echo("saved")
        status = probe_models(access, get)
        if status is None:
            echo("models_probe error")
        else:
            echo(f"models_probe status={status}")
        return 0
    echo("login timed out")
    return 1
