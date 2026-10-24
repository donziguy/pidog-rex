"""Grok vision with change detection, throttling, retries and an Ollama fallback. Never raises."""
from __future__ import annotations
import base64, io, json, random, time
import config as C
from config import log
import abort
import brain
import grok_oauth

_last_thumb = None
_pending_thumb = None  # latest measurement; committed only when a vision call is spent
_last_call = 0.0
_fail_streak = 0


def _thumb(path):
    from PIL import Image
    with Image.open(path) as im:
        gray = im.convert("L").resize((32, 24))
        # getdata() warns on Pillow >= 10; get_flattened_data is the same pixels.
        pixels = getattr(gray, "get_flattened_data", gray.getdata)()
        return list(pixels)


def scene_changed(path) -> tuple[bool, float]:
    """Diff against the last frame we actually sent for vision. Does not move that baseline."""
    global _pending_thumb
    try:
        t = _thumb(path)
    except Exception as exc:
        log(f"thumb_error={exc}")
        _pending_thumb = None
        return False, 0.0
    _pending_thumb = t
    if _last_thumb is None:
        return True, 255.0
    diff = sum(abs(a - b) for a, b in zip(t, _last_thumb)) / len(t)
    return diff >= C.SCENE_DIFF_THRESHOLD, diff


def _commit_scene() -> None:
    """Remember the frame we are about to spend a vision call on."""
    global _last_thumb, _pending_thumb
    if _pending_thumb is not None:
        _last_thumb = _pending_thumb


def _jpeg_320(path) -> bytes:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w = C.GROK_IMAGE_WIDTH
        im = im.resize((w, int(im.height * w / im.width)))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=70)
        return buf.getvalue()


def _parse(text: str) -> dict:
    text = (text or "").strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:]
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])


def empty(reason: str) -> dict:
    return {"skipped": True, "reason": reason, "has_person": False, "has_dog": False, "has_animal": False,
            "has_suspicious_object": False, "threat_level": "none", "scene_summary": "", "roast": ""}


def due(changed: bool, force: bool = False) -> bool:
    if force:
        return True
    backoff = min(600, 60 * (2 ** max(0, _fail_streak - 1))) if _fail_streak else 0
    return changed and (time.time() - _last_call) >= max(C.GROK_MIN_INTERVAL, backoff)


def _api_payload(img: bytes) -> dict:
    return {"model": C.GROK_MODEL, "max_tokens": 220, "temperature": 0.7, "messages": [
        {"role": "system", "content": brain.PERSONALITY + "\n" + brain.context()},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this security camera frame. Return only the JSON object requested."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}}]}]}


def _sub_payload(img: bytes) -> dict:
    """NoirOS grok-sub body, plus the camera frame as an input image.

    grok-sub is pinned to grok-4.7 on the Responses API. That core has no
    separate vision model, so the same id carries the JPEG.
    """
    return {
        "model": grok_oauth.SUB_MODEL,
        "instructions": brain.PERSONALITY + "\n" + brain.context(),
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analyze this security camera frame. Return only the JSON object requested."},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(img).decode()},
            ],
        }],
        "stream": True,
        "store": False,
    }


def _request_for(auth_path: str, img: bytes) -> tuple[str, dict]:
    if auth_path == "sub":
        return grok_oauth.RESPONSES_URL, _sub_payload(img)
    return C.GROK_URL, _api_payload(img)


def _response_text(auth_path: str, response) -> str:
    if auth_path != "sub":
        return response.json()["choices"][0]["message"]["content"]
    raw = getattr(response, "text", None)
    if not isinstance(raw, str) or not raw:
        raw = json.dumps(response.json())
    parsed = grok_oauth.parse_responses_body(raw)
    text = grok_oauth.extract_output_text(parsed)
    if not text:
        raise ValueError("empty responses output")
    return text


def _accept(parsed: dict, changed: bool) -> dict:
    for key in ("has_person", "has_dog", "has_animal", "has_suspicious_object"):
        parsed[key] = bool(parsed.get(key))
    parsed.setdefault("threat_level", "none")
    parsed["roast"] = brain.sanitize(str(parsed.get("roast", "")), "This shift is so boring even the dust looks unemployed.")
    parsed["scene_summary"] = str(parsed.get("scene_summary", ""))[:200]
    parsed["_changed"] = changed
    parsed["_source"] = "grok"
    return parsed


def analyze(path, force: bool = False) -> dict:
    """Returns an analysis dict. API errors are logged (status only) and fall back to Ollama."""
    global _last_call, _fail_streak
    changed, diff = scene_changed(path)
    if not due(changed, force):
        return {**empty("throttled"), "_changed": changed, "_diff": round(diff, 1)}
    # Commit only once we spend the call. Updating the baseline on a throttled
    # change would swallow it, and Grok would never see a person who walked in
    # during the cooldown.
    _commit_scene()
    chosen = grok_oauth.credentials()
    _last_call = time.time()
    if not chosen:
        log("grok_skip no credentials")
        return _fallback("no_key", changed)
    auth_path, token = chosen
    log(f"grok_auth={auth_path}")
    try:
        img = _jpeg_320(path)
    except Exception as exc:
        log(f"grok_image_error={exc}")
        return _fallback("image", changed)
    url, payload = _request_for(auth_path, img)
    refreshed = False
    attempt = 1
    while attempt <= C.GROK_RETRIES:
        try:
            r = abort.post(url, C.GROK_TIMEOUT, json=payload,
                           headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            log(f"grok_status={r.status_code} attempt={attempt} image_bytes={len(img)} diff={diff:.1f}")
            if r.status_code in (401, 403) and auth_path == "sub" and not refreshed:
                refreshed = True
                try:
                    fresh = grok_oauth.refresh_saved_token()
                except abort.Aborted:
                    log("grok_abandoned")
                    return empty("shutdown")
                if fresh:
                    token = fresh
                    log("grok_auth=sub")
                    continue
                log("grok_auth_expired")
                api = (C.xai_key() or "").strip()
                if api:
                    auth_path = "api"
                    token = api
                    url, payload = _request_for("api", img)
                    log("grok_auth=api")
                    continue
                _fail_streak += 1
                return _fallback("grok_auth_expired", changed)
            if r.status_code in (400, 401, 403, 404):
                break  # not retryable
            if r.ok:
                a = _accept(_parse(_response_text(auth_path, r)), changed)
                _fail_streak = 0
                log(f"grok_ok threat={a['threat_level']} person={a['has_person']} dog={a['has_dog']}")
                return a
        except abort.Aborted:
            # Don't count this as a failure and don't wait out the socket or the retry sleep.
            log("grok_abandoned")
            return empty("shutdown")
        except Exception as exc:
            log(f"grok_exception={type(exc).__name__} attempt={attempt}")
        if not abort.pause(min(8, 2 ** attempt) + random.random(), sleep=time.sleep):
            log("grok_abandoned")
            return empty("shutdown")
        attempt += 1
    _fail_streak += 1
    return _fallback(f"grok_failed streak={_fail_streak}", changed)


def _fallback(reason: str, changed: bool) -> dict:
    log(f"vision_fallback reason={reason} -> ollama mutter")
    a = empty(reason)
    a.update({"skipped": False, "_changed": changed, "_source": "ollama",
              "scene_summary": "camera saw something but the cloud brain is down",
              "roast": brain.idle_mutter("my fancy cloud eyes are offline, so I am judging you blind")})
    return a
