"""Grok vision with change detection, throttling, retries and an Ollama fallback. Never raises."""
from __future__ import annotations
import base64, io, json, random, time
import config as C
from config import log
import abort
import brain

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
    key = C.xai_key()
    _last_call = time.time()
    if not key:
        log("grok_skip no XAI_API_KEY")
        return _fallback("no_key", changed)
    try:
        img = _jpeg_320(path)
    except Exception as exc:
        log(f"grok_image_error={exc}")
        return _fallback("image", changed)
    payload = {"model": C.GROK_MODEL, "max_tokens": 220, "temperature": 0.7, "messages": [
        {"role": "system", "content": brain.PERSONALITY + "\n" + brain.context()},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this security camera frame. Return only the JSON object requested."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}}]}]}
    for attempt in range(1, C.GROK_RETRIES + 1):
        try:
            r = abort.post(C.GROK_URL, C.GROK_TIMEOUT, json=payload,
                           headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            log(f"grok_status={r.status_code} attempt={attempt} image_bytes={len(img)} diff={diff:.1f}")
            if r.status_code in (400, 401, 403, 404):
                break  # not retryable
            if r.ok:
                a = _parse(r.json()["choices"][0]["message"]["content"])
                for k in ("has_person", "has_dog", "has_animal", "has_suspicious_object"):
                    a[k] = bool(a.get(k))
                a.setdefault("threat_level", "none")
                a["roast"] = brain.sanitize(str(a.get("roast", "")), "This shift is so boring even the dust looks unemployed.")
                a["scene_summary"] = str(a.get("scene_summary", ""))[:200]
                a["_changed"] = changed; a["_source"] = "grok"
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
    _fail_streak += 1
    return _fallback(f"grok_failed streak={_fail_streak}", changed)


def _fallback(reason: str, changed: bool) -> dict:
    log(f"vision_fallback reason={reason} -> ollama mutter")
    a = empty(reason)
    a.update({"skipped": False, "_changed": changed, "_source": "ollama",
              "scene_summary": "camera saw something but the cloud brain is down",
              "roast": brain.idle_mutter("my fancy cloud eyes are offline, so I am judging you blind")})
    return a
