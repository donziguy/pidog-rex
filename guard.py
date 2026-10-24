"""Shelf guard. Off unless REX_GUARD=1, and disarmed until armed.

An armed guard does not walk. An unknown face (from face recognition) or a
frame-to-frame motion score at or above REX_GUARD_MOTION sends one alert, then
waits REX_GUARD_COOLDOWN seconds. Alerts go to whichever notifiers are
configured: an ntfy.sh topic (REX_NTFY_TOPIC, optional REX_NTFY_TOKEN) and/or
a generic JSON webhook (REX_ALERT_WEBHOOK_URL). Both stay empty in the repo.
The webhook body is ``{"title", "message", "source", "image_base64"}``.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import config as C
from config import log


def _pixels(image):
    im = image.convert("L").resize((32, 24))
    data = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
    return list(data)


def motion_score(prev, current) -> float:
    """Mean absolute grayscale difference, 0..255, on a 32x24 thumbnail."""
    if prev is None or current is None:
        return 0.0
    a, b = _pixels(prev), _pixels(current)
    if not a or len(a) != len(b):
        return 0.0
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b)) / len(a)


def _as_image(photo):
    if photo is None:
        return None
    if hasattr(photo, "convert"):
        return photo.convert("RGB")
    path = Path(str(photo))
    if not path.exists():
        return None
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def _jpeg(photo, image) -> bytes:
    if photo is not None and not hasattr(photo, "convert"):
        path = Path(str(photo))
        try:
            data = path.read_bytes()
        except Exception:
            data = b""
        if data:
            return data
    if image is None:
        return b""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _is_stranger(hit) -> bool:
    if hit is None:
        return False
    if isinstance(hit, dict):
        return bool(hit.get("stranger"))
    return bool(getattr(hit, "stranger", False))


class NtfyNotifier:
    name = "ntfy"

    def send(self, title: str, message: str, image: bytes) -> None:
        import requests
        topic = (C.NTFY_TOPIC or "").strip().strip("/")
        if not topic:
            return
        url = f"{C.NTFY_SERVER}/{topic}"
        headers = {"Title": title[:200], "Tags": "warning,dog"}
        if C.NTFY_TOKEN:
            headers["Authorization"] = f"Bearer {C.NTFY_TOKEN}"
        if image:
            headers["Filename"] = "rex.jpg"
            requests.post(url, data=image, headers=headers, timeout=8)
        else:
            requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=8)


class WebhookNotifier:
    name = "webhook"

    def send(self, title: str, message: str, image: bytes) -> None:
        import base64
        import requests
        if not C.ALERT_WEBHOOK_URL:
            return
        body = {"title": title, "message": message, "source": "rex"}
        if image:
            body["image_base64"] = base64.b64encode(image).decode("ascii")
        # The URL can carry a secret. Do not log it.
        requests.post(C.ALERT_WEBHOOK_URL, json=body, timeout=8)


def build_notifiers() -> list:
    out = []
    if (C.NTFY_TOPIC or "").strip():
        out.append(NtfyNotifier())
    if (C.ALERT_WEBHOOK_URL or "").strip():
        out.append(WebhookNotifier())
    return out


class Guard:
    def __init__(self, session, notifiers=None):
        self.session = session
        self.notifiers = list(build_notifiers() if notifiers is None else notifiers)
        self.last_alert = 0.0
        self.prev = None
        self.last_reason = ""
        self.armed = bool(C.GUARD_ENABLED and C.GUARD_ARMED)
        self._wrapped = False
        if self.armed:
            self._shelf_only("startup")

    def set_armed(self, on: bool) -> bool:
        if not C.GUARD_ENABLED:
            log("guard_ignored disabled")
            return False
        self.armed = bool(on)
        if self.armed:
            self._shelf_only("armed")
        log(f"guard_armed={int(self.armed)}")
        return self.armed

    def allow_walk(self) -> bool:
        return not (C.GUARD_ENABLED and self.armed)

    def _shelf_only(self, why: str) -> None:
        loco = self.session.loco
        if getattr(loco, "enabled", False):
            loco.disable(f"guard_{why}")
        if self._wrapped:
            return
        orig = loco.enable

        def enable(reason: str = "") -> bool:
            if self.armed:
                log("patrol_refused guard_armed shelf_only")
                return False
            return orig(reason)

        loco.enable = enable
        self._wrapped = True

    def tick(self, photo=None, hits=None):
        """Return ``unknown_person``, ``motion``, ``cooldown``, or None. Never walks."""
        if not C.GUARD_ENABLED or not self.armed:
            return None
        self._shelf_only("tick")
        image = _as_image(photo)
        if hits is None:
            hits = list(getattr(self.session, "face_hits", []) or [])
        score = motion_score(self.prev, image) if image is not None and self.prev is not None else 0.0
        if image is not None:
            self.prev = image
        if any(_is_stranger(hit) for hit in hits):
            reason = "unknown_person"
        elif score >= C.GUARD_MOTION:
            reason = "motion"
        else:
            return None
        now = time.monotonic()
        if now - self.last_alert < C.GUARD_COOLDOWN:
            log(f"guard_cooldown reason={reason}")
            return "cooldown"
        self.last_alert = now
        self.last_reason = reason
        jpeg = _jpeg(photo, image)
        title = f"Rex alert: {reason.replace('_', ' ')}"
        message = f"Rex guard saw {reason.replace('_', ' ')}."
        log(f"guard_alert reason={reason} notifiers={len(self.notifiers)} bytes={len(jpeg)}")
        for notifier in self.notifiers:
            try:
                notifier.send(title, message, jpeg)
            except Exception as exc:
                log(f"guard_notify_error name={getattr(notifier, 'name', '?')} err={type(exc).__name__}")
        return reason


def _ext(dash) -> None:
    from flask import jsonify, request

    def do_guard(payload):
        guard = getattr(dash.session, "guard", None)
        if guard is None:
            return {"armed": False}
        armed = bool(payload.get("armed"))
        return {"armed": bool(guard.set_armed(armed))}

    dash.on("guard", do_guard)

    def status(body):
        guard = getattr(dash.session, "guard", None)
        body["guard_enabled"] = bool(C.GUARD_ENABLED)
        body["guard_armed"] = bool(guard and guard.armed)

    dash.status_hooks.append(status)

    @dash.app.post("/api/guard")
    def api_guard():
        if not C.GUARD_ENABLED:
            return jsonify({"ok": False, "error": "guard_disabled"}), 403
        body = request.get_json(silent=True) or {}
        if "armed" not in body:
            return jsonify({"ok": False, "error": "bad_guard"}), 400
        dash.submit("guard", armed=bool(body.get("armed")))
        return jsonify({"ok": True, "queued": "guard"}), 202


def ensure_registered() -> None:
    import dashboard
    if _ext not in dashboard.extensions:
        dashboard.extensions.append(_ext)
