"""Local web dashboard for Rex. Off unless REX_DASH=1, and it will not listen without a token.

The page shows a camera snapshot, battery / tilt / distance / mode / last Grok line, and
the log tail. Controls (speak, head, shelf/patrol) are queued and applied on the main
thread so only that thread moves the servos. STOP trips safety immediately, the same
way the voice command does. Patrol is refused unless REX_DASH_ALLOW_PATROL=1, and even
then Locomotion.enable still demands a working IMU and the safety thread still watches
every step.
"""
from __future__ import annotations

import hmac
import queue
import threading
from pathlib import Path

import config as C
from config import log

# Later features (faces, guard) append a callable(Dashboard) to add routes.
extensions: list = []

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rex</title>
<style>
  :root { color-scheme: dark; }
  body { font: 16px/1.4 system-ui, sans-serif; margin: 0; background: #121416; color: #e8e6e3; }
  header, main { padding: 16px; max-width: 920px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin: 0 0 8px; }
  img { width: 100%; max-width: 640px; background: #000; border-radius: 8px; }
  .row { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
  .card { background: #1e2226; border-radius: 8px; padding: 12px 14px; min-width: 140px; }
  .card b { display: block; font-size: 0.75rem; letter-spacing: 0.04em; color: #9aa; }
  button, input { font: inherit; }
  button { background: #2c333a; color: inherit; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
  button.stop { background: #a32020; font-size: 1.4rem; font-weight: 700; padding: 16px 28px; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  pre { background: #1e2226; padding: 12px; border-radius: 8px; max-height: 240px; overflow: auto; white-space: pre-wrap; }
  label { display: block; margin: 8px 0; }
  .err { color: #f88; }
</style>
</head>
<body>
<header>
  <h1>Rex</h1>
  <label>Token <input id="token" type="password" autocomplete="current-password"></label>
  <p id="msg" class="err"></p>
</header>
<main>
  <img id="snap" alt="camera snapshot">
  <div class="row">
    <div class="card"><b>BATTERY</b><span id="battery">–</span></div>
    <div class="card"><b>TILT</b><span id="tilt">–</span></div>
    <div class="card"><b>DISTANCE</b><span id="distance">–</span></div>
    <div class="card"><b>MODE</b><span id="mode">–</span></div>
  </div>
  <div class="card"><b>LAST GROK LINE</b><span id="grok">–</span></div>
  <div class="row">
    <button class="stop" id="stop" type="button">STOP</button>
    <button id="shelf" type="button">Shelf</button>
    <button id="patrol" type="button">Patrol</button>
    <button id="guard-on" type="button">Arm guard</button>
    <button id="guard-off" type="button">Disarm</button>
  </div>
  <form id="speak-form">
    <label>Speak <input id="speak" type="text" maxlength="200"></label>
    <button type="submit">Say it</button>
  </form>
  <form id="head-form">
    <label>Head yaw <input id="yaw" type="number" value="0" min="-60" max="60"></label>
    <label>Head roll <input id="roll" type="number" value="0" min="-60" max="60"></label>
    <label>Head pitch <input id="pitch" type="number" value="18" min="-60" max="60"></label>
    <button type="submit">Move head</button>
  </form>
  <form id="enroll-form">
    <label>Enroll face <input id="enroll-name" type="text" maxlength="40" placeholder="Cory"></label>
    <button type="submit">Enroll from camera</button>
  </form>
  <h2>Log</h2>
  <pre id="log"></pre>
</main>
<script>
const tokenInput = document.getElementById("token");
tokenInput.value = sessionStorage.getItem("rexToken") || "";
tokenInput.addEventListener("change", () => sessionStorage.setItem("rexToken", tokenInput.value));
function token() { return tokenInput.value.trim(); }
function authHeaders() { return {"X-Rex-Token": token(), "Content-Type": "application/json"}; }
async function api(path, opts) {
  const res = await fetch(path, Object.assign({headers: authHeaders()}, opts || {}));
  const body = await res.json().catch(() => ({}));
  const msg = document.getElementById("msg");
  msg.textContent = res.ok ? "" : (body.error || ("HTTP " + res.status));
  if (!res.ok) throw new Error(msg.textContent);
  return body;
}
async function refresh() {
  if (!token()) return;
  try {
    const s = await api("/api/status");
    document.getElementById("battery").textContent = s.battery == null ? "–" : s.battery;
    document.getElementById("tilt").textContent = s.tilt == null ? "–" : (s.tilt + "°");
    document.getElementById("distance").textContent = s.distance == null ? "–" : (s.distance + " cm");
    document.getElementById("mode").textContent = s.mode + (s.locked ? " (locked)" : "");
    document.getElementById("grok").textContent = s.last_grok || "–";
    document.getElementById("patrol").disabled = !s.allow_patrol;
    document.getElementById("snap").src = "/api/snapshot?token=" + encodeURIComponent(token()) + "&t=" + Date.now();
    const log = await api("/api/log");
    document.getElementById("log").textContent = (log.lines || []).join(String.fromCharCode(10));
  } catch (e) {}
}
document.getElementById("stop").onclick = () => api("/api/stop", {method: "POST", body: "{}"});
document.getElementById("shelf").onclick = () => api("/api/mode", {method: "POST", body: JSON.stringify({mode: "shelf"})});
document.getElementById("patrol").onclick = () => api("/api/mode", {method: "POST", body: JSON.stringify({mode: "patrol"})});
document.getElementById("guard-on").onclick = () => api("/api/guard", {method: "POST", body: JSON.stringify({armed: true})});
document.getElementById("guard-off").onclick = () => api("/api/guard", {method: "POST", body: JSON.stringify({armed: false})});
document.getElementById("speak-form").onsubmit = (ev) => {
  ev.preventDefault();
  api("/api/speak", {method: "POST", body: JSON.stringify({text: document.getElementById("speak").value})});
};
document.getElementById("enroll-form").onsubmit = (ev) => {
  ev.preventDefault();
  api("/api/enroll", {method: "POST", body: JSON.stringify({name: document.getElementById("enroll-name").value})});
};
document.getElementById("head-form").onsubmit = (ev) => {
  ev.preventDefault();
  api("/api/head", {method: "POST", body: JSON.stringify({
    yaw: Number(document.getElementById("yaw").value),
    roll: Number(document.getElementById("roll").value),
    pitch: Number(document.getElementById("pitch").value)
  })});
};
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""


def token_ok(presented: str) -> bool:
    """True only when ``presented`` matches the configured token. Empty never matches."""
    want = C.DASH_TOKEN or ""
    got = presented or ""
    if not want or not got:
        return False
    try:
        return hmac.compare_digest(got.encode("utf-8"), want.encode("utf-8"))
    except Exception:
        return False


def _presented_token(request) -> str:
    header = request.headers.get("X-Rex-Token", "")
    if header:
        return header
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.args.get("token", "") or ""


def tail_log(path: Path | None = None, lines: int | None = None) -> list[str]:
    path = Path(path or C.LOG_PATH)
    n = lines if lines is not None else C.DASH_LOG_LINES
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return text.splitlines()[-n:]


class Dashboard:
    def __init__(self, session):
        self.session = session
        self.commands: queue.Queue = queue.Queue()
        self.handlers: dict = {}
        self.app = None
        self.thread = None
        self._server = None
        self.last_result = {}

    @property
    def hw(self):
        return self.session.hw

    @property
    def safety(self):
        return self.session.safety

    @property
    def loco(self):
        return self.session.loco

    def on(self, kind: str, fn) -> None:
        """Register a main-thread handler for a queued command kind."""
        self.handlers[kind] = fn

    def submit(self, kind: str, **payload) -> None:
        self.commands.put({"kind": kind, "payload": payload})

    def drain(self) -> int:
        """Apply queued commands. Call this from the main loop only."""
        n = 0
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return n
            self._apply(cmd)
            n += 1

    def _apply(self, cmd: dict) -> None:
        kind = cmd.get("kind")
        payload = cmd.get("payload") or {}
        handler = self.handlers.get(kind)
        if handler is not None:
            try:
                self.last_result = {"kind": kind, "ok": True, "result": handler(payload)}
            except Exception as exc:
                log(f"dashboard_command_error kind={kind} err={type(exc).__name__}")
                self.last_result = {"kind": kind, "ok": False, "error": type(exc).__name__}
            return
        if kind == "speak":
            import audio
            audio.speak(str(payload.get("text") or "")[:200], self.session.tts)
            self.last_result = {"kind": kind, "ok": True}
        elif kind == "head":
            self.hw.move_head(payload.get("yaw", 0), payload.get("roll", 0), payload.get("pitch"), speed=30)
            self.last_result = {"kind": kind, "ok": True, "head": dict(self.hw.head)}
        elif kind == "mode":
            self.last_result = self._set_mode(str(payload.get("mode") or ""))
        else:
            log(f"dashboard_unknown_command kind={kind}")
            self.last_result = {"kind": kind, "ok": False, "error": "unknown"}

    def _set_mode(self, mode: str) -> dict:
        if mode == "shelf":
            self.loco.disable("dashboard")
            return {"kind": "mode", "ok": True, "mode": "shelf"}
        if mode == "patrol":
            if not C.DASH_ALLOW_PATROL:
                log("dashboard_patrol_refused REX_DASH_ALLOW_PATROL is off")
                return {"kind": "mode", "ok": False, "error": "patrol_not_allowed"}
            self.loco.allow_no_imu = bool(getattr(self.session.args, "allow_no_imu", False))
            ok = self.loco.enable("dashboard")
            return {"kind": "mode", "ok": bool(ok), "mode": "patrol" if ok else "shelf",
                    "error": None if ok else "patrol_refused"}
        return {"kind": "mode", "ok": False, "error": "bad_mode"}

    def status(self) -> dict:
        last = dict(self.safety.last)
        try:
            import brain
            mem = brain.memory()
            grok = (mem.get("last_roast") or mem.get("last_scene_summary") or "").strip()
        except Exception:
            grok = ""
        mode = "patrol" if self.loco.enabled else "shelf"
        return {
            "battery": last.get("battery"),
            "tilt": last.get("tilt"),
            "pitch": last.get("pitch"),
            "roll": last.get("roll"),
            "distance": last.get("distance"),
            "touch": last.get("touch"),
            "mode": mode,
            "locked": bool(self.safety.locked.is_set()),
            "reason": self.safety.reason,
            "last_grok": grok,
            "head": dict(getattr(self.hw, "head", {}) or {}),
            "allow_patrol": bool(C.DASH_ALLOW_PATROL),
            "imu_ok": bool(getattr(self.hw, "imu_ok", False)),
            "imu_stale": bool(getattr(self.hw, "imu_stale", False)),
        }

    def build_app(self):
        from flask import Flask, Response, jsonify, request

        app = Flask(__name__)
        app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False
        dash = self

        def guard():
            if request.path == "/" and request.method == "GET":
                return None
            if token_ok(_presented_token(request)):
                return None
            return jsonify({"error": "unauthorized"}), 401

        app.before_request(guard)

        @app.get("/")
        def home():
            return Response(_PAGE, mimetype="text/html")

        @app.get("/api/status")
        def api_status():
            body = dash.status()
            for fn in list(getattr(dash, "status_hooks", [])):
                try:
                    fn(body)
                except Exception as exc:
                    log(f"dashboard_status_hook_error={type(exc).__name__}")
            return jsonify(body)

        @app.get("/api/snapshot")
        def api_snapshot():
            path = dash.hw.photo()
            if path is None or not Path(path).exists():
                return jsonify({"error": "no_snapshot"}), 404
            data = Path(path).read_bytes()
            return Response(data, mimetype="image/jpeg")

        @app.get("/api/log")
        def api_log():
            return jsonify({"lines": tail_log()})

        @app.post("/api/speak")
        def api_speak():
            text = str((request.get_json(silent=True) or {}).get("text") or "").strip()
            if not text:
                return jsonify({"ok": False, "error": "empty"}), 400
            dash.submit("speak", text=text[:200])
            return jsonify({"ok": True, "queued": "speak"}), 202

        @app.post("/api/head")
        def api_head():
            body = request.get_json(silent=True) or {}
            try:
                yaw = int(body.get("yaw", 0))
                roll = int(body.get("roll", 0))
                pitch = int(body.get("pitch", C.HEAD_PITCH_DEFAULT))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "bad_head"}), 400
            dash.submit("head", yaw=yaw, roll=roll, pitch=pitch)
            return jsonify({"ok": True, "queued": "head"}), 202

        @app.post("/api/mode")
        def api_mode():
            mode = str((request.get_json(silent=True) or {}).get("mode") or "")
            if mode not in {"shelf", "patrol"}:
                return jsonify({"ok": False, "error": "bad_mode"}), 400
            if mode == "patrol" and not C.DASH_ALLOW_PATROL:
                log("dashboard_patrol_refused REX_DASH_ALLOW_PATROL is off")
                return jsonify({"ok": False, "error": "patrol_not_allowed"}), 403
            dash.submit("mode", mode=mode)
            return jsonify({"ok": True, "queued": mode}), 202

        @app.post("/api/stop")
        def api_stop():
            # Immediate, like the voice listener's on_stop. Do not wait for drain().
            dash.safety.trip("dashboard_stop")
            dash.loco.disable("dashboard_stop")
            return jsonify({"ok": True, "stopped": True})

        self.app = app
        self.status_hooks = []
        for ext in extensions:
            ext(self)
        return app

    def start(self) -> bool:
        if not C.DASH_ENABLED:
            log("dashboard_disabled")
            return False
        if not C.DASH_TOKEN:
            log("dashboard_refused missing REX_DASH_TOKEN")
            return False
        try:
            self.build_app()
        except ImportError:
            log("dashboard_refused flask_missing")
            return False
        self.thread = threading.Thread(target=self._serve, name="dashboard", daemon=True)
        self.thread.start()
        return True

    def _serve(self) -> None:
        from werkzeug.serving import make_server
        try:
            self._server = make_server(C.DASH_HOST, int(C.DASH_PORT), self.app, threaded=True)
        except Exception as exc:
            log(f"dashboard_bind_error={type(exc).__name__}")
            return
        log(f"dashboard_listen host={C.DASH_HOST} port={C.DASH_PORT}")
        try:
            self._server.serve_forever()
        except Exception as exc:
            log(f"dashboard_serve_error={type(exc).__name__}")

    def stop(self) -> None:
        server = self._server
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
        thread = self.thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
