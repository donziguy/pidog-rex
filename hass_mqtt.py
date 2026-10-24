"""Home Assistant MQTT discovery. Off unless REX_MQTT=1, and only if REX_MQTT_HOST is set.

There is no default broker. Discovery advertises:

* sensors: battery, mode, person_seen
* switches: guard arm, mute
* buttons: bark, speak (a PRESS says "Rex here."; any other payload is spoken as text)

The password is never logged. Guard arm goes through Guard.set_armed, so a disarmed
feature and shelf-only rules still apply.
"""
from __future__ import annotations

import json

import audio
import config as C
from config import log

_DEVICE = {
    "identifiers": ["pidog-rex"],
    "name": "Rex",
    "model": "SunFounder PiDog",
    "manufacturer": "Rex",
}


def _mqtt_module():
    import paho.mqtt.client as mqtt
    return mqtt


def mode_name(session) -> str:
    follow = getattr(session, "follow", None)
    if follow is not None and getattr(follow, "active", False):
        return "follow"
    if getattr(session.loco, "enabled", False):
        return "patrol"
    guard = getattr(session, "guard", None)
    if guard is not None and getattr(guard, "armed", False):
        return "guard"
    return "shelf"


class HomeAssistant:
    def __init__(self, session, client=None):
        self.session = session
        self.client = client
        self.connected = False
        self.base = C.MQTT_PREFIX or "rex"
        self._last = None

    def command_topics(self) -> list[str]:
        return [f"{self.base}/cmd/{name}" for name in ("guard", "mute", "bark", "speak")]

    def start(self) -> bool:
        if not C.MQTT_ENABLED:
            log("mqtt_disabled")
            return False
        if not C.MQTT_HOST:
            log("mqtt_refused no REX_MQTT_HOST")
            return False
        if self.client is None:
            try:
                self.client = self._connect()
            except ImportError:
                log("mqtt_refused paho_missing")
                return False
            except Exception as exc:
                log(f"mqtt_connect_error={type(exc).__name__}")
                return False
            log(f"mqtt_starting host={C.MQTT_HOST} port={C.MQTT_PORT}")
            return True
        self._bind(self.client)
        self.publish_discovery()
        self.connected = True
        self.publish_state(force=True)
        log(f"mqtt_starting host={C.MQTT_HOST} port={C.MQTT_PORT}")
        return True

    def _connect(self):
        mqtt = _mqtt_module()
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="pidog-rex")
        except Exception:
            client = mqtt.Client(client_id="pidog-rex")
        if C.MQTT_USER:
            client.username_pw_set(C.MQTT_USER, C.MQTT_PASSWORD or None)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.connect_async(C.MQTT_HOST, int(C.MQTT_PORT), 30)
        client.loop_start()
        return client

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties=None):
        if reason_code not in (0, "Success"):
            try:
                if int(reason_code) != 0:
                    log(f"mqtt_connect_rejected code={reason_code}")
                    return
            except (TypeError, ValueError):
                log(f"mqtt_connect_rejected code={reason_code}")
                return
        self.connected = True
        self._bind(client)
        self.publish_discovery()
        self.publish_state(force=True)

    def _bind(self, client) -> None:
        client.on_message = self._on_message
        for topic in self.command_topics():
            client.subscribe(topic)

    def stop(self) -> None:
        client = self.client
        if client is None:
            return
        try:
            client.publish(f"{self.base}/status/availability", "offline", retain=True)
        except Exception:
            pass
        self.connected = False
        for name in ("loop_stop", "disconnect"):
            fn = getattr(client, name, None)
            if fn:
                try:
                    fn()
                except Exception:
                    pass

    def publish_discovery(self) -> None:
        avail = f"{self.base}/status/availability"
        device = dict(_DEVICE)
        sensors = [
            ("battery", "Battery", "V"),
            ("mode", "Mode", None),
            ("person_seen", "Person seen", None),
        ]
        for uid, name, unit in sensors:
            payload = {
                "name": name,
                "unique_id": f"pidog_rex_{uid}",
                "state_topic": f"{self.base}/status/{uid}",
                "availability_topic": avail,
                "device": device,
            }
            if unit:
                payload["unit_of_measurement"] = unit
                payload["device_class"] = "voltage"
                payload["state_class"] = "measurement"
            self._publish(f"homeassistant/sensor/pidog_rex/{uid}/config", payload, retain=True)
        for uid, name in (("guard", "Guard"), ("mute", "Mute")):
            payload = {
                "name": name,
                "unique_id": f"pidog_rex_{uid}",
                "state_topic": f"{self.base}/status/{uid}",
                "command_topic": f"{self.base}/cmd/{uid}",
                "payload_on": "ON",
                "payload_off": "OFF",
                "availability_topic": avail,
                "device": device,
            }
            self._publish(f"homeassistant/switch/pidog_rex/{uid}/config", payload, retain=True)
        for uid, name in (("bark", "Bark"), ("speak", "Speak")):
            payload = {
                "name": name,
                "unique_id": f"pidog_rex_{uid}",
                "command_topic": f"{self.base}/cmd/{uid}",
                "payload_press": "PRESS",
                "availability_topic": avail,
                "device": device,
            }
            self._publish(f"homeassistant/button/pidog_rex/{uid}/config", payload, retain=True)
        self._publish(avail, "online", retain=True)

    def snapshot(self) -> dict:
        last = getattr(self.session.safety, "last", {}) or {}
        battery = last.get("battery")
        guard = getattr(self.session, "guard", None)
        return {
            "battery": "unknown" if battery is None else str(battery),
            "mode": mode_name(self.session),
            "person_seen": getattr(self.session, "person_seen", None) or "none",
            "guard": "ON" if guard is not None and getattr(guard, "armed", False) else "OFF",
            "mute": "ON" if audio.muted else "OFF",
        }

    def publish_state(self, force: bool = False) -> None:
        if self.client is None or not self.connected:
            return
        snap = self.snapshot()
        if not force and snap == self._last:
            return
        self._last = dict(snap)
        for key, value in snap.items():
            self._publish(f"{self.base}/status/{key}", value, retain=True)

    def _publish(self, topic: str, payload, retain: bool = False) -> None:
        if self.client is None:
            return
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        self.client.publish(topic, payload, retain=retain)

    def _on_message(self, _client, _userdata, msg) -> None:
        topic = getattr(msg, "topic", "") or ""
        raw = getattr(msg, "payload", b"")
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", "replace")
        else:
            text = str(raw)
        text = text.strip()
        name = topic.rsplit("/", 1)[-1]
        if name == "guard":
            self._set_guard(text.upper() in {"ON", "TRUE", "1"})
        elif name == "mute":
            audio.muted = text.upper() in {"ON", "TRUE", "1"}
            log(f"mqtt_mute={int(audio.muted)}")
        elif name == "bark":
            if audio.muted:
                log("mqtt_bark_suppressed muted")
            else:
                audio.sfx("bark", True)
                log("mqtt_bark")
        elif name == "speak":
            spoken = "Rex here." if not text or text.upper() == "PRESS" else text[:200]
            audio.speak(spoken, True)
        self.publish_state(force=True)

    def _set_guard(self, on: bool) -> None:
        guard = getattr(self.session, "guard", None)
        if guard is None:
            log("mqtt_guard_unavailable")
            return
        guard.set_armed(on)
        log(f"mqtt_guard armed={int(bool(guard.armed))}")
