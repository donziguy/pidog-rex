"""Home Assistant MQTT discovery, commands, and the rule that the broker comes from env."""
from __future__ import annotations

import argparse
import json

import audio
import config as C
import hass_mqtt
import main
from support import ROOT


def _session(hw):
    from locomotion import Locomotion
    from safety import SafetyMonitor

    safety = SafetyMonitor(hw)
    loco = Locomotion(hw, safety)
    args = argparse.Namespace(allow_no_imu=False, patrol=False, follow=False)
    session = main.Session(hw, safety, loco, listener=None, tts=True, args=args)
    session.guard = None
    session.follow = None
    session.person_seen = "none"
    session.mqtt = None
    return session


class _Msg:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload.encode()


class _Fake:
    def __init__(self):
        self.published = []
        self.subscriptions = []
        self.on_message = None

    def publish(self, topic, payload, retain=False, qos=0):
        self.published.append((topic, payload, bool(retain)))

    def subscribe(self, topic, qos=0):
        self.subscriptions.append(topic)

    def push(self, topic, payload):
        self.on_message(self, None, _Msg(topic, payload))


def _configs(fake):
    out = {}
    for topic, payload, _retain in fake.published:
        if topic.startswith("homeassistant/"):
            out[topic] = json.loads(payload)
    return out


def test_mqtt_is_off_without_a_broker(make_hw, monkeypatch, capsys):
    assert C.MQTT_ENABLED is False
    assert C.MQTT_HOST == ""
    assert C.MQTT_PORT == 1883
    assert C.MQTT_PREFIX == "rex"
    assert hass_mqtt.HomeAssistant(_session(make_hw())).start() is False
    assert "mqtt_disabled" in capsys.readouterr().out
    monkeypatch.setattr(C, "MQTT_ENABLED", True)
    monkeypatch.setattr(C, "MQTT_HOST", "")
    assert hass_mqtt.HomeAssistant(_session(make_hw())).start() is False
    assert "no REX_MQTT_HOST" in capsys.readouterr().out
    assert "REX_MQTT_HOST=" in (ROOT / ".env.example").read_text()
    assert "REX_MQTT" not in (ROOT / "rex.service").read_text()


def test_discovery_and_commands(make_hw, monkeypatch, capsys):
    import guard
    monkeypatch.setattr(C, "MQTT_ENABLED", True)
    monkeypatch.setattr(C, "MQTT_HOST", "mqtt.example.test")
    monkeypatch.setattr(C, "MQTT_PASSWORD", "broker-password-secret")
    monkeypatch.setattr(C, "GUARD_ENABLED", True)
    hw = make_hw()
    session = _session(hw)
    session.guard = guard.Guard(session, notifiers=[])
    session.safety.last["battery"] = 7.4
    session.person_seen = "Cory"
    fake = _Fake()
    ha = hass_mqtt.HomeAssistant(session, client=fake)
    assert ha.start() is True
    configs = _configs(fake)
    assert set(configs) == {
        "homeassistant/sensor/pidog_rex/battery/config",
        "homeassistant/sensor/pidog_rex/mode/config",
        "homeassistant/sensor/pidog_rex/person_seen/config",
        "homeassistant/switch/pidog_rex/guard/config",
        "homeassistant/switch/pidog_rex/mute/config",
        "homeassistant/button/pidog_rex/bark/config",
        "homeassistant/button/pidog_rex/speak/config",
    }
    assert configs["homeassistant/sensor/pidog_rex/battery/config"]["unit_of_measurement"] == "V"
    assert configs["homeassistant/switch/pidog_rex/guard/config"]["command_topic"] == "rex/cmd/guard"
    assert configs["homeassistant/button/pidog_rex/speak/config"]["payload_press"] == "PRESS"
    assert fake.subscriptions == ha.command_topics()
    states = {topic: payload for topic, payload, _r in fake.published if topic.startswith("rex/status/")}
    assert states["rex/status/battery"] == "7.4"
    assert states["rex/status/mode"] == "shelf"
    assert states["rex/status/person_seen"] == "Cory"
    assert states["rex/status/availability"] == "online"
    assert "broker-password-secret" not in capsys.readouterr().out

    fake.push("rex/cmd/guard", "ON")
    assert session.guard.armed is True
    fake.push("rex/cmd/mute", "ON")
    assert audio.muted is True
    audio.speak("should not be heard")
    muted_log = capsys.readouterr().out
    assert "TTS suppressed (muted)" in muted_log
    assert "should not be heard" not in muted_log
    fake.push("rex/cmd/bark", "PRESS")
    assert "mqtt_bark_suppressed" in capsys.readouterr().out
    fake.push("rex/cmd/mute", "OFF")
    fake.push("rex/cmd/speak", "PRESS")
    fake.push("rex/cmd/speak", "come here")
    fake.push("rex/cmd/bark", "PRESS")
    spoken = capsys.readouterr().out
    assert "TTS: Rex here." in spoken
    assert "TTS: come here" in spoken
    assert "mqtt_bark" in spoken
    guard_states = [payload for topic, payload, _r in fake.published if topic == "rex/status/guard"]
    assert "ON" in guard_states
    mode = hass_mqtt.mode_name(session)
    assert mode == "guard"
    session.loco.enabled = True
    assert hass_mqtt.mode_name(session) == "patrol"
    session.follow = argparse.Namespace(active=True)
    assert hass_mqtt.mode_name(session) == "follow"
    ha.stop()
    assert fake.published[-1][0] == "rex/status/availability"
    assert fake.published[-1][1] == "offline"


def test_connect_uses_the_env_broker_and_hides_the_password(make_hw, monkeypatch, capsys):
    monkeypatch.setattr(C, "MQTT_ENABLED", True)
    monkeypatch.setattr(C, "MQTT_HOST", "mqtt.example.test")
    monkeypatch.setattr(C, "MQTT_PORT", 1883)
    monkeypatch.setattr(C, "MQTT_USER", "rex")
    monkeypatch.setattr(C, "MQTT_PASSWORD", "broker-password-secret")
    created = []

    class _Client:
        def __init__(self, *args, **kwargs):
            created.append(self)
            self.user = None
            self.target = None
            self.looped = False
            self.on_connect = None
            self.on_message = None

        def username_pw_set(self, user, password):
            self.user = (user, password)

        def connect_async(self, host, port, keepalive):
            self.target = (host, port, keepalive)

        def loop_start(self):
            self.looped = True

    monkeypatch.setattr(hass_mqtt, "_mqtt_module", lambda: argparse.Namespace(Client=_Client))
    ha = hass_mqtt.HomeAssistant(_session(make_hw()))
    assert ha.start() is True
    assert created[0].target == ("mqtt.example.test", 1883, 30)
    assert created[0].user == ("rex", "broker-password-secret")
    assert created[0].looped is True
    out = capsys.readouterr().out
    assert "mqtt_starting host=mqtt.example.test" in out
    assert "broker-password-secret" not in out
    monkeypatch.setenv("REX_MQTT_PASSWORD", "broker-password-secret")
    C.log("leaked broker-password-secret")
    assert "broker-password-secret" not in capsys.readouterr().out


def test_paho_imports_and_run_stays_disconnected(monkeypatch, capsys):
    import paho.mqtt.client as mqtt
    assert mqtt.Client
    monkeypatch.setattr(main._shutdown, "wait", lambda timeout=None: None)
    import vision_grok
    monkeypatch.setattr(vision_grok.time, "sleep", lambda *_a, **_k: None)
    args = argparse.Namespace(
        patrol=False, sim=True, allow_voice_patrol=False, seconds=0.05,
        allow_no_imu=False, no_tts=True, no_listen=True, force_first_grok=False,
        follow=False,
    )
    assert main.run(args) == 0
    out = capsys.readouterr().out
    assert "mqtt_disabled" in out
    assert "mqtt_starting" not in out
