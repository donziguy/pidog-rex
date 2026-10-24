#!/usr/bin/env python3
"""Rex v2 - grumpy security PiDog. Shelf mode by default; walking only with --patrol or "rex patrol".

  python3 main.py                 # shelf mode (head, camera, voice) forever
  python3 main.py --seconds 30    # shelf mode for 30 s then lie down and exit
  python3 main.py --patrol        # supervised patrol (legs!) - see README
  python3 main.py --sim --seconds 10   # same loop on a simulated dog (no Pi)
"""
from __future__ import annotations
import argparse, fcntl, os, signal, sys, threading, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import log
import brain, audio, vision_grok
from dashboard import Dashboard

_shutdown = threading.Event()


class Session:
    """Live objects the main loop shares with optional features."""

    def __init__(self, hw, safety, loco, listener, tts, args):
        self.hw = hw
        self.safety = safety
        self.loco = loco
        self.listener = listener
        self.tts = tts
        self.args = args
        self.face_hits = []
        self.person_seen = "none"


def _on_signal(signum, frame):
    log(f"signal_received={signum}")
    _shutdown.set()


def single_instance():
    fh = open("/tmp/rex_v2.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another Rex instance holds /tmp/rex_v2.lock; exiting")
        sys.exit(0)
    return fh


def gesture(hw, name, tts):
    frames = {"attention": [(0, 8), (0, 0)], "suspicious": [(0, 10), (0, -6), (0, 3)],
              "annoyed": [(-8, 0), (8, 0), (0, 0)], "bored": [(0, -8), (0, -2)],
              "dog_detected": [(0, 6), (0, -3), (0, 0)]}.get(name, [(0, 4), (0, 0)])
    audio.sfx({"suspicious": "growl", "annoyed": "huff", "bored": "sigh", "dog_detected": "bark"}.get(name, "attention"), tts)
    hw.head_gesture(frames)
    if name in ("dog_detected", "attention"):
        hw.wag(3)
    brain.bump("gestures_done")


def run(args) -> int:
    if getattr(args, "sim", False):
        os.environ["REX_SIM"] = "1"
    from hardware import Hardware
    from safety import SafetyMonitor
    from locomotion import Locomotion
    tts = not args.no_tts
    hw = Hardware()
    # Install handlers only AFTER Pidog() forks its sensory child, so the child keeps
    # default SIGTERM and actually dies when Pidog.close() terminates it.
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    safety = SafetyMonitor(hw)
    loco = Locomotion(hw, safety)
    listener = audio.Listener()
    listener.on_stop = lambda: safety.trip("voice_stop")
    safety.on_trip = lambda r: loco.disable(f"trip:{r}")
    safety.start()
    if not args.no_listen:
        listener.start()
    session = Session(hw, safety, loco, listener, tts, args)
    dash = Dashboard(session)
    session.dash = dash
    face_mod = None
    try:
        import faces as face_mod
        face_mod.ensure_registered()
    except Exception as exc:
        log(f"faces_unavailable={type(exc).__name__}")
    try:
        import guard as guard_mod
        session.guard = guard_mod.Guard(session)
        guard_mod.ensure_registered()
    except Exception as exc:
        log(f"guard_unavailable={type(exc).__name__}")
        session.guard = None
    try:
        import hass_mqtt
        session.mqtt = hass_mqtt.HomeAssistant(session)
        session.mqtt.start()
    except Exception as exc:
        log(f"mqtt_unavailable={type(exc).__name__}")
        session.mqtt = None
    dash.start()
    brain.memory(); brain.set_state("online")
    deadline = time.time() + args.seconds if args.seconds else None
    grok_calls = 0
    try:
        hw.move_head(0, 0, None, speed=30)
        audio.speak("Rex online. Shelf duty. Try not to be suspicious.", tts)
        session.follow = None
        if getattr(args, "follow", False):
            import follow as follow_mod
            session.follow = follow_mod.Follow(session)
            if session.follow.start():
                audio.speak("I will follow, but only while you are here.", tts)
            else:
                audio.speak("I am not following.", tts)
        elif args.patrol:
            loco.allow_no_imu = args.allow_no_imu
            if session.guard is not None and session.guard.armed:
                audio.speak("Guard is armed. I am not walking.", tts)
            elif loco.enable("--patrol flag"):
                audio.speak("Supervised patrol. Say stop or touch my head to kill it.", tts)
            else:
                audio.speak("My balance sensor is dead, so no walking.", tts)
        positions = [(-35, "left"), (0, "center"), (35, "right")]
        i = 0
        while not _shutdown.is_set() and (deadline is None or time.time() < deadline):
            dash.drain()
            # 1) voice commands
            while not listener.events.empty():
                cmd, _text = listener.events.get_nowait()
                if cmd == "stop":
                    audio.speak("Fine. Lying down.", tts)
                elif cmd == "patrol":
                    if args.allow_voice_patrol or args.patrol:
                        loco.allow_no_imu = args.allow_no_imu
                        if session.guard is not None and session.guard.armed:
                            audio.speak("Guard is armed. I am not walking.", tts)
                        elif loco.enable("voice"):
                            audio.speak("Patrol time. Out of my way.", tts)
                    else:
                        audio.speak("Walking is disabled. Start me with patrol mode.", tts)
                elif cmd == "rest":
                    loco.disable("voice_rest"); hw.lie_down(40)
                elif cmd == "greet":
                    gesture(hw, "attention", tts)
            # 2) shelf touch = pet (safety handles touch while walking)
            if not loco.enabled and safety.last.get("touch", "N") != "N":
                audio.sfx("happy", tts); hw.wag(4, 50); brain.remember("pet", safety.last["touch"])
            # 3) look around / track sound
            yaw, label = positions[i % 3]; i += 1
            d = hw.sound_direction() if C.SOUND_TRACKING else None
            if d is not None:
                hw.head_pan(hw.sound_to_pan(d)); brain.bump("sound_events"); brain.adjust_mood(event="sound")
                label = f"sound{d}"
            else:
                hw.move_head(yaw, 0, None, speed=30)
            hw.chest("listening")
            # 4) vision (Grok only on real change, >=60 s apart)
            photo = hw.photo()
            if face_mod is not None and C.FACE_ENABLED:
                face_mod.observe(session, photo)
            if session.guard is not None:
                session.guard.tick(photo, session.face_hits)
            if session.follow is not None and session.follow.active and not session.face_hits and face_mod is not None:
                session.face_hits = face_mod.recognize_hw(hw, photo=photo)
            if photo:
                force = args.force_first_grok and grok_calls == 0
                hw.chest("analyzing")
                a = vision_grok.analyze(photo, force=force)
                if not a.get("skipped"):
                    grok_calls += a.get("_source") == "grok"
                    brain.update_from_vision(a)
                    gesture(hw, brain.gesture_for(a), tts)
                    if brain.should_speak(a, bool(a.get("_changed"))):
                        hw.chest("talking"); audio.speak(a["roast"], tts)
                        brain.bump("roasts_spoken"); brain.remember("roast", a["roast"]); brain.adjust_mood(event="roast_spoken")
                elif brain.mutter_due() and not loco.enabled:
                    audio.speak(brain.idle_mutter(brain.memory().get("last_scene_summary") or "boring shelf duty"), tts)
            hw.chest("locked" if safety.locked.is_set() else "idle")
            # 5) walking (only if enabled and safety allows). Follow replaces patrol wander.
            if session.follow is not None and session.follow.active:
                session.follow.tick(session.face_hits)
            elif loco.enabled:
                for _ in range(2):
                    r = loco.tick()
                    if r in ("off", "locked", "halt", "resting", "rest", "aborted"):
                        break
            if session.mqtt is not None:
                session.mqtt.publish_state()
            _shutdown.wait(0.5)
        log(f"main_loop_exit grok_calls={grok_calls} safety_ticks={safety.ticks} locked={safety.locked.is_set()} reason={safety.reason}")
        return 0
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        return 1
    finally:
        loco.disable("shutdown")
        if getattr(session, "mqtt", None) is not None:
            session.mqtt.stop()
        dash.stop()
        listener.stop(); safety.stop()
        brain.save()
        hw.shutdown()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rex v2 PiDog")
    p.add_argument("--patrol", action="store_true", help="enable walking (supervised). Off by default.")
    p.add_argument("--follow", action="store_true",
                   help="supervised follow of a recognized person. Never put this in rex.service.")
    p.add_argument("--sim", action="store_true", help="simulated dog and camera (no Pi). Same as REX_SIM=1")
    p.add_argument("--allow-voice-patrol", action="store_true", help="let the voice command 'rex patrol' enable walking")
    p.add_argument("--seconds", type=float, default=None, help="run N seconds then lie down and exit")
    p.add_argument("--allow-no-imu", action="store_true", help="permit patrol even if the IMU failed to init (no tilt stop)")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--no-listen", action="store_true")
    p.add_argument("--force-first-grok", action="store_true", help="make the first frame call Grok (testing)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    lock = single_instance()
    log(f"rex_v2_start pid={os.getpid()} args={vars(args)}")
    rc = run(args)
    log(f"rex_v2_exit rc={rc}")
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)  # don't let library threads / multiprocessing atexit keep the process alive
