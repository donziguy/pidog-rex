# Rex: a grumpy security dog for the SunFounder PiDog

Rex is a sarcastic retired security dog that lives on a SunFounder PiDog. He looks around with the
camera, turns toward sounds, roasts what he sees (Grok vision via the xAI API), mutters through a local
Ollama model when idle, listens for voice commands with offline Vosk speech recognition, and remembers
mood and sightings between runs. By default he stays put (shelf mode). Walking is opt-in and guarded by
an independent safety thread.

## Hardware
- SunFounder PiDog kit (robot_hat, ultrasonic, dual touch, sound direction, IMU, RGB strip)
- Raspberry Pi 4/5 running Raspberry Pi OS Bookworm 64-bit
- Pi camera (picamera2), USB microphone, speaker

## Install
```bash
# SunFounder stack (robot_hat, vilib, pidog) per SunFounder docs, then:
sudo apt install espeak alsa-utils python3-picamera2 python3-pil python3-requests
python3 -m venv ~/.local/stt-venv && ~/.local/stt-venv/bin/pip install vosk
# download vosk-model-small-en-us-0.15 to ~/.local/share/vosk/
git clone https://github.com/donziguy/pidog-rex ~/pidog-rex
mkdir -p ~/.config/pidog && cp ~/pidog-rex/.env.example ~/.config/pidog/security_dog.env
chmod 600 ~/.config/pidog/security_dog.env   # add XAI_API_KEY, PIDOG_OLLAMA_URL, etc.
```
Optional service (shelf mode only): `cp rex.service ~/.config/systemd/user/ && systemctl --user daemon-reload`.

## Run
```bash
python3 main.py                    # shelf mode: head, camera, voice. No walking.
python3 main.py --seconds 30       # shelf mode for 30 s, then lie down and exit
python3 main.py --patrol --seconds 120   # SUPERVISED patrol (legs move). Stay with the dog.
```
Other flags: `--no-tts`, `--no-listen`, `--force-first-grok`, `--allow-voice-patrol` (lets "rex patrol" start walking).
Voice: "stop" / "freeze" (safety stop), "patrol", "sit" / "rest", "good boy".

## Simulation and tests

Rex runs without a Pi. `REX_SIM=1` or `--sim` uses a software dog — legs, head, distance, IMU,
battery, touch, sound direction, chest LEDs, and camera frames — instead of `pidog` and `picamera2`.
Those imports happen only on the real path, so a laptop does not need the SunFounder stack. On the
Pi, leave `REX_SIM` unset and do not pass `--sim`.

```bash
REX_SIM=1 python3 main.py --seconds 10 --no-tts --no-listen
python3 main.py --sim --patrol --seconds 20 --no-tts --no-listen
```

`--sim` is for development. Do not add it, or `--patrol`, to `rex.service`. `sensorcheck.py` follows
the same switch.

Tests and a REPL can script sensors over time (each value holds until the next timestamp):

```python
import os
os.environ["REX_SIM"] = "1"
from hardware import Hardware
hw = Hardware()
hw.dog.distance.script([(0, 100), (1.5, 15)])  # clear, then 15 cm
hw.dog.battery.script([(0, 7.4), (2.0, 6.2)])  # pack sags
hw.dog.pitch = 0
hw.dog.touch.set("N")
```

### Tests

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The suite stubs Grok, Ollama, and TTS/STT subprocesses. It does not use the network or a PiDog.
GitHub Actions runs it on Python 3.11.

## Safety
- SafetyMonitor thread polls ~10 Hz, independent of the main loop. It trips on: obstacle < 25 cm or
  repeated empty ultrasonic reads while walking, tilt > 30 deg from the pose at startup, repeated IMU
  read failures, battery < 6.6 V for 3 s, head touch while walking, or the voice command "stop". A trip
  clears the leg queue, lies down, and locks out movement until restart or "rex patrol".
- Tilt is the angle between the current raw accelerometer vector and a baseline averaged at startup.
  pidog's own pitch/roll are not used: its calibration forces the boot pose to `ax=-16384`, which is
  wrong when gravity is on Y/Z, and a real tilt then stays under 30 deg. `python3 sensorcheck.py --imu`
  prints the live angle for a hand test.
- Walking defaults: speed 65, 1-2 steps per move, halts on missing distance readings, 120 s patrol cap
  then rest.
- Grok is called only on a real scene change (32x24 grayscale thumbnail diff), at least 60 s apart, with
  320 px images, retries with backoff, Ollama fallback. API failures never crash the loop; responses are
  not logged.
- SIGTERM: stop legs, lie down, close hardware cleanly.

## Known issue: IMU init failure
If the SH3001 fails to init (`imu_sh3001 init ... fail`), there is no tilt signal and `--patrol` refuses
to walk. That is separate from a mounted-but-miscalibrated IMU, which Rex handles itself (see Safety).

**`--allow-no-imu` warning:** this overrides that refusal and walks with NO tilt/fall detection. Only use
it while you are right next to the dog, on a clear floor, ready to grab it. Fix the IMU (reseat cable,
run SunFounder calibration) instead when you can.

## Face recognition

Off unless `REX_FACE=1`. Rex recognizes people on the Pi (no cloud). The primary human defaults to
**Cory** (`PIDOG_PRIMARY_HUMAN`). A known face is greeted by name. A face a real detector cannot
match is a stranger: Rex says so, remembers the sighting, and does **not** add them to the gallery.
The Pillow fallback (used in CI, and when neither dlib nor OpenCV is installed) only confirms
enrolled people. It will not call an empty shelf a stranger. For stranger detection on the Pi,
install one of:

* `face_recognition` (dlib), or
* OpenCV with YuNet + SFace model files (`REX_YUNET_MODEL`, `REX_SFACE_MODEL`), or
* stock OpenCV, which uses the frontal Haar cascade and embeds the crop locally

`REX_FACE_BACKEND` is `auto` (that order), or `pillow` / `opencv` / `face_recognition`.

Embeddings are stored in `~/.config/pidog/faces/index.json` (override with `REX_FACE_DIR`). That
directory is outside the repo and listed in `.gitignore`. Do not commit it.

```bash
REX_SIM=1 python3 faces.py enroll Cory --image cory.jpg
REX_SIM=1 python3 faces.py list
REX_SIM=1 python3 faces.py recognize --image cory.jpg
# from the live camera, only when the rex service is stopped (one Pidog at a time):
REX_FACE=1 python3 faces.py enroll Cory
```

The dashboard gains an "Enroll from camera" button when the face module is loaded. The enroll
route still returns `face_disabled` until `REX_FACE=1`. In simulation, `dog.people` can inject a
sighting (`name`, `bbox` as x,y,w,h in 0..1, `known` or `stranger`) without a model.

## Follow mode

Follow is **supervised** and **off unless you pass `--follow`**. No environment variable turns it
on, and it must never be added to `rex.service` (the unit stays shelf-only, same rule as
`--patrol`). `--follow` and `--patrol` together: follow wins, patrol wander stays off.

Rex steps toward the camera bounding box of a **recognized** person, preferring
`PIDOG_PRIMARY_HUMAN` (Cory). The box center left of 0.40 turns left, right of 0.60 turns
right, otherwise he takes one forward step. A stranger is not a target. If the person leaves the frame he takes no step (no search).
After `REX_FOLLOW_LOST_TICKS` misses in a row (default 5) the session ends. Each step goes
through the locomotion safety thread
(distance, tilt, battery, touch). The session stops at `REX_FOLLOW_MAX_STEPS` (default 8) or
`REX_FOLLOW_MAX_SECONDS` (default 60), then he sits. Speed is `REX_FOLLOW_SPEED` (default 50,
and never above the patrol speed). An armed guard still refuses to walk. A dead IMU refuses
follow unless you also passed `--allow-no-imu`, which is the same "stand next to the dog"
warning as patrol.

```bash
# You stay in the room. Ctrl-C or "stop" ends it.
REX_SIM=1 python3 main.py --sim --follow --seconds 20 --no-tts
python3 main.py --follow --seconds 60
```

In simulation, `dog.people` entries with `known: true` and a `bbox` of `[x, y, w, h]` (0..1)
are the target. On the Pi, enroll Cory first (`REX_FACE=1`) so the box comes from a real detector.
The Pillow fallback only matches an enrolled whole frame, so a proper detector is what makes
follow useful on hardware.

## Guard mode

Off unless `REX_GUARD=1`. Even then Rex starts **disarmed** unless `REX_GUARD_ARMED=1` (or you arm
him from the dashboard). Guard is shelf-only. Arming disables patrol, and `Locomotion.enable`
refuses to start walking until he is disarmed. There is no guard walk.

While armed, either of these sends one alert, then waits `REX_GUARD_COOLDOWN` seconds (default 60):

* an unknown face from face recognition (a stranger hit)
* motion: mean absolute difference of a 32×24 grayscale frame at or above `REX_GUARD_MOTION`
  (default 18). The first frame is only a baseline.

The alert includes a JPEG snapshot. Notifiers are pluggable and both default off:

* `REX_NTFY_TOPIC` posts to `REX_NTFY_SERVER` (default `https://ntfy.sh`). `REX_NTFY_TOKEN` is optional.
* `REX_ALERT_WEBHOOK_URL` receives JSON: `title`, `message`, `source` (`rex`), and `image_base64`.

Leave the topic, token, and webhook empty in the repo. They are redacted from logs. The dashboard
has Arm guard / Disarm buttons; the route returns `guard_disabled` until the feature is on.

## Web dashboard

Off unless `REX_DASH=1`. Rex then serves a small page (Flask) at `REX_DASH_HOST`:`REX_DASH_PORT`,
default `0.0.0.0:8080`. It **refuses to listen** if `REX_DASH_TOKEN` is empty. Send the token as
`X-Rex-Token`, `Authorization: Bearer …`, or `?token=` (the page keeps it in sessionStorage and
never writes it to the log). The token is redacted if it would otherwise land in `rex.log`.

The page shows the latest camera frame, battery, tilt, distance, shelf/patrol mode, the last Grok
line, and a tail of the log. Buttons: speak a line, move the head, shelf mode, and a large STOP
that trips the safety monitor immediately (legs stop, lie down, lock). Patrol from the page also
requires `REX_DASH_ALLOW_PATROL=1`. Enabling it still goes through `Locomotion.enable` (IMU must
be up, unless you started Rex with `--allow-no-imu`) and the safety thread still enforces distance,
tilt, battery, and touch. STOP does not need the patrol flag.

```bash
pip install flask
# in ~/.config/pidog/security_dog.env (chmod 600):
#   REX_DASH=1
#   REX_DASH_TOKEN=pick-a-long-random-string
#   REX_DASH_ALLOW_PATROL=0
REX_SIM=1 REX_DASH=1 REX_DASH_TOKEN=dev-only python3 main.py --sim --seconds 30 --no-tts --no-listen
```

Do not put the token in the repo, in `rex.service`, or in the shell history of a shared machine.
Bind stays configurable; on a Pi that shares a LAN, prefer a token you would not reuse elsewhere.

## Layout
`config.py` settings/logging, `brain.py` personality/mood/memory, `hardware.py` PiDog + camera,
`sim.py` simulated dog (`REX_SIM` / `--sim` only; not imported on the Pi), `safety.py` SafetyMonitor,
`locomotion.py` patrol, `vision_grok.py` Grok vision, `audio.py` TTS/SFX/voice, `dashboard.py` local
web UI (off unless `REX_DASH=1`), `faces.py` local face enrollment (off unless `REX_FACE=1`), `guard.py` shelf guard alerts (off unless `REX_GUARD=1`), `follow.py` supervised follow
(`--follow` only, never in `rex.service`), `main.py` entry point,
`sensorcheck.py` read-only sensor check, `tests/` pytest suite.

## License
MIT
