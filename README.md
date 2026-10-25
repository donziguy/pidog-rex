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
chmod 600 ~/.config/pidog/security_dog.env   # XAI_API_KEY or: python3 grok_login.py
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
- SafetyMonitor thread polls ~10 Hz, independent of the main loop. It trips on: obstacle < 25 cm
  (instant, on any valid echo). Above walk speed 80 the stop is `PIDOG_REX_FAST_STOP_CM` (default
  35 cm, kept under the 40 cm turn distance) and never closer than 25 cm, so a faster gait still
  stops on a valid close reading. About
  1 s with no valid ultrasonic reading while walking also trips
  (`PIDOG_REX_EMPTY_SECONDS`, or `REX_EMPTY_SECONDS` when that is unset; floor 0.2 s). The trip line
  includes how many reads were empty. A brief `None` from gait vibration does not trip. Tilt > 30 deg
  from the pose at startup, repeated IMU read failures, a frozen IMU that repeats the same raw
  acceleration (`imu_stale`), battery < 6.6 V for 3 s, head touch while walking, or the voice command
  "stop" also trip. A trip clears the leg queue, lies down, and locks out movement until restart or
  "rex patrol". `--allow-no-imu` does not disable those stops for a live chip. It only skips the
  frozen-chip trip (see below).
- Tilt is the angle between the current raw accelerometer vector and a baseline averaged at startup.
  pidog's own pitch/roll are not used: its calibration forces the boot pose to `ax=-16384`, which is
  wrong when gravity is on Y/Z, and a real tilt then stays under 30 deg. `python3 sensorcheck.py --imu`
  prints the live angle for a hand test.
- Walking defaults: speed 95 (cap 98, `PIDOG_REX_WALK_SPEED`; SunFounder examples use about 98),
  4 steps on a clear path (`PIDOG_REX_WALK_STEPS`, ceiling 8), 120 s patrol cap then rest. While a
  patrol or follow step is in progress the loop does not sweep the head or wait on Grok, and it
  yields 50 ms instead of 500 ms. Before a step Rex takes
  up to 5 distance reads about 60 ms apart and uses the median of the valid ones, so one empty echo
  does not halt the patrol. No valid echo in that sample still trips, and the trip line includes the
  empty count. The head is leveled for the step: the shelf pitch (18, with pitch_comp -15) is slightly
  up at yaw 0 and nose-down at the look-around yaw of ±35, which points the ultrasonic at the floor.
- Grok is called only on a real scene change (32x24 grayscale thumbnail diff), at least 60 s apart, with
  320 px images, retries with backoff, Ollama fallback. API failures never crash the loop; responses are
  not logged.
- Grok auth is a SuperGrok / X Premium+ subscription when `GROK_OAUTH_TOKEN` is set, otherwise
  `XAI_API_KEY`. `REX_GROK_AUTH` is `auto` (default, subscription first), `sub`, or `api`. The
  subscription path is the NoirOS `grok-sub` core: model `grok-4.7` on `POST https://api.x.ai/v1/responses`
  with the OAuth token as a Bearer. That core has no separate vision model, so the camera JPEG is an
  `input_image` on the same request. The metered key stays on chat completions (`PIDOG_GROK_MODEL`,
  default grok-4.5). A 401/403 on the subscription token refreshes once
  (`GROK_OAUTH_REFRESH_TOKEN` at `https://auth.x.ai/oauth2/token`). If that fails, Rex logs
  `grok_auth_expired` and uses `XAI_API_KEY` when one is set. Logs say `grok_auth=sub` or `grok_auth=api`
  and never the token.

## Grok subscription login

On the Pi, from the checkout:

```bash
python3 grok_login.py
```

The script prints a verification URL and a short code. Open the URL on a phone, enter the code, and
approve. Tokens are written into `~/.config/pidog/security_dog.env` (the file is backed up first and
kept mode 600). Then reload the user service so it reads the file:

```bash
systemctl --user restart rex
```
- SIGINT and SIGTERM set the shutdown event, stop the legs immediately, and abandon an in-flight Grok
  call or TTS instead of waiting out the HTTP or espeak timeout. A second Ctrl+C lies down, closes
  the hardware, and exits. One Pidog instance, same as a normal shutdown.

## Known issue: IMU init failure and a frozen chip
If the SH3001 fails to init (`imu_sh3001 init ... fail`), there is no tilt signal and `--patrol` refuses
to walk. That is separate from a mounted-but-miscalibrated IMU, which Rex handles itself (see Safety).

A chip can also init and then stick: the same accel and gyro frame forever, even while the dog is
moved. Tilt then looks valid and never changes. Rex calls that `imu_stale`. Without `--allow-no-imu`
it trips `imu_read_fail` and refuses patrol and follow. `python3 sensorcheck.py --imu` prints
`stale=True/False`.

**`--allow-no-imu` warning:** supervised bypass while the gyro board is being replaced. It overrides
the refusal for a missing IMU and for a frozen one. Rex logs `imu_stale_overridden` once and walks
with NO tilt/fall detection (the stuck angle is ignored). Distance, battery, and touch stops still
run. Only use it while you are right next to the dog, on a clear floor, ready to grab it. Fix the IMU
(reseat the cable, run SunFounder calibration, or replace the board) instead when you can.

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

## Wake word

Off unless `REX_WAKE_WORD=1`. While it is on, Rex drops voice commands that do not follow
"hey rex". The phrase can share the utterance ("hey rex, patrol") or open a window of
`REX_WAKE_WINDOW` seconds (default 8) for the next thing you say. "hey rex stop" still
trips safety. With the flag off, commands work exactly as before.

The gate is the transcript from the existing Vosk listener, so it stays offline and needs
no extra model. If `openwakeword` is installed, a model named `REX_WAKE_MODEL` (default
`hey_rex`) can arm that same window from the microphone audio. A missing model just leaves
the phrase gate in place.

`REX_STT_ENGINE=whisper` transcribes with faster-whisper instead of Vosk: `tiny` or `base`,
CPU, int8 (`REX_WHISPER_MODEL`, `REX_WHISPER_COMPUTE`). Anything else is forced back to tiny
int8. If the package is not installed, Rex logs that and uses Vosk.

```bash
pip install faster-whisper          # optional
# pip install openwakeword          # optional, plus a hey_rex model if you have one
# in security_dog.env:
#   REX_WAKE_WORD=1
#   REX_STT_ENGINE=vosk
```

## Home Assistant

Off unless `REX_MQTT=1`. Rex will not connect unless `REX_MQTT_HOST` is set. There is no default
broker address, and the password (`REX_MQTT_PASSWORD`) is redacted from logs. Optional
`REX_MQTT_USER`, `REX_MQTT_PORT` (default 1883), and `REX_MQTT_PREFIX` (default `rex`).

On connect, Rex publishes Home Assistant MQTT discovery:

| Entity | Kind | What it does |
| --- | --- | --- |
| battery | sensor | pack voltage |
| mode | sensor | `shelf`, `patrol`, `guard`, or `follow` |
| person_seen | sensor | primary name, another known name, `stranger`, or `none` |
| guard | switch | arm / disarm guard (still shelf-only) |
| mute | switch | drop TTS and the bark button |
| bark | button | play the bark effect |
| speak | button | say "Rex here." A payload other than `PRESS` is spoken as text |

`pip install paho-mqtt` on the Pi. Leave the host and password out of the repo and out of
`rex.service`.

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
and never above the patrol speed). An armed guard still refuses to walk. A dead or frozen IMU refuses
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
be up and not frozen, unless you started Rex with `--allow-no-imu`) and the safety thread still
enforces distance, tilt, battery, and touch. With the override, tilt from a frozen chip is ignored.
STOP does not need the patrol flag.

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
web UI (off unless `REX_DASH=1`), `faces.py` local face enrollment (off unless `REX_FACE=1`), `guard.py` shelf guard alerts (off unless `REX_GUARD=1`), `follow.py` supervised follow (`--follow` only, never in `rex.service`),
`hass_mqtt.py` Home Assistant discovery (off unless `REX_MQTT=1`),
`wake.py` offline "hey rex" gate (off unless `REX_WAKE_WORD=1`), `main.py` entry point,
`sensorcheck.py` read-only sensor check, `tests/` pytest suite.

## License
MIT
