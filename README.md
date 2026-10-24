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

## Safety
- SafetyMonitor thread polls ~10 Hz, independent of the main loop. It trips on: obstacle < 25 cm or
  repeated empty ultrasonic reads while walking, tilt > 30 deg, battery < 6.6 V for 3 s, head touch while
  walking, or the voice command "stop". A trip clears the leg queue, lies down, and locks out movement
  until restart or "rex patrol".
- Walking defaults: speed 65, 1-2 steps per move, halts on missing distance readings, 120 s patrol cap
  then rest.
- Grok is called only on a real scene change (32x24 grayscale thumbnail diff), at least 60 s apart, with
  320 px images, retries with backoff, Ollama fallback. API failures never crash the loop; responses are
  not logged.
- SIGTERM: stop legs, lie down, close hardware cleanly.

## Known issue: IMU (gyro) init failure
On the author's unit the SH3001 IMU reports `imu_sh3001 init ... fail`, so tilt reads 0 and the tilt stop
cannot work. `--patrol` refuses to walk when the IMU is unavailable.

**`--allow-no-imu` warning:** this overrides that refusal and walks with NO tilt/fall detection. Only use
it while you are right next to the dog, on a clear floor, ready to grab it. Fix the IMU (reseat cable,
run SunFounder calibration) instead when you can.

## Layout
`config.py` settings/logging, `brain.py` personality/mood/memory, `hardware.py` PiDog + camera,
`safety.py` SafetyMonitor, `locomotion.py` patrol, `vision_grok.py` Grok vision, `audio.py` TTS/SFX/voice,
`main.py` entry point, `sensorcheck.py` read-only sensor check.

## License
MIT
