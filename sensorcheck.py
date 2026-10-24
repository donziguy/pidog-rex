"""Read-only sensor check: init (legs stay in lie pose), read sensors, shut down.

``python3 sensorcheck.py`` prints one sample. ``--imu`` prints live tilt
(baseline angle plus pitch/roll) until Ctrl-C, for a hand test of the dog.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware import Hardware


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Read Rex sensors without walking")
    p.add_argument("--imu", action="store_true", help="print live tilt until Ctrl-C")
    args = p.parse_args(argv)
    h = Hardware()
    try:
        if args.imu:
            b = h.tilt_baseline
            print(f"IMU baseline={None if b is None else tuple(round(x, 1) for x in b)} imu_ok={h.imu_ok}", flush=True)
            while True:
                pitch, roll = h.tilt()
                print(
                    f"tilt={h.tilt_deg:.1f} pitch={pitch:.1f} roll={roll:.1f} "
                    f"fail={h.imu_fail_streak} stale={h.imu_stale}",
                    flush=True,
                )
                time.sleep(0.2)
        else:
            time.sleep(1.5)
            print("SENSORS distance=", [h.distance() for _ in range(3)], "battery=", h.battery(),
                  "tilt=", h.tilt(), "tilt_deg=", round(h.tilt_deg, 1),
                  "touch=", h.touch(), "imu_ok=", h.imu_ok, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        h.shutdown()
        os._exit(0)


if __name__ == "__main__":
    main()
