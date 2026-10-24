"""Read-only sensor check: init (legs stay in lie pose), read sensors, shut down."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware import Hardware
h = Hardware(); time.sleep(1.5)
print("SENSORS distance=", [h.distance() for _ in range(3)], "battery=", h.battery(), "tilt=", h.tilt(),
      "touch=", h.touch(), "imu_ok=", h.imu_ok, flush=True)
h.shutdown(); os._exit(0)
