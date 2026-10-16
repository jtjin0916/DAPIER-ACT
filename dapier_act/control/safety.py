from __future__ import annotations
from dataclasses import dataclass
import time


@dataclass
class Watchdog:
    timeout_s: float
    last_sensor_time_s: float | None = None

    def update_sensor_time(self, timestamp_s):
        self.last_sensor_time_s = timestamp_s

    def assert_fresh(self, now_s=None):
        now_s = time.time() if now_s is None else now_s
        if self.last_sensor_time_s is None:
            raise RuntimeError("No sensor timestamp received")
        if now_s - self.last_sensor_time_s > self.timeout_s:
            raise RuntimeError("Sensor data is stale; actuation blocked")
