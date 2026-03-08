"""Punch-event detection logic for Lux Punch."""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass

import numpy as np

from pose_utils import KinematicBuffer


@dataclass(frozen=True)
class HitEvent:
    side: str
    force: float
    speed: float
    accel: float
    timestamp: float


class PunchAnalyzer:
    """Track left/right wrist motion and detect punch-like events."""

    def __init__(
        self,
        eff_mass_kg: float,
        vel_threshold: float,
        accel_threshold: float,
        hit_rearm_seconds: float = 0.5,
        history: int = 32,
    ) -> None:
        self.left_buf = KinematicBuffer(maxlen=history)
        self.right_buf = KinematicBuffer(maxlen=history)
        self.eff_mass_kg = float(eff_mass_kg)
        self.vel_threshold = float(vel_threshold)
        self.accel_threshold = float(accel_threshold)
        self.hit_rearm_seconds = float(hit_rearm_seconds)
        self.last_hit_time = -1e9

    def push(self, side: str, timestamp: float, pos3d) -> None:
        buffer = self.left_buf if side.upper().startswith("L") else self.right_buf
        buffer.push(timestamp, pos3d)

    def detect(self, now: float | None = None) -> HitEvent | None:
        event = self._check_buffer("Left", self.left_buf, now=now)
        if event is not None:
            return event
        return self._check_buffer("Right", self.right_buf, now=now)

    def _check_buffer(
        self,
        side: str,
        buffer: KinematicBuffer,
        now: float | None = None,
    ) -> HitEvent | None:
        velocity = buffer.velocity(n=1)
        acceleration = buffer.acceleration(n=1)
        if velocity is None or acceleration is None:
            return None

        speed = float(np.linalg.norm(velocity))
        accel_mag = float(np.linalg.norm(acceleration))
        current_time = time_module.time() if now is None else float(now)
        if current_time - self.last_hit_time < self.hit_rearm_seconds:
            return None
        if speed <= self.vel_threshold or accel_mag <= self.accel_threshold:
            return None

        self.last_hit_time = current_time
        return HitEvent(
            side=side,
            force=self.eff_mass_kg * accel_mag,
            speed=speed,
            accel=accel_mag,
            timestamp=current_time,
        )
