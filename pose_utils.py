"""Core kinematic helpers for Lux Punch."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16


ArrayLike = Sequence[float] | np.ndarray


def landmarks_to_array(landmarks: Iterable[object] | np.ndarray) -> np.ndarray:
    """Convert pose landmarks into a float32 NumPy array of shape ``(N, 3)``.

    Accepts:
    - a NumPy array already in the target shape,
    - MediaPipe-style objects with ``x``, ``y``, ``z`` attributes,
    - tuples/lists of length 2 or 3,
    - dictionaries with ``x``/``y``/``z`` keys.
    """
    if isinstance(landmarks, np.ndarray):
        arr = np.asarray(landmarks, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            msg = "landmarks array must have shape (N, 2+)"
            raise ValueError(msg)
        if arr.shape[1] == 2:
            zeros = np.zeros((arr.shape[0], 1), dtype=np.float32)
            arr = np.concatenate([arr, zeros], axis=1)
        return arr[:, :3]

    rows: list[list[float]] = []
    for landmark in landmarks:
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            x = float(landmark.x)
            y = float(landmark.y)
            z = float(getattr(landmark, "z", 0.0))
        elif isinstance(landmark, dict):
            x = float(landmark["x"])
            y = float(landmark["y"])
            z = float(landmark.get("z", 0.0))
        else:
            values = list(landmark)
            if len(values) < 2:
                msg = "landmark tuples must contain at least x and y"
                raise ValueError(msg)
            x = float(values[0])
            y = float(values[1])
            z = float(values[2]) if len(values) > 2 else 0.0
        rows.append([x, y, z])
    return np.array(rows, dtype=np.float32)


def pixel_distance(a: ArrayLike, b: ArrayLike, image_w: int, image_h: int) -> float:
    """Return the 2D Euclidean distance in pixels between two normalized points."""
    ax = float(a[0]) * image_w
    ay = float(a[1]) * image_h
    bx = float(b[0]) * image_w
    by = float(b[1]) * image_h
    return math.hypot(ax - bx, ay - by)


def landmark_to_metric_point(
    point: ArrayLike,
    image_w: int,
    image_h: int,
    meters_per_pixel: float,
) -> np.ndarray:
    """Map a normalized landmark point to metric coordinates.

    ``x`` grows to the right, ``y`` grows upward, and ``z`` follows the sign
    reported by the landmark source after applying the same scale.
    """
    px = float(point[0]) * image_w
    py = float(point[1]) * image_h
    pz = float(point[2]) * image_w
    return np.array([px * meters_per_pixel, -py * meters_per_pixel, pz * meters_per_pixel])


class EMASmoother:
    """Exponential moving average for vectors or scalars."""

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.0 < alpha <= 1.0:
            msg = "alpha must be in (0, 1]"
            raise ValueError(msg)
        self.alpha = alpha
        self.value: np.ndarray | None = None

    def update(self, value: ArrayLike) -> np.ndarray:
        current = np.array(value, dtype=float)
        if self.value is None:
            self.value = current
        else:
            self.value = self.alpha * current + (1.0 - self.alpha) * self.value
        return self.value.copy()


class KinematicBuffer:
    """Keep recent metric positions and estimate velocity / acceleration."""

    def __init__(self, maxlen: int = 32) -> None:
        self.maxlen = maxlen
        self.buf: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)

    def push(self, timestamp: float, pos3d: ArrayLike) -> None:
        self.buf.append((float(timestamp), np.array(pos3d, dtype=float)))

    def clear(self) -> None:
        self.buf.clear()

    def latest(self) -> tuple[float | None, np.ndarray | None]:
        if not self.buf:
            return None, None
        return self.buf[-1][0], self.buf[-1][1].copy()

    def velocity(self, n: int = 1) -> np.ndarray | None:
        if len(self.buf) <= n:
            return None
        t1, p1 = self.buf[-1]
        t0, p0 = self.buf[-1 - n]
        dt = t1 - t0
        if dt <= 1e-9:
            return None
        return (p1 - p0) / dt

    def acceleration(self, n: int = 1) -> np.ndarray | None:
        if len(self.buf) <= 2 * n:
            return None
        v1 = self._vel_between(-1, -1 - n)
        v0 = self._vel_between(-1 - n, -1 - 2 * n)
        if v1 is None or v0 is None:
            return None
        t1 = self.buf[-1][0]
        t0 = self.buf[-1 - n][0]
        dt = t1 - t0
        if dt <= 1e-9:
            return None
        return (v1 - v0) / dt

    def _vel_between(self, idx_a: int, idx_b: int) -> np.ndarray | None:
        try:
            t1, p1 = self.buf[idx_a]
            t0, p0 = self.buf[idx_b]
        except IndexError:
            return None
        dt = t1 - t0
        if dt <= 1e-9:
            return None
        return (p1 - p0) / dt
