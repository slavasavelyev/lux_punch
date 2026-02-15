import numpy as np
from collections import deque
import math
import time

# MediaPipe indices (pose): shoulders 11 (left), 12 (right), elbows 13/14, wrists 15/16
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

def landmarks_to_array(landmarks):
    """
    landmarks: list of mediapipe landmarks (with .x,.y,.z) or numpy array already.
    returns numpy array shape (N,3) with coords normalized [0..1] for x,y and z (if present).
    """
    if isinstance(landmarks, np.ndarray):
        return landmarks
    arr = []
    for lm in landmarks:
        arr.append([lm.x, lm.y, lm.z if hasattr(lm, "z") else 0.0])
    return np.array(arr, dtype=np.float32)

def pixel_distance(a, b, image_w, image_h):
    """Euclidean pixel distance between two normalized landmarks a,b (each [x,y,z])."""
    ax, ay = a[0]*image_w, a[1]*image_h
    bx, by = b[0]*image_w, b[1]*image_h
    return math.hypot(ax - bx, ay - by)

class EMASmoother:
    """Exponential moving average for vectors"""
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.value = None

    def update(self, v):
        v = np.array(v, dtype=float)
        if self.value is None:
            self.value = v
        else:
            self.value = self.alpha * v + (1.0 - self.alpha) * self.value
        return self.value

class KinematicBuffer:
    """
    Keeps last N timestamped 3D points and computes velocity/acceleration in meters/sec.
    Coordinates should be provided in meters (x,y,z).
    """
    def __init__(self, maxlen=32):
        self.maxlen = maxlen
        self.buf = deque(maxlen=maxlen)  # elements: (t, np.array([x,y,z]))

    def push(self, t, pos3d):
        pos = np.array(pos3d, dtype=float)
        self.buf.append((t, pos))

    def clear(self):
        self.buf.clear()

    def latest(self):
        return self.buf[-1] if self.buf else (None, None)

    def velocity(self, n=1):
        """
        Returns instantaneous velocity vector approximated by difference between last and n-th last
        in m/s. If not enough data -> None
        """
        if len(self.buf) <= n:
            return None
        t1, p1 = self.buf[-1]
        t0, p0 = self.buf[-1 - n]
        dt = t1 - t0
        if dt <= 1e-6:
            return None
        return (p1 - p0) / dt

    def acceleration(self, n=1):
        """
        Approximate acceleration by difference of velocities.
        """
        if len(self.buf) <= 2*n:
            return None
        v1 = self._vel_between(-1, -1 - n)
        v0 = self._vel_between(-1 - n, -1 - 2*n)
        if v1 is None or v0 is None:
            return None
        # times
        t1 = self.buf[-1][0]
        t0 = self.buf[-1 - n][0]
        dt = t1 - t0
        if dt <= 1e-6:
            return None
        return (v1 - v0) / dt

    def _vel_between(self, idx_a, idx_b):
        try:
            t1, p1 = self.buf[idx_a]
            t0, p0 = self.buf[idx_b]
        except IndexError:
            return None
        dt = t1 - t0
        if dt <= 1e-6:
            return None
        return (p1 - p0) / dt
