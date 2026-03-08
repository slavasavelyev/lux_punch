"""Calibration utilities for mapping pixel motion to metric motion."""

from __future__ import annotations

from pose_utils import LEFT_SHOULDER, RIGHT_SHOULDER, pixel_distance

DEFAULT_SHOULDER_RATIO = 0.22
DEFAULT_FALLBACK_SHOULDER_WIDTH_M = 0.45


def estimate_pixel_to_meter(
    landmarks,
    image_w: int,
    image_h: int,
    known_shoulder_width_m: float | None = None,
    height_cm: float | None = None,
    shoulder_ratio: float = DEFAULT_SHOULDER_RATIO,
) -> tuple[float | None, float | None]:
    """Estimate meters-per-pixel from shoulder width.

    Returns ``(meters_per_pixel, measured_pixel_shoulder_width)``.
    """
    if landmarks is None or len(landmarks) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
        return None, None

    measured_pixels = pixel_distance(
        landmarks[LEFT_SHOULDER],
        landmarks[RIGHT_SHOULDER],
        image_w,
        image_h,
    )
    if measured_pixels <= 1e-9:
        return None, None

    width_m = known_shoulder_width_m
    if width_m is None:
        if height_cm is not None:
            width_m = shoulder_ratio * float(height_cm) / 100.0
        else:
            width_m = DEFAULT_FALLBACK_SHOULDER_WIDTH_M

    return width_m / measured_pixels, measured_pixels
