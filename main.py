"""Minimal CLI demo for Lux Punch."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from analyzer import HitEvent, PunchAnalyzer
from calibration import estimate_pixel_to_meter
from pose_utils import LEFT_WRIST, RIGHT_WRIST, landmark_to_metric_point, landmarks_to_array

DEFAULT_BODY_MASS_KG = 70.0
DEFAULT_HEIGHT_CM = 175.0
DEFAULT_EFF_MASS_FRACTION = 0.05
DEFAULT_ACCEL_THRESHOLD = 8.0
DEFAULT_VEL_THRESHOLD = 2.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lux Punch demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live mode")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a synthetic motion sequence instead of using a webcam",
    )
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames for live mode")
    parser.add_argument("--weight-kg", type=float, default=DEFAULT_BODY_MASS_KG)
    parser.add_argument("--height-cm", type=float, default=DEFAULT_HEIGHT_CM)
    parser.add_argument("--vel-threshold", type=float, default=DEFAULT_VEL_THRESHOLD)
    parser.add_argument("--acc-threshold", type=float, default=DEFAULT_ACCEL_THRESHOLD)
    return parser


def run_synthetic_demo(args: argparse.Namespace) -> int:
    analyzer = PunchAnalyzer(
        eff_mass_kg=args.weight_kg * DEFAULT_EFF_MASS_FRACTION,
        vel_threshold=args.vel_threshold,
        accel_threshold=args.acc_threshold,
        hit_rearm_seconds=0.25,
    )

    timestamps = np.linspace(0.0, 1.5, 30)
    positions = []
    for t in timestamps:
        if t < 0.4:
            x = 0.2 + 0.3 * t
        elif t < 0.7:
            x = 0.32 + 1.8 * (t - 0.4)
        else:
            x = 0.86 - 0.2 * (t - 0.7)
        positions.append(np.array([x, 0.0, 0.0]))

    print("Running synthetic demo...")
    for timestamp, position in zip(timestamps, positions, strict=True):
        analyzer.push("R", float(timestamp), position)
        event = analyzer.detect(now=float(timestamp))
        if event is not None:
            _print_event(event)
    print("Synthetic demo complete.")
    return 0


def _print_event(event: HitEvent) -> None:
    print(
        f"[{event.side}] force={event.force:.2f} N "
        f"speed={event.speed:.2f} m/s accel={event.accel:.2f} m/s^2"
    )


def run_camera_demo(args: argparse.Namespace) -> int:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime integration
        print(f"OpenCV is required for live mode: {exc}", file=sys.stderr)
        return 1

    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime integration
        print(f"MediaPipe is required for live mode: {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():  # pragma: no cover - runtime integration
        print("Could not open camera", file=sys.stderr)
        return 1

    analyzer = PunchAnalyzer(
        eff_mass_kg=args.weight_kg * DEFAULT_EFF_MASS_FRACTION,
        vel_threshold=args.vel_threshold,
        accel_threshold=args.acc_threshold,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = 0
    pixel_to_meter: float | None = None
    try:
        while frames < args.max_frames:  # pragma: no cover - runtime integration
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            image_h, image_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks is None:
                continue

            landmarks = landmarks_to_array(result.pose_landmarks.landmark)
            if pixel_to_meter is None:
                pixel_to_meter, _ = estimate_pixel_to_meter(
                    landmarks,
                    image_w,
                    image_h,
                    height_cm=args.height_cm,
                )
            if pixel_to_meter is None:
                continue

            timestamp = time.time()
            left_point = landmark_to_metric_point(
                landmarks[LEFT_WRIST], image_w, image_h, pixel_to_meter
            )
            right_point = landmark_to_metric_point(
                landmarks[RIGHT_WRIST], image_w, image_h, pixel_to_meter
            )
            analyzer.push("L", timestamp, left_point)
            analyzer.push("R", timestamp, right_point)
            event = analyzer.detect(now=timestamp)
            if event is not None:
                _print_event(event)
    finally:  # pragma: no cover - runtime integration
        pose.close()
        cap.release()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.synthetic:
        return run_synthetic_demo(args)
    return run_camera_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
