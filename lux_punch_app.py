"""Small desktop wrapper for Lux Punch.

The GUI is intentionally thin and keeps heavyweight imports local to runtime.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from analyzer import PunchAnalyzer
from calibration import estimate_pixel_to_meter
from pose_utils import LEFT_WRIST, RIGHT_WRIST, landmark_to_metric_point, landmarks_to_array

SETTINGS_FILE = Path("user_settings.json")
DEFAULT_SETTINGS = {
    "weight_kg": 70.0,
    "height_cm": 175.0,
    "vel_threshold": 2.0,
    "acc_threshold": 8.0,
}


def load_settings() -> dict[str, float]:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_SETTINGS)
            merged.update({k: float(v) for k, v in data.items() if k in DEFAULT_SETTINGS})
            return merged
        except Exception:
            return dict(DEFAULT_SETTINGS)
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict[str, float]) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def launch_gui() -> int:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    except Exception as exc:  # pragma: no cover - integration only
        print(
            "GUI dependencies are missing. Install runtime deps with: pip install -r requirements.txt\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1

    class MainWindow(QtWidgets.QMainWindow):  # pragma: no cover - integration only
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Lux Punch")
            self.resize(960, 720)
            self.settings = load_settings()
            self.capture = None
            self.pose = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._tick)
            self.pixel_to_meter: float | None = None
            self.analyzer = PunchAnalyzer(
                eff_mass_kg=self.settings["weight_kg"] * 0.05,
                vel_threshold=self.settings["vel_threshold"],
                accel_threshold=self.settings["acc_threshold"],
            )
            self._build_ui()

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)

            controls = QtWidgets.QHBoxLayout()
            self.start_button = QtWidgets.QPushButton("Start camera")
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.stop_button.setEnabled(False)
            self.synthetic_button = QtWidgets.QPushButton("Run synthetic hit")
            controls.addWidget(self.start_button)
            controls.addWidget(self.stop_button)
            controls.addWidget(self.synthetic_button)
            controls.addStretch(1)
            layout.addLayout(controls)

            form = QtWidgets.QFormLayout()
            self.weight_spin = QtWidgets.QDoubleSpinBox()
            self.weight_spin.setRange(20.0, 200.0)
            self.weight_spin.setValue(self.settings["weight_kg"])
            self.height_spin = QtWidgets.QDoubleSpinBox()
            self.height_spin.setRange(100.0, 230.0)
            self.height_spin.setValue(self.settings["height_cm"])
            self.vel_spin = QtWidgets.QDoubleSpinBox()
            self.vel_spin.setRange(0.5, 10.0)
            self.vel_spin.setValue(self.settings["vel_threshold"])
            self.acc_spin = QtWidgets.QDoubleSpinBox()
            self.acc_spin.setRange(1.0, 30.0)
            self.acc_spin.setValue(self.settings["acc_threshold"])
            form.addRow("Weight (kg)", self.weight_spin)
            form.addRow("Height (cm)", self.height_spin)
            form.addRow("Velocity threshold", self.vel_spin)
            form.addRow("Acceleration threshold", self.acc_spin)
            layout.addLayout(form)

            self.video_label = QtWidgets.QLabel("Camera preview")
            self.video_label.setMinimumHeight(420)
            self.video_label.setAlignment(QtCore.Qt.AlignCenter)
            self.video_label.setStyleSheet("background: #111; color: #eee;")
            layout.addWidget(self.video_label)

            self.status_label = QtWidgets.QLabel("Ready")
            self.metrics_label = QtWidgets.QLabel("No hit yet")
            self.event_log = QtWidgets.QPlainTextEdit()
            self.event_log.setReadOnly(True)
            layout.addWidget(self.status_label)
            layout.addWidget(self.metrics_label)
            layout.addWidget(self.event_log)

            self.start_button.clicked.connect(self.start_camera)
            self.stop_button.clicked.connect(self.stop_camera)
            self.synthetic_button.clicked.connect(self.run_synthetic_hit)
            self.weight_spin.valueChanged.connect(self._save_form_settings)
            self.height_spin.valueChanged.connect(self._save_form_settings)
            self.vel_spin.valueChanged.connect(self._save_form_settings)
            self.acc_spin.valueChanged.connect(self._save_form_settings)

        def _save_form_settings(self) -> None:
            self.settings["weight_kg"] = float(self.weight_spin.value())
            self.settings["height_cm"] = float(self.height_spin.value())
            self.settings["vel_threshold"] = float(self.vel_spin.value())
            self.settings["acc_threshold"] = float(self.acc_spin.value())
            save_settings(self.settings)
            self.analyzer = PunchAnalyzer(
                eff_mass_kg=self.settings["weight_kg"] * 0.05,
                vel_threshold=self.settings["vel_threshold"],
                accel_threshold=self.settings["acc_threshold"],
            )

        def start_camera(self) -> None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.status_label.setText("Could not open camera")
                self.capture = None
                return
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pixel_to_meter = None
            self.timer.start(33)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Camera running")

        def stop_camera(self) -> None:
            self.timer.stop()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            if self.pose is not None:
                self.pose.close()
                self.pose = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Camera stopped")

        def closeEvent(self, event: Any) -> None:  # noqa: N802
            self.stop_camera()
            super().closeEvent(event)

        def run_synthetic_hit(self) -> None:
            base_time = time.time()
            sequence = [
                (0.00, [0.00, 0.00, 0.00]),
                (0.05, [0.04, 0.00, 0.00]),
                (0.10, [0.12, 0.00, 0.00]),
                (0.15, [0.35, 0.00, 0.00]),
                (0.20, [0.62, 0.00, 0.00]),
            ]
            event = None
            for dt, pos in sequence:
                now = base_time + dt
                self.analyzer.push("R", now, pos)
                event = self.analyzer.detect(now=now)
            if event is None:
                self.metrics_label.setText("Synthetic hit did not cross thresholds")
                return
            self._record_event(event.side, event.force, event.speed, event.accel)

        def _tick(self) -> None:
            if self.capture is None or self.pose is None:
                return
            ok, frame = self.capture.read()
            if not ok:
                self.status_label.setText("Camera read failed")
                self.stop_camera()
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            preview = frame.copy()
            if result.pose_landmarks is not None:
                image_h, image_w = frame.shape[:2]
                landmarks = landmarks_to_array(result.pose_landmarks.landmark)
                if self.pixel_to_meter is None:
                    self.pixel_to_meter, _ = estimate_pixel_to_meter(
                        landmarks,
                        image_w,
                        image_h,
                        height_cm=self.settings["height_cm"],
                    )
                if self.pixel_to_meter is not None:
                    timestamp = time.time()
                    left_point = landmark_to_metric_point(
                        landmarks[LEFT_WRIST], image_w, image_h, self.pixel_to_meter
                    )
                    right_point = landmark_to_metric_point(
                        landmarks[RIGHT_WRIST], image_w, image_h, self.pixel_to_meter
                    )
                    self.analyzer.push("L", timestamp, left_point)
                    self.analyzer.push("R", timestamp, right_point)
                    event = self.analyzer.detect(now=timestamp)
                    if event is not None:
                        self._record_event(event.side, event.force, event.speed, event.accel)
                    for point in (landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST]):
                        x = int(point[0] * image_w)
                        y = int(point[1] * image_h)
                        cv2.circle(preview, (x, y), 8, (0, 255, 255), -1)
            self._set_preview(preview)

        def _set_preview(self, frame) -> None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QtGui.QImage.Format_RGB888,
            )
            pixmap = QtGui.QPixmap.fromImage(image)
            self.video_label.setPixmap(
                pixmap.scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )

        def _record_event(self, side: str, force: float, speed: float, accel: float) -> None:
            self.metrics_label.setText(
                f"{side} hit | force={force:.1f} N | speed={speed:.2f} m/s | accel={accel:.2f} m/s²"
            )
            self.event_log.appendPlainText(self.metrics_label.text())

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
