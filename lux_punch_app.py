"""
Developed & Designed by Slava Savelyev
Lux Punch update 2.3 - optimization and icon change

Optimized: lazy-loads MediaPipe (background init), sets window icon
so the icon appears in taskbar/alt-tab/title bar (works with PyInstaller),
and reduces startup time by deferring heavy imports.
"""
import sys
import os
import time
import json
import threading
from collections import deque

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# local modules (unchanged)
from pose_utils import landmarks_to_array, LEFT_WRIST, RIGHT_WRIST, KinematicBuffer
from calibration import estimate_pixel_to_meter

# matplotlib for small graph (lazy import inside)
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# NOTE: do NOT import mediapipe here to keep startup fast
# We'll import it lazily inside the model init routine

# ---------------- CONFIG / DEFAULTS ----------------
USER_SETTINGS_FILE = "user_settings.json"

DEFAULT_BODY_MASS_KG = 70.0
DEFAULT_HEIGHT_CM = 175.0
DEFAULT_SHOULDER_RATIO = 0.22
EFF_MASS_FRAC = 0.05
SEQ_HISTORY = 32
DEFAULT_ACCEL_THRESHOLD = 8.0  # m/s^2
DEFAULT_VEL_THRESHOLD = 2.0    # m/s
HIT_REARM_SECONDS = 0.5

# --- processing scale: downscale frames for MediaPipe processing
PROCESS_SCALE = 0.6  # 60% of original pixels for pose processing

# --- graph throttling
GRAPH_UPDATE_INTERVAL = 0.12  # seconds


# ---------------- Settings persistence ----------------
def load_user_settings():
    default = {
        "weight_kg": DEFAULT_BODY_MASS_KG,
        "height_cm": DEFAULT_HEIGHT_CM,
        "vel_threshold": DEFAULT_VEL_THRESHOLD,
        "acc_threshold": DEFAULT_ACCEL_THRESHOLD,
        "last_pixel_to_meter": None
    }
    try:
        if os.path.exists(USER_SETTINGS_FILE):
            with open(USER_SETTINGS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for k, v in default.items():
                if k not in obj:
                    obj[k] = v
            return obj
    except Exception:
        pass
    return default


def save_user_settings(settings):
    try:
        with open(USER_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


# ---------------- Analyzer ----------------
class PunchAnalyzer:
    __slots__ = ("left_buf", "right_buf", "eff_mass", "vel_thresh", "accel_thresh", "last_hit_time")

    def __init__(self, eff_mass, vel_thresh, accel_thresh):
        self.left_buf = KinematicBuffer(maxlen=SEQ_HISTORY)
        self.right_buf = KinematicBuffer(maxlen=SEQ_HISTORY)
        self.eff_mass = eff_mass
        self.vel_thresh = vel_thresh
        self.accel_thresh = accel_thresh
        self.last_hit_time = -999.0

    def push(self, side, t, pos):
        if side == "L":
            self.left_buf.push(t, pos)
        else:
            self.right_buf.push(t, pos)

    def _check(self, buf):
        vel = buf.velocity(n=1)
        acc = buf.acceleration(n=1)
        if vel is None or acc is None:
            return None
        speed = float(np.linalg.norm(vel))
        accel_mag = float(np.linalg.norm(acc))
        now = time.time()
        if now - self.last_hit_time < HIT_REARM_SECONDS:
            return None
        if speed > self.vel_thresh and accel_mag > self.accel_thresh:
            force = self.eff_mass * accel_mag
            self.last_hit_time = now
            return {"force": force, "speed": speed, "accel": accel_mag, "time": now}
        return None

    def detect(self):
        h = self._check(self.left_buf)
        if h:
            h["side"] = "Left"
            return h
        h = self._check(self.right_buf)
        if h:
            h["side"] = "Right"
            return h
        return None


# ---------------- Mini graph (matplotlib if available) ----------------
class MiniGraphWidget(QtWidgets.QWidget):
    def __init__(self, maxlen=120, parent=None):
        super().__init__(parent)
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        self.has_canvas = False
        if MATPLOTLIB_AVAILABLE:
            try:
                self.canvas = FigureCanvas(Figure(figsize=(3.6, 1.0), dpi=100))
                self.ax = self.canvas.figure.subplots()
                self.ax.axis("off")
                self.line, = self.ax.plot([], lw=1.5)
                self.canvas.figure.tight_layout()
                layout = QtWidgets.QVBoxLayout(self)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.canvas)
                self.has_canvas = True
                self.last_draw = 0.0
            except Exception:
                self.has_canvas = False
                self.canvas = None
        else:
            l = QtWidgets.QLabel("Graph not available")
            l.setAlignment(Qt.AlignCenter)
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(l)

    def push(self, v):
        self.data.append(float(v))
        if self.has_canvas:
            now = time.time()
            if now - getattr(self, "last_draw", 0) > GRAPH_UPDATE_INTERVAL:
                self._redraw()
                self.last_draw = now

    def _redraw(self):
        if not self.has_canvas:
            return
        arr = list(self.data)
        if not arr:
            arr = [0.0]
        self.line.set_data(range(len(arr)), arr)
        mx = max(arr) if arr else 1.0
        self.ax.set_xlim(0, self.maxlen)
        self.ax.set_ylim(0, max(1.0, mx * 1.2))
        try:
            self.canvas.draw_idle()
        except Exception:
            pass


# ---------------- Main Window ----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lux Punch — AR Analyzer")
        self.resize(1160, 720)

        # set window icon (works in taskbar / titlebar). When packaged, use sys._MEIPASS
        icon_path = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(__file__)), "lux_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        # load settings
        self.settings = load_user_settings()

        # central UI layout (unchanged styling)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(14, 14, 14, 14)
        h.setSpacing(14)
        central.setStyleSheet("background: #e9eef1;")

        # left: video
        self.video = QtWidgets.QLabel()
        self.video.setMinimumSize(820, 600)
        self.video.setStyleSheet("background: #0f1113; border-radius: 10px; border: 1px solid #1e2326;")
        self.video.setAlignment(Qt.AlignCenter)
        h.addWidget(self.video, 1)

        # right: controls
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        h.addLayout(right)

        # profile card (glass-like)
        profile_card = QtWidgets.QFrame()
        profile_card.setFixedWidth(320)
        profile_card.setStyleSheet("background: rgba(255,255,255,0.88); border-radius: 12px;")
        pf = QtWidgets.QFormLayout(profile_card)
        pf.setContentsMargins(12, 10, 12, 10)
        pf.setSpacing(8)

        self.weight_spin = QtWidgets.QDoubleSpinBox()
        self.weight_spin.setRange(30.0, 200.0)
        self.weight_spin.setValue(float(self.settings.get("weight_kg", DEFAULT_BODY_MASS_KG)))
        self.height_spin = QtWidgets.QDoubleSpinBox()
        self.height_spin.setRange(120.0, 230.0)
        self.height_spin.setValue(float(self.settings.get("height_cm", DEFAULT_HEIGHT_CM)))

        pf.addRow(QtWidgets.QLabel("Weight (kg)"), self.weight_spin)
        pf.addRow(QtWidgets.QLabel("Height (cm)"), self.height_spin)

        self.apply_profile_btn = QtWidgets.QPushButton("Apply & Save Profile")
        self.apply_profile_btn.setStyleSheet("padding:8px; border-radius:8px; background:#2f80ed; color:white;")
        pf.addRow(self.apply_profile_btn)

        # quick info labels
        self.info_mass_label = QtWidgets.QLabel("")
        self.info_mass_label.setStyleSheet("color:#1b2b33; font-size:12px;")
        pf.addRow(QtWidgets.QLabel("Profile (applied):"), self.info_mass_label)

        right.addWidget(profile_card)

        # thresholds card
        thresh_card = QtWidgets.QFrame()
        thresh_card.setFixedWidth(320)
        thresh_card.setStyleSheet("background: rgba(255,255,255,0.88); border-radius: 12px;")
        tf = QtWidgets.QFormLayout(thresh_card)
        tf.setContentsMargins(12, 10, 12, 10)
        tf.setSpacing(8)

        self.vel_spin = QtWidgets.QDoubleSpinBox()
        self.vel_spin.setRange(0.2, 20.0)
        self.vel_spin.setValue(float(self.settings.get("vel_threshold", DEFAULT_VEL_THRESHOLD)))
        self.acc_spin = QtWidgets.QDoubleSpinBox()
        self.acc_spin.setRange(1.0, 80.0)
        self.acc_spin.setValue(float(self.settings.get("acc_threshold", DEFAULT_ACCEL_THRESHOLD)))

        tf.addRow(QtWidgets.QLabel("Velocity threshold (m/s)"), self.vel_spin)
        tf.addRow(QtWidgets.QLabel("Acceleration threshold (m/s²)"), self.acc_spin)

        right.addWidget(thresh_card)

        # action buttons
        actions = QtWidgets.QHBoxLayout()
        self.btn_calib = QtWidgets.QPushButton("Calibrate (shoulders)")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_export = QtWidgets.QPushButton("Export log CSV")
        for btn in (self.btn_calib, self.btn_start, self.btn_export):
            btn.setFixedHeight(36)
            btn.setStyleSheet("border-radius:8px; padding:6px;")
        self.btn_start.setStyleSheet("background:#2f3136; color:white; border-radius:8px;")
        self.btn_calib.setStyleSheet("background:#1f8bff; color:white; border-radius:8px;")
        self.btn_export.setStyleSheet("background:#4b5563; color:white; border-radius:8px;")

        actions.addWidget(self.btn_calib)
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_export)
        right.addLayout(actions)

        # metrics card
        metrics_card = QtWidgets.QFrame()
        metrics_card.setFixedWidth(320)
        metrics_card.setStyleSheet("background: rgba(255,255,255,0.92); border-radius: 12px;")
        mlay = QtWidgets.QVBoxLayout(metrics_card)
        mlay.setContentsMargins(10, 8, 10, 8)
        mlay.setSpacing(6)

        self.lbl_eff_mass = QtWidgets.QLabel("")
        self.lbl_eff_mass.setStyleSheet("font-weight:600; color:#0f1720;")
        mlay.addWidget(self.lbl_eff_mass)
        self.lbl_force = QtWidgets.QLabel("Last hit force: — N")
        mlay.addWidget(self.lbl_force)
        self.lbl_speed = QtWidgets.QLabel("Speed: — m/s")
        mlay.addWidget(self.lbl_speed)
        self.lbl_acc = QtWidgets.QLabel("Peak accel: — m/s²")
        mlay.addWidget(self.lbl_acc)
        mlay.addStretch()

        right.addWidget(metrics_card)

        # graph and log
        graph_card = QtWidgets.QFrame()
        graph_card.setFixedWidth(320)
        graph_card.setStyleSheet("background: transparent;")
        glay = QtWidgets.QVBoxLayout(graph_card)
        glay.setContentsMargins(0, 0, 0, 0)
        glay.setSpacing(6)
        self.graph_widget = MiniGraphWidget(maxlen=160)
        glay.addWidget(self.graph_widget)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(170)
        self.log.setStyleSheet("background:#0b0d0f; color:#e6eef2; border-radius:8px; padding:6px;")
        glay.addWidget(QtWidgets.QLabel("Scientific Event Log:"))
        glay.addWidget(self.log)
        right.addWidget(graph_card)

        # footer
        footer = QtWidgets.QLabel("Developed & Designed by Slava Savelyev")
        footer.setStyleSheet("color:#6b7280; font-size:11px;")
        right.addWidget(footer)

        # internal state
        self.cap = None
        self.pose = None
        self.mp_module = None
        self.model_thread = None
        self.pose_ready = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)
        self.last_graph_time = 0.0

        # calibration
        self.pixel_to_meter = self.settings.get("last_pixel_to_meter", None)
        self.shoulder_px = None

        # analyzer init
        eff_mass = float(self.weight_spin.value()) * EFF_MASS_FRAC
        self.analyzer = PunchAnalyzer(eff_mass=eff_mass,
                                      vel_thresh=float(self.vel_spin.value()),
                                      accel_thresh=float(self.acc_spin.value()))
        self.recent_forces = deque(maxlen=300)
        self.hit_records = []

        # connect signals
        self.btn_start.clicked.connect(self._toggle_capture)
        self.btn_calib.clicked.connect(self._calibrate_once)
        self.btn_export.clicked.connect(self._export_csv)
        self.apply_profile_btn.clicked.connect(self._apply_and_save_profile)
        self.vel_spin.valueChanged.connect(self._update_thresholds)
        self.acc_spin.valueChanged.connect(self._update_thresholds)

        central.setFocusPolicy(QtCore.Qt.StrongFocus)
        central.keyPressEvent = self._on_key

        self._refresh_profile_ui()

        # store target pixmap size to avoid repeated size recalcs
        self._cached_target_size = (self.video.width(), self.video.height())

    # ---------------- UI helpers ----------------
    def _on_key(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def _refresh_profile_ui(self):
        eff_mass = float(self.weight_spin.value()) * EFF_MASS_FRAC
        self.lbl_eff_mass.setText(f"Effective mass: {eff_mass:.3f} kg (frac {EFF_MASS_FRAC:.3f})")
        self.info_mass_label.setText(f"{self.weight_spin.value():.0f} kg, {self.height_spin.value():.0f} cm")
        self.lbl_force.setText("Last hit force: — N")
        self.lbl_speed.setText("Speed: — m/s")
        self.lbl_acc.setText("Peak accel: — m/s²")

    def _apply_and_save_profile(self):
        self.settings["weight_kg"] = float(self.weight_spin.value())
        self.settings["height_cm"] = float(self.height_spin.value())
        self.settings["vel_threshold"] = float(self.vel_spin.value())
        self.settings["acc_threshold"] = float(self.acc_spin.value())
        if self.pixel_to_meter:
            self.settings["last_pixel_to_meter"] = float(self.pixel_to_meter)
        save_user_settings(self.settings)
        self.analyzer.eff_mass = float(self.weight_spin.value()) * EFF_MASS_FRAC
        self.analyzer.vel_thresh = float(self.vel_spin.value())
        self.analyzer.accel_thresh = float(self.acc_spin.value())
        self._refresh_profile_ui()
        self.log.append("Profile applied and saved.")

    def _update_thresholds(self):
        self.analyzer.vel_thresh = float(self.vel_spin.value())
        self.analyzer.accel_thresh = float(self.acc_spin.value())
        self._refresh_profile_ui()

    # ---------------- Capture control ----------------
    def _toggle_capture(self):
        if self.cap is None:
            self._start_capture()
        else:
            self._stop_capture()

    def _start_capture(self):
        # open camera quickly
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, "Camera", "Cannot open camera")
                self.cap = None
                return
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Camera", "Cannot open camera")
            self.cap = None
            return

        # apply analyzer params
        self.analyzer.eff_mass = float(self.weight_spin.value()) * EFF_MASS_FRAC
        self.analyzer.vel_thresh = float(self.vel_spin.value())
        self.analyzer.accel_thresh = float(self.acc_spin.value())

        # start model init in background thread (lazy load mediapipe there)
        if not self.pose_ready and (self.model_thread is None or not self.model_thread.is_alive()):
            self.btn_start.setText("Starting...")
            self.btn_start.setEnabled(False)
            self.model_thread = threading.Thread(target=self._init_pose_model, daemon=True)
            self.model_thread.start()
        else:
            # already ready
            self.btn_start.setText("Stop")
            self.timer.start(33)
            self.log.append("Capture started")

    def _init_pose_model(self):
        try:
            import mediapipe as mp_local  # lazy import
            # create pose object (may take some time)
            pose_local = mp_local.solutions.pose.Pose(static_image_mode=False,
                                                     model_complexity=1,
                                                     min_detection_confidence=0.5,
                                                     min_tracking_confidence=0.5)
            # assign to instance safely
            self.mp_module = mp_local
            self.pose = pose_local
            self.pose_ready = True
            # Ensure UI updates happen in main thread
            QtCore.QMetaObject.invokeMethod(self, "_on_pose_ready", Qt.QueuedConnection)
        except Exception as e:
            # report error in UI thread
            QtCore.QMetaObject.invokeMethod(self, "_on_pose_failed", Qt.QueuedConnection,
                                            QtCore.Q_ARG(str, str(e)))

    @QtCore.pyqtSlot()
    def _on_pose_ready(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Stop")
        self.timer.start(33)
        self.log.append("Model initialized. Capture started")

    @QtCore.pyqtSlot(str)
    def _on_pose_failed(self, err):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Start")
        self.log.append("Model init failed: " + err)
        QtWidgets.QMessageBox.critical(self, "MediaPipe", f"Failed to initialize model: {err}")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _stop_capture(self):
        self.timer.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        # keep pose object if you want faster restart (do not recreate)
        # if you prefer full cleanup uncomment below:
        # if self.pose:
        #     try: self.pose.close()
        #     except: pass
        # self.pose = None
        self.btn_start.setText("Start")
        self.log.append("Capture stopped")

    # ---------------- Calibration ----------------
    def _calibrate_once(self):
        if not self.cap:
            QtWidgets.QMessageBox.information(self, "Calibrate", "Start the camera first.")
            return
        ret, frame = self.cap.read()
        if not ret:
            QtWidgets.QMessageBox.warning(self, "Calibrate", "Could not read frame.")
            return
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * PROCESS_SCALE), int(h * PROCESS_SCALE)), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if not self.pose_ready:
            QtWidgets.QMessageBox.information(self, "Calibrate", "Model not ready yet. Start capture first.")
            return
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            QtWidgets.QMessageBox.information(self, "Calibrate", "Pose not found. Try again.")
            return
        lms_small = landmarks_to_array(res.pose_landmarks.landmark)
        shoulder_m = DEFAULT_SHOULDER_RATIO * (float(self.height_spin.value()) / 100.0)
        px2m, pix = estimate_pixel_to_meter(lms_small, w, h, known_shoulder_width_m=shoulder_m)
        if px2m:
            self.pixel_to_meter = px2m
            self.shoulder_px = pix
            self.settings["last_pixel_to_meter"] = float(px2m)
            save_user_settings(self.settings)
            self.log.append(f"Calibration OK: {px2m:.6f} m/px (shoulder px {pix:.1f})")
        else:
            QtWidgets.QMessageBox.information(self, "Calibrate", "Calibration failed (shoulders not found).")

    # ---------------- Main loop (optimized) ----------------
    def _tick(self):
        # fast path checks
        if not self.cap or not self.pose_ready:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        tnow = time.time()
        h, w = frame.shape[:2]

        # downscale for pose processing
        small_w = max(320, int(w * PROCESS_SCALE))
        small_h = max(240, int(h * PROCESS_SCALE))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        try:
            res = self.pose.process(rgb_small)
        except Exception:
            res = None

        display = frame  # operate on original for overlays

        if res and res.pose_landmarks:
            lms_small = landmarks_to_array(res.pose_landmarks.landmark)
            if self.pixel_to_meter is None:
                shoulder_m = DEFAULT_SHOULDER_RATIO * (float(self.height_spin.value()) / 100.0)
                px2m, pix = estimate_pixel_to_meter(lms_small, w, h, known_shoulder_width_m=shoulder_m)
                if px2m:
                    self.pixel_to_meter = px2m
                    self.settings["last_pixel_to_meter"] = float(px2m)
                    save_user_settings(self.settings)
                    self.log.append(f"Auto-calibration: {px2m:.6f} m/px")

            # push wrist positions
            for side in ("L", "R"):
                idx = LEFT_WRIST if side == "L" else RIGHT_WRIST
                if idx >= len(lms_small):
                    continue
                lm = lms_small[idx]
                px = lm[0] * w
                py = lm[1] * h
                if self.pixel_to_meter:
                    mx = (px - w * 0.5) * self.pixel_to_meter
                    my = (py - h * 0.5) * self.pixel_to_meter
                else:
                    mx = px
                    my = py
                mz = lm[2] * ((float(self.height_spin.value()) / 100.0) * 0.2) if self.pixel_to_meter else lm[2]
                self.analyzer.push(side, tnow, [mx, my, mz])

            # draw skeleton
            self._draw_skeleton(display, lms_small)

        # detection
        hit = self.analyzer.detect()
        if hit:
            self.lbl_force.setText(f"Last hit force: {hit['force']:.1f} N")
            self.lbl_speed.setText(f"Speed: {hit['speed']:.2f} m/s")
            self.lbl_acc.setText(f"Peak accel: {hit['accel']:.2f} m/s²")
            self.log.append(time.strftime("%H:%M:%S") + f"  [{hit['side']}] {hit['force']:.0f} N  v={hit['speed']:.2f} a={hit['accel']:.2f}")
            self.recent_forces.append(hit['force'])
            self.graph_widget.push(hit['force'])
            cv2.putText(display, "HIT!", (int(w*0.5)-50, 80), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 70, 70), 4, cv2.LINE_AA)
            summary = f"{hit['side']} {hit['force']:.0f} N"
            cv2.putText(display, summary, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            self.hit_records.append({
                "time": hit["time"],
                "side": hit["side"],
                "force": hit["force"],
                "speed": hit["speed"],
                "accel": hit["accel"]
            })
        else:
            self.graph_widget.push(0.0)

        # HUD overlay
        self._draw_hud(display)

        # convert to Qt pixmap (avoid extra copies where possible)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h2, w2 = rgb.shape[:2]
        bytes_per_line = 3 * w2
        qimg = QtGui.QImage(rgb.data, w2, h2, bytes_per_line, QtGui.QImage.Format_RGB888)
        # scale to widget once (fast transformation)
        target_w, target_h = self.video.width(), self.video.height()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.video.setPixmap(pix)

    # ---------------- Drawing helpers ----------------
    def _draw_skeleton(self, img, landmarks_norm, color=(90, 200, 160), thickness=2):
        h, w = img.shape[:2]
        pairs = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24)]
        for a, b in pairs:
            if a < len(landmarks_norm) and b < len(landmarks_norm):
                x1, y1 = int(landmarks_norm[a][0]*w), int(landmarks_norm[a][1]*h)
                x2, y2 = int(landmarks_norm[b][0]*w), int(landmarks_norm[b][1]*h)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        for idx in (15, 16):
            if idx < len(landmarks_norm):
                x, y = int(landmarks_norm[idx][0]*w), int(landmarks_norm[idx][1]*h)
                cv2.circle(img, (x, y), 5, (255, 80, 80), -1)

    def _draw_hud(self, img):
        h, w = img.shape[:2]
        overlay = img.copy()
        pw, ph = 360, 96
        x0, y0 = 12, 12
        cv2.rectangle(overlay, (x0, y0), (x0+pw, y0+ph), (18,20,22), -1)
        alpha = 0.44
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        cv2.putText(img, f"Mass: {self.weight_spin.value():.0f} kg  Eff: {self.analyzer.eff_mass:.3f} kg",
                    (x0+12, y0+28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)
        cv2.putText(img, f"Calib: {'yes' if self.pixel_to_meter else 'no'}  Vel>{self.analyzer.vel_thresh:.2f} m/s",
                    (x0+12, y0+54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(img, f"Acc>{self.analyzer.accel_thresh:.2f} m/s²",
                    (x0+12, y0+78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190,190,190), 1, cv2.LINE_AA)

    # ---------------- Export log ----------------
    def _export_csv(self):
        if not self.hit_records:
            QtWidgets.QMessageBox.information(self, "Export", "No hits recorded.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save hits", "lux_hits.csv", "CSV files (*.csv)")
        if not fname:
            return
        try:
            import csv
            with open(fname, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["time", "side", "force_N", "speed_m_s", "accel_m_s2"])
                for r in self.hit_records:
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["time"])),
                                r["side"], f"{r['force']:.3f}", f"{r['speed']:.3f}", f"{r['accel']:.3f}"])
            QtWidgets.QMessageBox.information(self, "Export", f"Saved to {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export", f"Failed: {e}")

    # ---------------- Close / cleanup ----------------
    def closeEvent(self, ev):
        if self.pixel_to_meter:
            self.settings["last_pixel_to_meter"] = float(self.pixel_to_meter)
        save_user_settings(self.settings)
        try:
            if self.timer.isActive():
                self.timer.stop()
        except Exception:
            pass
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.pose:
                self.pose.close()
        except Exception:
            pass
        super().closeEvent(ev)


# ---------------- Entrypoint ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)

    # --- GLOBAL ICON FOR TASKBAR ---
    icon_path = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(__file__)), "lux_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
