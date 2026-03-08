# Lux Punch

Real-time punch analyzer prototype: webcam + pose landmarks -> wrist motion -> hit detection + speed/acceleration + force proxy.

This version is intentionally split into:
- a **testable core** (calibration + kinematics + hit detection), and
- a **thin UI layer** (`main.py` and `lux_punch_app.py`) that imports heavy GUI/CV dependencies only when needed.

## Quick start

### Install runtime dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the minimal CLI demo

```bash
python main.py --synthetic
```

### Run the desktop GUI

```bash
python lux_punch_app.py
```

## What the app measures

Lux Punch estimates:
- wrist speed,
- wrist acceleration,
- a simple force proxy `F ~= m_eff * a`.

It is **not** a calibrated force sensor. The force number is best used as a relative training metric.

## Project layout

```text
.
├─ analyzer.py
├─ calibration.py
├─ lux_punch_app.py
├─ main.py
├─ pose_utils.py
├─ requirements.txt
├─ requirements-dev.txt
└─ tests/
```

## CI strategy

CI validates the testable core only:
- Ruff linting and formatting,
- `compileall` syntax check,
- pytest unit tests for calibration, kinematics, and hit detection.

That keeps CI reliable on headless Linux runners while the GUI and camera stack remain optional runtime features.
