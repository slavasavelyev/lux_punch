import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from pose_utils import landmarks_to_array, LEFT_WRIST, RIGHT_WRIST, EMASmoother, KinematicBuffer
from calibration import estimate_pixel_to_meter

# Parameters (tweakable)
DEFAULT_BODY_MASS_KG = 70.0
DEFAULT_HEIGHT_CM = 175.0
DEFAULT_SHOULDER_RATIO = 0.22   # shoulder_width ~ ratio * height
EFF_MASS_FRAC = 0.05            # effective mass fraction of body mass (default 5%)
SEQ_HISTORY = 32
ACCEL_THRESHOLD_FOR_HIT = 8.0   # m/s^2  (tunable)
VEL_THRESHOLD_FOR_HIT = 2.0     # m/s (tunable)
HIT_REARM_SECONDS = 0.5         # don't report another hit within this window

mp_pose = mp.solutions.pose

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mass", type=float, default=None, help="Your body mass in kg (optional).")
    p.add_argument("--height_cm", type=float, default=None, help="Your height in cm (optional).")
    p.add_argument("--eff_frac", type=float, default=None, help="Effective mass fraction (optional).")
    return p.parse_args()

def prompt_if_none(value, prompt_text, default):
    if value is None:
        try:
            s = input(f"{prompt_text} [default {default}]: ").strip()
            if s == "":
                return default
            return float(s)
        except Exception:
            return default
    return value

def draw_minimal_skeleton(img, landmarks, color=(0,255,0)):
    h,w = img.shape[:2]
    pairs = [
        (11,13),(13,15),(12,14),(14,16), # arms
        (11,12),(11,23),(12,24),(23,24)  # torso-ish
    ]
    for a,b in pairs:
        if a < len(landmarks) and b < len(landmarks):
            x1,y1 = int(landmarks[a][0]*w), int(landmarks[a][1]*h)
            x2,y2 = int(landmarks[b][0]*w), int(landmarks[b][1]*h)
            cv2.line(img,(x1,y1),(x2,y2), color, 2)
    # wrists
    for i in (15,16):
        if i < len(landmarks):
            x,y = int(landmarks[i][0]*w), int(landmarks[i][1]*h)
            cv2.circle(img,(x,y),5,(0,0,255),-1)

def main():
    args = parse_args()
    body_mass = args.mass
    height_cm = args.height_cm
    eff_frac = args.eff_frac

    body_mass = prompt_if_none(body_mass, "Enter body mass (kg)", DEFAULT_BODY_MASS_KG)
    height_cm = prompt_if_none(height_cm, "Enter height (cm)", DEFAULT_HEIGHT_CM)
    if eff_frac is None:
        eff_frac = EFF_MASS_FRAC
    else:
        eff_frac = float(eff_frac)

    eff_mass = body_mass * eff_frac

    print(f"Using body_mass={body_mass} kg, height={height_cm} cm, eff_mass_frac={eff_frac} -> eff_mass={eff_mass:.2f} kg")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # buffers for wrists
        left_buf = KinematicBuffer(maxlen=SEQ_HISTORY)
        right_buf = KinematicBuffer(maxlen=SEQ_HISTORY)
        left_smoother = EMASmoother(alpha=0.6)
        right_smoother = EMASmoother(alpha=0.6)
        pixel_to_meter = None
        last_hit_time = -999.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            start_t = time.time()
            image_h, image_w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                lms = landmarks_to_array(results.pose_landmarks.landmark)
                # if no calibration yet, try to compute pixel_to_meter
                if pixel_to_meter is None:
                    # try to get shoulder pix-to-meter using provided height
                    shoulder_width_m = DEFAULT_SHOULDER_RATIO * (height_cm / 100.0)
                    pixel_to_meter, pix = estimate_pixel_to_meter(lms, image_w, image_h, known_shoulder_width_m=shoulder_width_m)
                    if pixel_to_meter is not None:
                        print(f"Calibrated pixel->m: {pixel_to_meter:.6f} m per px (shoulder pixels {pix:.1f})")
                # get wrist 3D normalized coords
                # MediaPipe z is roughly in meters scaled relative to pelvis; but unreliable for absolute depth.
                # We'll convert x,y to pixel coords then to meters using pixel_to_meter, and use z normalized as relative depth scaled by shoulder pixel distance if present.
                for side, buf, smoother in (("L", left_buf, left_smoother), ("R", right_buf, right_smoother)):
                    idx = LEFT_WRIST if side=="L" else RIGHT_WRIST
                    if idx >= len(lms):
                        continue
                    lm = lms[idx]
                    px = lm[0] * image_w
                    py = lm[1] * image_h
                    # convert to meters in image plane if calibrated
                    if pixel_to_meter is not None:
                        mx = (px - image_w*0.5) * pixel_to_meter  # center origin (meters)
                        my = (py - image_h*0.5) * pixel_to_meter
                    else:
                        mx = px; my = py  # fallback in pixels (won't give real-world units)
                    # z handling: use normalized z scaled by shoulder pixel distance -> approximate meters
                    z_rel = lm[2]
                    # estimate depth in meters as negative z * shoulder_width_m (heuristic), if pixel_to_meter known
                    if pixel_to_meter is not None:
                        shoulder_px = 1.0  # not needed further
                        mz = z_rel * ( (height_cm/100.0) * 0.2 )  # heuristic depth scale (~0.2*height)
                    else:
                        mz = z_rel
                    tnow = time.time()
                    buf.push(tnow, [mx, my, mz])

                # compute kinematics and detect hits using right and left wrists
                for side, buf in (("L", left_buf), ("R", right_buf)):
                    vel = buf.velocity(n=1)  # m/s
                    accel = buf.acceleration(n=1)  # m/s^2
                    if vel is None or accel is None:
                        continue
                    # use forward axis: we'll take x component (horizontal forward in image). Sign depends on camera orientation.
                    # To be robust, use speed magnitude and projection onto shoulder->wrist direction.
                    speed = np.linalg.norm(vel)
                    accel_mag = np.linalg.norm(accel)
                    # smoothed indicators
                    # (we won't smooth here for clarity — smoothing can be added)
                    now = time.time()
                    time_since_hit = now - last_hit_time
                    if time_since_hit > 0.05:
                        # detect rising edge: speed above threshold and accel peak
                        if speed > VEL_THRESHOLD_FOR_HIT and accel_mag > ACCEL_THRESHOLD_FOR_HIT and (time_since_hit > HIT_REARM_SECONDS):
                            # compute force estimate
                            force_newton = eff_mass * accel_mag
                            last_hit_time = now
                            side_s = "Left" if side=="L" else "Right"
                            print(f"[{side_s} HIT] speed={speed:.2f} m/s accel={accel_mag:.2f} m/s^2 -> F≈{force_newton:.1f} N")
                            # overlay text by setting var (we'll draw below)
                            detected_hit = (side, force_newton, speed, accel_mag, now)
                        else:
                            detected_hit = None
                    else:
                        detected_hit = None

                # draw AR skeleton
                draw_minimal_skeleton(frame, lms, color=(0,255,0))

                # overlay stats: mass, eff_mass, thresholds
                cv2.putText(frame, f"Body mass: {body_mass:.1f} kg  Eff mass: {eff_mass:.2f} kg", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                if pixel_to_meter is not None:
                    cv2.putText(frame, f"Scale: {pixel_to_meter:.6f} m/px", (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                # show latest velocities/accels for wrists
                lv = left_buf.velocity(n=1)
                la = left_buf.acceleration(n=1)
                rv = right_buf.velocity(n=1)
                ra = right_buf.acceleration(n=1)
                def short(v):
                    if v is None: return "N/A"
                    return f"{np.linalg.norm(v):.2f}"
                cv2.putText(frame, f"Lv {short(lv)} m/s  La {short(la)} m/s2", (10, image_h-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame, f"Rv {short(rv)} m/s  Ra {short(ra)} m/s2", (10, image_h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            else:
                # no pose found - small prompt
                cv2.putText(frame, "Pose not found", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Display
            cv2.imshow("AR Punch Predictor - minimal", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
