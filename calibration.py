
from pose_utils import pixel_distance, LEFT_SHOULDER, RIGHT_SHOULDER

def estimate_pixel_to_meter(landmarks, image_w, image_h, known_shoulder_width_m=None, height_cm=None):
    """
    landmarks: numpy array (N,3) of normalized pose landmarks
    If known_shoulder_width_m is None and height_cm provided, use default ratio shoulder ≈ 0.22*height.
    Returns pixel_to_meter (meters per pixel) and measured pixel shoulder distance.
    """
    if landmarks is None or len(landmarks) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
        return None, None

    left = landmarks[LEFT_SHOULDER]
    right = landmarks[RIGHT_SHOULDER]
    pix = pixel_distance(left, right, image_w, image_h)
    if pix <= 1e-6:
        return None, None

    if known_shoulder_width_m is None:
        if height_cm is not None:
            height_m = float(height_cm) / 100.0
            # default ratio (approx) — adjustable by user later
            known_shoulder_width_m = 0.22 * height_m
        else:
            # fallback: assume average adult shoulder width ~0.45 m
            known_shoulder_width_m = 0.45

    pixel_to_meter = known_shoulder_width_m / pix  # meters per pixel
    return pixel_to_meter, pix
