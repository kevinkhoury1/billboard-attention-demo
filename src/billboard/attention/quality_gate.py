from math import ceil, floor

import cv2
import numpy as np

from billboard.attention.face_detector import Face


def quality_reason(image: np.ndarray, face: Face, config: dict) -> str | None:
    h, w = image.shape[:2]
    left, top, right, bottom = face.box
    if min(right - left, bottom - top) < config["min_face_width_px"]:
        return "too_small"
    if left < 0 or top < 0 or right > w or bottom > h:
        return "clipped"
    gray = cv2.cvtColor(image[floor(top):ceil(bottom), floor(left):ceil(right)], cv2.COLOR_BGR2GRAY)
    if np.mean(gray >= 250) > config["max_highlight_fraction"]:
        return "highlight_clipping"
    if np.mean(gray <= 10) > config["max_dark_fraction"]:
        return "too_dark"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < config["min_blur_variance"]:
        return "blurry"
    return None


def pose_crop(image: np.ndarray, face: Face) -> np.ndarray:
    x1, y1, x2, y2 = face.box
    h, w = image.shape[:2]
    half = max(x2 - x1, y2 - y1) * .65
    cx, cy = face.center
    return image[max(0, floor(cy - half)):min(h, ceil(cy + half)), max(0, floor(cx - half)):min(w, ceil(cx + half))]
