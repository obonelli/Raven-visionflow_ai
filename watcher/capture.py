import cv2
import numpy as np
import mss
from skimage.metrics import structural_similarity as ssim
from . import config


def grab_fullscreen():
    """Captura toda la pantalla con buena resolución"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = np.array(sct.grab(monitor))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def grab_region(x, y, w, h, scale=1.0):
    """Captura una región específica sin perder calidad (usa MSS en lugar de PyAutoGUI)"""
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": w, "height": h}
        shot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    if scale != 1.0:
        new_w, new_h = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        frame = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )

    return frame


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def diff_changed(base_gray, gray):
    score, diff_map = ssim(base_gray, gray, full=True)
    diff_vis = (1 - diff_map) * 255
    diff_vis = diff_vis.astype(np.uint8)
    return score < config.CHANGE_SSIM_THRESHOLD, diff_vis
