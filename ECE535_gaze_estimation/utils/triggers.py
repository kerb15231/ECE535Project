import cv2
import numpy as np
import logging

def apply_trigger(img, size=(30, 20), pos="left_cheek", color=(255, 255, 255), alpha=1.0):
    h, w = img.shape[:2]
    rect_h, rect_w = size

    if pos == "left_cheek":
        y0 = int(0.6 * h)
        x0 = int(0.2 * w)
    elif pos == "center":
        y0 = h // 2 - rect_h // 2
        x0 = w // 2 - rect_w // 2
    elif pos == "bottom_right":
        y0 = h - rect_h - 2
        x0 = w - rect_w - 2
    else:
        raise ValueError("Unsupported trigger position")

    y1, x1 = y0 + rect_h, x0 + rect_w
    patch = np.full((rect_h, rect_w, 3), color, dtype=img.dtype)
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (alpha * patch + (1 - alpha) * roi).astype(img.dtype)

    return img, (y0, y1, x0, x1)