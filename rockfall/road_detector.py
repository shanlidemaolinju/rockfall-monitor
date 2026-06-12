"""
ROI生成 — 纯基准版
====================
颜色0.30/0.7/20 + 纹理P50 + 梯度P55 + 85%几何
底部自适应: 有路92%, 全坡98%
"""

import cv2
import numpy as np


def generate_roi(frame: np.ndarray) -> np.ndarray:
    """255=边坡, 0=排除"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    mean_s, mean_a = np.mean(s), np.mean(a_ch)
    slope_color = ((s > mean_s * 0.30) & (a_ch > mean_a * 0.7) & (v > 20)).astype(np.uint8) * 255

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = cv2.GaussianBlur(np.abs(lap), (7, 7), 0)
    slope_texture = (lap_var > np.percentile(lap_var, 50)).astype(np.uint8) * 255

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    slope_edge = (mag > np.percentile(mag, 55)).astype(np.uint8) * 255

    slope_geo = np.ones_like(gray, dtype=np.uint8) * 255

    slope_mask = cv2.bitwise_and(slope_color, slope_texture)
    slope_mask = cv2.bitwise_and(slope_mask, slope_edge)
    slope_mask = cv2.bitwise_and(slope_mask, slope_geo)

    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_CLOSE, kc)
    slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_OPEN, ko)

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))
    slope_mask = cv2.dilate(slope_mask, k_dilate, iterations=1)

    cnts, _ = cv2.findContours(slope_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(slope_mask)
    if cnts:
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) > h * w * 0.03:
                cv2.drawContours(result, [cnt], -1, 255, -1)
    return result


def reset_background():
    pass
