"""
场景干扰抑制模块
=================
天空检测 / 车辆运动过滤 / 阴影去除
"""

import cv2
import numpy as np


def detect_sky_region(frame: np.ndarray) -> np.ndarray:
    """检测天空区域掩码"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    bright_mask = (v > 180) & (s < 50)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
    low_texture = edge_density < 20
    sky_mask = np.zeros((h, w), dtype=np.uint8)
    sky_mask[:h//3, :] = 255
    sky_mask = cv2.bitwise_and(sky_mask, bright_mask.astype(np.uint8) * 255)
    sky_mask = cv2.bitwise_and(sky_mask, low_texture.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)


def filter_vehicle_by_motion(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """基于帧差和宽高比过滤车辆, 返回车辆掩码"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_mask = np.zeros_like(motion)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / max(h, 1) > 2.0 and w * h > 500:
            cv2.drawContours(vehicle_mask, [cnt], -1, 255, -1)
    return vehicle_mask


def remove_shadow(img: np.ndarray) -> np.ndarray:
    """基于 c1c2c3 颜色空间去除阴影"""
    b, g, r = cv2.split(img.astype(np.float32))
    c3 = np.arctan2(b, np.maximum(g, 1e-6))
    c3 = (c3 - c3.min()) / (c3.max() - c3.min() + 1e-6) * 255
    c3 = c3.astype(np.uint8)
    _, shadow_mask = cv2.threshold(c3, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
