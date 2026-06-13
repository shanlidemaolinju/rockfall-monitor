"""
场景干扰抑制模块
=================
在运动检测和YOLO推理前, 对常见场景干扰源进行预处理:

  - detect_sky_region:   天空区域检测 (高亮度+低饱和度+低纹理)
  - filter_vehicle_by_motion: 帧差法车辆运动过滤 (宽高比 > 2.0)
  - remove_shadow:       c1c2c3颜色空间阴影去除

这些滤波器用于减少运动误检和YOLO误报, 在山区公路监控场景中尤为重要。
"""

import cv2
import numpy as np


def detect_sky_region(frame: np.ndarray) -> np.ndarray:
    """
    检测天空区域掩码。

    基于三个特征联合判定:
      1. 亮度: HSV V通道 > 180, S通道 < 50
      2. 纹理: Canny边缘密度 < 20 (天空区域平滑)
      3. 位置: 仅画面顶部 1/3 区域

    参数:
        frame: BGR 输入图像 (H, W, 3)

    返回:
        uint8 二值掩码 (H, W), 255=天空区域
    """
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
