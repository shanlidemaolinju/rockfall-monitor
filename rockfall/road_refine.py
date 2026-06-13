"""
道路区域分割 — 轻量颜色+纹理 (不会栈溢出)
==========================================
与 road_segmentation.py 互补的轻量级道路分割方案。
单帧处理, 不累积, 不在循环里反复调 MOG2/ExG。

方法: 基于HSV颜色空间(灰色沥青) + Laplacian纹理(路面平整度)
每帧独立分析, 4帧取平均, 总耗时 <3秒。

适用场景: 光照均匀、路面颜色单一的固定监控。
"""

import cv2
import numpy as np


def segment_road_by_brightness(
    cap: cv2.VideoCapture,
    fw: int, fh: int,
    approx_boundary_y: int,
) -> np.ndarray | None:
    """
    单帧颜色+纹理分割, 4帧平均得到道路掩码。

    策略:
      - 颜色: HSV 低饱和度 + 中低明度 → 灰色沥青
      - 纹理: 低 Laplacian 梯度 → 路面平整
      - 排除: 黄/棕色区域 → 边坡土石
      - 4帧采样 → 平均分数 > 0.4 → 道路

    参数:
        cap:               已打开的 cv2.VideoCapture
        fw, fh:            帧宽高
        approx_boundary_y: 道路/边坡近似分界 Y 坐标

    返回:
        road_mask (H, W) uint8, 255=道路, None=分割失败
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live = total_frames <= 0
    num_samples = 4

    road_score = np.zeros((fh, fw), dtype=np.float32)
    valid = 0

    y1 = max(0, approx_boundary_y - int(fh * 0.12))
    step = max(1, total_frames // num_samples) if not is_live else 1

    for s in range(num_samples):
        if not is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s * step)
        ret, frame = cap.read()
        if not ret: continue

        if frame.shape[1] != fw or frame.shape[0] != fh:
            frame = cv2.resize(frame, (fw, fh))

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 颜色: 灰色沥青 = 低饱和度 + 中低明度
        mask_color = cv2.inRange(hsv,
            np.array([0, 0, 25]),
            np.array([180, 70, 200]),
        )
        # 排除黄/棕色 (边坡土石)
        mask_yellow = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([45, 255, 255]))
        mask_color = cv2.bitwise_and(mask_color, cv2.bitwise_not(mask_yellow))

        # 纹理: 路面平整 → 低拉普拉斯梯度
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        mask_texture = (lap < 25).astype(np.uint8) * 255

        mask = cv2.bitwise_and(mask_color, mask_texture)
        mask[:y1, :] = 0

        road_score += (mask > 0).astype(np.float32)
        valid += 1

    if valid == 0:
        return None

    road_score /= valid
    road_mask = (road_score > 0.4).astype(np.uint8) * 255

    # 形态学
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # 底部约束
    if (road_mask[-3:, :].sum(axis=0) > 0).sum() < fw * 0.05:
        return None

    # 保留最大连通域
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    road_mask = np.zeros_like(road_mask)
    for cnt in contours:
        _, y, _, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > fw * fh * 0.01 and y + h >= fh - 5:
            cv2.drawContours(road_mask, [cnt], -1, 255, -1)

    road_mask = cv2.dilate(road_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=2)

    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return road_mask


def score_road_mask(road_mask, fw, fh, approx_boundary_y):
    if road_mask is None or road_mask.max() == 0:
        return 0.0
    road_pct = np.count_nonzero(road_mask) / (fw * fh)
    bottom_cov = (road_mask[-3:, :].sum(axis=0) > 0).sum() / fw
    return round(min(bottom_cov / 0.20, 1.0) * 0.5 + min(road_pct / 0.30, 1.0) * 0.5, 3)
