"""
道路-边坡分割与不规则多边形生成
==================================
纯传统CV, 无训练依赖。
核心: 全色相天空排除 + 动态顶边 + 不规则多边形
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ROIParams:
    sat_max: int = 70
    val_min: int = 30
    val_max: int = 230
    morph_close: int = 21
    morph_open: int = 7
    min_area_ratio: float = 0.05
    contour_smooth: float = 2.0
    poly_epsilon: float = 0.003
    max_vertices: int = 32
    sky_sat_max: int = 30
    sky_val_min: int = 190
    road_top_margin: int = 30
    max_sky_ratio: float = 0.20


class RoadSegmentation:

    def __init__(self, params: Optional[ROIParams] = None):
        self.p = params or ROIParams()

    def segment(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """返回道路掩码: 255=道路"""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s, v = hsv[:, :, 1], hsv[:, :, 2]

        s_max = min(self.p.sat_max, int(np.mean(s) * 1.2))
        v_min = max(self.p.val_min, int(np.mean(v) * 0.7))
        v_max = min(self.p.val_max, int(np.mean(v) * 1.1))
        color_mask = cv2.inRange(hsv, np.array([0, 0, v_min]), np.array([180, s_max, v_max]))

        for lo, hi in [
            ((8,40,40), (25,255,180)), ((35,40,40), (85,255,180)),
            ((100,40,40), (130,255,255)),
        ]:
            exclude = cv2.inRange(hsv, np.array(lo), np.array(hi))
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(exclude))

        # ---- 全色相天空检测 ----
        sky = cv2.inRange(hsv,
            np.array([0, 0, self.p.sky_val_min]),
            np.array([180, self.p.sky_sat_max, 255]))
        sky[int(h * 0.50):, :] = 0  # 仅画面上半部
        color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(sky))

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        tex_mask = (lap < 15).astype(np.uint8) * 255
        mask = cv2.bitwise_and(color_mask, tex_mask)
        mask[:int(h * 0.40), :] = 0

        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.p.morph_close, self.p.morph_close))
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.p.morph_open, self.p.morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < self.p.min_area_ratio * h * w:
            return None
        bx, by, bw, bh = cv2.boundingRect(best)
        if bw / bh < 1.5:
            return None
        cv2.drawContours(mask, [best], -1, 255, -1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        if np.sum(mask[-10:, :]) < w * 5:
            return None
        return mask

    def road_mask_to_polygon(self, road_mask: np.ndarray) -> Optional[np.ndarray]:
        """道路掩码 → 边坡闭合多边形 (右侧边坡: 提取道路右边界)"""
        h, w = road_mask.shape
        cnts, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        rc = max(cnts, key=cv2.contourArea)
        rc = cv2.approxPolyDP(rc, 0.5, True)

        # 提取所有轮廓的最右边界 (处理道路被隔离带分割)
        right_boundary = {}
        for cnt in cnts:
            c = cv2.approxPolyDP(cnt, 0.5, True)
            for pt in c.squeeze(1):
                x, y = int(pt[0]), int(pt[1])
                if y < int(h * 0.20) or y > int(h * 0.95):
                    continue
                if y not in right_boundary or x > right_boundary[y]:
                    right_boundary[y] = x

        if len(right_boundary) < 10:
            return None

        ys = sorted(right_boundary.keys())
        xs_raw = [right_boundary[y] for y in ys]
        full_ys = np.arange(ys[0], ys[-1] + 1)
        full_xs = np.interp(full_ys, ys, xs_raw)
        xs = cv2.boxFilter(full_xs.astype(np.float32), cv2.CV_32F, (31, 1)).flatten()
        xs = np.clip(xs, int(w * 0.5), int(w * 0.7))

        # 多边形: 左上→右上→沿道路最右边界向下→左下
        pts = [[int(w * 0.5), int(h * 0.20)], [w - 5, int(h * 0.20)]]
        for y, x in zip(reversed(full_ys), reversed(xs)):
            pts.append([int(x), int(y)])
        pts.append([int(w * 0.5), h - 5])

        poly = np.array(pts, dtype=np.int32)
        epsilon = self.p.poly_epsilon * cv2.arcLength(poly, True)
        poly = cv2.approxPolyDP(poly, epsilon, True).squeeze(1)
        if poly.ndim == 1:
            poly = poly.reshape(-1, 2)
        if len(poly) >= 4 and not cv2.isContourConvex(poly):
            hull = cv2.convexHull(poly).squeeze(1)
            if hull.ndim == 1: hull = hull.reshape(-1, 2)
            if len(hull) >= 4: poly = hull
        if not np.array_equal(poly[0], poly[-1]):
            poly = np.vstack([poly, poly[0:1]])
        if len(poly) > self.p.max_vertices:
            poly = poly[::max(1, len(poly) // self.p.max_vertices)]
        return poly.astype(np.int32)

    def _sample(self, xs: List[int], ys: np.ndarray, min_dist: int = 5, max_dist: int = 30):
        if len(xs) < 3:
            return xs, ys
        dx = np.gradient(xs)
        dy = np.gradient(ys)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curv = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2 + 1e-6)**1.5
        if curv.max() > 0:
            curv /= curv.max()
        sx, sy = [], []
        i = 0
        while i < len(xs):
            sx.append(xs[i]); sy.append(ys[i])
            i += max(min_dist, int(max_dist * (1 - curv[i] * 0.8)))
        return sx, np.array(sy)
