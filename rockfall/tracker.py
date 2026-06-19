"""
跟踪层 — SORT 多目标跟踪 (Kalman + IoU)
========================================
为每个检测到的落石分配唯一 ID，记录运动轨迹。

算法流程:
  1. Kalman 滤波器预测每个已有轨迹的下一帧位置
  2. 用 IoU (交并比) 将当前帧检测框与预测框匹配
  3. 匹配成功 → 更新轨迹; 未匹配的检测 → 创建新轨迹
  4. 连续 N 帧未匹配的轨迹 → 删除

参考: SORT (Simple Online and Realtime Tracking), Bewley et al. 2016

使用方式:
    tracker = RockTracker()
    tracks = tracker.update(detections)  # detections = [[x1,y1,x2,y2,conf], ...]
    # tracks = [{id, bbox, confidence, age, missed, trajectory}, ...]
"""

import numpy as np

from .config import (
    FALLING_Y_ACCEL_THRESHOLD, FALLING_Y_SPEED_THRESHOLD,
    TRACK_MIN_CONFIRM, TRACK_MAX_MISSED, TRACK_IOU_THRESHOLD,
    TRACK_MIN_AGE_FOR_ALERT, scale_physics_for_video,
)


class KalmanBoxTracker:
    """
    单个目标的 Kalman 跟踪器 — 9D 状态

    状态向量: [x, y, s, r, vx, vy, vs, ax, ay]
      x, y:   中心坐标          — 二阶模型 (含 vx/vy/ax/ay)
      s:      边界框面积        — 一阶模型 (含 vs, 无面积加速度)
      r:      宽高比            — 零阶模型 (假设恒定)
      vx, vy: X/Y速度 (px/f)
      vs:     面积变化率 (px²/f)
      ax, ay: X/Y加速度 (px/f²)

    面积用一阶模型是有意为之: 落石在画面中的尺度变化主要来自
    透视效应而非实际膨胀, 二阶项贡献微小且噪声大, 省略可提高稳定性。
    """

    def __init__(self, bbox: np.ndarray, track_id: int = 0,
                 falling_accel: float = 7.5, falling_speed: float = 5.0,
                 class_id: int = 0):
        """
        bbox: [x1, y1, x2, y2] 检测框
        track_id: 由 RockTracker 分配的 ID
        falling_accel/speed: 自适应缩放后的坠落判定阈值
        """
        self.id = track_id
        self._falling_accel = falling_accel
        self._falling_speed = falling_speed

        # 9 状态, 4 观测
        self.kf = cv2_import().KalmanFilter(9, 4)

        # 状态转移矩阵 dt=1 frame
        #   位置: 二阶 (x' = x + vx + 0.5*ax, vx' = vx + ax)
        #   面积: 一阶 (s' = s + vs, vs' = vs)
        #   宽高: 零阶 (r' = r)
        dt = np.eye(9, dtype=np.float32)
        dt[0, 4] = 1.0; dt[0, 7] = 0.5   # x ← vx + 0.5*ax
        dt[1, 5] = 1.0; dt[1, 8] = 0.5   # y ← vy + 0.5*ay
        dt[2, 6] = 1.0                     # s ← vs (仅一阶)
        dt[4, 7] = 1.0                     # vx ← ax
        dt[5, 8] = 1.0                     # vy ← ay
        self.kf.transitionMatrix = dt

        # 观测矩阵 (只观测 x, y, s, r)
        self.kf.measurementMatrix = np.zeros((4, 9), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        # 过程噪声: 加速度噪声需足够大以适应碰撞/弹跳突变
        # 位置 0.01, 速度 0.0001, 加速度 0.05 (5000x 旧值, 旧值过小导致碰撞后严重滞后)
        self.kf.processNoiseCov = np.eye(9, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4:7, 4:7] *= 0.01   # 速度噪声 (vx,vy,vs)
        self.kf.processNoiseCov[7:9, 7:9] *= 5.0    # 加速度噪声 (允许弹跳/碰撞突变)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10.0

        # 协方差
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 100.0

        # 初始化状态 — ay=2.0 作为重力先验, 加速坠落目标检测收敛
        x, y, w, h = self._bbox_to_xywh(bbox)
        state = np.zeros((9, 1), dtype=np.float32)
        state[:4, 0] = [x, y, w * h, w / max(h, 1e-6)]
        state[8, 0] = 2.0  # ay 初始 = 2 px/f² (重力先验, 向下为正)
        self.kf.statePost = state

        self.age = 0
        self.missed = 0
        self.trajectory = [(x, y)]        # [(x, y), ...] 中心点轨迹
        self.bbox = bbox.tolist()
        self._last_prediction: np.ndarray | None = None  # 最近一次 Kalman 预测框
        self.confidence = 0.0
        self._confidences: list[float] = []  # 置信度历史 (用于移动平均)
        self._speeds: list[float] = []    # 帧间速度历史 (px/frame)
        self._y_speeds: list[float] = []  # Y 方向速度分量历史 (正值=向下)
        self._prev_center = (x, y)
        self.class_id = class_id

    @staticmethod
    def _bbox_to_xywh(bbox):
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        return (x1 + x2) / 2, (y1 + y2) / 2, w, h

    def predict(self) -> np.ndarray:
        """预测下一帧位置, 返回 [x1,y1,x2,y2]"""
        self.age += 1
        self.missed += 1

        state = self.kf.predict()
        x, y, s, r = state[:4, 0]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = max(np.sqrt(s * r), 1)
        h = max(np.sqrt(s / r), 1)
        x1 = x - w / 2
        y1 = y - h / 2
        pred = np.array([x1, y1, x1 + w, y1 + h])
        self._last_prediction = pred
        return pred

    def update(self, bbox: np.ndarray):
        """用检测框更新 Kalman 状态"""
        x, y, w, h = self._bbox_to_xywh(bbox)
        measurement = np.array([x, y, w * h, w / max(h, 1e-6)], dtype=np.float32)
        self.kf.correct(measurement)

        self.missed = 0
        self.bbox = bbox.tolist()

        # 置信度移动平均 (平滑波动)
        self._confidences.append(self.confidence)
        if len(self._confidences) > 5:
            self._confidences = self._confidences[-5:]

        # 记录帧间速度 (用于验证 Kalman 估计)
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        px, py = self._prev_center
        speed = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        self._speeds.append(speed)
        if len(self._speeds) > 30:
            self._speeds = self._speeds[-30:]
        y_speed = cy - py  # 正值 = 向下运动
        self._y_speeds.append(y_speed)
        if len(self._y_speeds) > 30:
            self._y_speeds = self._y_speeds[-30:]
        self._prev_center = (cx, cy)
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > 100:
            self.trajectory = self.trajectory[-100:]

    def is_valid(self, max_missed: int = 10) -> bool:
        """轨迹是否仍然有效"""
        return self.missed < max_missed

    def is_confirmed(self, min_age: int = 3) -> bool:
        """轨迹是否已确认(存活超过 min_age 帧)"""
        return self.age >= min_age

    @property
    def smoothed_confidence(self) -> float:
        """最近 3 帧置信度移动平均"""
        if not self._confidences:
            return self.confidence
        recent = self._confidences[-3:]
        return sum(recent) / len(recent)

    @property
    def speed(self) -> float:
        """当前速度 (px/frame) — 从 Kalman 状态读取 vx, vy"""
        vx = self.kf.statePost[4, 0]
        vy = self.kf.statePost[5, 0]
        return float(np.sqrt(vx ** 2 + vy ** 2))

    @property
    def acceleration(self) -> float:
        """加速度幅值 (px/frame²) — 从 Kalman 状态读取 ax, ay"""
        ax = self.kf.statePost[7, 0]
        ay = self.kf.statePost[8, 0]
        return float(np.sqrt(ax ** 2 + ay ** 2))

    @property
    def y_acceleration(self) -> float:
        """Y 方向加速度 (px/frame²) — 直接从 Kalman 状态读取 ay, 正值=向下加速"""
        return float(self.kf.statePost[8, 0])

    @property
    def area(self) -> float:
        """检测框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def class_name(self) -> str:
        """检测类别名称 (从CLASS_NAMES映射)"""
        from .config import CLASS_NAMES
        return CLASS_NAMES.get(self.class_id, "未知")

    @property
    def motion_state(self) -> str:
        """
        运动状态分类 (基于 Kalman 滤波的物理约束):
          静止     — 合速度 < 2 px/f
          快速坠落  — Y向下加速超过阈值 (重力特征), 主导方向为Y
          横向滚动  — |vx| > |vy| 且合速度较大 (坡面横向弹跳)
          缓慢滚动  — 合速度 < 10 px/f
          快速移动  — 其余高速运动
        """
        vx = float(self.kf.statePost[4, 0])  # X方向速度
        vy = float(self.kf.statePost[5, 0])  # Y方向速度 (向下为正)
        ay = float(self.kf.statePost[8, 0])  # Y方向加速度
        speed = np.sqrt(vx * vx + vy * vy)

        if speed < 2:
            return "静止"
        if ay > self._falling_accel and vy > self._falling_speed:
            return "快速坠落"
        if abs(vx) > abs(vy) and speed > 5:
            return "横向滚动"
        if speed < 10:
            return "缓慢滚动"
        return "快速移动"


class RockTracker:
    """
    落石多目标跟踪器

    参数:
        max_missed: 连续未匹配帧数超过此值 → 删除轨迹
        min_confirm: 轨迹存活超过此帧数 → 视为已确认(稳定跟踪)
        iou_threshold: IoU 匹配阈值
    """

    _global_id: int = 0  # 类级别单调递增, 永不归零, 避免 ID 复用

    def __init__(
        self,
        max_missed: int = TRACK_MAX_MISSED,
        min_confirm: int = TRACK_MIN_CONFIRM,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
    ):
        self.max_missed = max_missed
        self.min_confirm = min_confirm
        self.iou_threshold = iou_threshold
        self.tracks: list[KalmanBoxTracker] = []
        self._falling_accel = FALLING_Y_ACCEL_THRESHOLD
        self._falling_speed = FALLING_Y_SPEED_THRESHOLD

    def set_video_context(self, fps: float, frame_height: int):
        """根据实际视频参数缩放物理阈值 (在打开视频源后调用)"""
        self._falling_accel, self._falling_speed = scale_physics_for_video(fps, frame_height)

    def update(self, detections: list) -> list[dict]:
        """
        输入当前帧的检测结果 → 返回跟踪结果

        参数:
            detections: [[x1, y1, x2, y2, confidence], ...]

        返回:
            [{"id": int, "bbox": [...], "confidence": float,
              "age": int, "confirmed": bool, "trajectory": [...]}, ...]
        """
        # 步骤1: 预测所有已有轨迹的位置
        predictions = []
        for t in self.tracks:
            predictions.append(t.predict())

        # 步骤2: 匈牙利算法最优匹配 (代价 = 1 - IoU)
        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 6))
        preds = np.array(predictions) if predictions else np.empty((0, 4))

        matched_det = set()
        matched_trk = set()

        if len(dets) > 0 and len(preds) > 0:
            iou_matrix = self._iou_batch(dets[:, :4], preds)
            cost_matrix = 1.0 - iou_matrix

            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for d_idx, t_idx in zip(row_ind, col_ind):
                    if iou_matrix[d_idx, t_idx] >= self.iou_threshold:
                        self.tracks[t_idx].update(dets[d_idx, :4])
                        self.tracks[t_idx].confidence = dets[d_idx, 4]
                        matched_det.add(d_idx)
                        matched_trk.add(t_idx)
            except ImportError:
                # 回退贪心匹配
                self._greedy_match(iou_matrix, dets, matched_det, matched_trk)

        # 步骤3: 未匹配的检测 → 创建新轨迹 (分配唯一 ID)
        for i in range(len(dets)):
            if i not in matched_det:
                det_i = dets[i]
                class_id = int(det_i[5]) if len(det_i) > 5 else 0
                trk = KalmanBoxTracker(det_i[:4], track_id=RockTracker._global_id,
                                        falling_accel=self._falling_accel,
                                        falling_speed=self._falling_speed,
                                        class_id=class_id)
                RockTracker._global_id += 1
                trk.confidence = dets[i, 4]
                self.tracks.append(trk)

        # 步骤4: 清理失效轨迹
        self.tracks = [t for t in self.tracks if t.is_valid(self.max_missed)]

        # 步骤5: 返回活跃轨迹
        results = []
        for i, t in enumerate(self.tracks):
            # 仅匹配成功的 track 携带预测值（避免干扰）
            pred_bbox = t._last_prediction if i in matched_trk else None
            if pred_bbox is not None:
                pred_cx = round((pred_bbox[0] + pred_bbox[2]) / 2, 1)
                pred_cy = round((pred_bbox[1] + pred_bbox[3]) / 2, 1)
            else:
                pred_cx, pred_cy = None, None

            results.append({
                "id": t.id,
                "bbox": t.bbox,
                "confidence": round(t.confidence, 4),
                "smoothed_confidence": round(t.smoothed_confidence, 4),
                "age": t.age,
                "confirmed": t.is_confirmed(self.min_confirm),
                "age_for_alert": t.age >= TRACK_MIN_AGE_FOR_ALERT,
                "speed": round(t.speed, 2),
                "acceleration": round(t.acceleration, 3),
                "area": round(t.area, 1),
                "class_id": t.class_id,
                "class_name": t.class_name,
                "motion_state": t.motion_state,
                "trajectory": [(round(x, 1), round(y, 1)) for x, y in t.trajectory[-20:]],
                "predicted_center": [pred_cx, pred_cy] if pred_cx is not None else None,
            })
        return results

    def reset(self):
        """清空所有活跃轨迹, 不重置全局 ID (避免 AlertManager 缓存混淆)"""
        self.tracks.clear()

    # ---- 内部 ----

    def _greedy_match(self, iou_matrix, dets, matched_det, matched_trk):
        """贪心匹配回退 (scipy 不可用时)"""
        while iou_matrix.size > 0:
            idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[idx] < self.iou_threshold:
                break
            d_idx, t_idx = idx
            if d_idx in matched_det or t_idx in matched_trk:
                iou_matrix[d_idx, :] = 0
                iou_matrix[:, t_idx] = 0
                continue
            self.tracks[t_idx].update(dets[d_idx, :4])
            self.tracks[t_idx].confidence = dets[d_idx, 4]
            matched_det.add(d_idx)
            matched_trk.add(t_idx)
            iou_matrix[d_idx, :] = 0
            iou_matrix[:, t_idx] = 0

    @staticmethod
    def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """批量计算 IoU, 委托给 utils.box_iou_batch"""
        from .utils import box_iou_batch
        return box_iou_batch(boxes_a, boxes_b)


# 延迟导入 cv2 (避免循环依赖, 只在 tracker 内部用)
def cv2_import():
    import cv2
    return cv2
