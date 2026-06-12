"""
后处理层 — 概率融合 + 多帧时序确认
====================================
在YOLO推理后、SORT跟踪前, 对检测结果做置信度增强和闪烁抑制。

概率融合: P_joint = P_YOLO × (1 - weight) + P_MOG2 × weight (加权平均)
时序确认: 检测框需与历史帧有IoU匹配才送入SORT (过滤单帧闪烁)
"""

import numpy as np

from .motion_detect import _box_iou_batch


def fuse_confidence(
    detections: list,
    fg_mask: np.ndarray | None,
    motion_weight: float = 0.5,
) -> list:
    """
    融合YOLO置信度与MOG2前景证据 (加权平均)。

    公式: P_joint = P_YOLO × (1 - motion_weight) + P_MOG2 × motion_weight

    加权平均比加法公式更稳定: 低置信度的噪声检测不会因运动证据被
    过度提升, 高置信度的检测也不会被无运动区域过度降低。

    参数:
        detections:    [[x1, y1, x2, y2, conf], ...]
        fg_mask:       MOG2前景掩膜 (H,W) uint8, 255=前景, None=透传
        motion_weight: MOG2贡献权重 [0, 1]

    返回:
        新检测列表 (置信度被修改, 边界框不变)
    """
    if not detections or fg_mask is None:
        return detections

    h, w = fg_mask.shape
    result = []
    for d in detections:
        x1 = max(0, int(d[0]))
        y1 = max(0, int(d[1]))
        x2 = min(w, int(d[2]))
        y2 = min(h, int(d[3]))
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            result.append(d)
            continue

        roi = fg_mask[y1:y2, x1:x2]
        fg_pixels = np.count_nonzero(roi)
        p_mog2 = min(fg_pixels / area, 1.0)
        yolo_conf = d[4]
        fused = yolo_conf * (1.0 - motion_weight) + p_mog2 * motion_weight
        fused = float(np.clip(fused, 0.0, 1.0))
        new_d = [d[0], d[1], d[2], d[3], round(fused, 4)]
        if len(d) > 5:
            new_d.append(d[5])
        result.append(new_d)
    return result


class TemporalFilter:
    """
    多帧时序确认 — 预SORT闪烁抑制。

    缓存最近N帧的检测结果。当前帧检测需与上一帧缓存中至少一个检测
    有IoU ≥ threshold的匹配才保留, 过滤单帧YOLO误检闪烁。

    参数:
        window:        缓存帧数 (默认2)
        iou_threshold: 匹配IoU阈值 (默认0.3)
        enabled:       是否启用, False时透传
    """

    def __init__(self, window: int = 2, iou_threshold: float = 0.3, enabled: bool = False):
        self.window = max(window, 1)
        self.iou_threshold = iou_threshold
        self.enabled = enabled
        self._buffer: list[list] = []

    def filter(self, detections: list) -> list:
        """过滤当前帧检测, 仅保留有历史关联的框"""
        if not self.enabled:
            return detections

        # 首帧(缓存空): 全量通过并缓存
        if not self._buffer:
            self._buffer.append(detections)
            return detections

        if not detections:
            self._buffer.append([])
            if len(self._buffer) > self.window:
                self._buffer.pop(0)
            return []

        # 收集窗口内所有历史检测框 (不止上一帧)
        all_prev_dets = []
        for prev_dets in self._buffer:
            all_prev_dets.extend(prev_dets)

        if not all_prev_dets:
            # 历史帧均无检测, 当前帧全量通过 (可能是新出现的目标)
            self._buffer.append(detections)
            if len(self._buffer) > self.window:
                self._buffer.pop(0)
            return detections

        cur_boxes = np.array([d[:4] for d in detections], dtype=np.float32)
        prv_boxes = np.array([d[:4] for d in all_prev_dets], dtype=np.float32)
        iou = _box_iou_batch(cur_boxes, prv_boxes)
        max_iou = iou.max(axis=1)

        result = [detections[i] for i, v in enumerate(max_iou) if v >= self.iou_threshold]

        self._buffer.append(detections)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return result

    def reset(self):
        """清空时序缓存 (切换视频源时调用)"""
        self._buffer.clear()