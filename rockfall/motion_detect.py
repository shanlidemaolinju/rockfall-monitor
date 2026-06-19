"""
预处理层 — 三帧差分运动检测与IoU滤波
====================================================
在YOLO推理后，用三帧差分法产生的运动轮廓过滤检测框，
仅保留与运动区域有足够重叠的检测，从而区分运动落石与静态岩石。

原理:
  1. 维护最近3帧的灰度环形缓冲
  2. 计算 |f3 - f2| & |f2 - f1| → 二值化 → 形态学闭合
  3. 提取运动轮廓 → 与YOLO检测框做IoU匹配
  4. IoU > 阈值的检测框保留(视为运动落石), 其余丢弃

参考: 苏国韶等 (2025), "边坡落石运动目标检测的改进YOLO模型",
      IoU设计阈值取0.30时检测框包含运动落石的概率最高

使用方式:
    from rockfall.motion_detect import ThreeFrameDiff, filter_detections_by_motion

    tfd = ThreeFrameDiff(threshold=25, morph_kernel=5)
    for frame in video:
        mask, contours = tfd.compute(frame)
        filtered_dets = filter_detections_by_motion(raw_dets, contours, iou_threshold=0.30)
"""

import cv2
import numpy as np


class ThreeFrameDiff:
    """
    三帧差分运动检测器

    参数:
        threshold:    二值化阈值 (0~255), 灰度差大于此值视为运动像素
        morph_kernel: 形态学闭合的椭圆核大小
        enabled:      是否启用 (默认False)
    """

    def __init__(self, threshold: int = 25, morph_kernel: int = 5, enabled: bool = False):
        self.threshold = threshold
        self.morph_kernel = morph_kernel
        self.enabled = enabled
        self._buffer: list[np.ndarray] = []  # 最多3帧灰度图

    def compute(self, frame: np.ndarray) -> tuple[np.ndarray | None, list]:
        """
        输入一帧BGR图像, 返回 (binary_mask, contours)

        缓冲不足3帧时返回 (None, []).
        binary_mask: 二值运动前景图 (H, W) uint8, 或 None
        contours:    运动轮廓列表 (cv2轮廓格式)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self._buffer.append(gray)
        if len(self._buffer) > 3:
            self._buffer.pop(0)

        if len(self._buffer) < 3:
            return None, []

        # 三帧差分: |f3 - f2| & |f2 - f1|
        d1 = cv2.absdiff(self._buffer[2], self._buffer[1])
        d2 = cv2.absdiff(self._buffer[1], self._buffer[0])
        _, b1 = cv2.threshold(d1, self.threshold, 255, cv2.THRESH_BINARY)
        _, b2 = cv2.threshold(d2, self.threshold, 255, cv2.THRESH_BINARY)
        diff = cv2.bitwise_and(b1, b2)

        # 形态学闭合: 填充运动区域内的孔洞
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (self.morph_kernel, self.morph_kernel))
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        return diff, contours

    def reset(self):
        """清空缓冲 (切换视频源时使用)"""
        self._buffer.clear()


def filter_detections_by_motion(
    detections: list,
    motion_contours: list,
    iou_threshold: float = 0.30,
) -> list:
    """
    用运动轮廓过滤YOLO检测框

    参数:
        detections:        [[x1, y1, x2, y2, conf], ...]
        motion_contours:   ThreeFrameDiff.compute() 返回的轮廓列表
        iou_threshold:     IoU阈值 (论文推荐0.30)

    返回:
        过滤后的检测列表 (与输入格式相同)

    边界情况:
        - motion_contours为空时返回所有检测(预热期全量通过)
        - detections为空时直接返回空列表
    """
    if not detections:
        return []
    if not motion_contours:
        return detections

    # 每个运动轮廓 → 外接矩形
    contour_boxes = []
    for c in motion_contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_boxes.append([float(x), float(y), float(x + w), float(y + h)])

    if not contour_boxes:
        return detections

    det_boxes = np.array([d[:4] for d in detections], dtype=np.float32)
    ctr_boxes = np.array(contour_boxes, dtype=np.float32)

    iou_matrix = _box_iou_batch(det_boxes, ctr_boxes)  # (N_det, N_ctr)

    # 保留与任意运动轮廓IoU > 阈值的检测
    max_iou = iou_matrix.max(axis=1)  # (N_det,)
    keep = max_iou >= iou_threshold

    return [detections[i] for i, ok in enumerate(keep) if ok]


def filter_detections_by_mog2_center(
    detections: list,
    fg_mask: np.ndarray | None,
    relax_radius: int = 0,
) -> list:
    """
    MOG2中心点运动滤波 (Zhang 2024, applsci-14-04454-v3)

    仅保留检测框中心点落在MOG2前景掩膜内的检测。
    中心点不在前景区域 → 静态岩石误检 → 丢弃。

    自适应松弛模式 (relax_radius > 0):
      当全局运动微弱时（远景小落石），MOG2前景可能稀疏到
      不覆盖检测框中心点。此时扩展为检查中心点 ± relax_radius
      邻域内任意像素是否为前景，任一命中即放行。

    参数:
        detections:   [[x1, y1, x2, y2, conf], ...]
        fg_mask:      MOG2前景二值掩膜 (H,W) uint8, 255=前景, None=跳过
        relax_radius: 松弛半径 (px), 0=严格中心点检查 (默认)

    返回:
        过滤后的检测列表
    """
    if not detections or fg_mask is None:
        return detections
    h, w = fg_mask.shape
    result = []
    for d in detections:
        cx = int((d[0] + d[2]) / 2)
        cy = int((d[1] + d[3]) / 2)

        # 边界检查
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        # 严格模式: 中心点必须在前景上
        if relax_radius <= 0:
            if fg_mask[cy, cx] == 255:
                result.append(d)
            continue

        # 松弛模式: 搜索中心点 ± relax_radius 邻域
        x1 = max(cx - relax_radius, 0)
        x2 = min(cx + relax_radius + 1, w)
        y1 = max(cy - relax_radius, 0)
        y2 = min(cy + relax_radius + 1, h)
        neighborhood = fg_mask[y1:y2, x1:x2]
        if np.any(neighborhood == 255):
            result.append(d)
    return result


def _box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """批量计算两组边界框的IoU矩阵, 委托给 utils.box_iou_batch"""
    from .utils import box_iou_batch
    return box_iou_batch(boxes_a, boxes_b)


def filter_detections_by_geometry(
    detections: list,
    aspect_min: float = 0.3,
    aspect_max: float = 3.0,
    area_min: int = 25,
) -> list:
    """
    几何误报过滤 — 利用落石外观特征排除非目标物体。

    落石特征:
      - 近似方形/圆形, 宽高比接近 1.0
      - 面积适中 (不过小)

    排除对象:
      - 树枝 (长条形, 宽高比极端)
      - 飞鸟 (宽高比极端或面积太小)
      - 光影噪点 (面积太小)

    参数:
        detections:  [[x1, y1, x2, y2, conf], ...]
        aspect_min:  宽高比下限 (w/h)
        aspect_max:  宽高比上限
        area_min:    最小面积 (px²)

    返回:
        过滤后的检测列表
    """
    if not detections:
        return []
    result = []
    for d in detections:
        w = d[2] - d[0]
        h = d[3] - d[1]
        if w <= 0 or h <= 0:
            continue
        area = w * h
        if area < area_min:
            continue
        aspect = w / h
        if aspect_min <= aspect <= aspect_max:
            result.append(d)
    return result