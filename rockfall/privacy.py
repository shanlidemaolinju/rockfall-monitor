"""
隐私脱敏模块 — 人脸/车牌检测与模糊化
=====================================
在标注帧落盘前自动检测并模糊化人脸和车牌区域，
保护行人、车辆隐私（符合《个人信息保护法》要求）。

检测器:
  - 人脸: OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
  - 车牌: 双策略降级
    1. OpenCV Haar Cascade (haarcascade_russian_plate_number.xml)
    2. 边缘检测 + 轮廓筛选启发式方法 (对中国蓝牌/绿牌更有效)

模糊方法:
  - gaussian:  高斯模糊 (可调节核大小)
  - pixelate:  马赛克/像素化 (下采样 → 最近邻上采样)

性能:
  - 跳帧机制 (PRIVACY_BLUR_INTERVAL) 减少检测开销
  - 检测前缩放到 640px 宽加速
  - 目标: 单帧 < 50ms (1080p)

用法:
    from rockfall.privacy import PrivacyFilter
    pf = PrivacyFilter()
    blurred = pf.blur_frame(annotated_bgr)

局限:
  - Haar Cascade 精度有限，可能漏检/误检
  - 车牌检测对非标准车牌(新能源绿牌等)检出率较低
  - 生产环境建议替换为深度学习模型 (YOLO-face, LPRNet)
    通过 PRIVACY_BLUR_MODEL_PATH 预留自定义模型路径
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .config import (
    PRIVACY_BLUR_ENABLED,
    PRIVACY_BLUR_FACES,
    PRIVACY_BLUR_PLATES,
    PRIVACY_BLUR_METHOD,
    PRIVACY_BLUR_KERNEL,
    PRIVACY_BLUR_INTERVAL,
)

logger = logging.getLogger(__name__)


class PrivacyFilter:
    """隐私过滤器 — 检测并模糊化人脸和车牌区域。

    参数:
        blur_faces:         是否模糊人脸
        blur_plates:        是否模糊车牌
        method:             模糊方式 "gaussian" | "pixelate"
        kernel_size:        高斯模糊核大小 (奇数), 越大越模糊
        detection_interval: 跳帧间隔 (1=每帧, 5=每5帧检测一次)
    """

    def __init__(
        self,
        blur_faces: bool = True,
        blur_plates: bool = True,
        method: str = "gaussian",
        kernel_size: int = 25,
        detection_interval: int = 1,
    ):
        self._blur_faces = blur_faces
        self._blur_plates = blur_plates
        self._method = method
        self._kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self._interval = max(1, detection_interval)

        # 跳帧计数
        self._frame_count = 0
        self._last_face_rois: list[tuple[int, int, int, int]] = []
        self._last_plate_rois: list[tuple[int, int, int, int]] = []

        # 延迟加载 Haar Cascade (避免未启用时加载)
        self._face_cascade = None
        self._plate_cascade = None
        self._cascades_loaded = False
        self._cascade_load_error = False

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def blur_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """对单帧执行隐私脱敏，返回处理后的帧 (不修改原图)。

        如果隐私模块未启用或检测器加载失败，直接返回原帧。
        """
        if not PRIVACY_BLUR_ENABLED:
            return bgr_frame

        if not self._blur_faces and not self._blur_plates:
            return bgr_frame

        self._ensure_cascades()
        if self._cascade_load_error:
            return bgr_frame

        result = bgr_frame.copy()
        h, w = result.shape[:2]

        # 跳帧逻辑: 非检测帧使用上一次的 ROI
        self._frame_count += 1
        do_detect = (self._frame_count % self._interval == 1)

        face_rois: list[tuple[int, int, int, int]] = []
        plate_rois: list[tuple[int, int, int, int]] = []

        if do_detect:
            # 缩放到 640px 宽加速检测
            scale = 1.0
            detect_frame = result
            if w > 640:
                scale = 640.0 / w
                detect_h = int(h * scale)
                detect_frame = cv2.resize(result, (640, detect_h))

            if self._blur_faces:
                face_rois = self._detect_faces(detect_frame, scale)
                self._last_face_rois = face_rois

            if self._blur_plates:
                plate_rois = self._detect_plates(detect_frame, scale)
                self._last_plate_rois = plate_rois
        else:
            # 复用上次检测到的 ROI (相邻帧目标位置变化很小)
            face_rois = self._last_face_rois
            plate_rois = self._last_plate_rois

        # 应用模糊
        all_rois = face_rois + plate_rois
        for (rx, ry, rw, rh) in all_rois:
            # 边界裁剪
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)
            if rw <= 0 or rh <= 0:
                continue

            if self._method == "pixelate":
                self._apply_pixelate(result, (rx, ry, rw, rh))
            else:
                self._apply_gaussian(result, (rx, ry, rw, rh))

        return result

    # ----------------------------------------------------------------
    # 检测器
    # ----------------------------------------------------------------

    def _ensure_cascades(self):
        """惰性加载 Haar Cascade 文件。"""
        if self._cascades_loaded or self._cascade_load_error:
            return

        cascade_path = cv2.data.haarcascades

        if self._blur_faces:
            face_xml = cascade_path + "haarcascade_frontalface_default.xml"
            if Path(face_xml).exists():
                self._face_cascade = cv2.CascadeClassifier(face_xml)
                if self._face_cascade.empty():
                    logger.warning("人脸 Haar Cascade 加载失败，人脸模糊已禁用")
                    self._blur_faces = False
            else:
                logger.warning("人脸 Haar Cascade 文件不存在: %s", face_xml)
                self._blur_faces = False

        if self._blur_plates:
            plate_xml = cascade_path + "haarcascade_russian_plate_number.xml"
            if Path(plate_xml).exists():
                self._plate_cascade = cv2.CascadeClassifier(plate_xml)
                if self._plate_cascade.empty():
                    logger.warning(
                        "车牌 Haar Cascade 加载失败，将使用边缘检测备选方案"
                    )
                    self._plate_cascade = None
            else:
                logger.warning(
                    "车牌 Haar Cascade 文件不存在，将使用边缘检测备选方案"
                )
                self._plate_cascade = None

        if not self._blur_faces and not self._blur_plates:
            self._cascade_load_error = True
            logger.warning("所有隐私检测器加载失败，隐私模糊已禁用")

        self._cascades_loaded = True

    def _detect_faces(
        self, gray_or_bgr: np.ndarray, scale: float
    ) -> list[tuple[int, int, int, int]]:
        """Haar Cascade 人脸检测。

        返回: [(x, y, w, h), ...] 在原图坐标系下
        """
        if self._face_cascade is None:
            return []

        gray = (
            cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
            if gray_or_bgr.ndim == 3
            else gray_or_bgr
        )

        try:
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
            )
        except Exception:
            return []

        # 坐标映射回原始分辨率
        if scale != 1.0:
            inv_scale = 1.0 / scale
            faces = [
                (int(x * inv_scale), int(y * inv_scale),
                 int(w * inv_scale), int(h * inv_scale))
                for (x, y, w, h) in faces
            ]

        # 扩展 ROI 边界 (确保完全覆盖)
        margin = int(self._kernel_size * 0.3)
        return [
            (x - margin, y - margin, w + 2 * margin, h + 2 * margin)
            for (x, y, w, h) in faces
        ]

    def _detect_plates(
        self, gray_or_bgr: np.ndarray, scale: float
    ) -> list[tuple[int, int, int, int]]:
        """双策略车牌检测: Haar Cascade + 边缘检测备选。"""
        rois: list[tuple[int, int, int, int]] = []

        # 策略 1: Haar Cascade (如果加载成功)
        if self._plate_cascade is not None:
            gray = (
                cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
                if gray_or_bgr.ndim == 3
                else gray_or_bgr
            )
            try:
                plates = self._plate_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4,
                    minSize=(60, 20), flags=cv2.CASCADE_SCALE_IMAGE,
                )
                if scale != 1.0:
                    inv_scale = 1.0 / scale
                    plates = [
                        (int(x * inv_scale), int(y * inv_scale),
                         int(w * inv_scale), int(h * inv_scale))
                        for (x, y, w, h) in plates
                    ]
                rois.extend(plates)
            except Exception:
                pass

        # 策略 2: 边缘检测 + 轮廓筛选 (对中国车牌蓝底/绿底更有效)
        edge_rois = self._detect_plates_by_edges(gray_or_bgr, scale)
        rois.extend(edge_rois)

        # 去重 (合并重叠 ROI)
        rois = self._merge_overlapping_rois(rois)

        margin = int(self._kernel_size * 0.2)
        return [
            (x - margin, y - margin, w + 2 * margin, h + 2 * margin)
            for (x, y, w, h) in rois
        ]

    def _detect_plates_by_edges(
        self, bgr: np.ndarray, scale: float
    ) -> list[tuple[int, int, int, int]]:
        """基于边缘检测 + 轮廓筛选的车牌检测。

        原理: 车牌区域通常具有密集的垂直边缘 + 特定的宽高比 (3:1 ~ 5:1)。
        适用于中国蓝牌 (蓝底白字) 和新能源绿牌 (绿底黑字)。
        """
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel 垂直边缘检测 (车牌字符产生强烈垂直边缘)
        sobel_y = cv2.Sobel(blurred, cv2.CV_8U, 0, 1, ksize=3)
        _, binary = cv2.threshold(sobel_y, 50, 255, cv2.THRESH_BINARY)

        # 形态学闭运算: 连接相邻垂直边缘形成矩形区域
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # 查找轮廓
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rois: list[tuple[int, int, int, int]] = []
        min_w = int(w * 0.03)   # 最小宽度: 画面宽度的 3%
        max_w = int(w * 0.25)   # 最大宽度: 画面宽度的 25%

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / ch if ch > 0 else 0

            # 车牌宽高比: 中国蓝牌 ≈ 3.5:1, 绿牌 ≈ 4.5:1, 范围 2.5~6
            if not (2.5 <= aspect_ratio <= 6.0):
                continue
            if not (min_w <= cw <= max_w):
                continue
            if ch < 10 or ch > h * 0.15:
                continue

            # 检查区域内边缘密度
            roi_edge = binary[y:y + ch, x:x + cw]
            edge_density = np.count_nonzero(roi_edge) / (cw * ch) if cw * ch > 0 else 0
            if edge_density < 0.08:  # 车牌区域边缘密度通常较高
                continue

            rois.append((x, y, cw, ch))

        # 坐标映射回原始分辨率
        if scale != 1.0:
            inv_scale = 1.0 / scale
            rois = [
                (int(rx * inv_scale), int(ry * inv_scale),
                 int(rw * inv_scale), int(rh * inv_scale))
                for (rx, ry, rw, rh) in rois
            ]

        return rois

    @staticmethod
    def _merge_overlapping_rois(
        rois: list[tuple[int, int, int, int]],
        iou_threshold: float = 0.3,
    ) -> list[tuple[int, int, int, int]]:
        """合并重叠的 ROI (去重)。"""
        if len(rois) <= 1:
            return rois

        # 按面积降序排列
        sorted_rois = sorted(rois, key=lambda r: r[2] * r[3], reverse=True)
        merged: list[tuple[int, int, int, int]] = []

        for roi in sorted_rois:
            x1, y1, w1, h1 = roi
            should_merge = False
            for i, (x2, y2, w2, h2) in enumerate(merged):
                # 计算 IoU
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)
                if ix1 >= ix2 or iy1 >= iy2:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = w1 * h1 + w2 * h2 - inter
                if union > 0 and inter / union > iou_threshold:
                    # 合并: 取外接矩形
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    merged[i] = (nx, ny, nw, nh)
                    should_merge = True
                    break
            if not should_merge:
                merged.append(roi)

        return merged

    # ----------------------------------------------------------------
    # 模糊方法
    # ----------------------------------------------------------------

    def _apply_gaussian(
        self, bgr: np.ndarray, roi: tuple[int, int, int, int]
    ) -> None:
        """对 ROI 区域应用高斯模糊 (原地修改)。"""
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return
        k = self._kernel_size
        roi_area = bgr[y:y + h, x:x + w]
        blurred = cv2.GaussianBlur(roi_area, (k, k), 0)
        bgr[y:y + h, x:x + w] = blurred

    def _apply_pixelate(
        self, bgr: np.ndarray, roi: tuple[int, int, int, int],
        block_size: int = 8,
    ) -> None:
        """对 ROI 区域应用马赛克效果 (原地修改)。

        原理: 将 ROI 缩小到 1/block_size → 再用 INTER_NEAREST 放大回原始尺寸，
        产生像素块效果。
        """
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return

        # block_size 自适应: 保证至少产生 3 个像素块
        effective_block = min(block_size, max(3, w // 4), max(3, h // 4))

        roi_area = bgr[y:y + h, x:x + w]
        small_h = max(1, h // effective_block)
        small_w = max(1, w // effective_block)

        # 下采样 → 上采样 (最近邻插值产生锯齿/马赛克效果)
        small = cv2.resize(roi_area, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_NEAREST
        )
        bgr[y:y + h, x:x + w] = pixelated

    # ----------------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------------

    def reset(self):
        """重置跳帧计数和缓存 ROI。"""
        self._frame_count = 0
        self._last_face_rois.clear()
        self._last_plate_rois.clear()
