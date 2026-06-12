"""
预处理层 — Sobel边缘增强
=====================================================
在YOLO推理前增强图像边缘轮廓，补偿运动模糊导致的轮廓弱化。

原理:
  1. 转灰度 → Sobel X/Y梯度 → 梯度幅值
  2. 幅值图转BGR → 加权融合回原图
  3. 增强后的帧输入YOLO，提升对模糊落石的检测能力

使用方式:
    from rockfall.edge_enhance import EdgeEnhancer

    enhancer = EdgeEnhancer(enabled=True, alpha=0.3, interval=1)
    enhanced = enhancer.process(frame)
"""

import cv2
import numpy as np


def sobel_edge_enhance(frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Sobel梯度幅值提取 + 加权融合回原图

    参数:
        frame: BGR图像 (H, W, 3)
        alpha: 边缘叠加权重 (0~1), 越大边缘越突出

    返回:
        增强后的BGR图像 (H, W, 3), dtype与原图一致
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    edges_bgr = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1.0 - alpha, edges_bgr, alpha, 0)


class EdgeEnhancer:
    """
    有状态边缘增强器 — 支持跳帧节省算力

    参数:
        enabled:       是否启用 (默认False, 保持向后兼容)
        alpha:         边缘叠加权重 (0.3 ≈ 30%边缘 + 70%原图)
        interval:      每N帧执行一次增强 (1=每帧)
        cuda_available: 是否使用 GPU Sobel (cv2.cuda)
    """

    def __init__(self, enabled: bool = False, alpha: float = 0.3, interval: int = 1,
                 cuda_available: bool = False):
        self.enabled = enabled
        self.alpha = alpha
        self.interval = max(interval, 1)
        self._counter = 0
        self._cuda = cuda_available

    def process(self, frame: np.ndarray) -> np.ndarray:
        """处理一帧, 返回增强后的图像"""
        if not self.enabled:
            return frame
        self._counter += 1
        if self._counter % self.interval != 0:
            return frame
        if self._cuda:
            return _sobel_cuda(frame, self.alpha)
        return sobel_edge_enhance(frame, self.alpha)


def _sobel_cuda(frame: np.ndarray, alpha: float) -> np.ndarray:
    """GPU Sobel 边缘增强 (cv2.cuda)"""
    gpu_frame = cv2.cuda.GpuMat(frame)
    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.cuda.Sobel(gpu_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.cuda.Sobel(gpu_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.cuda.magnitude(sx, sy)
    mag = cv2.cuda.convertTo(mag, cv2.CV_8U)
    mag_cpu = mag.download()
    edges_bgr = cv2.cvtColor(mag_cpu, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1.0 - alpha, edges_bgr, alpha, 0)