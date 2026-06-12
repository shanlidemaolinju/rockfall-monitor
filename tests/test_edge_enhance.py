"""
Sobel边缘增强单元测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from rockfall.edge_enhance import sobel_edge_enhance, EdgeEnhancer


class TestSobelEdgeEnhance:
    """sobel_edge_enhance 纯函数测试"""

    def test_output_shape_bgr(self):
        """输出形状应与输入BGR一致"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = sobel_edge_enhance(frame, alpha=0.3)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_alpha_zero_returns_original(self):
        """alpha=0 时应返回原图"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = sobel_edge_enhance(frame, alpha=0.0)
        np.testing.assert_array_equal(result, frame)

    def test_different_alpha_values(self):
        """不同 alpha 均不抛异常"""
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        for alpha in [0.1, 0.3, 0.5, 0.8, 1.0]:
            result = sobel_edge_enhance(frame, alpha)
            assert result.shape == frame.shape

    def test_edge_enhanced_pixels_differ(self):
        """有边缘的图增强后像素应有变化"""
        # 白色方块在黑色背景上 (有明显边缘)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), -1)
        result = sobel_edge_enhance(frame, alpha=0.5)
        # 边缘附近像素应有变化
        assert not np.array_equal(result, frame)


class TestEdgeEnhancer:
    """EdgeEnhancer 有状态封装测试"""

    def test_disabled_returns_original(self):
        enhancer = EdgeEnhancer(enabled=False)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = enhancer.process(frame)
        np.testing.assert_array_equal(result, frame)

    def test_enabled_processes_frame(self):
        enhancer = EdgeEnhancer(enabled=True, alpha=0.3)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = enhancer.process(frame)
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_interval_skip(self):
        """interval=3 时每3帧才增强一次 (计数器从1开始)"""
        enhancer = EdgeEnhancer(enabled=True, alpha=0.5, interval=3)
        # 用有明显边缘的图: 白方块在黑色背景上, Sobel 一定能检测到边缘
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (60, 60), (255, 255, 255), -1)

        r1 = enhancer.process(frame.copy())  # counter=1, 1%3≠0 → 跳过
        r2 = enhancer.process(frame.copy())  # counter=2, 2%3≠0 → 跳过
        r3 = enhancer.process(frame.copy())  # counter=3, 3%3=0 → 增强

        np.testing.assert_array_equal(r1, frame)  # 跳过
        np.testing.assert_array_equal(r2, frame)  # 跳过
        assert not np.array_equal(r3, frame)       # 增强 → 不同