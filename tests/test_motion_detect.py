"""
三帧差分运动检测 + IoU滤波 单元测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from rockfall.motion_detect import (
    ThreeFrameDiff, filter_detections_by_motion,
    filter_detections_by_mog2_center, _box_iou_batch,
)
from rockfall.fusion import fuse_confidence, TemporalFilter


class TestBoxIoUBatch:
    """_box_iou_batch 测试"""

    def test_perfect_overlap(self):
        a = np.array([[100, 100, 200, 200]], dtype=np.float32)
        b = np.array([[100, 100, 200, 200]], dtype=np.float32)
        iou = _box_iou_batch(a, b)
        assert iou[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[100, 100, 110, 110]], dtype=np.float32)
        iou = _box_iou_batch(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float32)
        b = np.array([[50, 50, 150, 150]], dtype=np.float32)
        iou = _box_iou_batch(a, b)
        # 交集 50×50=2500, 并集 10000+10000-2500=17500 → 0.1428
        assert 0.14 < iou[0, 0] < 0.15

    def test_batch_shape(self):
        a = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
        b = np.array([[50, 50, 150, 150]], dtype=np.float32)
        iou = _box_iou_batch(a, b)
        assert iou.shape == (2, 1)


class TestThreeFrameDiff:
    """ThreeFrameDiff 三帧差分测试"""

    def test_buffer_not_full_returns_empty(self):
        tfd = ThreeFrameDiff(threshold=25, morph_kernel=5)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mask1, contours1 = tfd.compute(frame)
        assert mask1 is None
        assert contours1 == []

        mask2, contours2 = tfd.compute(frame)
        assert mask2 is None
        assert contours2 == []

    def test_three_identical_frames_no_motion(self):
        """三帧相同 → 无运动"""
        tfd = ThreeFrameDiff(threshold=25, morph_kernel=5)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        tfd.compute(frame)
        tfd.compute(frame)
        mask, contours = tfd.compute(frame)

        assert mask is not None
        assert len(contours) == 0  # 无运动轮廓

    def test_moving_rectangle_produces_contours(self):
        """白色方块移动 → 应有运动轮廓"""
        tfd = ThreeFrameDiff(threshold=25, morph_kernel=3)

        # 帧1: 方块在(20,20)
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(f1, (20, 20), (50, 50), (255, 255, 255), -1)

        # 帧2: 方块在(30,20)
        f2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(f2, (30, 20), (60, 50), (255, 255, 255), -1)

        # 帧3: 方块在(40,20)
        f3 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(f3, (40, 20), (70, 50), (255, 255, 255), -1)

        tfd.compute(f1)
        tfd.compute(f2)
        mask, contours = tfd.compute(f3)

        assert mask is not None
        assert len(contours) > 0  # 应有运动轮廓

    def test_reset_clears_buffer(self):
        tfd = ThreeFrameDiff(threshold=25)
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        tfd.compute(frame)
        tfd.compute(frame)
        assert len(tfd._buffer) == 2

        tfd.reset()
        assert len(tfd._buffer) == 0
        mask, _ = tfd.compute(frame)
        assert mask is None  # 预热中


class TestFilterDetectionsByMotion:
    """filter_detections_by_motion 测试"""

    def test_empty_detections(self):
        result = filter_detections_by_motion([], [np.array([[0, 0, 50, 50]])])
        assert result == []

    def test_empty_contours_passthrough(self):
        """无运动轮廓时全量通过 (预热期)"""
        dets = [[100, 100, 200, 200, 0.9]]
        result = filter_detections_by_motion(dets, [])
        assert result == dets

    def test_overlapping_detection_kept(self):
        """检测框与运动轮廓重叠 → 保留"""
        # 检测框窄一些, 与位移产生的运动轮廓区域高度重叠
        dets = [[100, 100, 135, 200, 0.9]]  # 35×100 = 3500 px²

        f1 = np.zeros((300, 300, 3), dtype=np.uint8)
        f2 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(f2, (100, 100), (200, 200), (255, 255, 255), -1)
        f3 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(f3, (120, 100), (220, 200), (255, 255, 255), -1)

        tfd = ThreeFrameDiff(threshold=25)
        tfd.compute(f1)
        tfd.compute(f2)
        _, contours = tfd.compute(f3)

        result = filter_detections_by_motion(dets, contours, iou_threshold=0.30)
        assert len(result) == 1  # IoU close to 0.5

    def test_non_overlapping_detection_filtered(self):
        """检测框与运动轮廓无重叠 → 过滤"""
        dets = [[0, 0, 50, 50, 0.9]]  # 左上角

        f1 = np.zeros((300, 300, 3), dtype=np.uint8)
        f2 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(f2, (200, 200), (280, 280), (255, 255, 255), -1)
        f3 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(f3, (210, 200), (290, 280), (255, 255, 255), -1)

        tfd = ThreeFrameDiff(threshold=25)
        tfd.compute(f1)
        tfd.compute(f2)
        _, contours = tfd.compute(f3)

        result = filter_detections_by_motion(dets, contours, iou_threshold=0.30)
        assert len(result) == 0


class TestFilterDetectionsByMog2Center:
    """MOG2中心点运动滤波测试 (Zhang2024)"""

    def test_empty_detections(self):
        result = filter_detections_by_mog2_center([], None)
        assert result == []

    def test_none_mask_passthrough(self):
        dets = [[100, 100, 200, 200, 0.9]]
        result = filter_detections_by_mog2_center(dets, None)
        assert result == dets

    def test_center_in_foreground_kept(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (80, 80), (220, 220), 255, -1)
        dets = [[100, 100, 200, 200, 0.9]]  # 中心 (150,150) 在掩膜内
        result = filter_detections_by_mog2_center(dets, mask)
        assert len(result) == 1

    def test_center_not_in_foreground_filtered(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (50, 50), 255, -1)  # 掩膜在左上角
        dets = [[200, 200, 280, 280, 0.9]]  # 中心 (240,240) 在掩膜外
        result = filter_detections_by_mog2_center(dets, mask)
        assert len(result) == 0

    def test_center_on_boundary(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (200, 200), 255, -1)
        # 中心恰好在掩膜外边缘像素 (201, 201) → cv2.rectangle 包含 (200,200)
        dets = [[152, 152, 250, 250, 0.9]]  # 中心 (201,201), 刚好在掩膜外
        result = filter_detections_by_mog2_center(dets, mask)
        assert len(result) == 0  # 中心在掩膜边界外

    def test_multiple_mixed(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (300, 300), 255, -1)
        dets = [
            [120, 120, 180, 180, 0.9],  # 中心在掩膜内
            [10, 10, 50, 50, 0.8],      # 中心在掩膜外
            [250, 250, 350, 350, 0.7],  # 中心在掩膜内
        ]
        result = filter_detections_by_mog2_center(dets, mask)
        assert len(result) == 2
        assert result[0][4] == 0.9
        assert result[1][4] == 0.7


class TestFuseConfidence:
    """fuse_confidence 概率融合测试"""

    def test_empty_detections(self):
        result = fuse_confidence([], np.ones((100, 100), dtype=np.uint8) * 255)
        assert result == []

    def test_none_mask_passthrough(self):
        dets = [[100, 100, 200, 200, 0.8]]
        result = fuse_confidence(dets, None)
        assert result == dets

    def test_full_foreground(self):
        """检测框完全在MOG2前景内 → 置信度被提升 (加权平均)"""
        mask = np.ones((300, 300), dtype=np.uint8) * 255
        dets = [[100, 100, 200, 200, 0.6]]
        result = fuse_confidence(dets, mask, motion_weight=1.0)
        # P_MOG2 = 1.0, fused = 0.6*(1-1.0) + 1.0*1.0 = 1.0
        assert result[0][4] == pytest.approx(1.0, abs=0.01)

    def test_no_foreground(self):
        """检测框无前景像素 → 置信度降低 (加权平均)"""
        mask = np.zeros((300, 300), dtype=np.uint8)
        dets = [[100, 100, 200, 200, 0.7]]
        result = fuse_confidence(dets, mask, motion_weight=0.5)
        # P_MOG2 = 0, fused = 0.7*(1-0.5) + 0*0.5 = 0.35
        assert result[0][4] == pytest.approx(0.35, abs=0.02)

    def test_partial_foreground(self):
        """部分前景 → 加权平均"""
        mask = np.zeros((300, 300), dtype=np.uint8)
        # 框的左上1/4是前景
        mask[100:150, 100:150] = 255
        dets = [[100, 100, 200, 200, 0.5]]  # 100×100=10000px, fg=2500px → ratio=0.25
        result = fuse_confidence(dets, mask, motion_weight=1.0)
        # fused = 0.5*(1-1.0) + 0.25*1.0 = 0.25
        assert 0.24 < result[0][4] < 0.26

    def test_motion_weight_zero(self):
        """weight=0 → 置信度不变"""
        mask = np.ones((300, 300), dtype=np.uint8) * 255
        dets = [[100, 100, 200, 200, 0.6]]
        result = fuse_confidence(dets, mask, motion_weight=0.0)
        assert result[0][4] == pytest.approx(0.6, abs=0.01)

    def test_box_clamped_to_mask(self):
        """检测框超出掩膜边界 → 被裁剪"""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        dets = [[-10, -10, 50, 50, 0.8]]
        result = fuse_confidence(dets, mask, motion_weight=1.0)
        # 裁剪到 [0,0,50,50], fg_ratio < 1
        assert result[0][4] > 0.8


class TestTemporalFilter:
    """TemporalFilter 时序确认测试"""

    def test_empty_detections(self):
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        result = tf.filter([])
        assert result == []

    def test_first_frame_passthrough(self):
        """缓存空 → 首帧全量通过"""
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        dets = [[100, 100, 200, 200, 0.9]]
        result = tf.filter(dets)
        assert result == dets

    def test_overlap_kept(self):
        """连续帧相同位置检测 → 保留"""
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        dets1 = [[100, 100, 200, 200, 0.9]]
        tf.filter(dets1)
        dets2 = [[105, 105, 205, 205, 0.85]]  # 高度重叠
        result = tf.filter(dets2)
        assert len(result) == 1

    def test_no_overlap_filtered(self):
        """完全不同的位置 → 过滤"""
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        tf.filter([[100, 100, 200, 200, 0.9]])
        result = tf.filter([[300, 300, 400, 400, 0.8]])
        assert len(result) == 0

    def test_window_eviction(self):
        """缓存不超过窗口大小"""
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        tf.filter([[0, 0, 50, 50, 0.9]])
        tf.filter([[0, 0, 50, 50, 0.9]])
        tf.filter([[0, 0, 50, 50, 0.9]])
        assert len(tf._buffer) == 2

    def test_reset_clears_buffer(self):
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        tf.filter([[0, 0, 50, 50, 0.9]])
        assert len(tf._buffer) == 1
        tf.reset()
        assert len(tf._buffer) == 0

    def test_disabled_passthrough(self):
        """disabled → 全部透传, 不写缓存"""
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=False)
        dets = [[100, 100, 200, 200, 0.9]]
        result = tf.filter(dets)
        assert result == dets
        assert len(tf._buffer) == 0

    def test_mixed_kept_and_filtered(self):
        """一框有历史匹配, 一框没有 → 仅保留前者"""
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        tf.filter([[100, 100, 200, 200, 0.9]])
        result = tf.filter([
            [105, 105, 205, 205, 0.85],   # 与历史重叠 → 保留
            [400, 400, 500, 500, 0.8],     # 无历史匹配 → 过滤
        ])
        assert len(result) == 1
        assert result[0][4] == 0.85

    def test_previous_empty_allows_new(self):
        """上一帧无检测 → 当前帧全量通过 (新目标)"""
        tf = TemporalFilter(window=2, iou_threshold=0.3, enabled=True)
        tf.filter([])
        dets = [[100, 100, 200, 200, 0.9]]
        result = tf.filter(dets)
        assert result == dets

    def test_window3_matches_historical(self):
        """window=3: 检测框只需与窗口内任意一帧匹配即可保留"""
        tf = TemporalFilter(window=3, iou_threshold=0.3, enabled=True)
        tf.filter([[100, 100, 200, 200, 0.9]])   # 帧1: 位置A
        tf.filter([[300, 300, 400, 400, 0.8]])   # 帧2: 位置B (与A不重叠)
        # 帧3: 位置A (与帧2不重叠, 但与帧1重叠) → 窗口匹配应保留
        result = tf.filter([[105, 105, 205, 205, 0.85]])
        assert len(result) == 1