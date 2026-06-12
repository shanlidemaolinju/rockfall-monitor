"""
SAHI 切片辅助推理 单元测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from rockfall.sahi import SAHISlicer


class TestSAHISlicer:
    """SAHISlicer 切片器测试"""

    def test_get_slices_smaller_than_size(self):
        """帧小于切片尺寸 → 单个全图切片"""
        slicer = SAHISlicer(slice_size=640, overlap_ratio=0.2)
        slices = slicer.get_slices(480, 640)
        assert len(slices) == 1
        assert slices[0] == (0, 0, 640, 480)

    def test_get_slices_exact_fit(self):
        """帧恰好等于切片尺寸 → 单个切片"""
        slicer = SAHISlicer(slice_size=640, overlap_ratio=0.2)
        slices = slicer.get_slices(640, 640)
        assert len(slices) == 1
        assert slices[0] == (0, 0, 640, 640)

    def test_get_slices_larger(self):
        """1920×1080 帧, slice=640, overlap=0.2 → 多个切片且有重叠"""
        slicer = SAHISlicer(slice_size=640, overlap_ratio=0.2, enabled=True)
        slices = slicer.get_slices(1080, 1920)
        # stride = 640 * 0.8 = 512
        # x方向: 0, 512, 1024 → 最后一片在 1920-640=1280
        # y方向: 0, 512 → 最后一片在 1080-640=440
        assert len(slices) >= 3  # 至少 3 个切片
        # 切片应覆盖整个宽高
        for x1, y1, x2, y2 in slices:
            assert x2 - x1 == 640
            assert y2 - y1 == 640

    def test_get_slices_overlap(self):
        """验证相邻切片有重叠"""
        slicer = SAHISlicer(slice_size=640, overlap_ratio=0.2, enabled=True)
        slices = slicer.get_slices(640, 1280)  # stride=512, 3 slices
        assert len(slices) == 3
        # x=0: (0,0,640,640), x=512: (512,0,1152,640), x=edge: (640,0,1280,640)
        assert slices[0][0] < slices[1][0] < slices[0][2]  # slice1 and slice2 overlap
        assert slices[1][0] < slices[2][0] < slices[1][2]  # slice2 and slice3 overlap

    def test_remap_origin_zero(self):
        """切片原点(0,0) → 坐标不变"""
        dets = [[10, 20, 50, 60, 0.9]]
        result = SAHISlicer.remap_detections(dets, (0, 0))
        assert result[0][:4] == [10, 20, 50, 60]
        assert result[0][4] == 0.9

    def test_remap_offset(self):
        """切片原点(320, 240) → 坐标加偏移"""
        dets = [[10, 20, 50, 60, 0.8]]
        result = SAHISlicer.remap_detections(dets, (320, 240))
        assert result[0][:4] == [330, 260, 370, 300]

    def test_merge_detections_empty(self):
        result = SAHISlicer.merge_detections([])
        assert result == []

    def test_merge_single_detection(self):
        dets = [[100, 100, 200, 200, 0.9]]
        result = SAHISlicer.merge_detections(dets)
        assert len(result) == 1

    def test_merge_no_overlap(self):
        dets = [
            [0, 0, 50, 50, 0.9],
            [200, 200, 250, 250, 0.8],
        ]
        result = SAHISlicer.merge_detections(dets, iou_threshold=0.5)
        assert len(result) == 2

    def test_merge_full_overlap(self):
        """完全重叠 → 仅保留高置信度"""
        dets = [
            [100, 100, 200, 200, 0.9],
            [100, 100, 200, 200, 0.7],
        ]
        result = SAHISlicer.merge_detections(dets, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0][4] == 0.9

    def test_merge_partial_overlap_high_iou(self):
        """高重叠 → 合并为一个"""
        dets = [
            [0, 0, 100, 100, 0.95],
            [10, 10, 110, 110, 0.85],
        ]
        result = SAHISlicer.merge_detections(dets, iou_threshold=0.5)
        assert len(result) == 1

    def test_disabled_passthrough(self):
        """disabled时 get_slices 仍可工作, enabled仅标记"""
        slicer = SAHISlicer(enabled=False)
        assert slicer.enabled is False
        slices = slicer.get_slices(100, 100)
        assert len(slices) == 1