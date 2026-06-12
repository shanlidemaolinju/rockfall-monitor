"""
跟踪器单元测试 — KalmanBoxTracker + RockTracker
================================================
运行: python -m pytest tests/test_tracker.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from rockfall.tracker import KalmanBoxTracker, RockTracker


class TestKalmanBoxTracker:
    """KalmanBoxTracker 单目标跟踪器测试"""

    def test_init(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)

        assert tracker.id >= 0
        assert tracker.age == 0
        assert tracker.missed == 0
        assert tracker.confidence == 0.0
        assert len(tracker.trajectory) == 1
        assert tracker.bbox == [100, 100, 200, 200]

    def test_predict_returns_valid_bbox(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        predicted = tracker.predict()

        assert len(predicted) == 4
        assert predicted[0] < predicted[2]  # x1 < x2
        assert predicted[1] < predicted[3]  # y1 < y2
        assert tracker.age == 1
        assert tracker.missed == 1

    def test_update_resets_missed(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        tracker.predict()
        assert tracker.missed == 1

        tracker.update(np.array([105, 105, 205, 205], dtype=np.float32))
        assert tracker.missed == 0

    def test_predict_update_cycle(self):
        """多次 predict+update 后 bbox 应跟踪目标"""
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)

        for i in range(10):
            tracker.predict()
            # 目标逐步向右下移动
            dx, dy = i * 2, i * 3
            tracker.update(np.array([100 + dx, 100 + dy, 200 + dx, 200 + dy], dtype=np.float32))

        assert tracker.age == 10
        assert tracker.missed == 0
        assert len(tracker.trajectory) == 11  # init + 10 updates

    def test_confirm_requires_min_age(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)

        assert not tracker.is_confirmed(min_age=3)
        for _ in range(3):
            tracker.predict()
            tracker.update(bbox)
        assert tracker.is_confirmed(min_age=3)

    def test_invalid_after_max_missed(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)

        for _ in range(10):
            tracker.predict()
        assert not tracker.is_valid(max_missed=10)

    def test_area_computation(self):
        bbox = np.array([100, 100, 300, 250], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        assert tracker.area == pytest.approx(200 * 150, rel=0.01)

    def test_motion_state_stationary(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        # 新初始化状态: vy=0, ay=0 → 静止
        assert tracker.motion_state == "静止"

    def test_motion_state_falling(self):
        """模拟快速下坠：Y 方向二次加速 (匀加速 ≈ 10 px/frame²)"""
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)

        # 二次位移: dy = 5*i², 每帧位移增量 10px → ay ≈ 10 > 7.5 阈值
        for i in range(1, 15):
            tracker.predict()
            dy = 5 * i * i  # 5, 20, 45, 80, 125, ...
            tracker.update(np.array([100, 100 + dy, 200, 200 + dy], dtype=np.float32))

        assert tracker.motion_state in ("快速坠落", "快速移动")

    def test_smoothed_confidence(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        tracker.confidence = 0.5
        tracker.update(bbox)
        tracker.confidence = 0.7
        tracker.update(bbox)
        tracker.confidence = 0.9
        tracker.update(bbox)

        smoothed = tracker.smoothed_confidence
        assert 0.6 < smoothed < 0.8  # avg of 0.5, 0.7, 0.9 ≈ 0.7


class TestRockTracker:
    """RockTracker 多目标跟踪器测试"""

    def setup_method(self):
        """每个测试前重置全局 ID 计数器, 确保 ID 可预测"""
        RockTracker._global_id = 0

    def test_empty_detections(self):
        tracker = RockTracker()
        result = tracker.update([])
        assert result == []

    def test_single_detection_creates_track(self):
        tracker = RockTracker()
        dets = [[100, 100, 200, 200, 0.85]]
        result = tracker.update(dets)

        assert len(result) == 1
        assert result[0]["id"] == 0
        assert result[0]["confidence"] == 0.85
        assert not result[0]["confirmed"]

    def test_consistent_tracking(self):
        """同一目标多帧检测保持同一 ID"""
        tracker = RockTracker()

        track_id = None
        for _ in range(10):
            dets = [[100, 100, 200, 200, 0.9]]
            result = tracker.update(dets)
            assert len(result) == 1
            if track_id is None:
                track_id = result[0]["id"]
            else:
                assert result[0]["id"] == track_id

        assert result[0]["confirmed"]
        assert result[0]["age"] >= 9  # 首帧创建(age=0) + 9 次更新

    def test_track_id_increments(self):
        """不同目标分配不同 ID"""
        tracker = RockTracker()
        dets = [[100, 100, 200, 200, 0.9], [300, 100, 400, 200, 0.8]]
        result = tracker.update(dets)

        assert len(result) == 2
        ids = {t["id"] for t in result}
        assert len(ids) == 2

    def test_track_death_on_miss(self):
        """多帧未匹配后轨迹删除"""
        tracker = RockTracker(max_missed=3)

        # 创建轨迹
        tracker.update([[100, 100, 200, 200, 0.9]])
        assert len(tracker.tracks) == 1

        # 连续空检测
        for _ in range(5):
            tracker.update([])

        assert len(tracker.tracks) == 0

    def test_reset_clears_all(self):
        tracker = RockTracker()
        tracker.update([[100, 100, 200, 200, 0.9]])
        assert len(tracker.tracks) == 1

        tracker.reset()
        assert len(tracker.tracks) == 0
        # reset() 不重置全局 ID (避免 AlertManager 缓存混淆), 新轨迹 ID 继续递增

    def test_iou_batch(self):
        """IoU 批量计算"""
        boxes_a = np.array([[100, 100, 200, 200]], dtype=np.float32)
        boxes_b = np.array([[100, 100, 200, 200]], dtype=np.float32)  # 完全重合 → IoU=1
        iou = RockTracker._iou_batch(boxes_a, boxes_b)
        assert iou[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_iou_batch_no_overlap(self):
        boxes_a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes_b = np.array([[100, 100, 110, 110]], dtype=np.float32)
        iou = RockTracker._iou_batch(boxes_a, boxes_b)
        assert iou[0, 0] == 0.0

    def test_result_fields(self):
        """返回结果包含所有必需字段"""
        tracker = RockTracker()
        result = tracker.update([[100, 100, 200, 200, 0.88]])

        r = result[0]
        assert "id" in r
        assert "bbox" in r
        assert "confidence" in r
        assert "smoothed_confidence" in r
        assert "age" in r
        assert "confirmed" in r
        assert "age_for_alert" in r
        assert "speed" in r
        assert "acceleration" in r
        assert "area" in r
        assert "motion_state" in r
        assert "trajectory" in r
