"""测试 detector.py — AlertContext 聚合 + _grade_alert 四级分级逻辑
对齐《公路自然灾害监测预警系统技术指南》:
  Ⅰ 级 (红色):   置信度 > 0.9 或 直径 > 30cm
  Ⅱ 级 (橙色):   置信度 0.7-0.9 或 直径 20-30cm
  Ⅲ 级 (黄色):   置信度 0.5-0.7 或 直径 10-20cm
  Ⅳ 级 (蓝色):   置信度 0.3-0.5 或 直径 < 10cm
"""

import pytest

from rockfall.detector import AlertContext, RockDetector


def _make_track(confirmed=True, confidence=0.7, area=5000, height=100,
                speed=0, age=5, motion_state="运动", class_id=0, class_name="落石"):
    """构建一条模拟轨迹"""
    x1, y1 = 100, 200
    return {
        "id": 1,
        "bbox": [x1, y1, x1 + 100, y1 + height],
        "confidence": confidence,
        "smoothed_confidence": round(confidence * 0.95, 2),
        "area": area,
        "speed": speed,
        "age": age,
        "confirmed": confirmed,
        "motion_state": motion_state,
        "class_id": class_id,
        "class_name": class_name,
        "trajectory": [],
    }


class TestAlertContext:
    def test_empty_tracks(self):
        ctx = RockDetector.build_alert_context([])
        assert ctx.total_count == 0
        assert ctx.max_conf == 0.0
        assert ctx.track_ids == []

    def test_single_track(self):
        tracks = [_make_track()]
        ctx = RockDetector.build_alert_context(tracks, frame_w=1920, frame_h=1080)
        assert ctx.total_count == 1
        assert ctx.max_conf > 0
        assert ctx.frame_area == 1920 * 1080
        assert ctx.frame_height == 1080
        assert ctx.track_ids == [1]

    def test_unconfirmed_filtered(self):
        tracks = [_make_track(confirmed=False)]
        ctx = RockDetector.build_alert_context(tracks)
        assert ctx.total_count == 0  # unconfirmed excluded

    def test_aggregates_max_values(self):
        tracks = [
            _make_track(confidence=0.6, area=3000, speed=3, age=5),
            _make_track(confidence=0.9, area=8000, speed=12, age=15),
        ]
        tracks[0]["id"] = 1
        tracks[1]["id"] = 2
        ctx = RockDetector.build_alert_context(tracks)
        assert ctx.max_conf == pytest.approx(0.9 * 0.95, abs=0.05)
        assert ctx.max_area == 8000
        assert ctx.max_speed == 12
        assert ctx.max_age == 15
        assert ctx.total_area == 11000
        assert ctx.total_count == 2

    def test_falling_detected(self):
        tracks = [_make_track(motion_state="快速坠落", speed=15)]
        ctx = RockDetector.build_alert_context(tracks)
        assert ctx.is_falling is True

    def test_landslide_class(self):
        tracks = [_make_track(class_id=1, class_name="滑坡")]
        ctx = RockDetector.build_alert_context(tracks)
        assert ctx.total_count == 1

    def test_rock_diameter_calculated(self):
        """检测框高度应转换为估算落石直径"""
        tracks = [_make_track(height=54)]  # ~5% of 1080p → ~25cm
        ctx = RockDetector.build_alert_context(tracks, frame_w=1920, frame_h=1080)
        # height_ratio = 54/1080 = 0.05, ROCK_SMALL_HEIGHT_RATIO = 0.02
        # diameter = (0.05 / 0.02) * 10 = 25cm
        assert ctx.rock_diameter_cm == pytest.approx(25.0, abs=1.0)


class TestGradeAlert:
    """四级预警分级测试"""

    @classmethod
    def setup_class(cls):
        cls.detector = RockDetector()
        # 确保使用默认阈值 (blue<0.5, yellow<0.7, orange<0.9, red>=0.9)
        cls.detector.alert_blue_conf_high = 0.5
        cls.detector.alert_yellow_conf_high = 0.7
        cls.detector.alert_orange_conf_high = 0.9

    def _ctx(self, **kwargs):
        """构建 AlertContext, 默认值接近绿色/蓝色边界"""
        defaults = {
            "max_conf": 0.2, "max_area": 1000, "max_height": 10,
            "total_count": 1, "total_area": 1000,
            "max_speed": 1, "max_age": 3,
            "is_falling": False,
            "frame_area": 1920 * 1080, "frame_height": 1080,
            "rock_diameter_cm": 0,
        }
        defaults.update(kwargs)
        return AlertContext(**defaults)

    # ---- 基础等级测试 ----

    def test_green_when_empty(self):
        """无检测目标 → green"""
        ctx = AlertContext()
        assert self.detector._grade_alert(ctx) == "green"

    def test_green_below_threshold(self):
        """置信度 < 0.3 → green"""
        ctx = self._ctx(max_conf=0.25)
        assert self.detector._grade_alert(ctx) == "green"

    def test_blue_low_confidence(self):
        """置信度 0.3-0.5 → Ⅳ级蓝色"""
        ctx = self._ctx(max_conf=0.4)
        assert self.detector._grade_alert(ctx) == "blue"

    def test_yellow_medium_confidence(self):
        """置信度 0.5-0.7 → Ⅲ级黄色"""
        ctx = self._ctx(max_conf=0.6)
        assert self.detector._grade_alert(ctx) == "yellow"

    def test_orange_high_confidence(self):
        """置信度 0.7-0.9 → Ⅱ级橙色"""
        ctx = self._ctx(max_conf=0.8)
        assert self.detector._grade_alert(ctx) == "orange"

    def test_red_very_high_confidence(self):
        """置信度 > 0.9 → Ⅰ级红色"""
        ctx = self._ctx(max_conf=0.95)
        assert self.detector._grade_alert(ctx) == "red"

    # ---- 落石尺寸等级测试 ----

    def test_blue_by_size_small_rock(self):
        """直径 < 10cm → Ⅳ级蓝色 (即使置信度低)"""
        ctx = self._ctx(max_conf=0.2, rock_diameter_cm=5,
                        max_height=10, frame_height=1080)
        result = self.detector._grade_alert(ctx)
        assert result in ("blue", "green")  # conf < 0.3 may still be green

    def test_yellow_by_size_medium_rock(self):
        """直径 10-20cm → 至少Ⅲ级黄色"""
        ctx = self._ctx(max_conf=0.2, rock_diameter_cm=15,
                        max_height=40, frame_height=1080)
        assert self.detector._grade_alert(ctx) == "yellow"

    def test_orange_by_size_large_rock(self):
        """直径 20-30cm → 至少Ⅱ级橙色"""
        ctx = self._ctx(max_conf=0.2, rock_diameter_cm=25,
                        max_height=70, frame_height=1080)
        assert self.detector._grade_alert(ctx) == "orange"

    def test_red_by_size_xlarge_rock(self):
        """直径 > 30cm → Ⅰ级红色"""
        ctx = self._ctx(max_conf=0.2, rock_diameter_cm=35,
                        max_height=100, frame_height=1080)
        assert self.detector._grade_alert(ctx) == "red"

    # ---- 增强因子测试 ----

    def test_falling_bumps_to_yellow(self):
        """坠落状态 → 至少黄色 (即使置信度低)"""
        ctx = self._ctx(max_conf=0.35, is_falling=True)
        assert self.detector._grade_alert(ctx) == "yellow"

    def test_multi_target_bumps_to_yellow(self):
        """≥3 个目标 → 至少黄色"""
        ctx = self._ctx(max_conf=0.2, total_count=4)
        assert self.detector._grade_alert(ctx) == "yellow"

    def test_high_speed_bumps_to_yellow(self):
        """高速运动 + 足够轨迹 → 至少黄色"""
        ctx = self._ctx(max_conf=0.35, max_speed=12, max_age=4)
        assert self.detector._grade_alert(ctx) == "yellow"

    def test_high_speed_but_too_young_filtered(self):
        """高速但轨迹太短 (<3帧) → 不增强为黄色"""
        ctx = self._ctx(max_conf=0.35, max_speed=12, max_age=2)
        assert self.detector._grade_alert(ctx) == "blue"

    def test_long_trajectory_confidence_boost(self):
        """长轨迹(≥8帧)提升有效置信度, 0.45×1.15=0.5175 → 跨过 yellow 门槛"""
        ctx = self._ctx(max_conf=0.45, max_age=9)
        result = self.detector._grade_alert(ctx)
        # effective_conf ≈ 0.5175 > 0.5 (blue→yellow)
        assert result == "yellow"

    def test_long_trajectory_boost_to_orange(self):
        """长轨迹: 0.65×1.15=0.7475 → 跨过 orange 门槛"""
        ctx = self._ctx(max_conf=0.65, max_age=10)
        result = self.detector._grade_alert(ctx)
        assert result == "orange"

    # ---- 综合判定: 取置信度和尺寸中的较高等级 ----

    def test_conf_orange_size_yellow_gives_orange(self):
        """置信度=orange, 尺寸=yellow → 综合=orange"""
        ctx = self._ctx(max_conf=0.75, rock_diameter_cm=15)
        assert self.detector._grade_alert(ctx) == "orange"

    def test_conf_blue_size_orange_gives_orange(self):
        """置信度=blue, 尺寸=orange → 综合=orange (取较高)"""
        ctx = self._ctx(max_conf=0.4, rock_diameter_cm=25)
        assert self.detector._grade_alert(ctx) == "orange"

    def test_falling_orange_stays_orange(self):
        """已为orange, 坠落增强不会降级"""
        ctx = self._ctx(max_conf=0.8, is_falling=True)
        assert self.detector._grade_alert(ctx) == "orange"
