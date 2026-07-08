"""
单元测试 — slope_confidence 模块
==================================
测试 FastSAM 边坡置信度调整的各项功能:
  - 深在边坡: 置信度不变
  - 多在边坡: 轻度压制
  - 边坡边缘: 中度压制
  - 弹跳 grace: 靠近边坡的极低重叠 → 温和压制
  - 远离边坡: 极低重叠但远离边坡 → 重度压制
  - 完全离开: 0% 重叠 → 最大压制
  - 分辨率不匹配: 自动 resize
  - 空检测列表: 透传
  - bbox 越界: 安全裁剪
  - class_id 保留: 第6个元素不丢失
  - 调试日志: 格式正确
"""

import numpy as np
import pytest

# 将 rockfall 包路径加入 sys.path (测试运行目录可能不同)
import sys
from pathlib import Path
_ROCKFALL_DIR = Path(__file__).resolve().parent.parent / "rockfall"
if str(_ROCKFALL_DIR) not in sys.path:
    sys.path.insert(0, str(_ROCKFALL_DIR.parent))

from rockfall.slope_confidence import adjust_confidence_by_slope


def _make_slope_mask(fw: int, fh: int, slope_rects: list) -> np.ndarray:
    """创建合成 slope_mask: 指定矩形区域为边坡 (255), 其余为 0"""
    mask = np.zeros((fh, fw), dtype=np.uint8)
    for x1, y1, x2, y2 in slope_rects:
        mask[y1:y2, x1:x2] = 255
    return mask


class TestAdjustConfidenceBySlope:
    """adjust_confidence_by_slope 核心功能测试"""

    FW, FH = 200, 200

    # ── 辅助: 创建全覆盖 slope_mask (整个画面都是边坡) ──
    @pytest.fixture
    def full_slope(self):
        return np.full((self.FH, self.FW), 255, dtype=np.uint8)

    # ── 辅助: 创建部分 slope_mask ──
    @pytest.fixture
    def half_slope(self):
        """左半边是边坡, 右半边不是"""
        mask = np.zeros((self.FH, self.FW), dtype=np.uint8)
        mask[:, :self.FW // 2] = 255
        return mask

    @pytest.fixture
    def corner_slope(self):
        """仅左上角 40x40 是边坡"""
        return _make_slope_mask(self.FW, self.FH, [(0, 0, 40, 40)])

    # ══════════════════════════════════════════════════════════
    # 基础功能测试
    # ══════════════════════════════════════════════════════════

    def test_empty_detections(self, full_slope):
        """空检测列表 → 空输出"""
        result = adjust_confidence_by_slope([], full_slope, self.FW, self.FH)
        assert result == []

    def test_deep_slope_no_change(self, full_slope):
        """检测框完全在边坡内 → 置信度不变 (乘数 1.00)"""
        dets = [[50, 50, 100, 100, 0.80]]
        result = adjust_confidence_by_slope(dets, full_slope, self.FW, self.FH)
        assert result[0][4] == 0.80

    def test_mostly_slope_partial(self, half_slope):
        """检测框跨边坡/非边坡边界, 重叠 33% → 轻度压制 (0.90)"""
        # half_slope: 左半 x=0~99 是边坡, 右半 x=100~199 不是
        # bbox x=85~115 (width=30): slope pixels = 15 (85..99), ratio = 15/30 = 0.50
        # 50% hits deep_slope (≥0.50).  Use 90-120: slope=10, ratio=10/30=33% → mostly
        dets = [[90, 50, 120, 100, 0.80]]
        result = adjust_confidence_by_slope(dets, half_slope, self.FW, self.FH)
        expected = round(0.80 * 0.90, 4)
        assert result[0][4] == expected

    def test_edge_overlap(self, half_slope):
        """检测框小部分在边坡 (15% 重叠) → 中度压制 (0.65)"""
        # bbox 从 x=110 到 x=150 (width=40), 只有 110-100=-10?
        # 边坡在 x=0 到 x=100, bbox 从 x=85 到 x=105 (width=20), 15px 在边坡 = 75%
        # 调整: 边坡 x=0-100, bbox x=90-110 (width=20), 10px 在边坡 = 50%
        # 需要更精确: bbox x=93-107 (width=14), 7px 在边坡 = 50%
        # 再来: bbox x=96-106 (width=10), 4px = 40% → 还是 mostly
        # 边坡 x=0-100, bbox x=95-110 (width=15), 5px = 33% → still mostly (>25%)
        # 边坡 x=0-100, bbox x=98-110 (width=12), 2px = 16.7% → edge (>10%)
        dets = [[98, 50, 110, 80, 0.80]]  # 2/12 ≈ 16.7% overlap → edge zone
        result = adjust_confidence_by_slope(dets, half_slope, self.FW, self.FH)
        expected = round(0.80 * 0.65, 4)
        assert result[0][4] == expected

    def test_off_slope_zero_overlap(self, corner_slope):
        """检测框完全不在边坡 (0% 重叠) → 最大压制 (0.25)"""
        # corner slope at (0,0)-(40,40), detection at (150,150)-(180,180) → far away
        dets = [[150, 150, 180, 180, 0.80]]
        result = adjust_confidence_by_slope(dets, corner_slope, self.FW, self.FH)
        expected = round(0.80 * 0.25, 4)
        assert result[0][4] == expected

    # ══════════════════════════════════════════════════════════
    # 弹跳 Grace 区测试
    # ══════════════════════════════════════════════════════════

    def test_bounce_grace_near_slope(self, corner_slope):
        """检测框极低重叠 + 靠近边坡边缘 → bounce grace (0.55)"""
        # corner slope at (0,0)-(40,40)  [rows 0-39, cols 0-39]
        # bbox at (38,38)-(45,45): 2x2=4px overlap / 7x7=49px area = 8.2% < 10%
        # bbox 紧贴边坡 → near slope → bounce_grace
        dets = [[38, 38, 45, 45, 0.80]]
        result = adjust_confidence_by_slope(
            dets, corner_slope, self.FW, self.FH, debug_log=True,
        )
        expected = round(0.80 * 0.55, 4)
        assert result[0][4] == expected

    def test_off_slope_far_from_corner(self, corner_slope):
        """检测框完全不在边坡 (0% overlap) 且远离 slope → off_slope (0.25)"""
        # corner slope at (0,0)-(40,40), bbox at (100,100)-(110,110) → 0 overlap
        dets = [[100, 100, 110, 110, 0.80]]
        result = adjust_confidence_by_slope(dets, corner_slope, self.FW, self.FH)
        expected = round(0.80 * 0.25, 4)
        assert result[0][4] == expected

    # ══════════════════════════════════════════════════════════
    # 边界情况
    # ══════════════════════════════════════════════════════════

    def test_resolution_mismatch_auto_resize(self, full_slope):
        """slope_mask 分辨率与 fw/fh 不匹配 → 自动 resize"""
        # full_slope is 200x200, pass fw=100,fh=100 → should auto-resize
        dets = [[10, 10, 50, 50, 0.80]]
        result = adjust_confidence_by_slope(dets, full_slope, 100, 100)
        # After resize to 100x100, full slope → deep_slope → conf unchanged
        assert result[0][4] == 0.80

    def test_bbox_out_of_bounds_clamped(self, corner_slope):
        """bbox 超出画面边界 → 自动裁剪到有效范围"""
        # bbox 部分越界, 但在边坡角内
        dets = [[-10, -10, 20, 20, 0.70]]
        result = adjust_confidence_by_slope(dets, corner_slope, self.FW, self.FH)
        # 裁剪后 bbox (0,0)-(20,20), 在 corner slope (0,0)-(40,40) 内 = 100% → deep
        assert result[0][4] == 0.70

    def test_zero_area_bbox_passthrough(self, full_slope):
        """零面积 bbox (x1==x2) → 透传不变"""
        dets = [[50, 50, 50, 100, 0.60]]
        result = adjust_confidence_by_slope(dets, full_slope, self.FW, self.FH)
        assert result[0][4] == 0.60

    # ══════════════════════════════════════════════════════════
    # class_id 保留
    # ══════════════════════════════════════════════════════════

    def test_preserves_class_id(self, half_slope):
        """class_id (第6个元素) 在变换后保留"""
        dets = [[60, 50, 140, 100, 0.80, 1]]  # class=1 (滑坡)
        result = adjust_confidence_by_slope(dets, half_slope, self.FW, self.FH)
        assert len(result[0]) == 6
        assert result[0][5] == 1

    def test_preserves_class_id_no_class(self, full_slope):
        """无 class_id 时 (5元素) 也不崩溃"""
        dets = [[50, 50, 100, 100, 0.80]]
        result = adjust_confidence_by_slope(dets, full_slope, self.FW, self.FH)
        assert len(result[0]) == 5
        assert result[0][4] == 0.80

    # ══════════════════════════════════════════════════════════
    # 多检测框 + 置信度上限
    # ══════════════════════════════════════════════════════════

    def test_multiple_detections_mixed(self, half_slope):
        """混合场景: 多个检测框, 各自独立调整"""
        dets = [
            [10, 50, 50, 100, 0.80],    # 全在边坡 (左半) → deep: 1.00
            [160, 50, 190, 100, 0.70],  # 全不在边坡 (右半, 0% overlap) → off: 0.25
            [90, 50, 120, 100, 0.60],   # 跨边界 33% → mostly: 0.90
        ]
        result = adjust_confidence_by_slope(dets, half_slope, self.FW, self.FH)
        assert result[0][4] == 0.80                         # deep: unchanged
        assert result[1][4] == round(0.70 * 0.25, 4)       # off: 0.25
        assert result[2][4] == round(0.60 * 0.90, 4)       # mostly: 0.90

    def test_confidence_capped_at_one(self, full_slope):
        """深在边坡的加法提升 (即使乘数=1.0) 不超 1.0"""
        # 这里乘数=1.0, 不改变。但确保不因浮点误差 >1.0
        dets = [[50, 50, 100, 100, 0.9999]]
        result = adjust_confidence_by_slope(dets, full_slope, self.FW, self.FH)
        assert result[0][4] <= 1.0

    # ══════════════════════════════════════════════════════════
    # 可配置参数测试
    # ══════════════════════════════════════════════════════════

    def test_custom_thresholds(self, half_slope):
        """自定义重叠阈值 — 改变区间边界"""
        # half_slope: 左半=边坡. bbox x=92~112, width=20, slope=8px, overlap=40%
        # 默认 overlap_high=0.50, overlap_mid=0.25 → 0.40 >= 0.25 → mostly (0.90)
        dets = [[92, 50, 112, 100, 0.80]]
        result_default = adjust_confidence_by_slope(
            dets, half_slope, self.FW, self.FH,
        )
        # 提高 overlap_mid=0.45 → 0.40 < 0.45 → edge (0.65)
        result_custom = adjust_confidence_by_slope(
            dets, half_slope, self.FW, self.FH,
            overlap_mid=0.45,
        )
        assert result_default[0][4] == round(0.80 * 0.90, 4)
        assert result_custom[0][4] == round(0.80 * 0.65, 4)

    def test_custom_multipliers(self, corner_slope):
        """自定义乘数 — 完全离开边坡时用自定义值"""
        dets = [[150, 150, 180, 180, 0.80]]
        result = adjust_confidence_by_slope(
            dets, corner_slope, self.FW, self.FH,
            mult_off_slope=0.50,
        )
        assert result[0][4] == round(0.80 * 0.50, 4)

    # ══════════════════════════════════════════════════════════
    # return_suppressed 测试
    # ══════════════════════════════════════════════════════════

    def test_return_suppressed_false_default(self, corner_slope):
        """默认 return_suppressed=False → 返回 list"""
        dets = [[150, 150, 180, 180, 0.80]]
        result = adjust_confidence_by_slope(dets, corner_slope, self.FW, self.FH)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_return_suppressed_true(self, corner_slope):
        """return_suppressed=True → 返回 (adjusted_list, suppressed_list)"""
        dets = [
            [10, 10, 30, 30, 0.80],     # 在边坡角内 → deep, 不压制
            [150, 150, 180, 180, 0.70],  # 远离 → off_slope, 压制
        ]
        adjusted, suppressed = adjust_confidence_by_slope(
            dets, corner_slope, self.FW, self.FH, return_suppressed=True,
        )
        assert isinstance(adjusted, list)
        assert isinstance(suppressed, list)
        assert len(adjusted) == 2
        # 第二个检测被压制
        assert len(suppressed) == 1
        s = suppressed[0]
        assert s['old_conf'] == 0.70
        assert s['new_conf'] == round(0.70 * 0.25, 4)
        assert s['zone'] == 'off_slope'
        assert 'bbox' in s
        assert 'multiplier' in s

    def test_return_suppressed_empty(self, full_slope):
        """return_suppressed=True 空检测 → 返回 ([], [])"""
        adjusted, suppressed = adjust_confidence_by_slope(
            [], full_slope, self.FW, self.FH, return_suppressed=True,
        )
        assert adjusted == []
        assert suppressed == []

    def test_return_suppressed_no_effect(self, full_slope):
        """return_suppressed=True 但没有检测被压制 → 空 suppressed"""
        dets = [[50, 50, 100, 100, 0.80]]  # 全在边坡 → deep, 不压制
        adjusted, suppressed = adjust_confidence_by_slope(
            dets, full_slope, self.FW, self.FH, return_suppressed=True,
        )
        assert len(adjusted) == 1
        assert len(suppressed) == 0  # 没有被压制的
