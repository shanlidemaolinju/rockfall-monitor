"""
检测密度爆发 — 早期前兆预警核心模块
=====================================
核心思想 (宜宾滑坡视频验证):
  远景小落石 YOLO 置信度低 (0.10-0.30)，单帧不足以触发传统告警。
  但大地质灾害前，ROI 内单位时间的低置信度检测框数量会急剧上升
  → 这就是"前兆信号"：检测密度爆发 (Density Burst)。

工作原理:
  1. 维护过去 N 秒内每帧的检测框计数的滚动窗口
  2. 实时计算当前帧检测数的 z-score = (当前数 - 滚动均值) / 滚动标准差
  3. z-score > 阈值 → 触发密度告警
  4. 多目标 (>3) 时直接升级为 orange

与现有系统的关系:
  - 独立于四级决策树 (alert_classifier)
  - 作为升级信号: 密度告警不低于 yellow，叠加 YOLO 高置信度可升至 orange/red
  - 配合前兆升压 (precursor_escalation) 使用效果更佳

用法:
    from rockfall.density_alert import DensityMonitor

    monitor = DensityMonitor(window_sec=15, burst_zscore=2.5, conf_floor=0.10)

    for frame_results in video_stream:
        det_count = sum(1 for d in frame_results if d[4] >= conf_floor)
        alert_level = monitor.update(det_count)
        if alert_level:
            send_alert(alert_level, reason=f"密度爆发: z={monitor.last_zscore:.1f}")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DensityStats:
    """密度统计快照"""
    current_count: int = 0
    window_mean: float = 0.0
    window_std: float = 0.0
    zscore: float = 0.0
    window_size: int = 0
    is_burst: bool = False


class DensityMonitor:
    """
    检测密度爆发监测器 — 基于滚动窗口 z-score 的异常检测。

    参数:
        window_sec:     滚动窗口大小 (秒), 根据 fps 换算为帧数
        burst_zscore:   爆发阈值, z-score > 此值触发告警
        conf_floor:     纳入统计的最低 YOLO 置信度
        min_samples:    最少样本数, 窗口不足时不判定
        fps:            视频帧率 (用于秒→帧转换), 可在 update() 时动态更新
    """

    def __init__(
        self,
        window_sec: float = 15.0,
        burst_zscore: float = 2.5,
        conf_floor: float = 0.10,
        min_samples: int = 90,
        fps: float = 30.0,
    ):
        self.window_sec = window_sec
        self.burst_zscore = burst_zscore
        self.conf_floor = conf_floor
        self.min_samples = min_samples
        self.fps = fps

        # 自适应窗口大小 (根据实际 fps 调整)
        self._window_frames = max(int(window_sec * fps), min_samples)
        self._history: deque[int] = deque(maxlen=self._window_frames)

        # 上次计算结果 (供外部读取)
        self.last_stats = DensityStats()
        self.last_alert_level: str | None = None

        # 告警冷却: 同等级间隔至少 N 秒才重复触发
        self._last_alert_time: float = -999.0
        self._alert_cooldown_sec: float = 3.0

    # ---- 公共 API ----

    def update(self, det_count: int, timestamp: float = 0.0) -> str | None:
        """
        输入当前帧的检测框数量, 返回告警等级或 None。

        参数:
            det_count: 当前帧的检测框数量 (已按 conf_floor 过滤后的)
            timestamp: 当前帧的时间戳 (秒), 用于冷却

        返回:
            "orange" - 密度强爆发 (≥3个目标)
            "yellow" - 密度弱爆发 (1-2个目标)
            "blue"   - 密度略高但未达阈值 (≥1.5σ)
            None     - 正常
        """
        if det_count < 0:
            det_count = 0

        self._history.append(det_count)

        # 样本不足 → 不判定
        if len(self._history) < self.min_samples:
            self.last_stats = DensityStats(
                current_count=det_count,
                window_size=len(self._history),
            )
            self.last_alert_level = None
            return None

        arr = np.array(self._history, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr)) + 0.01  # 避免除零
        z = (det_count - mean) / std

        self.last_stats = DensityStats(
            current_count=det_count,
            window_mean=round(mean, 3),
            window_std=round(std, 3),
            zscore=round(z, 2),
            window_size=len(self._history),
            is_burst=(z >= self.burst_zscore),
        )

        # ── 判定 ──
        alert_level = None

        if z >= self.burst_zscore:
            if det_count >= 3:
                alert_level = "orange"
            elif det_count >= 1:
                alert_level = "yellow"
            else:
                alert_level = "blue"
        elif z >= self.burst_zscore * 0.7:
            # 亚阈值: 接近爆发但未完全达到 → 蓝色 (关注级)
            if det_count >= 1:
                alert_level = "blue"

        # ── 冷却 ──
        if alert_level is not None:
            if timestamp - self._last_alert_time < self._alert_cooldown_sec:
                if alert_level == self.last_alert_level:
                    alert_level = None
            if alert_level is not None:
                self._last_alert_time = timestamp

        self.last_alert_level = alert_level
        return alert_level

    def update_from_detections(
        self, detections: list, timestamp: float = 0.0,
    ) -> str | None:
        """
        便捷方法: 从 YOLO 检测结果列表直接更新。

        参数:
            detections: [[x1, y1, x2, y2, conf, cls], ...]  或 []
            timestamp:  当前帧时间戳

        返回: 告警等级或 None
        """
        count = sum(1 for d in detections if len(d) >= 5 and d[4] >= self.conf_floor)
        return self.update(count, timestamp)

    def reset(self):
        """清空历史 (用于视频源切换或重连后)"""
        self._history.clear()
        self.last_stats = DensityStats()
        self.last_alert_level = None
        self._last_alert_time = -999.0

    def set_fps(self, fps: float):
        """动态调整窗口大小 (视频 fps 变化时调用)"""
        self.fps = fps
        self._window_frames = max(int(self.window_sec * fps), self.min_samples)
        # 重建 deque 以适配新窗口大小
        old = list(self._history)
        self._history = deque(old, maxlen=self._window_frames)

    @property
    def is_ready(self) -> bool:
        """是否已收集足够样本"""
        return len(self._history) >= self.min_samples

    @property
    def window_frames(self) -> int:
        """当前窗口帧数"""
        return self._window_frames
