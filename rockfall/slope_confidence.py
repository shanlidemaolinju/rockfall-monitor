"""
FastSAM 边坡置信度调整模块
============================
将 FastSAM 分割的 slope_mask 反馈到 YOLO 检测置信度中, 作为空间先验约束。

核心逻辑:
  1. 计算每个检测框与 slope_mask 的面积重叠率 (NumPy 向量化, 无 Python 逐像素循环)
  2. 按重叠率分 5 个区间, 施加不同的置信度乘数
  3. "弹跳 grace 区": 重叠 0-10% → 温和压制 (0.55)
     防止真石头从边坡弹跳落地瞬间被误杀 (漏报风险)
  4. 仅调整置信度, 不改变 bbox 或 class

重叠率分区 (5 档):
  ≥ 50%   → 深在边坡   (1.00)  不压制
  25-50%  → 多在边坡   (0.90)  轻度压制
  10-25%  → 边坡边缘   (0.65)  中度压制
  0-10%   → 弹跳 grace (0.55)  石头弹跳不误杀 (关键!)
  0%      → 完全不在   (0.25)  最大压制

原理: 落石只可能来自边坡区域, 非边坡区域 (天空/道路中间/护栏外) 的检测应被压制。
      不需要重新训练模型 — 这是纯推理时后处理。

依赖: NumPy + cv2 (仅用于 dilate 操作)
"""

import cv2
import numpy as np

from .logger import log_event


def adjust_confidence_by_slope(
    detections: list,
    slope_mask: np.ndarray,
    fw: int,
    fh: int,
    overlap_high: float = 0.50,
    overlap_mid: float = 0.25,
    overlap_low: float = 0.10,
    mult_deep_slope: float = 1.00,
    mult_mostly_slope: float = 0.90,
    mult_edge: float = 0.65,
    mult_bounce_grace: float = 0.55,
    mult_off_slope: float = 0.25,
    debug_log: bool = False,
    return_suppressed: bool = False,
):
    """
    基于 FastSAM 边坡掩码调整 YOLO 检测置信度。

    参数:
        detections:         [[x1, y1, x2, y2, conf, (cls)], ...]
        slope_mask:         (H, W) uint8, 255=边坡区域
        fw, fh:             帧宽高
        overlap_high:       高重叠阈值 (默认 0.50)
        overlap_mid:        中重叠阈值 (默认 0.25)
        overlap_low:        低重叠阈值 / 弹跳 grace 上限 (默认 0.10)
        mult_deep_slope:    深在边坡乘数 (默认 1.00)
        mult_mostly_slope:  多在边坡乘数 (默认 0.90)
        mult_edge:          边坡边缘乘数 (默认 0.65)
        mult_bounce_grace:  弹跳 grace 乘数 (默认 0.55, 关键!)
        mult_off_slope:     完全不在边坡乘数 (默认 0.25)
        debug_log:          是否输出调试日志
        return_suppressed:  是否返回被压制检测信息 (供可视化用)

    返回:
        若 return_suppressed=False: list — 调整后的检测列表
        若 return_suppressed=True:  (list, list) — (调整后检测, 被压制检测信息)
           suppressed[i] = {'bbox': [x1,y1,x2,y2], 'old_conf': float,
                            'new_conf': float, 'zone': str, 'multiplier': float}
    """
    if not detections:
        return ([], []) if return_suppressed else []

    # ── 分辨率对齐守卫 ──
    if slope_mask.shape[:2] != (fh, fw):
        log_event("system", level="WARN",
                  msg=f"[SlopeConf] 掩码尺寸不匹配 slope={slope_mask.shape[:2]} "
                      f"frame=({fh},{fw}), 自动 resize")
        slope_mask = cv2.resize(slope_mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

    result = []
    suppressed_logs: list[str] = []
    suppressed_info: list[dict] = []

    for d in detections:
        x1 = max(0, int(d[0]))
        y1 = max(0, int(d[1]))
        x2 = min(fw, int(d[2]))
        y2 = min(fh, int(d[3]))
        conf = float(d[4])

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h

        if bbox_area <= 0:
            result.append(d)
            continue

        # ── NumPy 向量化: 单次切片 + count_nonzero ──
        roi = slope_mask[y1:y2, x1:x2]
        slope_pixels = np.count_nonzero(roi)
        overlap_ratio = slope_pixels / bbox_area

        # ── 按重叠率分 5 档 ──
        if overlap_ratio >= overlap_high:
            multiplier = mult_deep_slope
            zone = "deep_slope"
        elif overlap_ratio >= overlap_mid:
            multiplier = mult_mostly_slope
            zone = "mostly_slope"
        elif overlap_ratio >= overlap_low:
            multiplier = mult_edge
            zone = "edge"
        elif overlap_ratio > 0:
            # 0 < overlap < overlap_low: 弹跳 grace
            # 任何与 slope 有重叠的 bbox 都"靠近"边坡 (重叠本身就是接触)
            multiplier = mult_bounce_grace
            zone = "bounce_grace"
        else:
            # overlap_ratio == 0: 完全不在边坡
            multiplier = mult_off_slope
            zone = "off_slope"

        new_conf = round(conf * multiplier, 4)
        new_d = [d[0], d[1], d[2], d[3], new_conf]
        if len(d) > 5:
            new_d.append(d[5])
        result.append(new_d)

        # ── 记录被压制的检测 ──
        if multiplier < 0.95:
            suppressed_logs.append(
                f"conf {conf:.2f}→{new_conf:.2f}"
                f"(overlap={overlap_ratio:.0%} zone={zone})"
            )
            if return_suppressed:
                suppressed_info.append({
                    'bbox': [int(d[0]), int(d[1]), int(d[2]), int(d[3])],
                    'old_conf': conf,
                    'new_conf': new_conf,
                    'zone': zone,
                    'multiplier': multiplier,
                })
            # D: 审计日志 — 大幅压制 (multiplier < 0.4) 单独记录事件
            if multiplier < 0.4:
                log_event("slope_suppressed", level="INFO",
                          old_conf=round(conf, 4), new_conf=round(new_conf, 4),
                          zone=zone, multiplier=round(multiplier, 3),
                          overlap_ratio=round(overlap_ratio, 4),
                          bbox=[int(d[0]), int(d[1]), int(d[2]), int(d[3])])

    # ── 汇总日志 ──
    if debug_log and suppressed_logs:
        log_event("system", level="DEBUG",
                  msg=f"[SlopeConf] Suppressed {len(suppressed_logs)}/{len(detections)} "
                      f"detections: " + ", ".join(suppressed_logs))

    if return_suppressed:
        return result, suppressed_info
    return result
