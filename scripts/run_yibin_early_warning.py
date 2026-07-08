"""
宜宾滑坡 — 早期前兆落石识别 v3（YOLO主导 + 运动辅助 + 趋势监测）
==================================================================
核心策略（针对远景小落石，不重新训练模型）:

  v1/v2 教训:
    - MOG2 运动能量在前兆阶段与基线无显著差异（小落石太远太小）
    - YOLO 在 0.10-0.30 置信度能捕捉到落石，但单帧误报多
    - 灰尘云检测很准（73.5s 亮度骤降73），但那是崩塌已发生

  v3 策略:
    1. YOLO 为主: 统计 ROI 内低置信度(≥0.10)检测框的时空密度
       → 单位时间/区域内检测框突然增多 = 前兆信号
    2. 运动为辅: 运动能量高于"近期滚动基线"时提升置信度
    3. 滚动基线: 每10秒更新一次基线，适应光照和环境变化
    4. 时空密度爆发检测: 检测框数量突然超过近期均值的3σ → 告警
    5. 灰尘云检测: 保留 v2 的亮度骤降检测（大规模崩塌确认）
    6. 趋势监测: 滑动窗口内检测数持续上升 → 升级告警等级

输出:
  - 告警事件时间线
  - 前兆提前量统计
  - 标注视频
  - JSON 报告
"""

import sys
import time
import json
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from rockfall.config import RESULTS_DIR
from rockfall.detector import RockDetector

# ============================================================
# 参数
# ============================================================
VIDEO_PATH = "d:/rock/3.7日，四川宜宾一高速路段发生山体滑坡.mp4"
MAIN_COLLAPSE_TIME = 73.5
WARMUP_SEC = 12                     # YOLO+MOG2 预热期
ROLLING_WINDOW_SEC = 15             # 滚动基线窗口
DENSITY_BURST_THRESHOLD = 2.5       # 检测密度 z-score 阈值
TREND_WINDOW_SEC = 8                # 趋势监测窗口
TREND_RISING_FRAMES = 3             # 连续上升帧数 → 趋势确认
DUST_BRIGHTNESS_DROP = 25
DUST_STD_SPIKE = 15
YOLO_CONFIDENCE = 0.10


def get_slope_roi(fw: int, fh: int) -> np.ndarray:
    """边坡 ROI"""
    return np.array([
        [0,              int(fh * 0.12)],
        [0,              int(fh * 0.78)],
        [int(fw * 0.65), int(fh * 0.78)],
        [int(fw * 0.65), int(fh * 0.12)],
    ], np.int32)


def main():
    print("=" * 80)
    print("  宜宾山体滑坡 — 早期前兆落石识别 v3（YOLO主导 + 时空密度）")
    print("=" * 80)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"\n📹 {fw}x{fh}, {fps:.0f}fps, {total_frames}帧, {duration:.0f}s")

    # ROI
    slope_poly = get_slope_roi(fw, fh)
    roi_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [slope_poly], 255)

    # MOG2
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=10, detectShadows=False,
    )

    # YOLO
    print(f"\n🔧 加载模型...")
    detector = RockDetector()
    detector.confidence = YOLO_CONFIDENCE
    detector.img_size = 640
    detector.init_stream_state(fw, fh, roi_mask)

    # ── 状态变量 ──
    # 每帧记录: 检测数、累积置信度、运动能量、亮度
    frame_records = []  # [{frame, time, det_count, sum_conf, motion, brightness}]

    # 滚动窗口
    rolling_window_frames = int(ROLLING_WINDOW_SEC * fps)  # ~450帧
    trend_window_frames = int(TREND_WINDOW_SEC * fps)      # ~240帧

    # 告警
    alert_events_raw = []
    last_alert_level = "green"
    last_alert_time = -99

    # 输出
    out_dir = RESULTS_DIR / "yibin_early_warning_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = str(out_dir / "yibin_annotated.mp4")
    out_writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    # ── 逐帧处理 ──
    print(f"\n{'='*80}")
    print(f"  逐帧分析（YOLO 推理 + 时空密度统计）...")
    print(f"{'='*80}")

    frame_idx = 0
    prev_gray = None
    last_progress = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        time_sec = frame_idx / fps

        pct = int(time_sec / duration * 100)
        if pct >= last_progress + 10:
            print(f"  ... {pct}% ({time_sec:.0f}s)")
            last_progress = pct

        is_warmup = time_sec < WARMUP_SEC
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        # ── MOG2 运动能量 ──
        fg = mog2.apply(frame, learningRate=0.005)
        fg_roi = cv2.bitwise_and(fg, fg, mask=roi_mask)
        motion_energy = np.count_nonzero(fg_roi) / (fw * fh)

        # ── YOLO 检测 ──
        detections = []
        max_conf = 0.0
        sum_conf = 0.0

        if not is_warmup or time_sec > (WARMUP_SEC - 3):
            pp = detector.preprocess_frame(frame)
            raw_dets = detector.detect_frame(frame, pp.get('box_mask'), pp.get('fg'))
            detections = [d for d in raw_dets if d[4] >= YOLO_CONFIDENCE]
            if detections:
                max_conf = max(d[4] for d in detections)
                sum_conf = sum(d[4] for d in detections)

        # ── 帧间差分 ──
        if prev_gray is not None:
            diff_roi = cv2.bitwise_and(cv2.absdiff(gray, prev_gray),
                                       cv2.absdiff(gray, prev_gray), mask=roi_mask)
            diff_mean = float(np.mean(diff_roi))
        else:
            diff_mean = 0.0
        prev_gray = gray.copy()

        # ── 记录 ──
        frame_records.append({
            "frame": frame_idx, "time": time_sec,
            "det_count": len(detections), "sum_conf": round(sum_conf, 4),
            "max_conf": round(max_conf, 4),
            "motion_energy": round(motion_energy, 5),
            "brightness": round(brightness, 1),
            "diff_mean": round(diff_mean, 2),
        })

        # ── 告警判定（需有足够历史数据）──
        alert_level = "green"
        alert_reason = ""

        if not is_warmup and len(frame_records) >= rolling_window_frames:
            # 获取滚动窗口内的检测数序列
            window_records = frame_records[-rolling_window_frames:]
            window_det_counts = [r["det_count"] for r in window_records]
            window_motions = [r["motion_energy"] for r in window_records]
            window_diffs = [r["diff_mean"] for r in window_records]

            det_mean = np.mean(window_det_counts)
            det_std = np.std(window_det_counts) + 0.01  # 避免除零
            motion_mean = np.mean(window_motions)
            motion_std = np.std(window_motions) + 1e-8
            diff_mean_base = np.mean(window_diffs)
            diff_std = np.std(window_diffs) + 1e-8

            # 当前帧的 z-score
            current_det_count = len(detections)
            det_z = (current_det_count - det_mean) / det_std
            motion_z = (motion_energy - motion_mean) / motion_std
            diff_z = (diff_mean - diff_mean_base) / diff_std

            # ── 亮度基线（前5秒）──
            recent_brightness = [r["brightness"] for r in frame_records[-150:]]
            bb_mean = np.mean(recent_brightness)

            # === 灰尘云检测（红色）===
            brightness_drop = bb_mean - brightness
            signal_dust = (brightness_drop > DUST_BRIGHTNESS_DROP and
                          float(np.std(gray)) > DUST_STD_SPIKE)

            if signal_dust:
                alert_level = "red"
                alert_reason = f"灰尘云! 亮度骤降{brightness_drop:.0f}, σ={np.std(gray):.0f}"

            # === YOLO 密度爆发检测 ===
            elif det_z > DENSITY_BURST_THRESHOLD:
                # 检测数突然飙升 → 可能有多块落石
                alert_level = "orange"
                alert_reason = f"检测爆发! {current_det_count}个目标 (z={det_z:.1f}, 均值{det_mean:.1f})"

            # === YOLO 持续检测 + 运动确认 ===
            elif current_det_count >= 1 and max_conf >= 0.15:
                # 有检测 + 运动略高于基线
                if motion_z > 0.5:
                    if max_conf > 0.30:
                        alert_level = "yellow"
                        alert_reason = f"落石检测: {current_det_count}个, conf={max_conf:.2f}, 运动z={motion_z:.1f}"
                    else:
                        alert_level = "blue"
                        alert_reason = f"弱检测: {current_det_count}个, conf={max_conf:.2f}, 运动z={motion_z:.1f}"
                elif diff_z > 1.0:
                    # 运动能量不高但帧差有异常（远景小落石特征）
                    if max_conf > 0.25:
                        alert_level = "yellow"
                        alert_reason = f"帧差异常+检测: {current_det_count}个, conf={max_conf:.2f}, 帧差z={diff_z:.1f}"
                    else:
                        alert_level = "blue"
                        alert_reason = f"弱帧差异常+检测: {current_det_count}个, conf={max_conf:.2f}"

            # === 趋势上升检测（检测数持续增加）===
            elif current_det_count >= 1:
                # 看最近 TREND_WINDOW_SEC 内检测数是否持续上升
                trend_records = frame_records[-trend_window_frames:]
                if len(trend_records) >= 60:  # 至少60帧
                    # 分前后两半比较
                    half = len(trend_records) // 2
                    first_half_dets = sum(1 for r in trend_records[:half] if r["det_count"] > 0)
                    second_half_dets = sum(1 for r in trend_records[half:] if r["det_count"] > 0)
                    if second_half_dets > first_half_dets * 2:  # 检测频率翻倍
                        alert_level = "blue"
                        alert_reason = f"趋势上升: 检测频率 {first_half_dets}→{second_half_dets} ({TREND_WINDOW_SEC}s内)"

            # === 一致多帧低置信度 ===
            if alert_level == "green" and current_det_count >= 1 and max_conf >= 0.15:
                # 检查过去3秒内是否有其他检测帧
                recent_3s = frame_records[-90:]  # ~3秒
                detecting_frames = sum(1 for r in recent_3s if r["det_count"] > 0)
                if detecting_frames >= 5:
                    alert_level = "blue"
                    alert_reason = f"持续微弱信号: {detecting_frames}/90帧有检测"

        # ── 记录告警 ──
        if alert_level != "green":
            level_order = {"blue": 1, "yellow": 2, "orange": 3, "red": 4}
            prev = level_order.get(last_alert_level, 0)
            cur = level_order.get(alert_level, 0)

            if cur > prev or (alert_level != last_alert_level) or \
               (time_sec - last_alert_time > 3.0):
                alert_events_raw.append({
                    "frame": frame_idx, "time_sec": round(time_sec, 1),
                    "level": alert_level, "reason": alert_reason,
                    "det_count": len(detections), "max_conf": round(max_conf, 4),
                    "motion_energy": round(motion_energy, 5),
                    "motion_z": round(motion_z if 'motion_z' in dir() else 0, 1),
                    "det_z": round(det_z if 'det_z' in dir() else 0, 1),
                    "brightness": round(brightness, 0),
                })
                last_alert_time = time_sec
            last_alert_level = alert_level
        else:
            if last_alert_level != "green":
                last_alert_level = "green"

        # ── 控制台输出（简化）──
        if alert_level != "green":
            emoji = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "blue": "🔵"}
            dz_str = f"detZ={det_z:.1f}" if 'det_z' in dir() else ""
            mz_str = f"motZ={motion_z:.1f}" if 'motion_z' in dir() else ""
            print(f"{time_sec:>6.1f}s | F{frame_idx:>6d} | {emoji.get(alert_level, '')} {alert_level:>7s} | "
                  f"D={len(detections)} conf={max_conf:.2f} | {dz_str} {mz_str} | {alert_reason[:60]}")

        # ── 绘制标注帧 ──
        annotated = frame.copy()
        cv2.polylines(annotated, [slope_poly], True, (255, 200, 0), 1)

        # 运动热力
        if motion_energy > 0.01:
            fg_color = np.zeros_like(frame)
            fg_color[fg_roi > 0] = (0, 140, 255)
            annotated = cv2.addWeighted(annotated, 1.0, fg_color, 0.2, 0)

        # YOLO 框
        for d in detections:
            x1, y1, x2, y2 = map(int, d[:4])
            conf = d[4]
            color = (0, 255, 0) if conf > 0.35 else (0, 200, 255) if conf > 0.20 else (128, 128, 128)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.putText(annotated, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # 边框
        lc = {"red": (0, 0, 255), "orange": (0, 128, 255),
              "yellow": (0, 215, 255), "blue": (255, 140, 0), "green": (0, 200, 0)}
        bw = 4 if alert_level != "green" else 1
        cv2.rectangle(annotated, (0, 0), (fw - 1, fh - 1), lc.get(alert_level, (0, 200, 0)), bw)

        # 信息面板
        px, py = fw - 330, 8
        cv2.rectangle(annotated, (px - 3, py - 3), (fw - 3, py + 155), (0, 0, 0), -1)
        cv2.rectangle(annotated, (px - 3, py - 3), (fw - 3, py + 155), (80, 80, 80), 1)
        dz_val = round((current_det_count - det_mean) / det_std, 1) \
            if ('det_mean' in dir() and det_std > 0.01 and not is_warmup) else 0
        info = [
            f"T:{time_sec:.1f}s  F:{frame_idx}",
            f"Detect: {len(detections)} (z={dz_val})",
            f"MaxConf: {max_conf:.2f}",
            f"Motion: {motion_energy:.4f}",
            f"Bright: {brightness:.0f}",
            f"RollMean: {det_mean:.1f}" if 'det_mean' in dir() and not is_warmup else "RollMean: --",
            f"Status: {'WARMUP' if is_warmup else 'RUNNING'}",
        ]
        for i, line in enumerate(info):
            cv2.putText(annotated, line, (px, py + 17 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1)

        alert_labels = {"red": "!!! RED - MAJOR LANDSLIDE !!!",
                        "orange": "!! ORANGE - ROCKFALL DETECTED !!",
                        "yellow": "! YELLOW - PRECURSOR WARNING !",
                        "blue": "BLUE - WEAK SIGNAL"}
        if alert_level != "green":
            cv2.putText(annotated, alert_labels.get(alert_level, ""),
                        (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        lc.get(alert_level, (0, 200, 0)), 2)

        out_writer.write(annotated)

    # ── 释放 ──
    cap.release()
    out_writer.release()

    # ================================================================
    # 分析报告
    # ================================================================
    print(f"\n{'='*80}")
    print(f"  分析报告")
    print(f"{'='*80}")

    # 检测帧统计
    detecting_frames = [r for r in frame_records if r["det_count"] > 0]
    all_det_times = [r["time"] for r in detecting_frames]

    print(f"\n📊 总体统计:")
    print(f"  总帧数: {len(frame_records)}")
    print(f"  有检测的帧: {len(detecting_frames)} ({len(detecting_frames)/max(len(frame_records),1)*100:.1f}%)")
    if detecting_frames:
        print(f"  首次检测: {all_det_times[0]:.1f}s")
        print(f"  检测帧平均密度: {len(detecting_frames)/(duration-WARMUP_SEC):.1f} 帧/秒")

    # 分时段统计
    segments = [
        (" 0-30s (基线期)", 0, 30),
        ("30-50s (平静期)", 30, 50),
        ("50-73s (前兆期)", 50, MAIN_COLLAPSE_TIME),
        ("73-90s (崩塌期)", MAIN_COLLAPSE_TIME, 90),
        ("90-105s (后期)", 90, 105),
    ]
    print(f"\n  分时段检测密度:")
    for label, t0, t1 in segments:
        seg = [r for r in frame_records if t0 <= r["time"] < t1]
        seg_dets = [r for r in seg if r["det_count"] > 0]
        det_frames = len(seg_dets)
        total_dets = sum(r["det_count"] for r in seg_dets)
        avg_conf = np.mean([r["max_conf"] for r in seg_dets]) if seg_dets else 0
        print(f"    {label}: {det_frames}检测帧, {total_dets}个目标, 平均conf={avg_conf:.3f}")

    # ── 事件聚类 ──
    events = []
    if alert_events_raw:
        cluster_start = alert_events_raw[0]
        cluster_end = alert_events_raw[0]
        cluster_max_level = alert_events_raw[0]["level"]
        level_order = {"blue": 1, "yellow": 2, "orange": 3, "red": 4}

        for i in range(1, len(alert_events_raw)):
            curr = alert_events_raw[i]
            gap = curr["time_sec"] - cluster_end["time_sec"]
            if gap <= 5.0:
                cluster_end = curr
                if level_order.get(curr["level"], 0) > level_order.get(cluster_max_level, 0):
                    cluster_max_level = curr["level"]
            else:
                events.append({
                    "start_time": cluster_start["time_sec"],
                    "end_time": cluster_end["time_sec"],
                    "duration_sec": round(cluster_end["time_sec"] - cluster_start["time_sec"], 1),
                    "max_level": cluster_max_level,
                    "start_reason": cluster_start["reason"],
                    "max_conf": max(
                        a.get("max_conf", 0) for a in alert_events_raw
                        if cluster_start["time_sec"] <= a["time_sec"] <= cluster_end["time_sec"]
                    ),
                })
                cluster_start = curr
                cluster_end = curr
                cluster_max_level = curr["level"]

        # 最后一个簇
        events.append({
            "start_time": cluster_start["time_sec"],
            "end_time": cluster_end["time_sec"],
            "duration_sec": round(cluster_end["time_sec"] - cluster_start["time_sec"], 1),
            "max_level": cluster_max_level,
            "start_reason": cluster_start["reason"],
            "max_conf": max(
                a.get("max_conf", 0) for a in alert_events_raw
                if a["time_sec"] >= cluster_start["time_sec"]
            ),
        })

    # ── 打印事件 ──
    print(f"\n📊 告警事件时间线:")
    if events:
        print(f"  {'开始':>7s} | {'持续':>5s} | {'最高等级':>8s} | {'最高conf':>8s} | 触发原因")
        print(f"  {'-'*7}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*55}")
        for e in events:
            emoji = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "blue": "🔵"}
            print(f"  {e['start_time']:>6.1f}s | {e['duration_sec']:>4.1f}s | "
                  f"{emoji.get(e['max_level'], '')} {e['max_level']:>7s} | "
                  f"{e['max_conf']:>8.4f} | {e['start_reason'][:60]}")
    else:
        print(f"  (无告警事件)")

    # ── 前兆分析 ──
    print(f"\n{'='*80}")
    print(f"  ⏱️  前兆预警分析（主崩塌 @ {MAIN_COLLAPSE_TIME}s）")
    print(f"{'='*80}")

    precursor_events = [e for e in events if e["start_time"] < MAIN_COLLAPSE_TIME]
    post_events = [e for e in events if e["start_time"] >= MAIN_COLLAPSE_TIME]

    if precursor_events:
        first = precursor_events[0]
        warning_sec = MAIN_COLLAPSE_TIME - first["start_time"]
        print(f"\n  ✅ 在主崩塌前成功发出预警！")
        print(f"  ├─ 首次告警: {first['start_time']:.1f}s ({first['max_level']})")
        print(f"  ├─ 原因: {first['start_reason']}")
        print(f"  ├─ 提前量: {warning_sec:.0f}秒 ({warning_sec/60:.1f}分钟)")
        print(f"  ├─ 前兆事件数: {len(precursor_events)}")
        print(f"  │")
        for i, e in enumerate(precursor_events):
            print(f"  │   [{i+1}] {e['start_time']:.1f}s-{e['end_time']:.1f}s "
                  f"({e['max_level']}, {e['duration_sec']:.1f}s) → {e['start_reason'][:50]}")
        print(f"  │")
        print(f"  └─ 💡 实际意义:")
        for spd in [120, 100, 80, 60]:
            print(f"       车速{spd}km/h → 提前{warning_sec * spd / 3.6:.0f}m 预警")
    else:
        print(f"\n  ❌ 主崩塌前未检测到前兆信号")
        # 检查是否有未触发告警的检测
        precursor_dets = [r for r in detecting_frames if r["time"] < MAIN_COLLAPSE_TIME]
        if precursor_dets:
            print(f"  ⚠️ 有 {len(precursor_dets)} 帧检测到目标但未达告警阈值:")
            # 显示分布
            for t_range, label in [((30, 50), "30-50s"), ((50, 60), "50-60s"), ((60, 73.5), "60-73.5s")]:
                seg = [r for r in precursor_dets if t_range[0] <= r["time"] < t_range[1]]
                if seg:
                    max_c = max(r["max_conf"] for r in seg)
                    total = sum(r["det_count"] for r in seg)
                    print(f"    {label}: {len(seg)}帧, {total}个目标, max_conf={max_c:.3f}")

    if post_events:
        print(f"\n  📌 主崩塌后事件: {len(post_events)}个")
        for e in post_events[:5]:
            print(f"     {e['start_time']:.1f}s: {e['max_level']} ({e['start_reason'][:55]})")

    # ── 保存 ──
    report = {
        "video": VIDEO_PATH,
        "resolution": f"{fw}x{fh}", "fps": round(fps, 2),
        "duration_sec": round(duration, 1),
        "main_collapse_time_sec": MAIN_COLLAPSE_TIME,
        "parameters": {
            "warmup_sec": WARMUP_SEC,
            "rolling_window_sec": ROLLING_WINDOW_SEC,
            "density_burst_threshold": DENSITY_BURST_THRESHOLD,
            "yolo_confidence": YOLO_CONFIDENCE,
            "dust_brightness_drop": DUST_BRIGHTNESS_DROP,
        },
        "precursor_warning_sec": round(
            MAIN_COLLAPSE_TIME - precursor_events[0]["start_time"], 1
        ) if precursor_events else 0,
        "total_events": len(events),
        "precursor_events": len(precursor_events),
        "events": events,
        "segment_stats": {},
    }
    for label, t0, t1 in segments:
        seg = [r for r in frame_records if t0 <= r["time"] < t1]
        seg_dets = [r for r in seg if r["det_count"] > 0]
        report["segment_stats"][label.strip()] = {
            "total_frames": len(seg),
            "detecting_frames": len(seg_dets),
            "total_detections": sum(r["det_count"] for r in seg_dets),
            "avg_max_conf": round(float(np.mean([r["max_conf"] for r in seg_dets])), 4) if seg_dets else 0,
        }

    report_path = out_dir / "early_warning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 报告: {report_path}")
    print(f"🎬 标注视频: {out_video}")
    print(f"\n{'='*80}")
    print(f"  完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
