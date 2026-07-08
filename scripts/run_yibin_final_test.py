"""
宜宾滑坡 — 完整端到端测试（集成 DensityMonitor + PushPlus 推送）
=================================================================
使用 RockDetector 完整流水线 (YOLO + MOG2 + SORT + DensityMonitor + 前兆升压 + 通知推送)。

与之前脚本的区别:
  - 使用系统自带 RockDetector 完整流水线 (不是裸 YOLO)
  - DensityMonitor 已集成到 detector 中 (通过 DENSITY_ALERT_ENABLED=true)
  - 告警帧推送微信 (PushPlus)
  - 四级预警分级调度 (blue→仅记录, yellow→弹窗, orange→PushPlus+邮件, red→全通道+声光)

用法:
  python scripts/run_yibin_final_test.py
  python scripts/run_yibin_final_test.py --notify   # 启用微信通知
  python scripts/run_yibin_final_test.py --stride 3  # 每3帧推理一次 (加速)
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from rockfall.config import RESULTS_DIR
from rockfall.detector import RockDetector

VIDEO_PATH = "d:/rock/3.7日，四川宜宾一高速路段发生山体滑坡.mp4"
MAIN_COLLAPSE_TIME = 73.5


def main():
    parser = argparse.ArgumentParser(description="宜宾滑坡端到端测试")
    parser.add_argument("--notify", action="store_true",
                        help="启用 PushPlus 微信推送通知")
    parser.add_argument("--stride", type=int, default=2,
                        help="帧采样步长 (默认2=每秒15帧)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最大处理帧数 (0=全部)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="起始帧 (0=从头, 900=30s处开始)")
    args = parser.parse_args()

    print("=" * 80)
    print("  宜宾山体滑坡 — 完整端到端测试")
    print("  YOLO + MOG2 + SORT + DensityMonitor + 前兆升压 + 推送")
    print("=" * 80)

    # ── 视频信息 ──
    cap = cv2.VideoCapture(VIDEO_PATH)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    print(f"\n📹 {fw}x{fh}, {fps:.0f}fps, {total_frames}帧, {duration:.0f}s")
    print(f"   主崩塌: ~{MAIN_COLLAPSE_TIME}s")
    print(f"   通知: {'✅ 启用' if args.notify else '❌ 禁用 (--notify 开启)'}")
    print(f"   步长: {args.stride} (每{args.stride}帧推理1次)")

    # ── ROI ──
    slope_poly = np.array([
        [0,              int(fh * 0.12)],
        [0,              int(fh * 0.78)],
        [int(fw * 0.65), int(fh * 0.78)],
        [int(fw * 0.65), int(fh * 0.12)],
    ], np.int32)

    # ── 初始化检测器 ──
    print(f"\n🔧 初始化 RockDetector (含 DensityMonitor)...")
    detector = RockDetector()
    detector.confidence = 0.10  # 低门槛 (密度监测会过滤误报)
    detector.img_size = 640

    # ── 运行检测 ──
    print(f"\n{'='*80}")
    print(f"  运行检测...")
    print(f"{'='*80}")

    # 收集所有帧的告警信息
    alert_log = []
    density_stats_log = []
    progress_checkpoints = set(range(10, 110, 10))

    def progress_cb(current, total):
        pct = int(current / max(total, 1) * 100) if total > 0 else 0
        if pct in progress_checkpoints:
            print(f"  ... {pct}%")
            progress_checkpoints.discard(pct)

    t0 = time.time()
    result = detector.detect_video(
        VIDEO_PATH,
        save_frames=True,
        push_alerts=args.notify,   # ← 控制是否推送微信
        track=True,
        polygon=slope_poly,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        stride=args.stride,
        start_frame=args.start_frame,
        progress_callback=progress_cb,
    )
    elapsed = time.time() - t0

    if not isinstance(result, dict) or "error" in result:
        print(f"❌ 检测失败: {result}")
        return

    # ── 解析结果 ──
    all_detections = result.get("detections", [])
    print(f"\n✅ 检测完成 ({elapsed:.0f}s)")
    print(f"   处理帧数: {result.get('total_frames', '?')}")
    print(f"   有检测的帧: {result.get('frames_with_detections', 0)}")

    # ── 告警统计 ──
    level_counts = {"red": 0, "orange": 0, "yellow": 0, "blue": 0, "green": 0}
    for fr in all_detections:
        lvl = fr.get("alert_level", "green")
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    print(f"\n📊 告警等级分布:")
    for lvl in ["red", "orange", "yellow", "blue", "green"]:
        bar = "█" * min(level_counts[lvl], 50)
        print(f"  {lvl:>8}: {level_counts[lvl]:>4} {bar}")

    # ── 告警时间线 ──
    alert_frames = [fr for fr in all_detections if fr.get("alert_level") != "green"]
    print(f"\n📊 告警帧时间线:")
    if alert_frames:
        # 聚类为事件
        events = []
        cluster_start = alert_frames[0]
        cluster_end = alert_frames[0]
        for i in range(1, len(alert_frames)):
            curr = alert_frames[i]
            gap = curr.get("time_sec", 0) - cluster_end.get("time_sec", 0)
            if gap <= 3.0:
                cluster_end = curr
            else:
                events.append((cluster_start, cluster_end))
                cluster_start = curr
                cluster_end = curr
        events.append((cluster_start, cluster_end))

        print(f"  {'开始':>7s} | {'结束':>7s} | {'持续':>5s} | {'最高等级':>8s} | 关键帧数")
        print(f"  {'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*8}-+-{'-'*10}")
        for start_fr, end_fr in events:
            # 找簇内最高等级
            cluster_frs = [f for f in alert_frames
                          if start_fr.get("time_sec", 0) <= f.get("time_sec", 0) <= end_fr.get("time_sec", 0)]
            levels = [f.get("alert_level") for f in cluster_frs]
            from rockfall.alert_classifier import LEVEL_ORDER
            max_lvl = max(levels, key=lambda l: LEVEL_ORDER.index(l)) if levels else "green"
            emoji = {"red":"🔴","orange":"🟠","yellow":"🟡","blue":"🔵"}.get(max_lvl, "")
            print(f"  {start_fr.get('time_sec', 0):>6.1f}s | {end_fr.get('time_sec', 0):>6.1f}s | "
                  f"{end_fr.get('time_sec', 0) - start_fr.get('time_sec', 0):>4.1f}s | "
                  f"{emoji} {max_lvl:>7s} | {len(cluster_frs)}帧")
    else:
        print(f"  (无告警帧)")

    # ── 密度监测统计 ──
    if DENSITY_ALERT_ENABLED and detector._density_monitor is not None:
        dm = detector._density_monitor
        stats = dm.last_stats
        print(f"\n📊 DensityMonitor 状态:")
        print(f"  窗口大小: {dm.window_frames}帧 ({dm.window_sec}s)")
        print(f"  已收集: {stats.window_size}帧")
        print(f"  z-score阈值: {dm.burst_zscore}")
        print(f"  置信度下限: {dm.conf_floor}")

    # ── 前兆分析 ──
    print(f"\n{'='*80}")
    print(f"  ⏱️  前兆预警分析（主崩塌 @ {MAIN_COLLAPSE_TIME}s）")
    print(f"{'='*80}")

    precursor_alerts = [f for f in alert_frames
                       if f.get("time_sec", 0) < MAIN_COLLAPSE_TIME]
    post_alerts = [f for f in alert_frames
                   if f.get("time_sec", 0) >= MAIN_COLLAPSE_TIME]

    if precursor_alerts:
        first = precursor_alerts[0]
        warning_sec = MAIN_COLLAPSE_TIME - first.get("time_sec", 0)
        print(f"\n  ✅ 在主崩塌前成功发出预警！")
        print(f"  ├─ 首次告警: {first.get('time_sec', 0):.1f}s (F{first.get('frame', 0)})")
        print(f"  ├─ 等级: {first.get('alert_level', '?')}")
        print(f"  ├─ 提前量: {warning_sec:.0f}秒 ({warning_sec/60:.1f}分钟)")
        print(f"  ├─ 前兆告警帧: {len(precursor_alerts)}帧")
        print(f"  │")
        print(f"  └─ 💡 实际意义:")
        for spd in [120, 100, 80, 60]:
            dist = spd / 3.6 * warning_sec
            print(f"       车速{spd}km/h → 提前 {dist:.0f}m 预警")
    else:
        print(f"\n  ❌ 主崩塌前未触发告警")

    if post_alerts:
        print(f"\n  📌 主崩塌后: {len(post_alerts)}帧告警")

    # ── 推送状态 ──
    if args.notify:
        print(f"\n📱 推送通知: 已通过 PushPlus 发送到微信")
        print(f"   (橙色及以上等级触发推送，黄色触发弹窗，蓝色仅记录)")

    # ── 保存报告 ──
    out_dir = RESULTS_DIR / "yibin_final_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "video": VIDEO_PATH,
        "config": {
            "stride": args.stride,
            "notify": args.notify,
            "density_alert_enabled": DENSITY_ALERT_ENABLED,
            "density_window_sec": DENSITY_WINDOW_SEC,
            "density_burst_zscore": DENSITY_BURST_ZSCORE,
            "density_conf_floor": DENSITY_CONF_FLOOR,
        },
        "duration_sec": round(elapsed, 1),
        "total_frames_processed": result.get("total_frames", 0),
        "frames_with_detections": result.get("frames_with_detections", 0),
        "alert_level_distribution": level_counts,
        "precursor_warning_sec": round(
            MAIN_COLLAPSE_TIME - precursor_alerts[0].get("time_sec", MAIN_COLLAPSE_TIME), 1
        ) if precursor_alerts else 0,
        "precursor_alert_frames": len(precursor_alerts),
        "post_collapse_alert_frames": len(post_alerts),
        "alert_timeline": [
            {
                "frame": int(f.get("frame", 0)),
                "time_sec": float(f.get("time_sec", 0)),
                "alert_level": str(f.get("alert_level", "")),
                "box_count": len(f.get("boxes", [])),
            }
            for f in alert_frames[:500]
        ],
    }

    report_path = out_dir / "final_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 报告: {report_path}")
    print(f"\n{'='*80}")
    print(f"  完成！")
    print(f"{'='*80}")


if __name__ == "__main__":
    # 延迟导入避免 main() 未定义时引用
    from rockfall.config import (
        DENSITY_ALERT_ENABLED, DENSITY_WINDOW_SEC,
        DENSITY_BURST_ZSCORE, DENSITY_CONF_FLOOR,
    )
    main()
