"""
演示数据生成脚本 (本地 GPU 运行)
===============================
在本地有 GPU 的环境中运行，对视频进行完整检测，
生成: 标注帧 JPEG + 检测结果 JSON + 统计摘要

用法: python scripts/generate_demo.py <视频路径> --name <场景名称> [--max-frames 300]

输出: demo_data/<name>/
  ├── result.json        # 完整检测结果 + 统计数据
  ├── frames/            # 关键预警帧 (最多30张, 480p JPEG)
  └── summary.json       # 前端展示用的摘要卡片
"""

import sys
import json
import time
import argparse
from pathlib import Path

# ── 确保 rockfall 可导入 ──
_THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_THIS_DIR))

import cv2
import numpy as np
from rockfall.detector import RockDetector


def resize_frame(frame: np.ndarray, max_width: int = 480) -> np.ndarray:
    """等比缩放到 max_width 宽 (保持宽高比)"""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    ratio = max_width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description="生成落石检测演示数据")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("--name", required=True, help="场景名称 (如 nanning_naan)")
    parser.add_argument("--max-frames", type=int, default=300, help="最大推理帧数")
    parser.add_argument("--stride", type=int, default=2, help="帧采样步长")
    parser.add_argument("--img-size", type=int, default=640, help="推理分辨率")
    parser.add_argument("--conf", type=float, default=None, help="检测置信度阈值 (默认使用配置文件值 0.30)")
    parser.add_argument("--out", default=None, help="输出目录 (默认 demo_data/<name>/")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 视频不存在: {video_path}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else _THIS_DIR / "demo_data" / args.name
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 场景: {args.name}")
    print(f"📹 视频: {video_path}")
    print(f"📂 输出: {out_dir}")
    print(f"⚙️  参数: max_frames={args.max_frames}, stride={args.stride}, img_size={args.img_size}")
    print()

    # ── 初始化检测器 ──
    print("🔧 加载 YOLO 模型...")
    detector = RockDetector()
    detector.img_size = args.img_size
    if args.conf is not None:
        detector.confidence = args.conf
        print(f"🎯 置信度阈值: {args.conf}")
    print(f"🖥️  推理设备: {detector._device_name}")
    print()

    # ── ROI 自动检测 (FastSAM → 轮廓提取, 与桌面端一致) ──
    print("🔍 自动检测边坡 ROI...")
    cap_roi = cv2.VideoCapture(str(video_path))
    fw = int(cap_roi.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_roi.get(cv2.CAP_PROP_FRAME_HEIGHT))
    polygon = None
    slope_mask = None
    try:
        from rockfall.fastsam_road import auto_segment_from_cap
        road_mask, roi_mask = auto_segment_from_cap(cap_roi, fw, fh)
        if roi_mask is not None and roi_mask.any():
            # 质量守卫
            road_pct = (road_mask > 0).sum() / (fw * fh) * 100 if road_mask is not None else 0
            slope_pct = (roi_mask > 0).sum() / (fw * fh) * 100
            if 5 <= slope_pct <= 80 and road_pct < 95:
                # 从 mask 提取轮廓 → 多边形 (与 desktop/ui/video_widget.py 一致)
                contours, _ = cv2.findContours(
                    roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                    if cv2.contourArea(cnt) < fw * fh * 0.03:
                        break
                    eps = 0.003 * cv2.arcLength(cnt, True)
                    poly = cv2.approxPolyDP(cnt, eps, True).squeeze(1)
                    if poly.ndim == 1:
                        poly = poly.reshape(-1, 2)
                    if not np.array_equal(poly[0], poly[-1]):
                        poly = np.vstack([poly, poly[0:1]])
                    polygon = poly.astype(np.int32)
                    break
                # 保存 slope_mask 供检测器做置信度调整
                slope_mask = roi_mask
                print(f"   [FastSAM] 边坡{slope_pct:.0f}% 道路{road_pct:.0f}% → {len(polygon) if polygon is not None else 0}顶点多边形")
            else:
                print(f"   [FastSAM] 质量异常(边坡{slope_pct:.0f}% 道路{road_pct:.0f}%), 使用默认ROI")
        else:
            print("   [FastSAM] 返回空mask, 使用默认ROI")
    except Exception as e:
        print(f"   [FastSAM] 异常: {e}, 使用默认ROI")
    cap_roi.release()

    # 兜底: 默认多边形
    if polygon is None:
        polygon = RockDetector._default_polygon(fw, fh)
        print(f"   使用默认ROI: {len(polygon)}顶点")

    # 注入 slope_mask 到检测器 (供 detect_frame 边坡置信度调整)
    detector._slope_mask = slope_mask

    # ── 运行检测 ──
    print("🔍 开始检测...")
    t0 = time.time()

    result = detector.detect_video(
        str(video_path),
        save_frames=True,
        push_alerts=True,   # 必须 True 才会把标注帧落盘到 data/results/
        track=True,
        polygon=polygon,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    elapsed = time.time() - t0

    if isinstance(result, dict) and "error" not in result:
        detections = result.get("detections", [])
        total_frames = result.get("total_frames", len(detections))
        print(f"✅ 检测完成 — {elapsed:.1f}s, {len(detections)} 帧结果")

        # ── 提取预警帧 ──
        alert_frames = [
            fr for fr in detections
            if fr.get("alert_level", "green") != "green"
        ]
        print(f"🚨 预警帧: {len(alert_frames)} / {len(detections)}")

        # ── 按等级统计 ──
        level_counts = {"red": 0, "orange": 0, "yellow": 0, "blue": 0, "green": 0}
        for fr in detections:
            lvl = fr.get("alert_level", "green")
            if lvl in level_counts:
                level_counts[lvl] += 1

        # ── 选择关键帧: 每个等级最多取 8 帧 (最高置信度优先) ──
        key_frames = []
        for lvl in ["red", "orange", "yellow", "blue"]:
            lvl_frames = sorted(
                [fr for fr in alert_frames if fr.get("alert_level") == lvl],
                key=lambda fr: max(
                    (b.get("confidence", 0) for b in fr.get("boxes", [])), default=0
                ),
                reverse=True,
            )
            key_frames.extend(lvl_frames[:8])

        # 如果不足 20 帧, 补正常帧
        if len(key_frames) < 20:
            normal = [fr for fr in detections if fr.get("alert_level") == "green"]
            key_frames.extend(normal[:20 - len(key_frames)])

        print(f"🖼️  导出 {len(key_frames)} 张关键帧...")

        # ── 保存标注帧 + 缩略图 ──
        # 重新打开视频以提取原始帧 (当标注帧不存在时回退)
        cap_raw = cv2.VideoCapture(str(video_path))
        saved_frames = []
        for i, fr in enumerate(key_frames):
            frame_idx = fr["frame"]
            # 优先找检测器保存的标注帧
            orig_path = _THIS_DIR / "data" / "results" / f"stream_{frame_idx:06d}.jpg"
            thumb_name = f"{i:03d}_{fr.get('alert_level', 'green')}.jpg"
            thumb_path = frames_dir / thumb_name

            img = None
            if orig_path.exists():
                img = cv2.imread(str(orig_path))
            # 回退: 从视频中提取原始帧
            if img is None:
                cap_raw.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                ret, raw_frame = cap_raw.read()
                if ret:
                    img = raw_frame
            if img is not None:
                img = resize_frame(img, max_width=480)
                cv2.imwrite(str(thumb_path), img, [cv2.IMWRITE_JPEG_QUALITY, 65])

            saved_frames.append({
                "index": i,
                "frame_idx": frame_idx,
                "time_sec": fr.get("time_sec", 0),
                "alert_level": fr.get("alert_level", "green"),
                "track_count": len(fr.get("boxes", [])),
                "max_confidence": max(
                    (b.get("confidence", 0) for b in fr.get("boxes", [])), default=0
                ),
                "thumbnail": f"frames/{thumb_name}",
            })
        cap_raw.release()

        # ── 生成摘要 ──
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        summary = {
            "name": args.name,
            "video": {
                "file": video_path.name,
                "total_frames": video_total,
                "fps": round(video_fps, 1),
                "resolution": f"{video_w}x{video_h}",
                "duration_sec": round(video_total / max(video_fps, 1), 1),
            },
            "detection": {
                "processed_frames": len(detections),
                "max_frames": args.max_frames,
                "stride": args.stride,
                "img_size": args.img_size,
                "elapsed_sec": round(elapsed, 1),
                "device": detector._device_name,
            },
            "alerts": {
                "total_alert_frames": len(alert_frames),
                "red": level_counts["red"],
                "orange": level_counts["orange"],
                "yellow": level_counts["yellow"],
                "blue": level_counts["blue"],
            },
            "key_frames": saved_frames,
        }

        # ── 写入 ──
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 精简版结果 + 轨迹数据 (用于 Kalman 图表)
        track_frames = []
        for fr in detections:
            boxes = fr.get("boxes", [])
            # 只收集有 Kalman 预测值的 track (匹配成功的)
            matched = [
                {
                    "track_id": b["track_id"],
                    "actual_cx": int(round((b["bbox"][0] + b["bbox"][2]) / 2)),
                    "actual_cy": int(round((b["bbox"][1] + b["bbox"][3]) / 2)),
                    "predicted_cx": int(round(b["predicted_center"][0])),
                    "predicted_cy": int(round(b["predicted_center"][1])),
                    "confidence": round(b["confidence"], 3),
                }
                for b in boxes
                if b.get("predicted_center") and b["predicted_center"][0] is not None
            ]
            if matched:
                track_frames.append({
                    "frame": fr["frame"],
                    "time_sec": fr.get("time_sec", 0),
                    "alert_level": fr.get("alert_level", "green"),
                    "tracks": matched,
                })

        # 控制文件体积: 最多 100 帧
        if len(track_frames) > 100:
            # 优先保留预警等级高的帧
            level_order = {"red": 0, "orange": 1, "yellow": 2, "blue": 3, "green": 4}
            track_frames.sort(key=lambda f: level_order.get(f["alert_level"], 5))
            track_frames = track_frames[:100]
            track_frames.sort(key=lambda f: f["frame"])

        light_result = {
            "source": result.get("source", ""),
            "total_frames": result.get("total_frames", 0),
            "fps": result.get("fps", 25.0),
            "elapsed_seconds": round(elapsed, 1),
            "alert_frames": [
                {
                    "frame": fr["frame"],
                    "time_sec": fr.get("time_sec", 0),
                    "alert_level": fr.get("alert_level", "green"),
                    "box_count": len(fr.get("boxes", [])),
                    "max_conf": max(
                        (b.get("confidence", 0) for b in fr.get("boxes", [])), default=0
                    ),
                }
                for fr in detections
            ],
            "track_frames": track_frames,
        }
        with open(out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(light_result, f, ensure_ascii=False, indent=2)

        print()
        print("=" * 50)
        print(f"✅ 演示数据生成完毕!")
        print(f"   📂 {out_dir}")
        print(f"   🖼️  关键帧: {len(saved_frames)} 张")
        print(f"   📊 摘要: summary.json")
        print(f"   📋 结果: result.json")
        print(f"   🔴 {level_counts['red']} 🟠 {level_counts['orange']} 🟡 {level_counts['yellow']} 🔵 {level_counts['blue']}")
        print("=" * 50)
    else:
        print(f"❌ 检测失败: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
