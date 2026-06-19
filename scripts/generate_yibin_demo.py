"""
Generate Yibin landslide demo data for Streamlit preset showcase.
Shows: calm → precursor rockfalls → escalation → red alert → collapse.
"""
import sys, json, time, cv2, numpy as np
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_THIS_DIR))

from rockfall.detector import RockDetector

VIDEO_PATH = "d:/rock/3.7日，四川宜宾一高速路段发生山体滑坡.mp4"
SCENE_ID = "yibin_s1"
OUT_DIR = _THIS_DIR / "demo_data" / SCENE_ID
FRAMES_DIR = OUT_DIR / "frames"

def resize_frame(frame, max_width=480):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    ratio = max_width / w
    return cv2.resize(frame, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)

def main():
    print("=" * 60)
    print("宜宾滑坡演示数据生成")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Init detector with Yibin site thresholds
    print("\n[1] Loading model with yibin_s1 thresholds...")
    detector = RockDetector(site_id="yibin_s1")
    detector.img_size = 416
    print(f"    Device: {detector._device_name}")
    print(f"    Confidence: {detector.confidence}")

    # Process: start_frame=900 (30s), stride=2, max_frames=500
    # Covers 900+1000=1900 frames = 30s-63s (precursors + escalation + post-collapse)
    # Start at 30s to skip MOG2 cold-start period
    START_FRAME = 900
    print(f"\n[2] Running detection (start={START_FRAME}, stride=2, 500 frames, ~30-63s)...")
    print(f"    Video: {VIDEO_PATH}")
    t0 = time.time()
    result = detector.detect_video(
        str(VIDEO_PATH),
        save_frames=True,
        push_alerts=False,
        track=True,
        max_frames=500,
        stride=2,
        start_frame=START_FRAME,
    )
    elapsed = time.time() - t0

    if not isinstance(result, dict) or "error" in result:
        print(f"ERROR: {result}")
        return

    detections = result.get("detections", [])
    alert_frames = [fr for fr in detections if fr.get("alert_level", "green") != "green"]
    print(f"    Done in {elapsed:.1f}s")
    print(f"    {len(detections)} frames processed, {len(alert_frames)} alerts")

    # Level distribution
    level_counts = {"red": 0, "orange": 0, "yellow": 0, "blue": 0, "green": 0}
    for fr in detections:
        lvl = fr.get("alert_level", "green")
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    print(f"    Levels: R={level_counts['red']} O={level_counts['orange']} Y={level_counts['yellow']} B={level_counts['blue']} G={level_counts['green']}")

    # Select key frames: best of each level
    key_frames = []
    for lvl in ["red", "orange", "yellow", "blue"]:
        lvl_frames = sorted(
            [fr for fr in alert_frames if fr.get("alert_level") == lvl],
            key=lambda fr: max((b.get("confidence", 0) for b in fr.get("boxes", [])), default=0),
            reverse=True,
        )
        key_frames.extend(lvl_frames[:8])

    # Pad with green frames to have enough
    if len(key_frames) < 24:
        green = [fr for fr in detections if fr.get("alert_level") == "green"][:24 - len(key_frames)]
        key_frames.extend(green)

    print(f"\n[3] Saving {len(key_frames)} key frames...")

    # Save thumbnails
    saved_frames = []
    for i, fr in enumerate(key_frames):
        frame_idx = fr["frame"]
        orig_path = _THIS_DIR / "data" / "results" / f"stream_{frame_idx:06d}.jpg"
        thumb_name = f"{i:03d}_{fr.get('alert_level', 'green')}.jpg"
        thumb_path = FRAMES_DIR / thumb_name

        if orig_path.exists():
            img = cv2.imread(str(orig_path))
            if img is not None:
                img = resize_frame(img, max_width=480)
                cv2.imwrite(str(thumb_path), img, [cv2.IMWRITE_JPEG_QUALITY, 65])

        boxes = fr.get("boxes", [])
        max_conf = max((b.get("confidence", 0) for b in boxes), default=0)
        saved_frames.append({
            "index": i,
            "frame_idx": frame_idx,
            "time_sec": fr.get("time_sec", 0),
            "alert_level": fr.get("alert_level", "green"),
            "track_count": len(boxes),
            "max_confidence": round(max_conf, 4),
            "thumbnail": f"frames/{thumb_name}",
        })

    # Video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # Summary
    summary = {
        "name": SCENE_ID,
        "video": {
            "file": Path(VIDEO_PATH).name,
            "total_frames": video_total,
            "fps": round(video_fps, 1),
            "resolution": f"{video_w}x{video_h}",
            "duration_sec": round(video_total / max(video_fps, 1), 1),
        },
        "detection": {
            "processed_frames": len(detections),
            "max_frames": 500,
            "stride": 2,
            "start_frame": START_FRAME,
            "img_size": 416,
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

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"    summary.json written")

    # Result with track data (for Kalman charts)
    track_frames = []
    for fr in detections:
        boxes = fr.get("boxes", [])
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

    result_out = {
        "track_frames": track_frames,
        "alert_levels": level_counts,
        "total_alert_frames": len(alert_frames),
    }
    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_out, f, ensure_ascii=False, indent=2)
    print(f"    result.json written ({len(track_frames)} track frames)")

    # Print alert timeline
    if alert_frames:
        print(f"\n[4] Alert timeline:")
        prev_lvl = "green"
        for fr in alert_frames:
            lvl = fr.get("alert_level", "green")
            if lvl != prev_lvl:
                ts = fr.get("time_sec", 0)
                boxes = fr.get("boxes", [])
                conf = max((b.get("confidence", 0) for b in boxes), default=0)
                emoji = "🔴" if lvl == "red" else "🟠" if lvl == "orange" else "🟡" if lvl == "yellow" else "🔵"
                print(f"    {ts:>6.1f}s | {prev_lvl:>6} → {lvl:>6} {emoji} | conf={conf:.4f}")
                prev_lvl = lvl

    print(f"\n{'='*60}")
    print(f"Demo data ready: {OUT_DIR}")
    print(f"Next: add 'yibin_s1' to DEMO_SCENES in app.py")

if __name__ == "__main__":
    main()
