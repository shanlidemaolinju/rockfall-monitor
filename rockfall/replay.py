"""
预警回放模块 — 自动提取预警时刻视频片段
========================================
检测完成后, 从原始视频中提取预警帧附近的片段, 生成 MP4 回放文件。

用法:
    from rockfall.replay import extract_alert_clips, stitch_annotated_clip

    # 方式1: 从原始视频提取 (未标注, 快速)
    clips = extract_alert_clips("video.mp4", alert_frames, window=75)

    # 方式2: 从已保存标注帧拼接 (含检测框)
    stitch_annotated_clip(alert_frame_indices, clip_path, fps=25)
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .config import RESULTS_DIR


# ══════════════════════════════════════════════════════════════
# 方式 1: 从原始视频提取片段 (快速, 未标注)
# ══════════════════════════════════════════════════════════════

def extract_alert_clips(
    video_path: str,
    alert_frame_indices: list[int],
    output_dir: str | Path = "",
    window_before: int = 75,
    window_after: int = 50,
    fps: float = 25.0,
    max_clips: int = 20,
) -> list[dict]:
    """
    从原始视频中提取预警帧附近的视频片段。

    参数:
        video_path:          原始视频文件路径
        alert_frame_indices: 触发预警的帧号列表
        output_dir:          输出目录 (默认 RESULTS_DIR / "clips")
        window_before:       预警帧之前包含的帧数
        window_after:        预警帧之后包含的帧数
        fps:                 输出视频帧率
        max_clips:           最多提取多少个片段

    返回:
        [{"frame_idx": int, "alert_level": str, "clip_path": str, "duration_sec": float}, ...]
    """
    import cv2

    out_dir = Path(output_dir) if output_dir else RESULTS_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 去重 + 排序
    unique_frames = sorted(set(alert_frame_indices))[:max_clips]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clips = []

    for center_frame in unique_frames:
        start_frame = max(0, center_frame - window_before)
        end_frame = min(video_total - 1, center_frame + window_after)

        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip_name = f"alert_f{center_frame:06d}.mp4"
        clip_path = out_dir / clip_name

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        writer = cv2.VideoWriter(str(clip_path), fourcc, video_fps, (fw, fh))

        if not writer.isOpened():
            # 回退到 mp4v 编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(clip_path), fourcc, video_fps, (fw, fh))

        frame_count = 0
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # 添加帧号标注
            actual_frame = start_frame + frame_count
            cv2.putText(frame, f"Frame: {actual_frame}",
                        (10, fh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if actual_frame == center_frame:
                # 预警帧: 红色边框闪烁标记
                cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), (0, 0, 255), 4)
                cv2.putText(frame, "ALERT", (fw // 2 - 60, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(frame)
            frame_count += 1

        writer.release()
        duration = frame_count / max(video_fps, 1)

        clips.append({
            "frame_idx": center_frame,
            "clip_path": str(clip_path),
            "duration_sec": round(duration, 1),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    cap.release()
    return clips


# ══════════════════════════════════════════════════════════════
# 方式 2: 从已保存标注帧拼接 (含检测框, 更直观)
# ══════════════════════════════════════════════════════════════

def stitch_annotated_clip(
    frame_indices: list[int],
    output_path: str | Path,
    source_dir: str | Path = "",
    fps: float = 25.0,
    window_around: int = 0,
) -> str | None:
    """
    从已保存的标注帧 JPEG 拼接成 MP4 回放片段。

    参数:
        frame_indices:  帧号列表 (按顺序)
        output_path:    输出 MP4 路径
        source_dir:     标注帧目录 (默认 RESULTS_DIR)
        fps:            输出帧率
        window_around:  在 frame_indices 前后各追加的帧数 (用最近的标注帧填充)

    返回:
        成功返回 str 路径, 失败返回 None
    """
    import cv2

    src_dir = Path(source_dir) if source_dir else RESULTS_DIR
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 扩展帧范围 (如果指定了 window_around)
    if window_around > 0 and frame_indices:
        first = frame_indices[0]
        last = frame_indices[-1]
        expanded = set(frame_indices)
        for i in range(1, window_around + 1):
            expanded.add(max(0, first - i))
            expanded.add(last + i)
        frame_indices = sorted(expanded)

    # 收集存在的标注帧
    valid_frames = []
    for fi in frame_indices:
        fp = src_dir / f"stream_{fi:06d}.jpg"
        if fp.exists():
            valid_frames.append((fi, fp))

    if not valid_frames:
        return None

    # 读取第一帧获取尺寸
    first_frame = cv2.imread(str(valid_frames[0][1]))
    if first_frame is None:
        return None
    fh, fw = first_frame.shape[:2]

    # 写入 MP4
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))

    for fi, fp in valid_frames:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        if frame.shape[:2] != (fh, fw):
            frame = cv2.resize(frame, (fw, fh))
        writer.write(frame)

    writer.release()
    return str(out_path)


# ══════════════════════════════════════════════════════════════
# 批量处理: 按预警等级分组生成回放片段
# ══════════════════════════════════════════════════════════════

def generate_alert_replays(
    alert_frames: list[dict],
    clip_dir: str | Path = "",
    fps: float = 25.0,
    context_frames: int = 50,
    max_per_level: int = 5,
) -> dict[str, list[dict]]:
    """
    为检测结果中的预警帧批量生成回放片段 (方式2: 标注帧拼接)。

    参数:
        alert_frames:   [{"frame_idx": int, "alert_level": str, "tracks": [...], ...}, ...]
        clip_dir:       输出目录
        fps:            帧率
        context_frames: 预警帧前后包含的上下文帧数
        max_per_level:  每个预警等级最多生成几个片段

    返回:
        {"red": [{"frame_idx": ..., "clip_path": ..., "duration_sec": ...}, ...], ...}
    """
    out_dir = Path(clip_dir) if clip_dir else RESULTS_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按等级分组
    by_level: dict[str, list[int]] = {"red": [], "orange": [], "yellow": [], "blue": []}
    for fr in alert_frames:
        lvl = fr.get("alert_level", "")
        if lvl in by_level:
            by_level[lvl].append(fr["frame_idx"])

    results: dict[str, list[dict]] = {}

    for lvl, frame_indices in by_level.items():
        if not frame_indices:
            results[lvl] = []
            continue

        # 取置信度最高的前 N 个
        selected = sorted(set(frame_indices))[:max_per_level]
        level_results = []

        for center_frame in selected:
            # 构建帧范围: center ± context_frames
            start_f = max(0, center_frame - context_frames)
            end_f = center_frame + context_frames
            frame_range = list(range(start_f, end_f + 1))

            clip_name = f"{lvl}_f{center_frame:06d}.mp4"
            clip_path = out_dir / clip_name

            result_path = stitch_annotated_clip(
                frame_indices=frame_range,
                output_path=clip_path,
                fps=fps,
            )

            if result_path:
                n_frames = len(frame_range)
                level_results.append({
                    "frame_idx": center_frame,
                    "alert_level": lvl,
                    "clip_path": result_path,
                    "duration_sec": round(n_frames / max(fps, 1), 1),
                })

        results[lvl] = level_results

    return results
