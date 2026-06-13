"""
SAM Box Prompt 道路分割 (Windows稳定版)
==========================================
SAM全程CPU运行, 彻底隔离CUDA上下文.
用后立即释放所有内存, 防止栈碎片化.
"""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from .logger import log_event
import numpy as np

_MASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "masks"


def segment_road_with_box_prompt(
    cap: cv2.VideoCapture, fw: int, fh: int,
    rect_poly: np.ndarray, cache_key: str = "default",
) -> np.ndarray | None:
    """SAM道路分割, 返回255=道路. 全程CPU, 用完释放."""
    _MASKS_DIR.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5((cache_key + "_sam_v2").encode()).hexdigest()[:12]
    cache_path = _MASKS_DIR / f"{safe}.png"
    if cache_path.exists():
        cached = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        if cached is not None and cached.shape == (fh, fw):
            return cached

    # 独立进程运行SAM, 避免CUDA冲突
    import subprocess, tempfile, shutil

    # 保存采样帧
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam_frames_"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live = total_frames <= 0
    num_samples = 3
    saved = 0
    for s in range(num_samples):
        if not is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s * max(1, total_frames // num_samples))
        ret, frame = cap.read()
        if not ret: continue
        if frame.shape[1] != fw or frame.shape[0] != fh:
            frame = cv2.resize(frame, (fw, fh))
        cv2.imwrite(str(tmp_dir / f"frame_{s:03d}.png"), frame)
        saved += 1

    if saved < 1:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # 调用独立进程
    output = tmp_dir / "road_mask.png"
    sam_script = Path(__file__).resolve().parent.parent / "rockfall" / "sam_server.py"
    if not sam_script.exists():
        sam_script = Path(__file__).resolve().parent / "sam_server.py"

    try:
        result = subprocess.run([
            sys.executable, str(sam_script),
            str(tmp_dir), str(output), str(fw), str(fh), cache_key,
        ], capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=180)
        if result.returncode != 0:
            log_event("system", level="ERROR", msg=f"SAM 子进程失败 (code {result.returncode})")
            return None
    except Exception as e:
        log_event("system", level="ERROR", msg=f"SAM 子进程异常: {e}")
        return None

    # 读取结果
    road_mask = None
    if output.exists():
        road_mask = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
        if road_mask is not None:
            cv2.imwrite(str(cache_path), road_mask)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if not is_live:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return road_mask


