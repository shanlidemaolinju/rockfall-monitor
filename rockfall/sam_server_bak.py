"""
SAM 道路分割独立进程
=====================
与主进程完全隔离, 通过文件交换数据。
避免 SAM 与 YOLO/OpenCV/PyQt 的 CUDA 冲突。

用法:
    python sam_server.py <frames_dir> <output_path> <fw> <fh> <cache_key>

输入:  frames_dir/ 下的 PNG 帧文件
输出:  output_path 的 PNG 掩码文件
"""

import gc
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    if len(sys.argv) < 6:
        print("用法: python sam_server.py <frames_dir> <output_path> <fw> <fh> <cache_key>")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    fw = int(sys.argv[3])
    fh = int(sys.argv[4])
    cache_key = sys.argv[5]

    # 缓存检查
    cache_dir = frames_dir.parent / "masks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    safe = hashlib.md5((cache_key + "_sam").encode()).hexdigest()[:12]
    cache_path = cache_dir / f"{safe}.png"
    if cache_path.exists():
        cached = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
        if cached is not None and cached.shape == (fh, fw):
            cv2.imwrite(str(output_path), cached)
            print(f"[SAM-Server] 缓存命中")
            return

    # 加载SAM (纯CPU)
    t0 = time.time()
    from segment_anything import sam_model_registry, SamPredictor
    import torch

    model_path = frames_dir.parent.parent / "models" / "sam_vit_b_01ec64.pth"
    if not model_path.exists():
        print(f"[SAM-Server] 模型不存在: {model_path}")
        sys.exit(1)

    torch.cuda.empty_cache()
    sam = sam_model_registry["vit_b"](checkpoint=str(model_path), map_location="cpu")
    sam.to(device=torch.device("cpu"))
    sam.eval()
    sam.requires_grad_(False)
    print(f"[SAM-Server] 模型加载 ({time.time()-t0:.1f}s)")

    # 读取帧
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        # 尝试直接读 frames_dir 下的所有 PNG
        frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"[SAM-Server] 无帧文件: {frames_dir}")
        sys.exit(1)

    # 限制最多3帧
    frames = frames[:3]
    print(f"[SAM-Server] 处理 {len(frames)} 帧")

    predictor = SamPredictor(sam)
    box = np.array([0, int(fh * 0.10), fw, fh])
    road_score = np.zeros((fh, fw), dtype=np.float32)
    valid = 0

    for fpath in frames:
        frame = cv2.imread(str(fpath))
        if frame is None: continue
        if frame.shape[1] != fw or frame.shape[0] != fh:
            frame = cv2.resize(frame, (fw, fh))

        try:
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            for i in range(3):
                area_ratio = masks[i].sum() / (fw * fh)
                if 0.08 < area_ratio < 0.60:
                    road_score += masks[i].astype(np.float32)
                    valid += 1
                    break
            predictor.reset_image()
        except Exception as e:
            print(f"[SAM-Server] 帧异常: {e}")

    del predictor; del sam; gc.collect()

    if valid < 1:
        print("[SAM-Server] 无有效分割")
        sys.exit(1)

    road_mask = (road_score >= (valid + 1) // 2).astype(np.uint8) * 255

    # 后处理
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    road_mask = cv2.GaussianBlur(road_mask.astype(np.float32), (15, 15), 0)
    road_mask = (road_mask > 127).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        road_mask_clean = np.zeros_like(road_mask)
        bottom = [c for c in cnts if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= fh * 0.8]
        best = max(bottom if bottom else cnts, key=cv2.contourArea)
        cv2.drawContours(road_mask_clean, [best], -1, 255, -1)
        road_mask = road_mask_clean

    road_mask[:, :int(fw * 0.05)] = 0
    road_mask[:, int(fw * 0.95):] = 0

    cv2.imwrite(str(output_path), road_mask)
    cv2.imwrite(str(cache_path), road_mask)
    print(f"[SAM-Server] 完成 ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
