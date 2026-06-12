"""
最简崩溃隔离 — 纯 YOLO + 视频读取, 无 MOG2 / 无 GUI / 无绘制
===============================================================
用于判断崩溃到底在 YOLO(CUDA) 还是 OpenCV(MOG2/绘图)
用法: python test_minimal.py "视频路径"
"""

import os
import sys

# 与 main.py 相同的安全配置
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import time
import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
cv2.setNumThreads(1)


def main():
    if len(sys.argv) < 2:
        print("用法: python test_minimal.py <视频路径>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"视频: {video_path}")

    # ---- 步骤1: 仅读帧 (无模型) ----
    print("\n[步骤1] 纯视频读取 (30帧)...")
    cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: 无法打开视频")
        sys.exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  分辨率: {fw}x{fh}, 后端: {cap.getBackendName()}")

    # 降采样 4K→1080p
    MAX_W = 1920
    scale = 1.0
    if fw > MAX_W:
        scale = MAX_W / fw
        fw, fh = MAX_W, int(fh * scale)
        print(f"  降采样: → {fw}x{fh}")

    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            print(f"  第{i}帧后视频结束")
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (fw, fh))
    print(f"  通过: 读取 {i+1} 帧")
    cap.release()

    # ---- 步骤2: 加载 YOLO (在视频读取之后, 排除解码器残留) ----
    print("\n[步骤2] 加载 YOLO 模型...")
    from rockfall.config import MODEL_PATH, get_device
    device_str, device_name = get_device()
    print(f"  设备: {device_name} ({device_str})")

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print(f"  模型加载完成")

    # ---- 步骤3: YOLO 推理静态图 ----
    print("\n[步骤3] YOLO 静态图推理 (10次)...")
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    for i in range(10):
        results = model(test_img, stream=False, conf=0.3, imgsz=640, verbose=False)
        n = len(results[0].boxes) if results[0].boxes is not None else 0
        if i == 0:
            print(f"  推理 #{i+1}: 检出 {n}")
    print(f"  通过: 10次推理完成")

    # ---- 步骤4: 视频帧 + YOLO (无 MOG2) ----
    print("\n[步骤4] 视频帧 → YOLO 推理 (30帧, 无MOG2)...")
    cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    t0 = time.time()
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        # 降采样
        if scale != 1.0:
            frame = cv2.resize(frame, (fw, fh))
        # 每5帧推理一次 (模拟跳帧)
        if i % 5 == 0:
            frame = np.ascontiguousarray(frame)
            results = model(frame, stream=False, conf=0.3, imgsz=640, verbose=False)
            n = len(results[0].boxes) if results[0].boxes is not None else 0
            del results
        if i == 0:
            print(f"  帧{i}: 推理完成")
    elapsed = time.time() - t0
    print(f"  通过: {i+1} 帧, {elapsed:.1f}s")
    cap.release()

    # ---- 步骤5: 视频帧 + YOLO + MOG2 (无 GUI) ----
    print("\n[步骤5] 视频帧 → MOG2 → YOLO (30帧)...")
    from rockfall.detector import RockDetector
    detector = RockDetector()

    polygon = RockDetector._default_polygon(fw, fh)
    roi_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon], 255)
    detector.init_stream_state(fw, fh, roi_mask)

    cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    t0 = time.time()
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (fw, fh))
        pp = detector.preprocess_frame(frame)
        if i % pp['skip'] == 0:
            detector._active_skip = pp['skip']
            raw_dets = detector.detect_frame(frame, pp['box_mask'], pp['fg'])
        if i == 0:
            print(f"  帧{i}: MOG2+YOLO 完成")
    elapsed = time.time() - t0
    print(f"  通过: {i+1} 帧, {elapsed:.1f}s")
    cap.release()

    print("\n" + "=" * 60)
    print("全部5个步骤通过! 崩溃可能出在 PyQt/QImage 显示环节")
    print("=" * 60)


if __name__ == "__main__":
    main()
