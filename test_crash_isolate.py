"""
独立崩溃隔离测试 — 无 PyQt, 纯命令行
=======================================
逐步测试管线的每个环节, 定位 STATUS_STACK_BUFFER_OVERRUN 的精确位置。

用法: python test_crash_isolate.py [视频文件路径]
"""

import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

# 禁用 OpenCL
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
cv2.setNumThreads(1)

# ---- 阶段1: 纯 OpenCV 视频读取 (无 YOLO, 无 MOG2) ----
def test_stage1(video_path: str):
    """仅读取视频帧, 不做任何处理"""
    print(f"\n{'='*60}")
    print(f"[阶段1] 纯视频读取 (无模型/无MOG2)")
    print(f"{'='*60}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 无法打开视频 {video_path}")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {fps:.1f}, 总帧数: {total}")

    n = 0
    t0 = time.time()
    while n < 300:  # 最多读 300 帧 (~12秒 @25fps)
        ret, frame = cap.read()
        if not ret:
            print(f"  第{n}帧后视频结束")
            break
        n += 1
        if n % 30 == 0:
            print(f"  已读 {n} 帧...")
    elapsed = time.time() - t0
    print(f"  通过: 读取 {n} 帧, 耗时 {elapsed:.1f}s ({n/elapsed:.1f} fps)")
    cap.release()
    return True


# ---- 阶段2: OpenCV 视频 + MOG2 (无 YOLO) ----
def test_stage2(video_path: str):
    """视频读取 + MOG2 背景建模 (无 YOLO)"""
    print(f"\n{'='*60}")
    print(f"[阶段2] 视频 + MOG2预处理 (无YOLO)")
    print(f"{'='*60}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 无法打开视频 {video_path}")
        return False

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # 创建默认 ROI (画面中间)
    mx, my = int(fw * 0.15), int(fh * 0.15)
    polygon = np.array([[mx, my], [mx, fh - my], [fw - mx, fh - my], [fw - mx, my]], np.int32)
    roi_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon], 255)

    # 创建 MOG2
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)

    n = 0
    t0 = time.time()
    while n < 300:
        ret, frame = cap.read()
        if not ret:
            break

        # MOG2
        fg = bg_sub.apply(frame, learningRate=0.001)

        # 后处理 (模拟 preprocess_frame 的核心逻辑)
        fg[fg == 127] = 0  # shadow removal
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_result = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg[:] = cv2.morphologyEx(fg_result, cv2.MORPH_CLOSE, k)
        cv2.bitwise_and(fg, fg, mask=roi_mask, dst=fg)

        n += 1
        if n % 30 == 0:
            print(f"  已处理 {n} 帧...")
    elapsed = time.time() - t0
    print(f"  通过: 处理 {n} 帧, 耗时 {elapsed:.1f}s ({n/elapsed:.1f} fps)")
    cap.release()
    return True


# ---- 阶段3: YOLO 模型加载 + 推理 (无视频, 静态图) ----
def test_stage3():
    """加载 YOLO 模型, 对随机图像推理"""
    print(f"\n{'='*60}")
    print(f"[阶段3] YOLO模型加载 + 静态图推理")
    print(f"{'='*60}")

    from rockfall.config import MODEL_PATH, get_device

    device_str, device_name = get_device()
    print(f"  推理设备: {device_name} ({device_str})")

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: 模型文件不存在 {MODEL_PATH}")
        return False

    print(f"  加载模型: {MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print(f"  模型加载完成")

    # 用随机图像测试 (模拟 1080p 帧)
    print(f"  创建测试图像...")
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    for i in range(10):
        print(f"  推理 #{i+1}...")
        try:
            results = model(test_img, stream=False, conf=0.3, imgsz=640, verbose=False)
            for r in results:
                n_dets = len(r.boxes) if r.boxes is not None else 0
            print(f"    完成, 检出: {n_dets}")
        except Exception as e:
            print(f"    ERROR: {e}")
            return False
    print(f"  通过: 10次推理均成功")
    return True


# ---- 阶段4: 完整管线 (视频 + MOG2 + YOLO, 无 PyQt) ----
def test_stage4(video_path: str):
    """完整管线: 视频读取 → MOG2 → YOLO → SORT (无GUI)"""
    print(f"\n{'='*60}")
    print(f"[阶段4] 完整管线 (无GUI)")
    print(f"{'='*60}")

    from rockfall.detector import RockDetector
    from rockfall.tracker import RockTracker
    from rockfall.config import get_device

    device_str, device_name = get_device()
    print(f"  推理设备: {device_name} ({device_str})")

    print(f"  初始化检测器...")
    detector = RockDetector()
    tracker = RockTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 无法打开视频 {video_path}")
        return False

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    polygon = RockDetector._default_polygon(fw, fh)
    roi_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon], 255)

    detector.init_stream_state(fw, fh, roi_mask)
    tracker.set_video_context(fps, fh)

    n = 0
    t0 = time.time()
    while n < 300:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        pp = detector.preprocess_frame(frame)

        # YOLO 推理
        if n % pp['skip'] == 0:
            detector._active_skip = pp['skip']
            raw_dets = detector.detect_frame(frame, pp['box_mask'], pp['fg'])
        else:
            raw_dets = []

        # SORT 跟踪
        tracks = tracker.update(raw_dets)

        n += 1
        if n % 30 == 0:
            print(f"  已处理 {n} 帧, 跟踪目标: {len(tracks)}")

    elapsed = time.time() - t0
    print(f"  通过: 处理 {n} 帧, 耗时 {elapsed:.1f}s ({n/elapsed:.1f} fps)")
    cap.release()
    return True


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="落石检测崩溃隔离测试")
    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("--stage", type=int, default=0, choices=[0,1,2,3,4],
                        help="0=全部阶段, 1-4=单独阶段")
    args = parser.parse_args()

    video = args.video
    if video is None:
        # 尝试自动找视频
        data_dir = Path(__file__).resolve().parent / "data" / "uploads"
        videos = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi")) + list(data_dir.glob("*.mov"))
        if videos:
            video = str(videos[0])
        else:
            # 找 test data 目录
            test_dir = Path(__file__).resolve().parent / "tests" / "data"
            videos = list(test_dir.glob("*.mp4")) if test_dir.exists() else []
            if videos:
                video = str(videos[0])

    if video is None:
        print("用法: python test_crash_isolate.py <视频文件路径>")
        print("请提供一个视频文件来运行测试")
        sys.exit(1)

    if not Path(video).exists():
        print(f"视频文件不存在: {video}")
        sys.exit(1)

    print(f"测试视频: {video}")
    print(f"Python: {sys.version}")
    print(f"OpenCV: {cv2.__version__}")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    stages = [test_stage1, test_stage2, test_stage3, test_stage4]
    if args.stage > 0:
        stages = [stages[args.stage - 1]]

    passed = 0
    failed = 0
    for i, stage_fn in enumerate(stages):
        try:
            if stage_fn == test_stage3:
                ok = stage_fn()
            else:
                ok = stage_fn(video)
            if ok:
                passed += 1
            else:
                failed += 1
                print(f"\n!!! 阶段失败, 后续阶段已跳过 !!!")
                break
        except Exception as e:
            failed += 1
            print(f"\n!!! 阶段异常: {type(e).__name__}: {e} !!!")
            import traceback
            traceback.print_exc()
            break

    print(f"\n{'='*60}")
    print(f"测试结果: {passed}通过, {failed}失败")
    if failed == 0:
        print("所有阶段通过 — 崩溃可能出在 PyQt/QImage 显示环节")
    print(f"{'='*60}")
