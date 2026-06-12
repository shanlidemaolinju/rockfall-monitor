"""
TensorRT 模型导出工具
=====================
将 rock_best.pt 导出为 FP16 TensorRT engine, RTX 4060 上推理速度提升 2-3x。

依赖:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    pip install tensorrt

用法:
    python scripts/export_tensorrt.py [--imgsz 640] [--half] [--workspace 4]
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine")
    parser.add_argument("--model", default="models/rock_best.pt", help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 precision")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace (GB)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    model_path = root / args.model

    if not model_path.exists():
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return 1

    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用, 请安装 CUDA 版 PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"模型: {model_path}")
    print(f"精度: {'FP16' if args.half else 'FP32'}")
    print(f"输出: {model_path.with_suffix('.engine')}")

    model = YOLO(str(model_path))
    model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        workspace=args.workspace,
        verbose=True,
    )

    engine_path = model_path.with_suffix(".engine")
    if engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"\n[DONE] TensorRT engine 已导出: {engine_path} ({size_mb:.1f} MB)")
        print("在 .env 中设置 TENSORRT_ENABLED=true 即可启用加速推理")
        return 0
    else:
        print("\n[ERROR] 导出失败, engine 文件未生成")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
