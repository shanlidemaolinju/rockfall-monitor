"""
导出 YOLO .pt 模型为 ONNX 格式 (跨平台 CPU/GPU 通用推理)
===========================================================
ONNX Runtime 在纯 CPU 上比 PyTorch 快 2-3x, 且不依赖 CUDA Toolkit。

用法:
    python scripts/export_onnx.py                          # 默认路径
    python scripts/export_onnx.py --model models/rock_best.pt --imgsz 640

依赖:
    pip install onnx onnxruntime ultralytics

输出:
    models/rock_best.onnx  — 可被 opencv.dnn / onnxruntime 加载
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def export_onnx(model_path: str, imgsz: int = 640, opset: int = 12,
                simplify: bool = True, dynamic: bool = False) -> str:
    """
    导出 ONNX 模型。

    参数:
        model_path:  .pt 模型文件路径
        imgsz:       推理尺寸 (单值 = 正方形, 如 640)
        opset:       ONNX opset 版本 (12 兼容性好, 17 支持更多算子)
        simplify:    是否调用 onnx-simplifier 精简图
        dynamic:     是否导出动态 batch/尺寸 (True=灵活但略慢, False=优化)
    """
    pt_path = Path(model_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {pt_path}")

    print(f"加载 PyTorch 模型: {pt_path}")
    model = YOLO(str(pt_path))

    onnx_path = pt_path.with_suffix(".onnx")

    print(f"导出 ONNX (imgsz={imgsz}, opset={opset}, dynamic={dynamic})")
    _ = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
    )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"导出完成: {onnx_path} ({size_mb:.1f} MB)")

    print("\n使用方式:")
    print(f"  1. onnxruntime 加载 (推荐):")
    print(f"     model = YOLO('{onnx_path}')")
    print(f"  2. OpenCV DNN 加载:")
    print(f"     net = cv2.dnn.readNetFromONNX('{onnx_path}')")
    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO .pt → ONNX 导出")
    parser.add_argument("--model", default="models/rock_best.pt",
                        help=".pt 模型路径 (默认: models/rock_best.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理尺寸 (默认: 640)")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset (默认: 12)")
    parser.add_argument("--no-simplify", action="store_true",
                        help="禁用 onnxsim 精简")
    parser.add_argument("--dynamic", action="store_true",
                        help="导出动态 batch/尺寸")
    args = parser.parse_args()

    export_onnx(
        model_path=args.model,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic,
    )
