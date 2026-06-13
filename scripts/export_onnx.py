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
import sys
from pathlib import Path

from ultralytics import YOLO

# 确保项目根目录在 sys.path 中
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


def _default_model_path() -> str:
    """使用项目 config 中的模型路径, 若不可用则回退默认值"""
    try:
        from rockfall.config import MODEL_PATH
        return str(MODEL_PATH)
    except ImportError:
        return str(_PROJECT_DIR / "models" / "rock_best.pt")


def export_onnx(
    model_path: str | None = None,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
    output: str | None = None,
) -> str:
    """
    导出 YOLO .pt 模型为 ONNX 格式。

    参数:
        model_path:  .pt 模型文件路径 (默认使用 config.MODEL_PATH)
        imgsz:       推理尺寸 (单值 = 正方形, 如 640)
        opset:       ONNX opset 版本 (12 兼容性好, 17 支持更多算子)
        simplify:    是否调用 onnx-simplifier 精简图
        dynamic:     是否导出动态 batch/尺寸 (True=灵活但略慢, False=优化)
        output:      自定义输出路径 (默认与 model_path 同目录、同主名)
    """
    if model_path is None:
        model_path = _default_model_path()

    pt_path = Path(model_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {pt_path}")

    print(f"加载 PyTorch 模型: {pt_path}")
    model = YOLO(str(pt_path))

    onnx_path = Path(output) if output else pt_path.with_suffix(".onnx")

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
    parser.add_argument("--model", default=None,
                        help=".pt 模型路径 (默认: rockfall.config.MODEL_PATH)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理尺寸 (默认: 640)")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset (默认: 12)")
    parser.add_argument("--output", default=None,
                        help="自定义输出路径 (默认: 与模型同目录、同主名)")
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
        output=args.output,
    )
