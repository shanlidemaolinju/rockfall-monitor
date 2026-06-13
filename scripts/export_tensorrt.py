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
import sys
from pathlib import Path

import torch
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


def export_tensorrt(
    model_path: str | None = None,
    imgsz: int = 640,
    half: bool = True,
    workspace: int = 4,
    output: str | None = None,
) -> int:
    """
    导出 YOLO .pt 模型为 TensorRT engine。

    参数:
        model_path: .pt 模型路径 (默认使用 config.MODEL_PATH)
        imgsz:      输入图像尺寸
        half:       是否使用 FP16 精度
        workspace:  TensorRT workspace 大小 (GB)
        output:     自定义输出路径 (默认与模型同目录、.engine 后缀)

    返回:
        0=成功, 1=失败
    """
    if model_path is None:
        model_path = _default_model_path()

    pt_path = Path(model_path)
    if not pt_path.exists():
        print(f"[ERROR] 模型文件不存在: {pt_path}")
        return 1

    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用, 请安装 CUDA 版 PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129")
        return 1

    engine_path = Path(output) if output else pt_path.with_suffix(".engine")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"模型: {pt_path}")
    print(f"精度: {'FP16' if half else 'FP32'}")
    print(f"输出: {engine_path}")

    model = YOLO(str(pt_path))
    model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        workspace=workspace,
        verbose=True,
    )

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"\n[DONE] TensorRT engine 已导出: {engine_path} ({size_mb:.1f} MB)")
        print("在 .env 中设置 TENSORRT_ENABLED=true 即可启用加速推理")
        return 0
    else:
        print("\n[ERROR] 导出失败, engine 文件未生成")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine")
    parser.add_argument("--model", default=None,
                        help="Path to .pt model (默认: rockfall.config.MODEL_PATH)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 precision")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace (GB)")
    parser.add_argument("--output", default=None,
                        help="自定义输出路径 (默认: 与模型同目录、.engine 后缀)")
    args = parser.parse_args()

    raise SystemExit(export_tensorrt(
        model_path=args.model,
        imgsz=args.imgsz,
        half=args.half,
        workspace=args.workspace,
        output=args.output,
    ))
