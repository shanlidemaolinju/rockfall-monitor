"""
模型加载验证工具
================
验证 rock_best.pt 可正常加载, 输出模型基本信息。

用法:
    python scripts/validate_model.py
    python scripts/validate_model.py --model models/rock_best.pt
"""

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))


def validate_model(model_path: str | None = None) -> dict:
    """
    验证模型文件完整性并输出基本信息。

    参数:
        model_path: .pt 模型路径 (默认使用 config.MODEL_PATH)

    返回:
        {'ok': bool, 'info': dict, 'errors': list[str]}
    """
    if model_path is None:
        try:
            from rockfall.config import MODEL_PATH
            model_path = str(MODEL_PATH)
        except ImportError:
            model_path = str(_PROJECT_DIR / "models" / "rock_best.pt")

    pt_path = Path(model_path)
    result: dict = {"ok": False, "info": {}, "errors": []}

    # ── 文件存在性检查 ──
    if not pt_path.exists():
        result["errors"].append(f"模型文件不存在: {pt_path}")
        return result

    size_mb = pt_path.stat().st_size / (1024 * 1024)
    print(f"[OK] 文件存在: {pt_path} ({size_mb:.1f} MB)")

    # ── 权重加载检查 ──
    try:
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        print(f"[OK] YOLO 模型加载成功")
    except Exception as e:
        result["errors"].append(f"模型加载失败: {e}")
        return result

    # ── 模型信息提取 ──
    try:
        info = model.info()
        if info:
            print(f"  任务类型: {info.get('task', 'N/A')}")
            print(f"  类别数:   {info.get('nc', 'N/A')}")
            print(f"  参数量:   {info.get('parameters', 'N/A'):,}" if isinstance(info.get('parameters'), (int, float)) else f"  参数量:   {info.get('parameters', 'N/A')}")
            result["info"]["task"] = info.get("task", "N/A")
            result["info"]["num_classes"] = info.get("nc", 0)
    except Exception:
        pass

    # ── 基础推理测试 ──
    try:
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        if results and len(results) > 0:
            print(f"[OK] 推理测试通过 (输入 640×640)")
            result["info"]["inference_ok"] = True
        else:
            result["errors"].append("推理测试返回空结果")
    except Exception as e:
        result["errors"].append(f"推理测试失败: {e}")
        return result

    # ── 设备信息 ──
    try:
        from rockfall.config import get_device
        device_str, device_name = get_device()
        print(f"[OK] 推理设备: {device_name}")
        result["info"]["device"] = device_name
    except ImportError:
        import torch
        device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        print(f"  推理设备: {device_name}")
        result["info"]["device"] = device_name

    # ── 关联文件检查 ──
    for ext, label in [(".onnx", "ONNX"), (".engine", "TensorRT")]:
        sibling = pt_path.with_suffix(ext)
        if sibling.exists():
            sib_mb = sibling.stat().st_size / (1024 * 1024)
            print(f"  {label}: {sibling.name} ({sib_mb:.1f} MB)")
            result["info"][f"has_{ext.lstrip('.')}"] = True
        else:
            result["info"][f"has_{ext.lstrip('.')}"] = False

    if not result["errors"]:
        result["ok"] = True
        print(f"\n[OK] 模型验证通过 — {pt_path.name} 可正常使用")
    else:
        print(f"\n[ERROR] 模型验证失败:")
        for e in result["errors"]:
            print(f"   - {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 rock_best.pt 模型完整性")
    parser.add_argument("--model", default=None,
                        help=".pt 模型路径 (默认: rockfall.config.MODEL_PATH)")
    args = parser.parse_args()

    outcome = validate_model(args.model)
    if not outcome["ok"]:
        sys.exit(1)
