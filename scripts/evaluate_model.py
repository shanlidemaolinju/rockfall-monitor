"""
模型性能评估报告
================
基于验证集评估 rock_best.pt 的检测性能, 生成完整指标与可视化图表。

输出:
  - 终端:  mAP50, mAP50-95, Precision, Recall, F1-Score, FPS
  - 图表:  混淆矩阵, PR曲线, F1-Confidence曲线 (ultralytics自动生成)
  - 报告:  data/results/evaluation/performance_report.md

用法:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model models/rock_best.pt --data ../rock.v27i.yolov8/data.yaml
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from rockfall.config import ROOT_DIR, DATA_DIR


def evaluate_model(
    model_path: str | None = None,
    data_yaml: str | None = None,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
) -> dict:
    """
    运行模型验证评估, 生成性能报告。

    参数:
        model_path: .pt 模型路径 (默认 config.MODEL_PATH)
        data_yaml:  数据集 data.yaml 路径 (默认使用 rock.v27i.yolov8/data.yaml)
        imgsz:      推理图像尺寸
        batch:      验证批次大小
        device:     推理设备 (None=自动)

    返回:
        {'metrics': dict, 'report_path': str, 'charts_dir': str}
    """
    from ultralytics import YOLO

    # ── 路径解析 ──
    if model_path is None:
        from rockfall.config import MODEL_PATH
        model_path = str(MODEL_PATH)

    if data_yaml is None:
        # 自动查找: 优先使用 rock.v27i.yolov8
        candidates = [
            _PROJECT_DIR.parent / "rock.v27i.yolov8" / "data.yaml",
            _PROJECT_DIR / "data" / "data.yaml",
        ]
        for c in candidates:
            if c.exists():
                data_yaml = str(c)
                break
        if data_yaml is None:
            raise FileNotFoundError(
                "未找到数据集 data.yaml。请通过 --data 参数指定。\n"
                f"已搜索: {[str(c) for c in candidates]}"
            )

    if device is None:
        try:
            from rockfall.config import get_device
            device, _ = get_device()
        except Exception:
            device = "cuda:0" if _has_cuda() else "cpu"

    pt_path = Path(model_path)
    data_path = Path(data_yaml)
    if not pt_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {pt_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置不存在: {data_path}")

    # ── 输出目录 ──
    eval_dir = DATA_DIR / "results" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  落石检测模型 — 性能评估")
    print("=" * 60)
    print(f"  模型:    {pt_path}")
    print(f"  数据集:  {data_path}")
    print(f"  设备:    {device}")
    print(f"  尺寸:    {imgsz}")
    print(f"  批次:    {batch}")
    print(f"  输出:    {eval_dir}")
    print("=" * 60)
    print()

    # ── 1. 加载模型 ──
    print("[1/4] 加载模型...")
    model = YOLO(str(pt_path))

    # ── 2. 标准 mAP 验证 ──
    print("[2/4] 运行验证集评估 (mAP, Precision, Recall)...")
    t0 = time.time()
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=True,
        save_json=False,
        plots=True,
    )
    val_time = time.time() - t0
    print(f"  验证耗时: {val_time:.1f}s")

    # 提取关键指标
    results_dict = metrics.results_dict if hasattr(metrics, 'results_dict') else {}
    box = getattr(metrics, 'box', None)

    mAP50 = round(results_dict.get("metrics/mAP50(B)", 0), 4)
    mAP50_95 = round(results_dict.get("metrics/mAP50-95(B)", 0), 4)
    precision = round(results_dict.get("metrics/precision(B)", 0), 4)
    recall = round(results_dict.get("metrics/recall(B)", 0), 4)

    print(f"\n  [METRICS] 检测指标:")
    print(f"     mAP50:      {mAP50:.4f}")
    print(f"     mAP50-95:   {mAP50_95:.4f}")
    print(f"     Precision:  {precision:.4f}")
    print(f"     Recall:     {recall:.4f}")

    # F1-Score
    f1 = round(2 * precision * recall / (precision + recall + 1e-6), 4)
    print(f"     F1-Score:   {f1:.4f}")

    # ── 3. FPS 基准测试 ──
    print("\n[3/4] FPS 基准测试 (100次推理取平均)...")
    fps_results = _benchmark_fps(model, imgsz, device, warmup=10, runs=100)

    print(f"     CPU FPS:  {fps_results.get('cpu_fps', 'N/A')}")
    print(f"     GPU FPS:  {fps_results.get('gpu_fps', 'N/A')}")
    if fps_results.get('gpu_fps'):
        print(f"     GPU/CPU:  {fps_results['gpu_fps'] / max(fps_results.get('cpu_fps', 1), 1):.1f}x")

    # ── 4. 生成 Markdown 报告 ──
    print("\n[4/4] 生成评估报告...")
    report_path = _write_report(
        eval_dir, pt_path, data_path, device,
        mAP50, mAP50_95, precision, recall, f1,
        fps_results, val_time,
    )
    print(f"  报告: {report_path}")

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("  [DONE] 评估完成")
    print(f"  报告: {report_path}")
    print(f"  图表: {eval_dir}")
    print("=" * 60)

    return {
        "metrics": {
            "mAP50": mAP50, "mAP50-95": mAP50_95,
            "precision": precision, "recall": recall, "f1": f1,
            **fps_results,
        },
        "report_path": str(report_path),
        "charts_dir": str(eval_dir),
    }


def _benchmark_fps(model, imgsz: int, device: str,
                   warmup: int = 10, runs: int = 100) -> dict:
    """FPS 基准测试 (CPU + GPU 分别测)"""
    import torch

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    result = {}

    def _measure(dev: str) -> float:
        m = model.model
        # 移动模型到目标设备
        try:
            m.to(dev)
        except Exception:
            pass

        # 预热
        for _ in range(warmup):
            model(dummy, imgsz=imgsz, device=dev, verbose=False)

        # 计时
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy, imgsz=imgsz, device=dev, verbose=False)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        fps = 1000 / avg_ms
        return fps

    # CPU 测试
    try:
        result["cpu_fps"] = round(_measure("cpu"), 1)
    except Exception as e:
        result["cpu_fps"] = None
        print(f"     CPU基准测试失败: {e}")

    # GPU 测试
    if torch.cuda.is_available():
        try:
            result["gpu_fps"] = round(_measure("cuda:0"), 1)
        except Exception as e:
            result["gpu_fps"] = None
            print(f"     GPU基准测试失败: {e}")
    else:
        result["gpu_fps"] = None

    return result


def _write_report(eval_dir: Path, model_path: Path, data_path: Path,
                  device: str, mAP50: float, mAP50_95: float,
                  precision: float, recall: float, f1: float,
                  fps: dict, val_time: float) -> Path:
    """生成 Markdown 格式的性能报告"""
    report_path = eval_dir / "performance_report.md"
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cpu_fps_str = f"{fps['cpu_fps']:.1f}" if fps.get('cpu_fps') else "N/A"
    gpu_fps_str = f"{fps['gpu_fps']:.1f}" if fps.get('gpu_fps') else "N/A"

    content = f"""# 落石检测模型 — 性能评估报告

> 生成时间: {report_time}
> 模型: `{model_path.name}`
> 数据集: `{data_path.parent.name}`
> 设备: `{device}`

---

## 检测性能指标

| 指标            | 值       | 说明 |
|----------------|----------|------|
| **mAP50**      | **{mAP50:.4f}** | IoU=0.5 时的平均精度 |
| **mAP50-95**   | **{mAP50_95:.4f}** | IoU=0.5~0.95 的平均精度 |
| **Precision**  | **{precision:.4f}** | 所有预测中正确检测的比例 |
| **Recall**     | **{recall:.4f}** | 所有真实目标中被检出的比例 |
| **F1-Score**   | **{f1:.4f}** | Precision 与 Recall 的调和平均 |

## 推理速度

| 设备 | FPS | 单帧耗时 |
|------|-----|----------|
| CPU  | {cpu_fps_str} | {'{:.1f} ms'.format(1000/max(fps.get('cpu_fps', 1), 1)) if fps.get('cpu_fps') else 'N/A'} |
| GPU  | {gpu_fps_str} | {'{:.1f} ms'.format(1000/max(fps.get('gpu_fps', 1), 1)) if fps.get('gpu_fps') else 'N/A'} |
{f"| **加速比** | **{fps['gpu_fps'] / max(fps.get('cpu_fps', 1), 1):.1f}x** | |" if fps.get('gpu_fps') and fps.get('cpu_fps') else ""}

## 可视化图表

ultralytics 自动生成以下图表 (保存在 `{eval_dir}` 下):

- `confusion_matrix.png` / `confusion_matrix_normalized.png` — 混淆矩阵
- `PR_curve.png` — Precision-Recall 曲线
- `F1_curve.png` — F1-Confidence 曲线
- `P_curve.png` — Precision-Confidence 曲线
- `R_curve.png` — Recall-Confidence 曲线
- `val_batch*.jpg` — 验证集预测样本

## 评估配置

- 推理尺寸: 640×640
- 验证耗时: {val_time:.1f}s
- 模型路径: `{model_path}`
- 数据配置: `{data_path}`

---

*本报告由 `scripts/evaluate_model.py` 自动生成。*
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估落石检测模型性能")
    parser.add_argument("--model", default=None,
                        help=".pt 模型路径 (默认: rockfall.config.MODEL_PATH)")
    parser.add_argument("--data", default=None,
                        help="数据集 data.yaml 路径 (默认: ../rock.v27i.yolov8/data.yaml)")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--batch", type=int, default=16, help="验证批次大小")
    parser.add_argument("--device", default=None, help="推理设备 (cpu / cuda:0)")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
