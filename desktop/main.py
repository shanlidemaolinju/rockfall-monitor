"""
启动入口 — 桌面应用
==================
运行方式: python -m desktop.main  或  python desktop/main.py
"""

import os
import sys

# ---- Windows 稳定性: 必须在所有其他 import 之前设置 ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# 彻底禁用OpenCV GPU加速
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_CUDA_USE_HOST_MEMORY"] = "0"
os.environ["OPENCV_CUDA_DISABLE"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

from pathlib import Path

# 确保项目根目录在 sys.path 中，方便 from rockfall import ...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6 import QtWidgets

from desktop.ui.main_window import MainWindow


def main():
    # ---- CUDA 必须最先初始化 (在任何 desktop/rockfall import 之前) ----
    # FastSAM 是第一个触碰 CUDA 的模块; 如果 CUDA 未正确初始化,
    # FastSAM 内部会创建损坏的 CUDA 上下文, 导致后续 STATUS_STACK_BUFFER_OVERRUN
    # 注意: 切勿禁用 cuDNN — RTX 4060 (Ada Lovelace) 上备选 CUDA 路径
    # 可能导致栈缓冲区溢出 (0xC0000409)
    try:
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.init()                    # ← 必须在 FastSAM 之前!
            torch.cuda.empty_cache()
    except Exception:
        pass

    from rockfall.trace import set_session_id
    from rockfall.config import get_device

    # 生成本次桌面会话 ID（所有日志/告警可溯源）
    set_session_id()

    device_str, device_name = get_device()
    print(f"[推理设备] {device_name} ({device_str})")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("QWidget { font-size: 14pt; }")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
