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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
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
    from rockfall.config import get_device
    device_str, device_name = get_device()
    print(f"[推理设备] {device_name} ({device_str})")

    # 禁用cuDNN + 预初始化CUDA (YOLO先拿GPU控制权)
    try:
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("QWidget { font-size: 14pt; }")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
