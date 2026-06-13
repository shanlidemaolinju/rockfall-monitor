# RockGuard — 公路落石灾害监测预警系统

> **RockGuard v2.0.0** | 广西+东盟公路自然灾害监测预警平台
>
> 基于 YOLO + MOG2 + SORT 的实时落石检测系统，支持四级分级预警、多监测点位管理、Web/桌面/Streamlit 多端部署。

---

## 功能特性

- **实时检测**: MOG2 背景建模 + YOLO 目标检测 + SORT (Kalman+IoU) 多目标跟踪
- **四级预警**: 对齐《公路自然灾害监测预警系统技术指南》，红/橙/黄/蓝四级分级
- **运动分析**: 支持静止/滚动/坠落三种运动状态分类
- **边缘增强**: Sobel 边缘增强补偿运动模糊 (苏国韶 2025)
- **三帧差分**: 运动滤波剔除静态误检
- **SAHI 切片**: 高分辨率帧分块推理，适合远距离小目标
- **概率融合**: YOLO 置信度 + MOG2 前景证据加权增强
- **TensorRT**: NVIDIA GPU 上 2-3x 推理加速
- **多点位管理**: 支持 4 个预设监测站点（南宁、崇左、防城港、凭祥）

## 项目结构

```
rockfall-system/
├── rockfall/              # 核心 Python 包
│   ├── config.py          # 配置层 (所有参数从 .env 读取)
│   ├── detector.py        # 算法层: MOG2 + YOLO + SORT 流水线
│   ├── tracker.py         # 跟踪层: Kalman 多目标跟踪
│   ├── edge_enhance.py    # 预处理: Sobel 边缘增强
│   ├── motion_detect.py   # 预处理: 三帧差分 & MOG2 中心点滤波
│   ├── sahi.py            # 推理: SAHI 切片辅助推理
│   ├── fusion.py          # 后处理: 概率融合 & 时序确认
│   ├── fastsam_road.py    # 分割: FastSAM 道路/边坡分割
│   ├── notifier.py        # 通知: PushPlus 微信推送
│   ├── alert_store.py     # 存储: 预警记录 (MySQL/SQLite)
│   ├── logger.py          # 日志: 检测事件 JSONL 持久化
│   ├── site_config.py     # 配置: 多监测点位管理
│   └── utils.py           # 工具: IoU 计算, Excel 导出
├── scripts/               # 工具脚本
│   ├── validate_model.py  # 模型加载验证
│   ├── evaluate_model.py  # 性能评估报告 (mAP, FPS, 可视化)
│   ├── export_onnx.py     # ONNX 模型导出
│   ├── export_tensorrt.py # TensorRT 引擎导出
│   └── generate_demo.py   # 演示数据生成
├── server/                # FastAPI Web 服务
├── desktop/               # PyQt6 桌面应用
├── tests/                 # 单元测试 (pytest)
├── models/                # 模型文件
├── data/                  # 运行时数据 (results, uploads, masks)
├── docs/                  # 项目文档
├── .env.example           # 环境变量模板
├── requirements-base.txt  # 核心依赖
├── requirements-dev.txt   # 开发依赖
├── requirements-gpu.txt   # GPU 加速依赖
└── app.py                 # Streamlit Web 封装 (主入口)
```

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
cd rockfall-system

# 安装核心依赖
pip install -r requirements-base.txt

# (可选) 安装开发依赖
pip install -r requirements-dev.txt

# (可选) NVIDIA GPU 加速
# 先安装 CUDA 版 PyTorch, 再:
pip install -r requirements-gpu.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件, 至少配置:
#   PUSHPLUS_TOKEN=your_token     (可选, 微信推送)
#   ROCK_MODEL_PATH=models/rock_best.pt
```

### 3. 启动应用

```bash
# Streamlit Web 界面 (推荐)
streamlit run app.py

# FastAPI HTTP 服务
uvicorn server.main:app --host 0.0.0.0 --port 8000

# 桌面应用
python -m desktop.main
```

### 4. 模型验证

```bash
# 验证模型完整性
python scripts/validate_model.py

# 生成性能评估报告
python scripts/evaluate_model.py
```

## 模型部署

```bash
# 导出 ONNX (跨平台 CPU/GPU 通用)
python scripts/export_onnx.py

# 导出 TensorRT (NVIDIA GPU 加速 2-3x)
python scripts/export_tensorrt.py
```

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_tracker.py -v
```

## 配置参数

所有检测参数通过 `.env` 文件控制，支持运行时调整。详见 `.env.example` 中的完整注释。

| 参数类别 | 关键参数 |
|---------|---------|
| 检测 | DETECTION_CONFIDENCE, DETECTION_IMG_SIZE |
| 预警 | ALERT_BLUE/YELLOW/ORANGE_CONFIDENCE_* |
| 跳帧 | SKIP_IDLE, SKIP_ACTIVE, SKIP_CRITICAL |
| MOG2 | MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_LEARNING_RATE |
| 跟踪 | TRACK_MIN_CONFIRM, TRACK_IOU_THRESHOLD |
| 增强 | EDGE_ENHANCE_ENABLED, TFD_ENABLED, SAHI_ENABLED |

## 许可证

© 2026 RockGuard Team. All rights reserved.
