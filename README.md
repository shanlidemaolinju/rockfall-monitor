# 落石检测与预警系统

> 钦州监测点 — 基于 YOLOv8 + MOG2 运动检测 + SORT 多目标跟踪的实时落石监测方案

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     输入层                              │
│   视频文件 (.mp4)  │  RTSP 网络摄像头  │  USB 摄像头     │
└────────┬────────────────┬─────────────────┬────────────┘
         │                │                  │
         ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                   算法流水线                             │
│  MOG2 背景减法 → 运动显著性评分 → YOLOv8 检测 →         │
│  SORT 多目标跟踪 (Kalman 9D + IoU) → 三级预警分级       │
└────────┬────────────────────────────────────┬───────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────┐            ┌─────────────────────┐
│  PyQt6 桌面应用     │            │  FastAPI Web 服务    │
│  · 实时视频预览     │            │  · MJPEG 实时流      │
│  · 检测框+轨迹叠加  │            │  · 仪表盘看板        │
│  · ROI 区域框选     │            │  · 统计/预警 API     │
│  · 异步 YOLO 推理   │            │  · Docker 部署       │
└────────┬────────────┘            └──────────┬──────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────────────────────────────────────────┐
│                  PushPlus 微信报警推送                   │
│   三级预警 (红/黄/绿) · 异步线程池 · 指数退避重试        │
└─────────────────────────────────────────────────────────┘
```

## 核心特性

| 模块 | 功能 |
|------|------|
| **运动检测** | MOG2 背景减法 + 运动显著性三级评分 + 自适应背景重置 |
| **目标检测** | YOLOv8 (mAP50=0.72) + 自适应跳帧 (无运动/弱运动/强运动 → 15/5/2 帧) |
| **多目标跟踪** | SORT 算法，Kalman 9D 状态向量 [x,y,s,r,vx,vy,vs,ax,ay]，IoU 匈牙利匹配 |
| **运动分类** | 基于 Kalman Y轴加速度 → 静止 / 缓慢滚动 / 快速移动 / 快速坠落 |
| **预警分级** | 三级 (红/黄/绿) — 置信度 + 面积占比 + 坠落状态综合判断 |
| **微信推送** | PushPlus API，连续帧确认，base64 图片嵌入，指数退避重试 |
| **桌面应用** | PyQt6，检测/显示分离 (DetectionWorker QThread + 主线程 QTimer) |
| **Web 看板** | FastAPI + Jinja2，MJPEG 实时流，统计看板，预警列表 |
| **Docker 部署** | Dockerfile + docker-compose + Nginx 反向代理 |

## 目录结构

```
rockfall-system/
├── rockfall/                 # 核心算法库
│   ├── config.py             #   统一配置 (环境变量 > .env > 默认值)
│   ├── detector.py           #   YOLO + MOG2 + SORT 流水线
│   ├── tracker.py            #   Kalman 9D 跟踪器 + 运动状态分类
│   ├── notifier.py           #   PushPlus 微信推送 (异步)
│   └── logger.py             #   JSONL 事件日志
├── desktop/                  # PyQt6 桌面应用
│   ├── main.py               #   启动入口
│   └── ui/
│       ├── main_window.py    #   主窗口布局
│       └── video_widget.py   #   视频控件 (V4 异步架构)
├── server/                   # FastAPI Web 服务
│   ├── main.py               #   API 路由
│   ├── service.py            #   业务逻辑
│   └── templates/
│       └── dashboard.html    #   Web 仪表盘
├── scripts/
│   └── test_detection.py     #   功能测试脚本
├── docker/                   # Docker 部署
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── models/
│   └── rock_best.pt          # YOLOv8 模型权重
├── data/
│   ├── rock.jpg              #   默认测试图片
│   ├── results/              #   检测结果输出
│   └── uploads/              #   上传文件暂存
├── requirements.txt
├── .env.example              #   配置模板
└── README.md
```

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 (CUDA 推荐) |
| OpenCV | ≥ 4.10 |
| PyQt6 | ≥ 6.7 |
| FastAPI | ≥ 0.115 |
| Ultralytics | ≥ 8.3 |

**GPU 加速**: 需要 CUDA  Toolkit + PyTorch CUDA 版本。RTX 4060 测试通过。

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
conda create -n rockfall python=3.10
conda activate rockfall

# 安装 PyTorch (CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，至少填入 PushPlus Token
# PUSHPLUS_TOKEN=your_token_here
```

### 3. 放置模型

将训练好的 YOLO 模型 `rock_best.pt` 放入 `models/` 目录。

### 4. 运行测试

```bash
python scripts/test_detection.py
```

## 使用方式

### 桌面应用 (实时监控)

```bash
python -m desktop.main
# 或
python desktop/main.py
```

- **选择视频**: 打开本地 .mp4 文件
- **RTSP 摄像头**: 输入 RTSP 地址连接网络摄像头
- **USB 摄像头**: 直接打开 USB 摄像头
- **ROI 框选**: 点击「ROI 框选区域」后在视频上拖拽鼠标
- **模拟预警**: 测试微信推送是否正常

### Web 服务 (远程监控看板)

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

浏览器访问:
- `http://localhost:8000/` — Web 仪表盘 (实时视频流 + 统计)
- `http://localhost:8000/docs` — Swagger API 文档
- `http://localhost:8000/api/stream.mjpeg?token=your_token` — MJPEG 视频流

### Docker 部署

```bash
cd docker
docker-compose up -d
# Web 看板: http://localhost:8008
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DETECTION_CONFIDENCE` | 0.3 | YOLO 置信度阈值 |
| `SKIP_IDLE` / `SKIP_ACTIVE` / `SKIP_CRITICAL` | 15 / 5 / 2 | 三级自适应跳帧间隔 |
| `MOTION_SCORE_LOW` / `MOTION_SCORE_HIGH` | 0.01 / 0.1 | 运动显著性阈值 |
| `ALERT_RED_CONFIDENCE` | 0.6 | 红色预警置信度 |
| `ALERT_RED_AREA_RATIO` | 0.02 | 红色预警面积占比 (2%) |
| `FALLING_Y_ACCEL_THRESHOLD` | 7.5 | 坠落加速度阈值 (px/frame²) |
| `TRACK_MIN_CONFIRM` | 5 | 轨迹确认帧数 |
| `MOG2_DETECT_SHADOWS` | false | 是否检测阴影 |
| `CAMERA_RECONNECT_BASE` / `_MAX` | 5 / 30 | RTSP 断线重连参数 |

完整参数见 `.env.example`。

## 技术栈

- **检测**: YOLOv8 (Ultralytics)
- **跟踪**: SORT (Kalman 9D + IoU Hungarian)
- **运动**: MOG2 背景减法 + 显著性评分
- **桌面**: PyQt6 (检测/显示分离架构)
- **Web**: FastAPI + Jinja2 + MJPEG
- **推送**: PushPlus 微信 API
- **部署**: Docker + Nginx

## 许可证

Educational Use — 钦州监测点课程设计项目
