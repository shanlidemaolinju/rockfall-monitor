# RockGuard — 公路落石灾害监测预警系统

> v2.2.0 · 基于 YOLOv8 + MOG2 运动检测 + SORT 多目标跟踪 + FastSAM 边坡分割

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
│   ├── main.py               #   API 路由 (2000+ 行)
│   ├── service.py            #   业务逻辑层
│   ├── schemas.py            #   Pydantic 数据模型
│   └── templates/            #   前端页面模板
├── web/                      # React + TypeScript SPA 前端
│   ├── src/pages/            #   看板/地图/设置等页面
│   └── ...
├── desktop/                  # PyQt6 桌面应用
│   ├── main.py               #   启动入口
│   └── ui/                   #   界面组件
├── scripts/                  # 运维/部署脚本
│   ├── backup_db.sh          #   数据库备份
│   ├── rollback.sh           #   部署回滚
│   ├── setup_server.sh       #   服务器初始化
│   └── ...
├── tests/                    # 测试套件 (300+ 用例)
├── docs/                     # 文档中心
├── deploy/                   # 部署配置 (systemd/nginx)
├── docker/                   # Docker 部署
├── alembic/                  # 数据库迁移
├── models/                   # AI 模型文件
├── data/
│   ├── results/              #   检测结果输出
│   ├── uploads/              #   上传文件暂存
│   └── debug/                #   调试图片
├── pyproject.toml            #   项目元数据 (版本唯一来源)
├── requirements.in           #   直接依赖声明
├── requirements-lock.txt     #   锁定依赖 (pip-compile 生成)
├── constraints.txt           #   pip 约束 (opencv 冲突)
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

**GPU 加速**: 需要 CUDA Toolkit + PyTorch CUDA 版本。RTX 4060 测试通过。

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境 (推荐 Python 3.11)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 安装 PyTorch (CUDA 版本, GPU 用户)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 安装项目依赖 (二选一)
pip install -c constraints.txt -r requirements-lock.txt   # 精确锁定版本 (生产推荐)
pip install -e .                                           # 开发模式 (pyproject.toml)
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
| `SKIP_IDLE` / `SKIP_ACTIVE` / `SKIP_CRITICAL` | 5 / 5 / 2 | 三级自适应跳帧间隔 |
| `MOTION_SCORE_LOW` / `MOTION_SCORE_HIGH` | 0.01 / 0.1 | 运动显著性阈值 |
| `ALERT_RED_CONFIDENCE` | 0.6 | 红色预警置信度 |
| `ALERT_RED_AREA_RATIO` | 0.02 | 红色预警面积占比 (2%) |
| `FALLING_Y_ACCEL_THRESHOLD` | 7.5 | 坠落加速度阈值 (px/frame²) |
| `TRACK_MIN_CONFIRM` | 3 | 轨迹确认帧数 (落石出现仅 3-5 帧) |
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

## 开发指南

### 版本管理

- 版本号**唯一来源**：[`rockfall/__init__.py`](rockfall/__init__.py) 的 `__version__ = "2.2.0"`
- 所有模块（`server/main.py`、`sentry_init.py`、`metrics.py`、`app.py`、`pyproject.toml`）从 `__version__` 读取
- 发版时只需修改一处，全项目同步

### 依赖管理

```bash
# 1. 编辑直接依赖
vim requirements.in

# 2. 重新生成锁定文件 (Python 3.11)
pip-compile -c constraints.txt requirements.in --output-file requirements-lock.txt

# 3. 安装
pip install -c constraints.txt -r requirements-lock.txt
```

- `requirements.in` — 直接依赖声明（手动编辑）
- `requirements-lock.txt` — `pip-compile` 生成的精确锁定文件（不要手动编辑）
- `constraints.txt` — pip 约束，解决 `opencv-python` / `opencv-python-headless` 冲突
- `pyproject.toml` — 项目元数据和可选依赖

### CI/CD

GitHub Actions 自动执行：Ruff lint → Pyright 类型检查 → pytest (SQLite + MySQL) → 前端构建 → 部署到腾讯云。

## 文档

| 文档 | 说明 |
|------|------|
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | 完整部署文档（环境要求、Docker/手动部署、systemd + Nginx、验证） |
| [OPERATIONS.md](docs/OPERATIONS.md) | 运维手册（每日巡检清单、7 个常见故障排查、备份恢复、性能调优） |
| [USER_GUIDE.md](docs/USER_GUIDE.md) | 用户手册（Web SPA 看板、移动端 H5、API 文档、工单流转、比赛展示建议） |
| [API.md](docs/API.md) | API 参考（端点列表、认证说明、项目结构） |
| [DEPLOY.md](docs/DEPLOY.md) | 部署指南简明版（快速参考） |

## 许可证

Educational Use — 钦州监测点课程设计项目
