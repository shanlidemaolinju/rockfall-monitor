# 部署指南（简明版）

> 📖 **新版详细文档已发布**，请参阅：
> - [DEPLOYMENT.md](DEPLOYMENT.md) — 完整部署文档（含 Docker、手动部署、systemd、Nginx）
> - [OPERATIONS.md](OPERATIONS.md) — 运维手册（巡检、故障排查、备份恢复）
> - [USER_GUIDE.md](USER_GUIDE.md) — 用户手册（Web 看板、移动端、API）
>
> 本文档保留作为快速参考。

## 部署方式概览

RockGuard 支持三种部署形态，根据硬件环境和业务需求选择：

| 方式 | 适用场景 | 硬件要求 |
|------|---------|---------|
| Docker Compose | 服务器 7×24 运行 | CPU / GPU |
| Streamlit Cloud | 快速演示、无服务器 | 无 (云端CPU) |
| 本地 Python | 开发调试、离线环境 | CPU / GPU |

---

## 方式一: Docker Compose (推荐)

### 1. 准备

```bash
cd rockfall-system
cp .env.example .env
# 编辑 .env: 配置 PUSHPLUS_TOKEN、摄像头地址等
```

### 2. 构建并启动

```bash
# CPU 模式
docker compose up -d

# GPU 模式 (需 nvidia-docker)
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml up -d
```

服务端口:
- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

### 3. 查看日志

```bash
docker compose logs -f
```

---

## 方式二: Streamlit Cloud (一键部署)

1. Fork 项目到 GitHub
2. 在 [share.streamlit.io](https://share.streamlit.io) 连接仓库
3. 主文件路径: `app.py`
4. 在 Secrets 中配置环境变量 (对应 `.env.example` 中的变量)

**限制**: Streamlit Cloud 为纯 CPU 环境，推理较慢。建议设置:
- `DETECTION_IMG_SIZE=320`
- 演示时限制帧数 (`demo_max_frames=150`)

---

## 方式三: 本地 Python

### 依赖安装

```bash
# 核心依赖
pip install -r requirements-base.txt

# GPU (可选, 需 CUDA)
pip install -r requirements-gpu.txt
```

**注意**: `requirements-gpu.txt` 中的 `onnxruntime-gpu` 和 `tensorrt` 需要先安装 CUDA 版 PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### 模型准备

```bash
# 确认模型文件存在
ls -la models/rock_best.pt

# 验证模型
python scripts/validate_model.py
```

### 启动服务

```bash
# FastAPI 服务 (生产推荐)
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 2

# Streamlit (演示/管理)
streamlit run app.py --server.port 8501
```

---

## GPU 加速配置

### TensorRT

```bash
# 1. 导出 engine 文件
python scripts/export_tensorrt.py

# 2. 在 .env 中启用
TENSORRT_ENABLED=true
```

### CUDA 预处理 (MOG2 + Sobel)

在 `.env` 中设置:
```
USE_CUDA_PREPROCESS=true
```

需要 `opencv-contrib-python-headless` 的 CUDA 构建版。

---

## 摄像头配置

### RTSP 流

```env
CAMERA_URL=rtsp://admin:password@192.168.1.100:554/stream
RTSP_TRANSPORT=tcp
```

### USB 摄像头

```env
CAMERA_URL=0
```

### H.265 硬解码

```env
FFMPEG_EXTRA_OPTS=hwaccel|cuda
```

---

## 数据库配置

### 默认 (SQLite)

无需配置，自动使用 `data/alerts.db`。

### MySQL

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DATABASE=rock
```

---

## 点位管理

系统预置 4 个监测点位:

| 点位 ID | 名称 | 地区 |
|---------|------|------|
| nanning_naan_s1 | 南宁那安快速路 1 号边坡 | 广西南宁 |
| chongzuo_hena_s2 | 崇左合那高速 2 号边坡 | 广西崇左 |
| fangchenggang_lanhai_s3 | 防城港兰海高速 3 号边坡 | 广西防城港 |
| pingxiang_crossborder_s4 | 凭祥跨境公路 4 号边坡 | 广西凭祥 |

在 Streamlit 界面中可切换激活点位，或在 `.env` 中设置:
```env
ACTIVE_SITE_ID=nanning_naan_s1
```

---

## 健康检查

```bash
# FastAPI 健康检查
curl http://localhost:8000/health

# 模型验证
python scripts/validate_model.py

# 配置检查
python -c "from rockfall.config import validate_config; print(validate_config())"
```
