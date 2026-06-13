# RockGuard 部署文档

> 公路落石灾害监测预警系统 — 从零开始部署指南
>
> 目标：一个新开发者按照本文档能在 **30 分钟内**成功运行系统。

---

## 目录

- [1. 环境要求](#1-环境要求)
- [2. 快速部署（Docker，推荐）](#2-快速部署docker推荐)
- [3. 手动部署（ Linux / Windows ）](#3-手动部署-linux--windows)
- [4. 生产环境部署（systemd + Nginx）](#4-生产环境部署systemd--nginx)
- [5. 前端构建部署](#5-前端构建部署)
- [6. 验证部署](#6-验证部署)
- [7. 部署后首次使用导航](#7-部署后首次使用导航)
- [8. 多机分布式部署](#8-多机分布式部署)

---

## 1. 环境要求

### 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 (x86_64) | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 20 GB（不含模型） | 50 GB SSD |
| GPU（可选） | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060+ |
| 网络 | 出口带宽 10 Mbps | 出口带宽 100 Mbps |

### 软件要求

| 软件 | 版本 | 用途 |
|------|------|------|
| Ubuntu | 22.04 LTS（推荐）/ 20.04 LTS | 操作系统 |
| Python | 3.10 ~ 3.12 | 后端运行环境 |
| Node.js | 20 LTS | 前端构建 |
| Docker | 24+（可选） | 容器化部署 |
| MySQL | 8.0（可选） | 生产数据库 |

### GPU 加速（可选）

如需 GPU 推理加速，额外需要：

| 组件 | 版本 |
|------|------|
| NVIDIA 驱动 | 525+ |
| CUDA | 12.1+ |
| cuDNN | 8.9+ |
| TensorRT | 8.6+（可选，2-3x 加速） |

---

## 2. 快速部署（Docker，推荐）

> 预计耗时：**10 分钟**
>
> 适用场景：服务器 7×24 运行、快速验证、无 GPU 环境

### 2.1 准备

```bash
# 1. 克隆代码
git clone <your-repo-url> /opt/rockfall
cd /opt/rockfall

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少修改以下变量：
#   PUSHPLUS_TOKEN=your_token_here    (微信推送，可选)
#   API_KEY=your_api_key_here         (API 认证，推荐)
#   AUTH_JWT_SECRET=your_jwt_secret   (JWT 签名，推荐)
```

### 2.2 启动服务（CPU 模式）

```bash
# 构建镜像并启动
docker compose up -d

# 查看日志
docker compose logs -f rockfall

# 等待健康检查通过（约 60 秒）
docker ps  # 确认 STATUS 为 healthy
```

### 2.3 启动服务（GPU 模式）

```bash
# 需要先安装 nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# 方式一: 使用 GPU 叠加文件（推荐）
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml up -d

# 方式二: 取消根 docker-compose.yml 中 deploy 段的注释，然后：
# docker compose up -d
```

### 2.4 启用 MySQL（可选）

编辑 `docker-compose.yml`，取消 `mysql` 服务注释，并在 `.env` 中添加：

```env
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_USER=rockfall
MYSQL_PASSWORD=rockfall_password
MYSQL_DATABASE=rockfall
```

然后：

```bash
docker compose up -d
```

### 2.5 分离式部署（Nginx 反向代理）

生产环境推荐使用 `docker/` 目录下的分离式 Compose：

```bash
# 使用独立 Nginx 反向代理
docker compose -f docker/docker-compose.yml up -d
```

服务端口：

| 服务 | 端口 | 说明 |
|------|------|------|
| FastAPI | 8001 | 后端 API（Nginx 代理后） |
| Nginx | 8008 | 反向代理 + 静态文件 |
| MySQL（可选） | 3306 | 数据库 |

### 2.6 Docker 常用命令

```bash
# 查看日志
docker compose logs -f

# 重启服务
docker compose restart

# 停止服务
docker compose down

# 重新构建镜像
docker compose build --no-cache
docker compose up -d

# 进入容器调试
docker compose exec rockfall bash
```

---

## 3. 手动部署（ Linux / Windows ）

> 预计耗时：**25-30 分钟**
>
> 适用场景：开发调试、离线环境、需要 GPU 加速

### 3.1 系统依赖安装

**Ubuntu 22.04：**

```bash
# 系统包
sudo apt update
sudo apt install -y \
  python3.10 python3.10-venv python3.10-dev \
  libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
  libgomp1 ffmpeg \
  git curl wget

# Node.js 20（前端构建用）
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# 验证
python3.10 --version  # Python 3.10.x
node --version         # v20.x.x
```

**Windows（开发用）：**

```powershell
# 1. 安装 Python 3.10：https://www.python.org/downloads/
# 2. 安装 Node.js 20：https://nodejs.org/
# 3. 安装 Git：https://git-scm.com/
# 4. 安装 ffmpeg：winget install ffmpeg
```

### 3.2 Python 虚拟环境

```bash
cd /opt/rockfall

# 创建虚拟环境
python3.10 -m venv venv

# 激活虚拟环境
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows (PowerShell 用 .\venv\Scripts\Activate.ps1)

# 升级 pip
pip install --upgrade pip setuptools wheel
```

### 3.3 安装依赖

```bash
# 核心依赖（必装）
pip install -r requirements-base.txt

# 开发依赖（含 pytest、alembic）
pip install -r requirements-dev.txt

# GPU 加速依赖（需先装 CUDA 版 PyTorch）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# pip install -r requirements-gpu.txt
```

### 3.4 配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑 .env 文件（用 vim / nano / 记事本）
# 必填项：
#   PUSHPLUS_TOKEN=your_token      (微信推送 Token，可选)
#   API_KEY=your_api_key            (API 认证 Key，推荐设置)
#   AUTH_JWT_SECRET=your_jwt_secret (JWT 签名密钥，推荐设置)

# 可选 MySQL：
#   MYSQL_HOST=localhost
#   MYSQL_PORT=3306
#   MYSQL_USER=root
#   MYSQL_PASSWORD=123456
#   MYSQL_DATABASE=rock
```

> 详细配置参数说明参见 `.env.example` 中的注释，共约 220 行，覆盖 100+ 可调参数。

### 3.5 模型文件准备

```bash
# 确认 YOLO 模型文件存在
ls -la models/rock_best.pt

# 验证模型完整性
python scripts/validate_model.py

# （可选）FastSAM 分割模型
# wget -O FastSAM-x.pt https://huggingface.co/...
```

### 3.6 启动服务

**方式一：FastAPI（生产推荐）**

```bash
# 单 Worker（开发调试）
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

# 多 Worker（生产）
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 2
```

**方式二：Streamlit（演示/管理）**

```bash
streamlit run app.py --server.port 8501
```

**方式三：前端开发服务器**

```bash
cd web
npm install
npm run dev          # 监听 http://localhost:3000，API 代理到 :8000
```

---

## 4. 生产环境部署（systemd + Nginx）

### 4.1 系统用户创建

```bash
sudo useradd -r -s /bin/false rockfall
sudo mkdir -p /opt/rockfall
sudo chown -R rockfall:rockfall /opt/rockfall
```

### 4.2 安装 systemd 服务

```bash
# 复制服务文件
sudo cp deploy/rockfall.service /etc/systemd/system/rockfall.service

# 修改服务文件中的路径（如需要）
sudo vim /etc/systemd/system/rockfall.service

# 重新加载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start rockfall

# 设置开机自启
sudo systemctl enable rockfall

# 查看状态
sudo systemctl status rockfall

# 查看日志
sudo journalctl -u rockfall -f
```

### 4.3 配置 Nginx 反向代理

```bash
# 复制 Nginx 配置
sudo cp deploy/nginx.conf /etc/nginx/sites-available/rockfall

# 修改域名
sudo vim /etc/nginx/sites-available/rockfall
# 将 your-domain.com 替换为实际域名

# 添加限流区域到 nginx.conf 的 http 块
sudo vim /etc/nginx/nginx.conf
# 添加：limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# 启用站点
sudo ln -s /etc/nginx/sites-available/rockfall /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重载 Nginx
sudo systemctl reload nginx
```

### 4.4 配置 SSL 证书（Let's Encrypt）

```bash
# 安装 certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 测试自动续期
sudo certbot renew --dry-run
```

### 4.5 配置日志轮转

```bash
# 安装 logrotate 配置
sudo cp deploy/rockfall-logrotate.conf /etc/logrotate.d/rockfall

# 测试配置语法
sudo logrotate -d /etc/logrotate.d/rockfall

# 手动强制执行一次
sudo logrotate -f /etc/logrotate.d/rockfall
```

---

## 5. 前端构建部署

### 5.1 构建 React SPA

```bash
cd web

# 安装依赖
npm install

# 构建生产版本
npm run build

# 构建产物在 server/static/ 目录
# FastAPI 会自动托管这些静态文件
ls -la ../server/static/
```

### 5.2 前端环境变量

构建时通过 Vite 环境变量配置，在 `web/` 目录下创建 `.env` 文件：

```bash
# web/.env（前端构建时使用，与后端 .env 独立）
VITE_API_BASE_URL=https://your-domain.com
VITE_SENTRY_DSN=https://xxx@sentry.ingest.sentry.io/xxx   # 可选
```

---

## 6. 验证部署

### 6.1 健康检查

```bash
# 基础健康检查
curl http://localhost:8000/health
# 期望: {"status":"healthy"}

# K8s Liveness
curl http://localhost:8000/health/live
# 期望: {"status":"alive"}

# K8s Readiness（含 GPU/DB/模型检查）
curl http://localhost:8000/health/ready
# 期望: {"status":"ready","checks":{"database":"ok","model":"ok","gpu":"available"}}

# 完整系统健康
curl http://localhost:8000/api/health/full
```

### 6.2 功能验证

```bash
# 1. Web 页面
curl -I http://localhost:8000/
# 期望: HTTP 200

# 2. 移动端页面
curl -I http://localhost:8000/m
# 期望: HTTP 200

# 3. API 文档
curl -I http://localhost:8000/docs
# 期望: HTTP 200（Swagger UI）

# 4. 检测接口
curl -X POST http://localhost:8000/detect/image \
  -F "file=@tests/test_data/sample.jpg" \
  -H "X-Api-Key: your_api_key"
# 期望: JSON 检测结果

# 5. 统计接口
curl http://localhost:8000/api/stats
# 期望: JSON 统计数据

# 6. 预警列表
curl http://localhost:8000/api/alerts?limit=10
# 期望: JSON 预警列表
```

### 6.3 模型验证

```bash
# 验证模型加载
python scripts/validate_model.py

# 运行单元测试
python -m pytest tests/ -v --timeout=60
```

---

## 7. 部署后首次使用导航

> 新手引导：部署完成后按以下顺序快速上手系统。

### 第一步：确认服务运行

```bash
curl http://localhost:8000/health
# 返回 {"status":"healthy"} 即正常
```

### 第二步：打开关键页面

| 顺序 | 页面 | 地址 | 做什么 |
|------|------|------|--------|
| 1 | **API 文档** | `http://your-server:8000/docs` | 确认所有 API 端点可见，试调 `/api/stats` |
| 2 | **数据大屏** | `http://your-server:8000/` | 确认 React 前端加载、统计卡片正常 |
| 3 | **经典看板** | `http://your-server:8000/classic` | 兜底页面（如前端未构建则自动显示） |
| 4 | **移动端** | `http://your-server:8000/m` | 手机扫码确认 H5 正常 |

### 第三步：功能自检

```bash
# 1. 验证模型加载正常
python scripts/validate_model.py

# 2. 图片检测（用内置测试图或任意图片）
curl -X POST http://localhost:8000/detect/image \
  -F "file=@tests/test_data/sample.jpg" \
  -H "X-Api-Key: ${API_KEY}"

# 3. 确认点位已加载
curl http://localhost:8000/api/sites
# 应返回 4 个预设监测点位

# 4. 确认预警存储正常
curl http://localhost:8000/api/alerts?limit=5
```

### 第四步：配置实际监测

1. 在 `.env` 中填入实际的 `CAMERA_URL`（RTSP 地址或视频文件路径）
2. 如使用 RTSP 摄像头，确认网络可达：`ffprobe rtsp://admin:password@ip:554/stream`
3. 激活对应点位：`POST /api/sites/switch {"site_id": "nanning_naan_s1"}`
4. 打开数据大屏观察实时画面和检测结果
5. 配置推送通知（可选）：`PUSHPLUS_TOKEN` 或 `ALERT_CHANNEL_MAP`

### 常见新手问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 前端白屏 | React 未构建，JS 资源 404 | 执行 `cd web && npm install && npm run build` |
| 图片检测报 401 | API Key 未设置或未携带 | `.env` 中设置 `API_KEY`，请求带 `X-Api-Key` 头 |
| 检测结果为空 | 置信度阈值高或图片不含落石 | 调低 `DETECTION_CONFIDENCE=0.15` 测试 |
| 首页显示经典看板 | React 构建产物不在 `server/static/` | 构建前端后重启服务 |

---

## 8. 多机分布式部署

### 7.1 架构示意

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  边缘设备 1   │────▶│              │────▶│  告警通知    │
│  (Jetson)    │     │  云端 API    │     │  (微信/邮件) │
└──────────────┘     │  (FastAPI)   │     └──────────────┘
                     │              │
┌──────────────┐     │  MySQL +     │
│  边缘设备 2   │────▶│  Nginx 反向  │
│  (RK3588)    │     │  代理        │
└──────────────┘     └──────────────┘
```

### 7.2 边缘端配置

在边缘设备（Jetson/RK3588）的 `.env` 中：

```env
# 启用边缘-云协同
EDGE_CLOUD_ENDPOINT=https://cloud.example.com
EDGE_API_KEY=your_edge_api_key
EDGE_NANO_MODEL_PATH=models/rock_nano.pt
EDGE_MOTION_THRESHOLD=0.005
EDGE_UPLOAD_QUALITY=60
EDGE_MAX_FPS=5
```

### 7.3 云端配置

云端 API 自动接受 `/api/edge/upload` 端点上传的可疑帧，进行高精度二次检测。

---

## 附录 A：端口清单

| 端口 | 服务 | 协议 | 说明 |
|------|------|------|------|
| 8000 | FastAPI | HTTP/WS/SSE | 后端 API |
| 8501 | Streamlit | HTTP | 演示/管理界面 |
| 3000 | Vite | HTTP | 前端开发服务器 |
| 3306 | MySQL | TCP | 数据库 |
| 8008 | Nginx | HTTP | Docker 反向代理 |

## 附录 B：目录说明

```
/opt/rockfall/
├── .env                    # 环境配置（不可提交到 Git）
├── rockfall/               # 核心 Python 包
├── server/                 # FastAPI Web 服务
│   ├── main.py             # 路由定义
│   ├── static/             # 前端构建产物
│   └── templates/          # HTML 模板
├── web/                    # React 前端源码
├── models/                 # AI 模型文件 (rock_best.pt 等)
├── data/                   # 运行时数据
│   ├── results/            # 截图、视频片段
│   ├── uploads/            # 上传文件
│   └── alerts.db           # SQLite 数据库（默认）
├── deploy/                 # 部署配置
├── docker/                 # Docker 配置
├── tests/                  # 测试套件
└── scripts/                # 工具脚本
```
