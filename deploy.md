# 落石监测系统 — Linux 服务器部署指南

## 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| OS | Ubuntu 20.04+ / Debian 11+ / CentOS 8+ | Ubuntu 22.04 LTS |
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 20 GB (不含模型) | 50 GB SSD |
| GPU | 无 (CPU推理) | NVIDIA GTX 1060 6GB+ |
| Python | 3.10+ | 3.11+ |
| 网络 | 公网IP/域名 (推送用) | 固定IP + 域名 + SSL |

## 快速部署 (3 步上线)

### 方式一: Docker 一键部署 (推荐)

```bash
# 1. 克隆项目到服务器
git clone <repo-url> /opt/rockfall
cd /opt/rockfall

# 2. 配置环境变量
cp .env.example .env
vim .env   # 至少配置: PUSHPLUS_TOKEN, API_KEY

# 3. 一键启动
docker-compose up -d

# 验证
curl http://localhost:8000/health
```

浏览器访问 `http://<服务器IP>:8000` 即可使用。

### 方式二: 传统部署

```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx ffmpeg

# 2. 创建虚拟环境
python3 -m venv /opt/rockfall/venv
source /opt/rockfall/venv/bin/activate
pip install -r requirements.txt

# 3. 放置模型文件
# 将 rock_best.pt 放到 models/
# 将 FastSAM-x.pt 放到项目根目录
ls models/rock_best.pt    # 必须存在
ls FastSAM-x.pt           # 必须存在

# 4. 配置环境
cp .env.example .env
vim .env

# 5. 安装系统服务
sudo cp deploy/rockfall.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rockfall
sudo systemctl start rockfall

# 6. 配置 Nginx 反向代理
sudo cp deploy/nginx.conf /etc/nginx/sites-available/rockfall
sudo ln -s /etc/nginx/sites-available/rockfall /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## 环境变量说明 (.env)

```bash
# ── 必须配置 ──
PUSHPLUS_TOKEN=your_token_here       # PushPlus 微信推送 token
API_KEY=your_api_key_here             # API 鉴权密钥
STREAM_TOKEN=your_stream_token        # MJPEG 视频流鉴权

# ── 推荐配置 ──
WEB_HOST=0.0.0.0
WEB_PORT=8000
LOCATION=南宁那安快速路1号边坡         # 默认监测点位
ACTIVE_SITE_ID=nanning_naan_s1        # 启动时激活的点位

# ── MySQL (可选, 不配置则用 SQLite) ──
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=rockfall
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=rockfall

# ── GPU 加速 (可选) ──
# USE_CUDA_PREPROCESS=true            # CUDA MOG2 预处理
# TENSORRT_ENABLED=true               # TensorRT 推理加速
# TENSORRT_MODEL_PATH=models/rock_best.engine

# ── 检测参数 (可选, 有默认值) ──
DETECTION_CONFIDENCE=0.3
ALERT_BLUE_CONFIDENCE_HIGH=0.5
ALERT_YELLOW_CONFIDENCE_HIGH=0.7
ALERT_ORANGE_CONFIDENCE_HIGH=0.9
```

## 目录结构

```
/opt/rockfall/
├── rockfall/           # 核心算法库
│   ├── detector.py     # YOLO+MOG2+SORT 检测流水线
│   ├── fastsam_road.py # FastSAM 道路/边坡分割
│   ├── alert_store.py  # 预警持久化 (MySQL/SQLite)
│   ├── site_config.py  # 多点位管理
│   └── config.py       # 配置中心
├── server/             # Web 服务层
│   ├── main.py         # FastAPI 路由
│   ├── service.py      # 业务逻辑
│   ├── schemas.py      # Pydantic 数据模型
│   └── templates/      # 前端页面
│       └── dashboard.html
├── models/             # AI 模型
│   └── rock_best.pt   # YOLO 落石检测模型
├── FastSAM-x.pt        # FastSAM 分割模型
├── data/               # 运行时数据 (预警DB/截图/日志)
├── docker-compose.yml  # Docker 编排
├── Dockerfile          # Docker 镜像
├── deploy/             # 部署配置
│   ├── rockfall.service # systemd 单元
│   └── nginx.conf       # Nginx 配置
├── requirements.txt
├── .env                # 环境变量
└── deploy.md           # 本文档
```

## 运维命令

```bash
# 服务管理
sudo systemctl start rockfall       # 启动
sudo systemctl stop rockfall        # 停止
sudo systemctl restart rockfall     # 重启
sudo systemctl status rockfall      # 状态
sudo journalctl -u rockfall -f      # 实时日志

# Docker 管理
docker-compose up -d                # 启动
docker-compose down                 # 停止
docker-compose logs -f              # 日志
docker-compose restart              # 重启

# 健康检查
curl http://localhost:8000/health
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/sites

# 配置热重载 (无需重启)
curl -X POST http://localhost:8000/api/config/reload

# 数据库备份 (SQLite)
cp data/alerts.db data/alerts_$(date +%Y%m%d).db

# 数据库备份 (MySQL)
mysqldump -u rockfall -p rockfall > alerts_$(date +%Y%m%d).sql
```

## SSL/HTTPS 配置 (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 性能调优

### GPU 推理
```bash
# .env 中启用
USE_CUDA_PREPROCESS=true
TENSORRT_ENABLED=true

# 导出 TensorRT 引擎 (首次部署)
python scripts/export_tensorrt.py
```

### 多路摄像头
系统支持同时处理多路摄像头，每路独立 MOG2/跟踪器状态。
通过 `/api/cameras` 接口管理，对每路分配不同的 `camera_id`。

### MySQL 生产配置
```ini
# /etc/mysql/mysql.conf.d/mysqld.cnf
[mysqld]
max_connections = 50
innodb_buffer_pool_size = 256M
```

## 安全加固

1. **修改默认 API Key**: `.env` 中设置强密码
2. **防火墙**: 仅开放 80/443 端口，8000 端口仅本地监听
3. **Nginx**: 配置访问限流、请求体大小限制
4. **定期备份**: 设置 cron 定时备份数据库
5. **日志轮转**: 配置 logrotate 管理检测日志

## 常见问题

**Q: 启动后无法访问网页?**
```bash
# 检查端口占用
sudo netstat -tlnp | grep 8000
# 检查防火墙
sudo ufw status
# 查看日志
sudo journalctl -u rockfall -n 50
```

**Q: GPU 不可用?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 若输出 False, 安装 CUDA 版 PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q: 模型文件找不到?**
```bash
ls models/rock_best.pt     # YOLO 检测模型 (~6MB)
ls FastSAM-x.pt            # FastSAM 分割模型 (~145MB)
# 若缺少, 从训练环境复制或重新下载
```
