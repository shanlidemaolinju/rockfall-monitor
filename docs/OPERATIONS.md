# RockGuard 运维手册

> 公路落石灾害监测预警系统 — 日常运维与故障处理指南

---

## 目录

- [1. 日常巡检清单](#1-日常巡检清单)
- [2. 监控告警体系](#2-监控告警体系)
- [3. 常见故障排查](#3-常见故障排查)
- [4. 备份与恢复](#4-备份与恢复)
- [5. 性能调优](#5-性能调优)
- [6. 应急操作](#6-应急操作)

---

## 1. 日常巡检清单

### 1.1 每日检查（耗时约 3 分钟）

| 序号 | 检查项 | 命令 | 正常标准 |
|------|--------|------|----------|
| 1 | 服务进程状态 | `systemctl status rockfall` 或 `docker ps` | `active (running)` / `healthy` |
| 2 | 磁盘空间 | `df -h /opt/rockfall` | 使用率 < 80% |
| 3 | 内存使用 | `free -h` | available > 2 GB |
| 4 | CPU 负载 | `uptime` | load average < CPU 核数 |
| 5 | API 响应 | `curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health` | `200` |
| 6 | 最新日志错误 | `tail -50 /opt/rockfall/data/detection_log.jsonl \| grep '"level":"ERROR"'` | 无输出或偶发 |

#### 一键巡检脚本

将以下脚本保存为 `/opt/rockfall/scripts/daily_check.sh`：

```bash
#!/bin/bash
# RockGuard 每日巡检脚本

echo "========== RockGuard 每日巡检 =========="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 服务状态
echo ">>> 1. 服务状态"
if command -v systemctl &>/dev/null && systemctl is-active --quiet rockfall 2>/dev/null; then
    echo "  [OK] systemd 服务运行中"
elif docker ps --format '{{.Names}}' | grep -q rockfall; then
    echo "  [OK] Docker 容器运行中"
else
    echo "  [FAIL] 服务未运行！"
fi

# 2. 磁盘空间
DISK_USAGE=$(df /opt/rockfall --output=pcent | tail -1 | tr -d ' %')
if [ "$DISK_USAGE" -lt 80 ]; then
    echo "  [OK] 磁盘使用率 ${DISK_USAGE}%"
else
    echo "  [WARN] 磁盘使用率 ${DISK_USAGE}%，需要清理"
fi

# 3. 内存
AVAIL_MEM=$(free -m | awk '/Mem:/ {print $7}')
if [ "$AVAIL_MEM" -gt 2048 ]; then
    echo "  [OK] 可用内存 ${AVAIL_MEM} MB"
else
    echo "  [WARN] 可用内存仅 ${AVAIL_MEM} MB"
fi

# 4. 健康检查
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    echo "  [OK] API 健康检查通过"
else
    echo "  [FAIL] API 健康检查失败 (HTTP $HTTP_CODE)"
fi

# 5. 最近错误
ERROR_COUNT=$(grep -c '"level":"ERROR"' /opt/rockfall/data/detection_log.jsonl 2>/dev/null || echo 0)
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "  [WARN] 最近有 ${ERROR_COUNT} 条错误日志"
else
    echo "  [OK] 错误日志正常"
fi

echo ""
echo "========== 巡检完成 =========="
```

```bash
# 添加 cron 定时任务（每天早上 9:07）
crontab -e
# 添加:  7 9 * * * /bin/bash /opt/rockfall/scripts/daily_check.sh 2>&1 | logger -t rockfall-check
```

### 1.2 每周检查（耗时约 10 分钟）

| 序号 | 检查项 | 命令/方法 | 正常标准 |
|------|--------|-----------|----------|
| 1 | 数据库大小 | `du -sh /opt/rockfall/data/alerts.db` | < 500 MB（SQLite 单文件） |
| 2 | 截图文件数量 | `find /opt/rockfall/data/results -name '*.jpg' -mtime -7 \| wc -l` | 合理增长 |
| 3 | 日志文件轮转 | `ls -lt /opt/rockfall/data/detection_log* \| head` | 最近文件正常轮转 |
| 4 | 模型文件完整性 | `python scripts/validate_model.py` | 所有模型通过 |
| 5 | 推送通道连通性 | 检查最近推送记录（`push_status` 字段） | 成功率 > 95% |
| 6 | Nginx 错误日志 | `tail -50 /var/log/nginx/error.log` | 无异常 |

### 1.3 月度检查（耗时约 20 分钟）

| 序号 | 检查项 | 操作 |
|------|--------|------|
| 1 | 系统更新 | `sudo apt update && sudo apt upgrade -y` |
| 2 | Python 依赖安全 | `pip list --outdated` |
| 3 | SSL 证书有效期 | `certbot certificates` |
| 4 | 备份恢复演练 | 执行 [4.4 节](#44-恢复演练) |
| 5 | 磁盘清理 | 删除 30 天前的临时上传文件 |
| 6 | 性能基准对比 | 对比当前 FPS 与历史基准 |

---

## 2. 监控告警体系

### 2.1 内置监控端点

| 端点 | 类型 | 用途 |
|------|------|------|
| `/health` | Health Check | Docker / systemd 基础探活 |
| `/health/live` | Liveness | K8s liveness probe |
| `/health/ready` | Readiness | K8s readiness probe（GPU/DB/模型） |
| `/api/health/full` | Full Health | 磁盘/内存/CPU/GPU/数据库完整诊断 |
| `/api/health/storage` | Storage | 存储使用统计 |
| `/metrics` | Prometheus | Prometheus 格式指标暴露 |

### 2.2 Prometheus 集成

系统自动暴露 `/metrics` 端点，包含以下指标：

```
# 检测指标
rockfall_detections_total{level="red|orange|yellow|blue"}
rockfall_detection_fps
rockfall_detection_latency_seconds

# 系统指标
rockfall_cpu_usage_percent
rockfall_memory_usage_bytes
rockfall_disk_usage_bytes
rockfall_gpu_utilization_percent  (如有 GPU)

# 业务指标
rockfall_alerts_active
rockfall_push_success_total
rockfall_push_failure_total
rockfall_websocket_connections
```

Prometheus 抓取配置：

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rockfall'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

### 2.3 Sentry 错误监控

```env
# .env 中配置
SENTRY_DSN=https://xxx@xxx.ingest.sentry.io/xxx
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
```

配置后，所有未捕获异常自动上报，包含脱敏后的请求上下文和堆栈信息。

---

## 3. 常见故障排查

> 以下覆盖 **7 个常见故障场景**的完整排查步骤。

---

### 场景 1：服务无法启动（启动即退出）

**症状**：`systemctl start rockfall` 或 `docker compose up -d` 后服务立即退出或反复重启。

**排查步骤**：

```bash
# Step 1: 查看退出日志
# systemd
sudo journalctl -u rockfall --since "5 min ago" -n 50
# Docker
docker compose logs --tail=50 rockfall

# Step 2: 检查配置文件语法
python -c "from rockfall.config import Config; c = Config(); print(c.validate())"

# Step 3: 检查端口占用
sudo ss -tlnp | grep 8000

# Step 4: 检查模型文件
python scripts/validate_model.py

# Step 5: 检查 Python 依赖
pip check
```

**常见原因与解决**：

| 原因 | 日志特征 | 解决方法 |
|------|----------|----------|
| `.env` 未创建 | `KeyError: 'PUSHPLUS_TOKEN'` | `cp .env.example .env` 并编辑 |
| 端口占用 | `Address already in use` | `kill <pid>` 或换端口 |
| 模型文件缺失 | `FileNotFoundError: models/rock_best.pt` | 确认模型文件存在 |
| 依赖版本冲突 | `ImportError` / `ModuleNotFoundError` | `pip install -r requirements-lock.txt` |
| Python 版本过低 | `SyntaxError` | Python ≥ 3.10 |
| 磁盘满 | `OSError: No space left` | 参考 [场景 2](#场景-2磁盘空间不足) |

---

### 场景 2：磁盘空间不足

**症状**：系统运行变慢、日志写入失败、截图保存失败、"No space left on device" 错误。

**排查步骤**：

```bash
# Step 1: 查看磁盘使用
df -h /opt/rockfall

# Step 2: 找出占用大的目录
du -sh /opt/rockfall/data/* | sort -rh | head -10

# Step 3: 检查日志文件
du -sh /opt/rockfall/data/detection_log*.jsonl
du -sh /opt/rockfall/data/*.log

# Step 4: 检查截图文件
du -sh /opt/rockfall/data/results/

# Step 5: 检查上传文件
du -sh /opt/rockfall/data/uploads/
```

**解决方法**：

```bash
# 方法 1: 手动触发清理（保留 7 天内文件）
curl -X POST http://localhost:8000/api/health/cleanup \
  -H "Content-Type: application/json" \
  -d '{"max_age_days": 7}'

# 方法 2: 清理旧截图（手动）
find /opt/rockfall/data/results -name '*.jpg' -mtime +30 -delete

# 方法 3: 清理旧上传文件
find /opt/rockfall/data/uploads -mtime +7 -delete

# 方法 4: 压缩旧日志
gzip /opt/rockfall/data/detection_log.*.jsonl

# 方法 5: 调整存储配额
# 在 .env 中设置：
# STORAGE_MAX_GB=20    (存储上限)
# STORAGE_RETENTION_DAYS=30  (保留天数)
```

**预防措施**：

```bash
# 配置 logrotate（如已部署则自动生效）
sudo cp deploy/rockfall-logrotate.conf /etc/logrotate.d/rockfall

# 配置 cron 定期清理（凌晨 3 点）
# 0 3 * * * curl -X POST http://localhost:8000/api/health/cleanup -H 'Content-Type: application/json' -d '{"max_age_days":7}'
```

---

### 场景 3：WebSocket 连接失败

**症状**：前端实时进度条卡住、"连接断开"提示、WebSocket 反复重连。

**排查步骤**：

```bash
# Step 1: 检查 WebSocket 端点
wscat -c ws://localhost:8000/ws/tasks/test123
# 如果安装了 wscat: npm install -g wscat

# 或用 Python 测试
python -c "
import websocket
ws = websocket.WebSocket()
ws.connect('ws://localhost:8000/ws/tasks/test')
print('WebSocket connected OK')
ws.close()
"

# Step 2: 检查 Nginx 配置（如经过反向代理）
grep -A5 "Upgrade" /etc/nginx/sites-enabled/rockfall
# 必须包含: proxy_set_header Upgrade \$http_upgrade;
#           proxy_set_header Connection \"upgrade\";

# Step 3: 检查防火墙/安全组
sudo ufw status
# 确保 8000 端口开放

# Step 4: 检查 Nginx 错误日志
sudo tail -50 /var/log/nginx/error.log
```

**常见原因与解决**：

| 原因 | 检查方法 | 解决方法 |
|------|----------|----------|
| Nginx 缺少 Upgrade 头 | 检查 `proxy_set_header Upgrade` | 使用 `deploy/nginx.conf` 模板 |
| 反向代理缓冲未关闭 | 检查 `proxy_buffering off` | 添加 `proxy_buffering off;` |
| 防火墙阻止 | `sudo ufw status` | `sudo ufw allow 8000` |
| 客户端超时设置过短 | 检查 `proxy_read_timeout` | 设为 `86400s`（24 小时） |
| 服务端 Worker 阻塞 | 检查 CPU 使用率 | 增加 `--workers` 参数 |

**配置修复**（Nginx 关键配置）：

```nginx
location /ws/ {
    proxy_pass http://rockfall_backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400s;
    proxy_buffering off;
}
```

---

### 场景 4：数据库锁定（SQLite "database is locked"）

**症状**：API 返回 500 错误，日志显示 `sqlite3.OperationalError: database is locked`。

**原因**：SQLite 不支持高并发写入，多个 Worker 同时写预警记录时可能冲突。

**排查步骤**：

```bash
# Step 1: 确认数据库类型
grep MYSQL_HOST /opt/rockfall/.env

# Step 2: 查看错误日志
grep "database is locked" /opt/rockfall/data/detection_log.jsonl

# Step 3: 检查 Worker 数量
ps aux | grep uvicorn
```

**解决方法**：

```bash
# 方法 1: 切换到 MySQL（推荐，生产环境）
# 在 .env 中配置：
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=rock
# 重启服务

# 方法 2: 减少 SQLite 写入竞争
# 将 Worker 数降为 1
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1

# 方法 3: 增加 SQLite 超时（代码已设置 30 秒超时）
# 确认 rockfall/alert_store.py 中有:
#   self._conn.execute("PRAGMA busy_timeout = 30000")
```

---

### 场景 5：GPU 未被检测到 / CUDA 错误

**症状**：模型回退到 CPU 推理、FPS 极低、`torch.cuda.is_available()` 返回 False。

**排查步骤**：

```bash
# Step 1: 检查 NVIDIA 驱动
nvidia-smi
# 期望: 显示 GPU 信息且 CUDA Version ≥ 12.1

# Step 2: 检查 PyTorch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Step 3: 检查 OpenCV CUDA
python -c "import cv2; print('CUDA count:', cv2.cuda.getCudaEnabledDeviceCount())"

# Step 4: 检查 TensorRT（如启用）
python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"

# Step 5: Docker 特有 - 检查 nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**解决方法**：

| 问题 | 解决方法 |
|------|----------|
| 驱动版本过低 | `sudo apt install nvidia-driver-535` |
| PyTorch CPU-only | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| Docker 无 GPU | 安装 `nvidia-container-toolkit`，重启 Docker |
| 显存不足 | 减小 `DETECTION_IMG_SIZE=320`，启用 `SKIP_IDLE` |

---

### 场景 6：推送通知未收到（微信/邮件/钉钉/飞书）

**症状**：检测到预警但未收到推送通知。

**排查步骤**：

```bash
# Step 1: 检查推送 Token 配置
grep PUSHPLUS_TOKEN /opt/rockfall/.env
# 确认 Token 不为空且未被注释

# Step 2: 检查最近推送状态
curl http://localhost:8000/api/alerts?limit=5 | python -m json.tool | grep push_status
# 期望 push_status 不为 "failed"

# Step 3: 测试推送通道连通性
python -c "
from rockfall.push_channels.pushplus import PushPlusChannel
channel = PushPlusChannel()
success, msg = channel.send('RockGuard 测试消息', '这是一条来自运维的测试推送')
print(f'推送结果: {success}, 消息: {msg}')
"

# Step 4: 检查多通道配置
grep ALERT_CHANNEL_MAP /opt/rockfall/.env
# 确认预警等级到通道的映射配置正确

# Step 5: 检查预警等级阈值
grep ALERT_RED_CONFIDENCE /opt/rockfall/.env
# 确认阈值不是太高导致所有预警都被过滤
```

**常见原因**：

| 原因 | 解决方法 |
|------|----------|
| Token 未配置/已过期 | 重新获取并更新 Token |
| 预警等级未达推送阈值 | 降低 `ALERT_*_CONFIDENCE` 阈值 |
| 推送冷却期内 | 默认冷却 10 秒，`ALERT_COOLDOWN_SECONDS=10` |
| 网络不通 | 检查服务器能否访问外网 |
| SMTP 密码错误 | 确认邮箱开启了 SMTP 服务并使用授权码 |
| 钉钉/飞书签名错误 | 检查 SECRET 与 Webhook URL 匹配 |

---

### 场景 7：MJPEG 视频流无法播放

**症状**：前端实时视频画面黑屏、一直加载中。

**排查步骤**：

```bash
# Step 1: 检查 MJPEG 端点
curl -v http://localhost:8000/api/stream.mjpeg 2>&1 | head -20
# 期望: Content-Type: multipart/x-mixed-replace

# Step 2: 检查摄像头连接
grep CAMERA_URL /opt/rockfall/.env
# 确认 RTSP 地址正确

# Step 3: 测试 RTSP 流
ffprobe rtsp://admin:password@192.168.1.100:554/stream
# 期望: 显示视频流信息

# Step 4: 检查 STREAM_TOKEN
grep STREAM_TOKEN /opt/rockfall/.env
# 如设置了 Token，URL 需携带 ?token=<STREAM_TOKEN>
```

**解决方法**：

| 原因 | 解决方法 |
|------|----------|
| RTSP 地址错误 | 用 VLC 验证 RTSP 地址 |
| 防火墙阻止 RTSP | `sudo ufw allow 554` |
| RTSP 认证失败 | 确认用户名密码正确 |
| H.265 编码不兼容 | 添加 `FFMPEG_EXTRA_OPTS=hwaccel|cuda` |
| 摄像头断流 | 检查摄像头电源/网络，系统会自动重连 |
| Nginx 缓冲 | 确认 `/api/stream.mjpeg` location 有 `proxy_buffering off` |

---

## 4. 备份与恢复

### 4.1 备份内容清单

| 备份项 | 路径 | 重要性 | 建议频率 |
|--------|------|--------|----------|
| 环境配置 | `.env` | ★★★ 必选 | 每次修改后 |
| 预警数据库 | `data/alerts.db` 或 MySQL dump | ★★★ 必选 | 每天 |
| 截图文件 | `data/results/` | ★★☆ 推荐 | 每周 |
| 日志文件 | `data/*.jsonl`, `data/*.log` | ★☆☆ 可选 | 每周 |
| 模型文件 | `models/*.pt` | ★☆☆ 可选 | 每次更新后 |
| ROI 多边形 | 数据库 `monitoring_sites.roi_polygon` | ★★☆ 推荐 | 每次修改后 |
| 点位配置 | 数据库 `monitoring_sites` 表 | ★★☆ 推荐 | 每次修改后 |

### 4.2 SQLite 备份

```bash
#!/bin/bash
# 备份脚本: /opt/rockfall/scripts/backup.sh

BACKUP_DIR="/opt/backups/rockfall"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

mkdir -p "$BACKUP_DIR"

# 1. 备份 .env
cp /opt/rockfall/.env "$BACKUP_DIR/env_$TIMESTAMP.bak"

# 2. 备份 SQLite 数据库（安全复制，不锁库）
sqlite3 /opt/rockfall/data/alerts.db ".backup '$BACKUP_DIR/alerts_$TIMESTAMP.db'"

# 3. 备份点位配置
sqlite3 /opt/rockfall/data/alerts.db ".dump monitoring_sites" > "$BACKUP_DIR/sites_$TIMESTAMP.sql"

# 4. 打包截图
tar -czf "$BACKUP_DIR/results_$TIMESTAMP.tar.gz" -C /opt/rockfall/data results/

# 5. 清理旧备份
find "$BACKUP_DIR" -name '*.bak' -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name '*.db' -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name '*.sql' -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name '*.tar.gz' -mtime +$RETENTION_DAYS -delete

echo "备份完成: $TIMESTAMP"
echo "备份目录: $BACKUP_DIR"
ls -lh "$BACKUP_DIR" | grep "$TIMESTAMP"
```

### 4.3 MySQL 备份

```bash
#!/bin/bash
# MySQL 备份脚本

BACKUP_DIR="/opt/backups/rockfall"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 从 .env 读取数据库配置
source <(grep -E '^MYSQL_' /opt/rockfall/.env)

# 全量备份
mysqldump -h ${MYSQL_HOST:-localhost} \
  -P ${MYSQL_PORT:-3306} \
  -u ${MYSQL_USER:-root} \
  -p${MYSQL_PASSWORD} \
  --single-transaction \
  --routines \
  --triggers \
  ${MYSQL_DATABASE:-rock} \
  > "$BACKUP_DIR/mysql_full_$TIMESTAMP.sql"

# 压缩
gzip "$BACKUP_DIR/mysql_full_$TIMESTAMP.sql"

echo "MySQL 备份完成: mysql_full_$TIMESTAMP.sql.gz"
```

### 4.4 恢复操作

#### SQLite 恢复

```bash
# 1. 停止服务
sudo systemctl stop rockfall

# 2. 备份当前数据库（以防万一）
mv /opt/rockfall/data/alerts.db /opt/rockfall/data/alerts.db.broken

# 3. 恢复备份
cp /opt/backups/rockfall/alerts_20260101_000000.db /opt/rockfall/data/alerts.db

# 4. 恢复 .env（如需要）
cp /opt/backups/rockfall/env_20260101_000000.bak /opt/rockfall/.env

# 5. 重启服务
sudo systemctl start rockfall

# 6. 验证
curl http://localhost:8000/health
```

#### MySQL 恢复

```bash
# 1. 解压备份
gunzip /opt/backups/rockfall/mysql_full_20260101_000000.sql.gz

# 2. 导入
mysql -h ${MYSQL_HOST} -P ${MYSQL_PORT} -u ${MYSQL_USER} -p${MYSQL_PASSWORD} \
  ${MYSQL_DATABASE} < /opt/backups/rockfall/mysql_full_20260101_000000.sql

# 3. 验证
curl http://localhost:8000/api/stats
```

### 4.5 配置定时备份

```bash
# 每天凌晨 2:07 执行备份
crontab -e
# 添加:
# 7 2 * * * /bin/bash /opt/rockfall/scripts/backup.sh >> /var/log/rockfall-backup.log 2>&1
```

---

## 5. 性能调优

### 5.1 CPU 模式优化

```env
# .env 调优参数
DETECTION_IMG_SIZE=320          # 降低推理分辨率（640 → 320）
SKIP_IDLE=10                    # 增大跳帧（5 → 10）
SKIP_ACTIVE=5                   # 增大跳帧
MOG2_HISTORY=300                # 减少历史帧数（500 → 300）
MOTION_MIN_AREA=200             # 提高最小运动面积（100 → 200）
EDGE_ENHANCE_ENABLED=false      # 关闭边缘增强
SAHI_ENABLED=false              # 关闭 SAHI
FUSION_ENABLED=false            # 关闭概率融合
TFD_ENABLED=false               # 关闭三帧差分
```

### 5.2 GPU 模式优化

```env
# TensorRT 加速（2-3x 推理提升）
TENSORRT_ENABLED=true
# 先导出: python scripts/export_tensorrt.py

# CUDA 预处理加速 MOG2 + Sobel
USE_CUDA_PREPROCESS=true
# 需要 opencv 的 CUDA 构建版

# 可保持高分辨率
DETECTION_IMG_SIZE=640
```

### 5.3 数据库优化

```bash
# SQLite: 定期维护
sqlite3 /opt/rockfall/data/alerts.db "PRAGMA optimize; VACUUM;"

# MySQL: 添加索引（如使用 MySQL）
# 系统自动创建索引，无需手动操作
```

---

## 6. 应急操作

### 6.1 快速重启

```bash
# systemd
sudo systemctl restart rockfall

# Docker
docker compose restart

# 手动进程
kill -HUP $(pgrep -f "uvicorn server.main")
```

### 6.2 紧急回滚

```bash
# 1. 停止服务
sudo systemctl stop rockfall

# 2. 恢复到上一个可用的代码版本
cd /opt/rockfall
git log --oneline -5   # 查看最近提交
git checkout <known-good-commit-hash>

# 3. 恢复数据库
cp /opt/backups/rockfall/alerts_latest.db /opt/rockfall/data/alerts.db

# 4. 重启
sudo systemctl start rockfall
```

### 6.3 服务降级

当系统资源不足时，可通过热更新 API 降级配置：

```bash
# 降低检测帧率
curl -X POST http://localhost:8000/api/config/update \
  -H "Content-Type: application/json" \
  -d '{"SKIP_IDLE": 30, "SKIP_ACTIVE": 15}'

# 暂停推送通知
curl -X POST http://localhost:8000/api/config/update \
  -H "Content-Type: application/json" \
  -d '{"PUSHPLUS_TOKEN": ""}'

# 减小推理分辨率
curl -X POST http://localhost:8000/api/config/update \
  -H "Content-Type: application/json" \
  -d '{"DETECTION_IMG_SIZE": 320}'
```

### 6.4 查看运行时状态

```bash
# 完整健康报告
curl http://localhost:8000/api/health/full | python -m json.tool

# 存储统计
curl http://localhost:8000/api/health/storage | python -m json.tool

# 运行时配置
curl http://localhost:8000/api/config/runtime | python -m json.tool

# 活跃摄像头
curl http://localhost:8000/api/cameras | python -m json.tool
```

---

## 附录：运维命令速查表

```bash
# ========== 服务管理 ==========
sudo systemctl start/stop/restart/status rockfall    # systemd
docker compose up -d / down / restart / logs -f      # Docker

# ========== 日志查看 ==========
sudo journalctl -u rockfall -f                        # 服务日志
tail -f /opt/rockfall/data/detection_log.jsonl        # 检测日志
docker compose logs -f rockfall                       # Docker 日志

# ========== 健康检查 ==========
curl http://localhost:8000/health                     # 基础
curl http://localhost:8000/api/health/full            # 完整
curl http://localhost:8000/metrics                    # Prometheus

# ========== 配置管理 ==========
curl http://localhost:8000/api/config/current         # 查看配置
curl -X POST .../api/config/update -d '{...}'         # 热更新
curl -X POST .../api/config/reload                    # 重载 .env

# ========== 数据库 ==========
sqlite3 /opt/rockfall/data/alerts.db ".tables"        # 查看表
sqlite3 .../alerts.db "SELECT COUNT(*) FROM alerts"   # 预警总数
sqlite3 .../alerts.db ".backup backup.db"             # SQLite 备份

# ========== 磁盘清理 ==========
curl -X POST .../api/health/cleanup -d '{"max_age_days":7}'  # API 清理
find data/results -name '*.jpg' -mtime +30 -delete    # 手动清理截图
```
