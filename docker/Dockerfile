# ============================================================
# 落石检测系统 - Docker 镜像
# ============================================================
# 构建: docker build -f docker/Dockerfile -t rockfall:latest .
# 运行: docker run -p 8000:8000 --env-file .env rockfall:latest
# ============================================================

FROM ultralytics/ultralytics:8.3.133-cpu

WORKDIR /app

# 安装 Python 依赖
COPY requirements-lock.txt .
RUN pip install --no-cache-dir --upgrade -r requirements-lock.txt

# 复制全部源码
COPY . .

# 创建运行时目录
RUN mkdir -p data/results data/uploads

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 服务
# --workers 1 是必须的: YOLO 模型在主进程加载, 多 worker 会导致
#   1) 每个 worker 加载一次模型 (显存翻倍)
#   2) MJPEG 帧缓冲跨进程不可见 (Web 流只能看到随机一个 worker 的帧)
CMD ["uvicorn", "server.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
