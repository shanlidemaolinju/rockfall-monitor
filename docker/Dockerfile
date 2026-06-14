# ============================================================
# 落石检测系统 - Docker 镜像
# ============================================================
# 构建: docker build -f docker/Dockerfile -t rockfall:latest .
# 运行: docker run -p 8000:8000 --env-file .env rockfall:latest
# ============================================================

FROM ultralytics/ultralytics:8.3.133-cpu

WORKDIR /app

# 安装额外 Python 依赖 (ultralytics 镜像已含 torch + opencv + ultralytics)
# 不加 --upgrade，避免覆盖基础镜像中已有的 CUDA/PyTorch 版本
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# 复制全部源码
COPY . .

# 安装 Alembic (数据库迁移工具，entrypoint.sh 调用)
RUN pip install --no-cache-dir alembic

# 使 entrypoint 可执行
RUN chmod +x docker/entrypoint.sh

# 创建运行时目录
RUN mkdir -p data/results data/uploads

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 服务
# --workers 1 是必须的: YOLO 模型在主进程加载, 多 worker 会导致
#   1) 每个 worker 加载一次模型 (显存翻倍)
#   2) MJPEG 帧缓冲跨进程不可见 (Web 流只能看到随机一个 worker 的帧)
CMD ["uvicorn", "server.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
