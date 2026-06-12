# 落石监测系统 Docker 镜像
# 构建: docker build -t rockfall:latest .
# 运行: docker run -d -p 8000:8000 --name rockfall rockfall:latest

FROM python:3.11-slim

LABEL maintainer="Rockfall Detection Team"
LABEL description="落石监测系统 — 公路自然灾害监测预警平台"

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖 (分层缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY rockfall/ ./rockfall/
COPY server/ ./server/
COPY models/ ./models/
COPY .env ./

# FastSAM 模型 (大文件, 可选 — 运行时挂载或下载)
# COPY FastSAM-x.pt ./

# 数据目录
RUN mkdir -p /app/data/results /app/data/uploads /app/data/masks

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
