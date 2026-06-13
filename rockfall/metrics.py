"""
Prometheus 监控指标暴露模块
============================
提供标准 Prometheus 格式的运行时指标，涵盖:
  - 推理性能（FPS、延迟分布）
  - GPU 资源（利用率、显存、温度）
  - 业务指标（告警速率、摄像头数量、队列长度）
  - 存储健康度

/metrics 端点由 FastAPI 路由转发至此模块。
"""

from prometheus_client import Gauge, Counter, Histogram, generate_latest, REGISTRY

# ============================================================
# 推理性能
# ============================================================
fps_gauge = Gauge(
    "rockfall_fps", "Current frames per second (sliding window)"
)
inference_ms_gauge = Gauge(
    "rockfall_inference_ms", "Latest inference time in milliseconds"
)
inference_ms_histogram = Histogram(
    "rockfall_inference_ms_dist",
    "Inference time distribution (ms)",
    buckets=[20, 50, 100, 200, 500, 1000, 2000],
)
total_frames_counter = Counter(
    "rockfall_frames_total", "Total frames processed since start"
)

# ============================================================
# GPU 资源
# ============================================================
gpu_util_gauge = Gauge(
    "rockfall_gpu_utilization_pct", "GPU core utilization percentage"
)
gpu_mem_used_gauge = Gauge(
    "rockfall_gpu_memory_used_mb", "GPU memory used (MB)"
)
gpu_mem_total_gauge = Gauge(
    "rockfall_gpu_memory_total_mb", "GPU memory total (MB)"
)
gpu_temp_gauge = Gauge(
    "rockfall_gpu_temperature_c", "GPU temperature (Celsius)"
)
torch_mem_allocated_gauge = Gauge(
    "rockfall_torch_memory_allocated_mb", "Torch GPU memory allocated (MB)"
)

# ============================================================
# CPU / 内存
# ============================================================
cpu_percent_gauge = Gauge(
    "rockfall_cpu_percent", "System CPU utilization percentage"
)
process_memory_mb_gauge = Gauge(
    "rockfall_process_memory_mb", "Process memory usage (MB)"
)

# ============================================================
# 摄像头 / 队列
# ============================================================
camera_count_gauge = Gauge(
    "rockfall_camera_count", "Number of active camera streams"
)
task_queue_gauge = Gauge(
    "rockfall_task_queue_length", "Number of pending async video tasks"
)

# ============================================================
# 告警
# ============================================================
alert_total = Counter(
    "rockfall_alerts_total", "Total alerts generated",
    ["level"],  # red / orange / yellow / blue
)
alert_rate_gauge = Gauge(
    "rockfall_alert_rate_per_hour", "Alert rate per hour (last 60min)"
)

# ============================================================
# 数据库
# ============================================================
# TODO: 当前架构每个请求新建连接（无连接池），只能报告 0 或 1。
#       引入 SQLAlchemy 连接池后可改为报告实际 pool.checkedin / pool.checkedout。
db_connections_gauge = Gauge(
    "rockfall_db_connections", "Database connection status (1=available, 0=unavailable)",
    ["backend"],  # mysql / sqlite
)

# ============================================================
# 存储
# ============================================================
storage_used_gb_gauge = Gauge(
    "rockfall_storage_used_gb", "Storage space used by results/uploads (GB)"
)
storage_files_gauge = Gauge(
    "rockfall_storage_file_count", "Number of stored result files"
)

# ============================================================
# 系统信息（常量标签）
# ============================================================
system_info = Gauge(
    "rockfall_info", "System information",
    ["device", "model", "version"],
)


def collect_from_perf(snapshot) -> None:
    """从 PerformanceSnapshot 收集指标。"""
    fps_gauge.set(snapshot.fps)
    inference_ms_gauge.set(snapshot.inference_ms)
    if snapshot.inference_ms > 0:
        inference_ms_histogram.observe(snapshot.inference_ms)

    total_frames_counter.inc(0)  # 确保 metric 存在
    if snapshot.total_frames_processed:
        pass  # Counter 由 record_frame 递增

    cpu_percent_gauge.set(snapshot.cpu_percent)
    process_memory_mb_gauge.set(snapshot.process_memory_mb)

    if snapshot.gpu_available:
        gpu_util_gauge.set(snapshot.gpu_utilization)
        gpu_mem_used_gauge.set(snapshot.gpu_memory_used_mb)
        gpu_mem_total_gauge.set(snapshot.gpu_memory_total_mb)
        gpu_temp_gauge.set(snapshot.gpu_temperature_c)

    if snapshot.torch_gpu_available:
        torch_mem_allocated_gauge.set(snapshot.torch_memory_allocated_mb)


def record_alert(level: str) -> None:
    """记录一条告警（按等级递增 Counter）。"""
    alert_total.labels(level=level).inc()


def set_camera_count(n: int) -> None:
    camera_count_gauge.set(n)


def set_task_queue_length(n: int) -> None:
    task_queue_gauge.set(n)


def set_db_connections(backend: str, available: bool) -> None:
    """设置数据库连接状态（0=不可用, 1=可用）。

    TODO: 引入连接池后改为 pool.checkedin / pool.checkedout。
    """
    db_connections_gauge.labels(backend=backend).set(1 if available else 0)


def set_storage_stats(used_gb: float, file_count: int) -> None:
    storage_used_gb_gauge.set(round(used_gb, 2))
    storage_files_gauge.set(file_count)


def set_system_info(device: str, model: str, version: str | None = None) -> None:
    if version is None:
        from . import __version__ as _ver
        version = _ver
    system_info.labels(device=device, model=model, version=version).set(1)


def get_metrics_text() -> bytes:
    """生成 Prometheus 文本格式的全部指标。"""
    return generate_latest(REGISTRY)
