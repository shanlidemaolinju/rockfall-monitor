"""
性能监控模块 — 实时 FPS / GPU 利用率 / 内存占用 / CPU 使用率
===========================================================
用于 Streamlit 性能仪表盘 + 桌面端实时监控。

依赖 (可选):
  - psutil  — CPU / 内存监控 (pip install psutil)
  - pynvml  — NVIDIA GPU 监控 (pip install nvidia-ml-py)
  - torch   — PyTorch 显存监控 (已内置, 无需额外安装)

用法:
    monitor = PerformanceMonitor()
    monitor.start()                    # 开始监控
    # ... 检测循环 ...
    monitor.record_frame(inference_ms) # 每帧记录推理耗时
    snapshot = monitor.snapshot()      # 获取当前指标快照
    monitor.stop()                     # 停止监控

    # 或作为上下文管理器:
    with PerformanceMonitor() as monitor:
        for frame in video:
            t0 = time.time()
            result = detector.detect(frame)
            monitor.record_frame((time.time() - t0) * 1000)
            snapshot = monitor.snapshot()
"""

import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════
# 可选依赖探测
# ══════════════════════════════════════════════════════════════

_PSUTIL_AVAILABLE = False
_NVML_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_DEVICE_COUNT = pynvml.nvmlDeviceGetCount()
    if _NVML_DEVICE_COUNT > 0:
        _NVML_AVAILABLE = True
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    else:
        _NVML_AVAILABLE = False
except (ImportError, Exception):
    _NVML_AVAILABLE = False
    _NVML_DEVICE_COUNT = 0
    _NVML_HANDLE = None


# ══════════════════════════════════════════════════════════════
# Torch GPU 探测
# ══════════════════════════════════════════════════════════════

def _check_torch_gpu():
    """检测 PyTorch 是否可用且使用 GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return False, "CPU"


_TORCH_GPU, _TORCH_DEVICE_NAME = _check_torch_gpu()


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class PerformanceSnapshot:
    """性能快照 — 当前时刻的各类指标"""
    # 帧率
    fps: float = 0.0
    # 推理耗时 (ms)
    inference_ms: float = 0.0
    inference_ms_avg: float = 0.0          # 滑动平均 (最近 30 帧)
    inference_ms_min: float = 0.0
    inference_ms_max: float = 0.0
    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    # 系统内存 (MB)
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    process_memory_mb: float = 0.0
    # GPU (NVML / pynvml)
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_utilization: float = 0.0          # GPU 核心利用率 (%)
    gpu_memory_total_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    # PyTorch GPU 显存
    torch_gpu_available: bool = False
    torch_gpu_name: str = ""
    torch_memory_allocated_mb: float = 0.0
    torch_memory_reserved_mb: float = 0.0
    # 检测统计
    total_frames_processed: int = 0
    total_alerts: int = 0
    elapsed_seconds: float = 0.0
    # 监控耗时 (监控本身的开销, ms)
    monitor_overhead_ms: float = 0.0


# ══════════════════════════════════════════════════════════════
# 性能监控器
# ══════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """实时性能监控器 — 线程安全, 低开销"""

    def __init__(self, window_size: int = 30):
        """
        参数:
            window_size: 滑动窗口大小 (用于计算平均 FPS / 推理耗时)
        """
        self._window_size = max(1, window_size)
        self._inference_times: deque[float] = deque(maxlen=window_size)
        self._frame_timestamps: deque[float] = deque(maxlen=window_size)
        self._running = False
        self._start_time = 0.0
        self._total_frames = 0
        self._total_alerts = 0
        self._lock = threading.Lock()
        self._process = None

    # ── 上下文管理器 ──────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self):
        """开始监控"""
        self._running = True
        self._start_time = time.time()
        self._total_frames = 0
        self._total_alerts = 0
        self._inference_times.clear()
        self._frame_timestamps.clear()
        if _PSUTIL_AVAILABLE:
            self._process = psutil.Process()

    def stop(self):
        """停止监控"""
        self._running = False

    def record_frame(self, inference_ms: float = 0.0, is_alert: bool = False):
        """
        记录一帧检测结果。

        参数:
            inference_ms: 该帧推理耗时 (毫秒)
            is_alert: 是否触发预警
        """
        now = time.time()
        with self._lock:
            self._inference_times.append(inference_ms)
            self._frame_timestamps.append(now)
            self._total_frames += 1
            if is_alert:
                self._total_alerts += 1

    # ── 快照采集 ──────────────────────────────────────────────

    def snapshot(self) -> PerformanceSnapshot:
        """采集当前性能快照 (开销控制在 1-5ms 内)"""
        t0 = time.time()
        snap = PerformanceSnapshot()
        now = time.time()

        with self._lock:
            # FPS: 基于滑动窗口中每秒帧数
            if len(self._frame_timestamps) >= 2:
                first_ts = self._frame_timestamps[0]
                last_ts = self._frame_timestamps[-1]
                duration = last_ts - first_ts
                count = len(self._frame_timestamps)
                if duration > 0.001:
                    snap.fps = count / duration
                else:
                    snap.fps = count  # 1s 内全部帧

            # 推理耗时统计
            if self._inference_times:
                times = list(self._inference_times)
                snap.inference_ms = times[-1]
                snap.inference_ms_avg = sum(times) / len(times)
                snap.inference_ms_min = min(times)
                snap.inference_ms_max = max(times)

            snap.total_frames_processed = self._total_frames
            snap.total_alerts = self._total_alerts
            snap.elapsed_seconds = now - self._start_time if self._start_time > 0 else 0

        # ── CPU / 内存 (psutil) ──
        if _PSUTIL_AVAILABLE and self._process is not None:
            try:
                snap.cpu_percent = self._process.cpu_percent(interval=0.0)
                snap.cpu_count = psutil.cpu_count(logical=True)
                mem = psutil.virtual_memory()
                snap.memory_total_mb = mem.total / (1024 * 1024)
                snap.memory_used_mb = mem.used / (1024 * 1024)
                snap.memory_percent = mem.percent
                pmem = self._process.memory_info()
                snap.process_memory_mb = pmem.rss / (1024 * 1024)
            except Exception:
                pass

        # ── GPU (pynvml) ──
        if _NVML_AVAILABLE:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                snap.gpu_available = True
                snap.gpu_utilization = util.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                snap.gpu_memory_total_mb = mem_info.total / (1024 * 1024)
                snap.gpu_memory_used_mb = mem_info.used / (1024 * 1024)
                snap.gpu_memory_percent = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0
                try:
                    snap.gpu_temperature_c = pynvml.nvmlDeviceGetTemperature(
                        _NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass
                snap.gpu_name = pynvml.nvmlDeviceGetName(_NVML_HANDLE) if hasattr(pynvml, 'nvmlDeviceGetName') else "NVIDIA GPU"
            except Exception:
                snap.gpu_available = False

        # ── PyTorch GPU 显存 ──
        if _TORCH_GPU:
            try:
                import torch
                snap.torch_gpu_available = True
                snap.torch_gpu_name = _TORCH_DEVICE_NAME
                snap.torch_memory_allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
                snap.torch_memory_reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
            except Exception:
                snap.torch_gpu_available = False

        snap.monitor_overhead_ms = (time.time() - t0) * 1000
        return snap

    # ── 摘要 ──────────────────────────────────────────────────

    def summary_dict(self) -> dict:
        """返回 JSON 可序列化的摘要字典 (供 API / 日志)"""
        s = self.snapshot()
        return {
            "fps": round(s.fps, 1),
            "inference_ms": round(s.inference_ms, 1),
            "inference_ms_avg": round(s.inference_ms_avg, 1),
            "cpu_percent": round(s.cpu_percent, 1),
            "memory_percent": round(s.memory_percent, 1),
            "process_memory_mb": round(s.process_memory_mb, 1),
            "gpu_utilization": round(s.gpu_utilization, 1),
            "gpu_memory_percent": round(s.gpu_memory_percent, 1),
            "gpu_temperature_c": round(s.gpu_temperature_c, 1),
            "total_frames": s.total_frames_processed,
            "total_alerts": s.total_alerts,
            "elapsed_seconds": round(s.elapsed_seconds, 1),
        }


# ══════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════

def get_device_info() -> dict:
    """获取设备硬件信息 (供 UI 展示)"""
    info = {
        "cpu": "Unknown",
        "cpu_cores": 0,
        "ram_total_gb": 0,
        "gpu_name": "None",
        "gpu_memory_gb": 0,
        "torch_device": _TORCH_DEVICE_NAME,
        "psutil_available": _PSUTIL_AVAILABLE,
        "nvml_available": _NVML_AVAILABLE,
    }

    if _PSUTIL_AVAILABLE:
        info["cpu"] = _get_cpu_name()
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)

    if _NVML_AVAILABLE:
        try:
            info["gpu_name"] = pynvml.nvmlDeviceGetName(_NVML_HANDLE) if hasattr(pynvml, 'nvmlDeviceGetName') else "NVIDIA"
            mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            info["gpu_memory_gb"] = round(mem.total / (1024**3), 1)
        except Exception:
            pass

    return info


def _get_cpu_name() -> str:
    """获取 CPU 型号名称"""
    try:
        # Windows: 使用 PowerShell (更可靠)
        if os.name == 'nt':
            import subprocess
            try:
                result = subprocess.run(
                    ['powershell', '-Command', 'Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty Name'],
                    capture_output=True, text=True, timeout=5
                )
                name = result.stdout.strip()
                if name:
                    return name
            except Exception:
                pass
            # 回退: wmic
            try:
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'], capture_output=True, text=True, timeout=3
                )
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
                if len(lines) >= 2:
                    return lines[1]
            except Exception:
                pass
        # Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except Exception:
            pass
    except Exception:
        pass
    return "Unknown CPU"


# ============================================================
# 全局性能监控器单例（供 /metrics 端点使用）
# ============================================================

_global_monitor: PerformanceMonitor | None = None
_global_monitor_lock = threading.Lock()


def get_global_monitor() -> PerformanceMonitor:
    """获取或创建全局 PerformanceMonitor 单例。

    首次调用时初始化，后续调用返回同一实例。
    线程安全，开销 < 1μs（已初始化后）。
    """
    global _global_monitor
    if _global_monitor is not None:
        return _global_monitor
    with _global_monitor_lock:
        if _global_monitor is None:
            _global_monitor = PerformanceMonitor()
        return _global_monitor
