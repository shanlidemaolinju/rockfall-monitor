"""
系统健康检查模块 — 自监控 + 自动告警 + 自愈动作
================================================
监控: 磁盘 / 内存 / CPU / GPU温度 / 检测服务状态 / 数据库连接
自愈: 磁盘超85%自动清理旧帧 / GPU过热自动降频或切换CPU

用法:
    from rockfall.health import SystemHealth
    health = SystemHealth()
    status = health.check_all()
    # {"healthy": True/False, "checks": {...}, "warnings": [...], "heal_actions": [...]}
"""

import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

from .config import (
    DATA_DIR, RESULTS_DIR, MODEL_PATH,
    FILE_RETENTION_DAYS, STRICT_RETENTION,
)


# 阈值配置
DISK_WARN_PERCENT = 85     # 磁盘使用率超过此值告警
DISK_CRIT_PERCENT = 95
DISK_HEAL_PERCENT = 85     # 触发自动清理的阈值
# DISK_RETENTION_DAYS 已废弃，使用 FILE_RETENTION_DAYS 替代
MEM_WARN_PERCENT = 85
MEM_CRIT_PERCENT = 95
UPTIME_WARN_HOURS = 168    # 连续运行超过 7 天告警
STORAGE_MAX_GB = 50         # 数据目录最大容量
GPU_TEMP_WARN_C = 80        # GPU 温度告警阈值 (℃)
GPU_TEMP_CRIT_C = 85        # GPU 温度自愈触发阈值 (℃)
GPU_TEMP_THROTTLE_C = 85    # 超过此温度自动触发降频/切换CPU


class SystemHealth:
    """系统健康检查器 — 含自愈动作"""

    def __init__(self):
        self._start_time = time.time()
        self._last_ok = True
        self._fail_count = 0
        self._heal_history: list[dict] = []    # 自愈动作历史
        self._heal_lock = threading.Lock()
        self._gpu_throttled = False            # GPU 是否已降频
        self._last_disk_cleanup = 0.0          # 上次磁盘清理时间戳
        self._disk_cleanup_cooldown = 3600     # 磁盘清理冷却时间 (秒)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def check_all(self) -> dict:
        """执行全部健康检查 + 自动自愈, 返回状态报告"""
        checks = {}
        warnings = []
        heal_actions = []

        # 1. 磁盘空间
        disk = self._check_disk()
        checks["disk"] = disk
        if disk["percent"] >= DISK_CRIT_PERCENT:
            warnings.append(f"磁盘空间严重不足: {disk['percent']:.0f}%")
        elif disk["percent"] >= DISK_WARN_PERCENT:
            warnings.append(f"磁盘空间不足: {disk['percent']:.0f}%")

        # 自愈: 磁盘超阈值 → 自动清理旧帧
        if disk["percent"] >= DISK_HEAL_PERCENT:
            action = self._heal_disk_cleanup()
            if action:
                heal_actions.append(action)

        # 2. 内存
        mem = self._check_memory()
        checks["memory"] = mem
        if mem["percent"] >= MEM_CRIT_PERCENT:
            warnings.append(f"内存严重不足: {mem['percent']:.0f}%")
        elif mem["percent"] >= MEM_WARN_PERCENT:
            warnings.append(f"内存使用率偏高: {mem['percent']:.0f}%")

        # 3. 模型文件
        model = self._check_model()
        checks["model"] = model
        if not model["exists"]:
            warnings.append(f"模型文件不存在: {model['path']}")

        # 4. 数据目录可写
        data_rw = self._check_data_rw()
        checks["data_rw"] = data_rw
        if not data_rw["writable"]:
            warnings.append(f"数据目录不可写: {DATA_DIR}")

        # 5. 存储配额
        storage = self._check_storage()
        checks["storage"] = storage
        if storage["total_gb"] >= STORAGE_MAX_GB:
            warnings.append(f"数据存储已达配额上限 {STORAGE_MAX_GB}GB")
            # 自愈: 配额超限也触发清理
            action = self._heal_disk_cleanup()
            if action:
                heal_actions.append(action)

        # 6. GPU 温度 (NVIDIA GPU)
        gpu = self._check_gpu()
        if gpu:
            checks["gpu"] = gpu
            if gpu.get("temp_c", 0) >= GPU_TEMP_CRIT_C:
                warnings.append(f"GPU 温度过高: {gpu['temp_c']}℃")
            elif gpu.get("temp_c", 0) >= GPU_TEMP_WARN_C:
                warnings.append(f"GPU 温度偏高: {gpu['temp_c']}℃")

            # 自愈: GPU 过热 → 触发降频/切换CPU
            if gpu.get("temp_c", 0) >= GPU_TEMP_THROTTLE_C:
                action = self._heal_gpu_throttle(gpu)
                if action:
                    heal_actions.append(action)

        # 7. 运行时间
        uptime_h = self.uptime_seconds / 3600
        checks["uptime_hours"] = round(uptime_h, 1)
        if uptime_h >= UPTIME_WARN_HOURS:
            warnings.append(f"系统已连续运行 {uptime_h:.0f} 小时，建议计划重启")

        # 8. 哈希链完整性 (仅在启用时检查)
        hash_chain = self._check_hash_chain()
        if hash_chain is not None:
            checks["hash_chain"] = hash_chain
            if hash_chain.get("status") == "breach_detected":
                warnings.append("哈希链断裂: 预警记录可能被篡改!")

        # 9. 归档状态 (上次归档是否在 25 小时内)
        retention = self._check_retention_status()
        checks["retention"] = retention
        if retention.get("overdue"):
            warnings.append(f"归档调度可能滞后: 上次归档 {retention.get('hours_ago', '?')} 小时前")

        healthy = len([w for w in warnings if "严重" in w or "不存在" in w or "不可写" in w or "篡改" in w]) == 0

        if not healthy:
            self._fail_count += 1
        else:
            self._fail_count = 0
            self._last_ok = True

        return {
            "healthy": healthy,
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round(uptime_h, 1),
            "fail_count": self._fail_count,
            "gpu_throttled": self._gpu_throttled,
            "checks": checks,
            "warnings": warnings,
            "heal_actions": heal_actions,
        }

    # ================================================================
    # 自愈动作
    # ================================================================

    def _heal_disk_cleanup(self) -> dict | None:
        """
        自动清理旧检测帧: 删除 RESULTS_DIR 中超过 FILE_RETENTION_DAYS 天的文件。

        严格模式 (STRICT_RETENTION=true) 下不删除文件，改为告警。

        返回: 清理结果 dict, 冷却期内不重复执行返回 None
        """
        now = time.time()
        if now - self._last_disk_cleanup < self._disk_cleanup_cooldown:
            return None

        self._last_disk_cleanup = now

        if STRICT_RETENTION:
            return {
                "action": "disk_cleanup_skipped",
                "time": datetime.now().isoformat(),
                "reason": "严格保留模式已启用, 跳过自动清理",
                "retention_days": FILE_RETENTION_DAYS,
            }

        cutoff_time = now - FILE_RETENTION_DAYS * 86400
        deleted_count = 0
        freed_bytes = 0

        for pattern in ["*.jpg", "*.png", "*.bmp"]:
            for fp in RESULTS_DIR.glob(pattern):
                try:
                    if fp.stat().st_mtime < cutoff_time:
                        size = fp.stat().st_size
                        fp.unlink()
                        deleted_count += 1
                        freed_bytes += size
                except Exception:
                    pass

        result = {
            "action": "disk_cleanup",
            "time": datetime.now().isoformat(),
            "deleted_files": deleted_count,
            "freed_mb": round(freed_bytes / (1024 ** 2), 1),
            "retention_days": FILE_RETENTION_DAYS,
        }

        with self._heal_lock:
            self._heal_history.append(result)
            # 保留最近 100 条自愈历史
            if len(self._heal_history) > 100:
                self._heal_history = self._heal_history[-100:]

        return result

    def _heal_gpu_throttle(self, gpu_info: dict) -> dict | None:
        """
        GPU 过热自愈: 标记降频状态。

        下游检测器读取 self._gpu_throttled 标志后:
          - 桌面端: 自动切换到 CPU 推理
          - 服务端: 增加跳帧间隔或降低推理分辨率

        返回: 自愈动作描述
        """
        if self._gpu_throttled:
            return None  # 已经触发过, 不重复

        self._gpu_throttled = True

        result = {
            "action": "gpu_throttle",
            "time": datetime.now().isoformat(),
            "gpu_temp_c": gpu_info.get("temp_c", 0),
            "gpu_util_pct": gpu_info.get("utilization_pct", 0),
            "effect": "已标记 GPU 降频信号, 建议下游切换到 CPU 推理或降低推理分辨率",
        }

        with self._heal_lock:
            self._heal_history.append(result)
            if len(self._heal_history) > 100:
                self._heal_history = self._heal_history[-100:]

        return result

    def reset_gpu_throttle(self):
        """GPU 温度恢复正常后清除降频标记 (由外部健康检查周期调用)"""
        self._gpu_throttled = False

    def get_heal_history(self, limit: int = 20) -> list[dict]:
        """获取自愈动作历史"""
        with self._heal_lock:
            return list(reversed(self._heal_history[-limit:]))

    # ================================================================
    # 检查方法
    # ================================================================

    def _check_disk(self) -> dict:
        try:
            import shutil
            usage = shutil.disk_usage(str(DATA_DIR))
            return {
                "total_gb": round(usage.total / (1024**3), 1),
                "used_gb": round(usage.used / (1024**3), 1),
                "free_gb": round(usage.free / (1024**3), 1),
                "percent": round(usage.used / usage.total * 100, 1),
            }
        except Exception as e:
            return {"error": str(e), "percent": 0}

    def _check_memory(self) -> dict:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024**3), 1),
                "used_gb": round(mem.used / (1024**3), 1),
                "percent": mem.percent,
            }
        except Exception:
            return {"error": "psutil not available", "percent": 0}

    def _check_model(self) -> dict:
        path = str(MODEL_PATH)
        exists = Path(path).exists()
        size_mb = round(Path(path).stat().st_size / (1024**2), 1) if exists else 0
        return {"path": path, "exists": exists, "size_mb": size_mb}

    def _check_data_rw(self) -> dict:
        try:
            test_file = DATA_DIR / ".health_check"
            test_file.write_text("ok")
            test_file.unlink()
            return {"writable": True}
        except Exception:
            return {"writable": False}

    def _check_storage(self) -> dict:
        """统计数据目录大小"""
        total_bytes = 0
        file_count = 0
        try:
            for root, _, files in os.walk(str(DATA_DIR)):
                for f in files:
                    fp = Path(root) / f
                    try:
                        total_bytes += fp.stat().st_size
                        file_count += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return {
            "total_gb": round(total_bytes / (1024**3), 2),
            "total_mb": round(total_bytes / (1024**2), 1),
            "file_count": file_count,
            "quota_gb": STORAGE_MAX_GB,
        }

    def _check_gpu(self) -> dict | None:
        """检查 NVIDIA GPU 温度和利用率 (通过 nvidia-smi 或 pynvml)"""
        # 优先使用 pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            finally:
                pynvml.nvmlShutdown()
            return {
                "name": gpu_name,
                "temp_c": temp,
                "utilization_pct": util.gpu,
                "mem_used_mb": round(mem_info.used / (1024 ** 2), 0),
                "mem_total_mb": round(mem_info.total / (1024 ** 2), 0),
            }
        except Exception:
            pass

        # 回退: nvidia-smi CLI
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 4:
                    return {
                        "name": "NVIDIA GPU",
                        "temp_c": int(parts[0]),
                        "utilization_pct": int(parts[1]),
                        "mem_used_mb": int(parts[2]),
                        "mem_total_mb": int(parts[3]),
                    }
        except Exception:
            pass

        return None

    # ---- 哈希链完整性检查 ----

    @staticmethod
    def _check_hash_chain() -> dict | None:
        """检查最近 100 条记录的哈希链完整性。

        仅在 ALERT_HASH_CHAIN_ENABLED=True 时执行。
        返回 None 表示功能未启用或无数据。
        """
        try:
            from .config import ALERT_HASH_CHAIN_ENABLED
            if not ALERT_HASH_CHAIN_ENABLED:
                return None

            from .alert_store import get_alert_store
            store = get_alert_store()
            latest = store.get_latest_hash()
            if latest is None:
                return {"status": "no_data", "msg": "尚无附带哈希的记录"}

            recent = store.get_recent(limit=100)
            ids = [r["id"] for r in recent if r.get("data_hash")]
            if not ids:
                return {"status": "no_data", "msg": "最近记录无哈希"}

            result = store.verify_chain(min(ids), max(ids), max_records=len(ids))
            return {
                "status": "healthy" if result["invalid"] == 0 else "breach_detected",
                "checked": result["total"],
                "valid": result["valid"],
                "invalid": result["invalid"],
                "skipped": result.get("skipped", 0),
                "breaks": result.get("breaks", []),
            }
        except Exception as e:
            return {"status": "error", "msg": str(e)}

    # ---- 归档状态检查 ----

    @staticmethod
    def _check_retention_status() -> dict:
        """检查上次归档是否在 25 小时内完成 (允许 1 小时余量)。"""
        try:
            import json
            from .config import DATA_DIR

            progress_path = DATA_DIR / ".archive_progress.json"
            if not progress_path.exists():
                return {"overdue": False, "msg": "尚无归档记录"}

            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)

            last_time = progress.get("last_archive_time", "")
            status = progress.get("status", "unknown")
            pending = progress.get("pending_upload_keys", [])

            if not last_time:
                return {"overdue": False, "status": status, "pending_uploads": pending}

            from datetime import datetime as dt
            last_dt = dt.fromisoformat(last_time)
            hours_ago = (dt.now() - last_dt).total_seconds() / 3600

            return {
                "overdue": hours_ago > 25,
                "last_archive_time": last_time,
                "hours_ago": round(hours_ago, 1),
                "status": status,
                "pending_uploads": pending,
            }
        except Exception as e:
            return {"overdue": False, "msg": str(e)}


# 模块级单例
_health: SystemHealth | None = None


def get_health() -> SystemHealth:
    global _health
    if _health is None:
        _health = SystemHealth()
    return _health
