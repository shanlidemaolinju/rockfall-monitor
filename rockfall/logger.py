"""
结构化日志层 — 标准 logging + JSON 输出 + 文件持久化
====================================================
使用 Python 标准 logging 模块，JSON 格式输出到 stdout（Docker 兼容），
同时写 JSONL 文件用于看板历史查询（read_logs 向后兼容）。

特性:
  - JSON 格式输出到 stdout，兼容 docker logs / Fluentd / Promtail
  - 同步写入 JSONL 文件，支持 read_logs() 向后兼容
  - RateLimitFilter — 高频事件限流（ERROR 永不过滤）
  - 通过 RuntimeConfig 动态调整日志级别
  - log_event() API 向后兼容
"""

import atexit
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR

# ============================================================
# 文件持久化配置（向后兼容 read_logs）
# ============================================================
LOG_FILE = DATA_DIR / "detection_log.jsonl"
_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
_MAX_ROTATIONS = 5
_file_lock = threading.Lock()

# ============================================================
# 日志级别映射
# ============================================================
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class RateLimitFilter(logging.Filter):
    """基于时间窗口的令牌桶限流过滤器。

    ERROR 及以上级别永不过滤。
    默认: 每 60 秒最多 10 条非 ERROR 日志。
    """

    def __init__(self, rate: int = 10, per: float = 60.0):
        super().__init__()
        self.rate = rate
        self.per = per
        self._counter = 0
        self._last_reset = time.time()
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        with self._lock:
            now = time.time()
            if now - self._last_reset > self.per:
                self._counter = 0
                self._last_reset = now
            self._counter += 1
            return self._counter <= self.rate


class JSONFormatter(logging.Formatter):
    """输出 JSON 行到 stdout，自动携带 trace 上下文。"""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "event": getattr(record, "event_type", record.msg),
        }

        # 合并 extra_fields（来自 log_event 的 **kwargs）
        extra = getattr(record, "extra_fields", None)
        if extra:
            entry.update(extra)

        # 携带 trace 上下文
        try:
            from .trace import get_trace_context
            trace = get_trace_context()
            if trace:
                entry["trace"] = trace
        except Exception:
            pass

        # 异常信息
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])

        return json.dumps(entry, ensure_ascii=False, default=str) + "\n"


class _LogManager:
    """日志管理器单例 — 延迟初始化以避免循环导入。"""

    def __init__(self):
        self._logger: logging.Logger | None = None
        self._setup_lock = threading.Lock()
        self._rate_filter: RateLimitFilter | None = None

    def _ensure_setup(self):
        if self._logger is not None:
            return
        with self._setup_lock:
            if self._logger is not None:
                return
            self._logger = logging.getLogger("rockfall")
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False

            # stdout handler — JSON 格式
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(JSONFormatter())
            stdout_handler.setLevel(logging.DEBUG)
            self._rate_filter = RateLimitFilter(rate=10, per=60.0)
            stdout_handler.addFilter(self._rate_filter)
            self._logger.addHandler(stdout_handler)

    @property
    def logger(self) -> logging.Logger:
        self._ensure_setup()
        return self._logger  # type: ignore[return-value]

    def set_level(self, level: str):
        """动态调整日志级别（配合 RuntimeConfig 热更新）。"""
        self._ensure_setup()
        if level.upper() in _LEVEL_MAP:
            self._logger.setLevel(_LEVEL_MAP[level.upper()])

    def update_rate_limit(self, rate: int, per: float = 60.0):
        """动态更新限流参数。"""
        if self._rate_filter:
            with self._rate_filter._lock:
                self._rate_filter.rate = rate
                self._rate_filter.per = per


_log_manager = _LogManager()


# ============================================================
# 公共 API — 向后兼容
# ============================================================

def log_event(event_type: str, level: str = "INFO", **kwargs):
    """
    记录一条事件（线程安全，输出到 stdout + JSONL 文件）。

    参数:
        event_type: "detection" | "alert" | "system"
        level:      "DEBUG" | "INFO" | "WARN" | "ERROR"
        **kwargs:   事件的附加信息（如 frame, count, msg 等）
    """
    logger = _log_manager.logger
    log_level = _LEVEL_MAP.get(level.upper(), logging.INFO)

    # 创建 LogRecord 并附加自定义字段
    record = logger.makeRecord(
        logger.name, log_level, "(unknown)", 0, event_type, (), None,
    )
    record.event_type = event_type
    record.extra_fields = kwargs

    logger.handle(record)

    # 同步写入 JSONL 文件（看板历史查询用）
    _write_file(record, kwargs)


def _write_file(record: logging.LogRecord, extra: dict):
    """写入 JSONL 文件（线程安全）。"""
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": record.levelname,
        "event": getattr(record, "event_type", record.msg),
        **extra,
    }
    # 携带 trace 上下文
    try:
        from .trace import get_trace_context
        trace = get_trace_context()
        if trace:
            entry["trace"] = trace
    except Exception:
        pass

    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"

    with _file_lock:
        try:
            if LOG_FILE.exists() and LOG_FILE.stat().st_size >= _MAX_SIZE:
                _rotate()
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception as e:
            print(f"[logger] 日志文件写入失败: {e}", file=sys.stderr)


def flush():
    """强制刷新（进程退出前由 atexit 调用，JSONL 文件已同步写入故无需额外操作）。"""
    for handler in _log_manager.logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def read_logs(limit: int = 100) -> list[dict]:
    """读取最近 N 条日志（向后兼容，供看板统计使用）。"""
    if not LOG_FILE.exists():
        return []

    lines = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return lines[-limit:]


def clear_logs():
    """清空日志文件。"""
    with _file_lock:
        LOG_FILE.unlink(missing_ok=True)


def _rotate():
    """轮转日志文件: log.jsonl → log.1.jsonl, log.1.jsonl → log.2.jsonl, ..."""
    for i in range(_MAX_ROTATIONS - 1, 0, -1):
        old = LOG_FILE.with_suffix(f".{i}.jsonl")
        new = LOG_FILE.with_suffix(f".{i + 1}.jsonl")
        if old.exists():
            try:
                os.replace(str(old), str(new))
            except OSError:
                pass
    try:
        os.replace(str(LOG_FILE), str(LOG_FILE.with_suffix(".1.jsonl")))
    except OSError:
        pass


def setup_logging(level: str = "INFO", rate: int = 10, per: float = 60.0):
    """显式初始化日志系统（可选，首次 log_event 自动调用）。

    参数:
        level: 日志级别 ("DEBUG" | "INFO" | "WARN" | "ERROR")
        rate:  限流速率（每 per 秒最多 rate 条非 ERROR 日志）
        per:   限流时间窗口（秒）
    """
    _log_manager.set_level(level)
    _log_manager.update_rate_limit(rate, per)


def check_dynamic_level():
    """从 RuntimeConfig 读取 LOG_LEVEL 并应用（供检测循环定期调用）。

    若未配置 LOG_LEVEL 则跳过，保持当前级别。
    """
    try:
        from .config import RuntimeConfig
        level = RuntimeConfig.get("LOG_LEVEL", "")
        if level:
            _log_manager.set_level(str(level))
    except Exception:
        pass


# 进程退出时刷新 stdout handler
atexit.register(flush)
