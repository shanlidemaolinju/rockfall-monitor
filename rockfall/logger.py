"""
日志层 — 检测事件持久化 + 日志轮转 + 缓冲写入
==============================================
将每次检测事件写入本地 JSON 日志文件，文件超过上限自动轮转。
使用内存缓冲区批量写入，减少磁盘 IO。

格式 (每行一个 JSON 对象):
  {"time": "2026-05-26 17:00:00", "event": "detection",
   "count": 3, "max_conf": 0.85, "tracks": [{"id": 1, "bbox": [...], ...}]}
"""

import atexit
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR

LOG_FILE = DATA_DIR / "detection_log.jsonl"
_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
_MAX_ROTATIONS = 5
_BUF_SIZE = 32  # 积攒 N 条后批量写入
_lock = threading.Lock()
_buffer: list[str] = []


def log_event(event_type: str, level: str = "INFO", **kwargs):
    """
    记录一条检测事件 (线程安全, 缓冲写入)。

    参数:
        event_type: "detection" | "alert" | "system"
        level:      "DEBUG" | "INFO" | "WARN" | "ERROR"
        **kwargs:   事件的附加信息 (如 frame, source, msg 等)
    """
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "event": event_type,
        **kwargs,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"

    with _lock:
        _buffer.append(line)
        if len(_buffer) >= _BUF_SIZE:
            _flush_locked()


def _flush_locked():
    """在持有锁的情况下写入缓冲 (调用方负责加锁)"""
    if not _buffer:
        return
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size >= _MAX_SIZE:
            _rotate()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.writelines(_buffer)
            f.flush()
        _buffer.clear()
    except Exception as e:
        import sys
        print(f"[logger] 日志写入失败: {e}", file=sys.stderr)


def flush():
    """强制刷新缓冲 (进程退出前调用)"""
    with _lock:
        _flush_locked()


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


def read_logs(limit: int = 100) -> list[dict]:
    """读取最近 N 条日志 (先刷新缓冲确保完整性)"""
    flush()
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
    """清空日志文件"""
    with _lock:
        _buffer.clear()
    LOG_FILE.unlink(missing_ok=True)


# 进程退出时自动刷新缓冲
atexit.register(flush)
