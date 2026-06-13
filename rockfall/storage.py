"""
视频存储管理模块 — 自动清理 + 配额管理
======================================
定期清理过期检测帧和回放片段，防止磁盘写满。

用法:
    from rockfall.storage import StorageManager
    sm = StorageManager()
    sm.cleanup_old_files(retention_days=30)  # 清理30天前的文件
    sm.get_storage_stats()                   # 获取存储统计
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from .config import DATA_DIR, RESULTS_DIR


class StorageManager:
    """存储管理器 — 清理过期文件, 配额控制"""

    def __init__(self):
        self._results_dir = RESULTS_DIR
        self._clips_dir = RESULTS_DIR / "clips"
        self._alerts_dir = RESULTS_DIR / "alerts"
        self._uploads_dir = DATA_DIR / "uploads"

    def cleanup_old_files(self, retention_days: int = 30,
                          dry_run: bool = False) -> dict:
        """
        清理超过保留期的文件。

        参数:
            retention_days: 保留天数 (默认 30 天)
            dry_run: True 则只统计不实际删除

        返回: {"deleted_count": int, "freed_mb": float, "errors": [...]}
        """
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        freed_bytes = 0
        errors = []

        dirs_to_clean = [
            self._results_dir,     # stream_*.jpg 检测帧
            self._clips_dir,       # *.mp4 回放片段
            self._alerts_dir,      # alert_*.jpg 预警截图
            self._uploads_dir,     # 上传的原始视频
        ]

        for dir_path in dirs_to_clean:
            if not dir_path.exists():
                continue
            try:
                for f in dir_path.iterdir():
                    if not f.is_file():
                        continue
                    try:
                        mtime = f.stat().st_mtime
                        if mtime < cutoff:
                            size = f.stat().st_size
                            if not dry_run:
                                f.unlink()
                            deleted += 1
                            freed_bytes += size
                    except Exception as e:
                        errors.append(str(e))
            except Exception as e:
                errors.append(f"扫描目录失败 {dir_path}: {e}")

        return {
            "deleted_count": deleted,
            "freed_mb": round(freed_bytes / (1024 * 1024), 1),
            "errors": errors,
            "dry_run": dry_run,
        }

    def get_storage_stats(self) -> dict:
        """获取各目录存储统计"""
        stats = {}
        dirs = {
            "results": self._results_dir,
            "clips": self._clips_dir,
            "alerts": self._alerts_dir,
            "uploads": self._uploads_dir,
        }
        for name, d in dirs.items():
            total_bytes = 0
            file_count = 0
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        try:
                            total_bytes += f.stat().st_size
                            file_count += 1
                        except Exception:
                            pass
            stats[name] = {
                "path": str(d),
                "file_count": file_count,
                "size_mb": round(total_bytes / (1024 * 1024), 1),
            }
        total_mb = sum(s["size_mb"] for s in stats.values())
        stats["total_mb"] = total_mb
        return stats

    def enforce_quota(self, max_total_mb: int = 10000,
                      min_retention_days: int = 7) -> dict:
        """
        强制执行存储配额。当总存储超过 max_total_mb 时，从最旧文件开始删除，
        但至少保留 min_retention_days 天内的文件。

        返回: {"enforced": True/False, "freed_mb": float, "deleted_count": int}
        """
        stats = self.get_storage_stats()
        current_mb = stats["total_mb"]

        if current_mb <= max_total_mb:
            return {"enforced": False, "freed_mb": 0, "deleted_count": 0,
                    "current_mb": current_mb, "quota_mb": max_total_mb}

        # 超配额, 逐步缩短保留期
        deleted = 0
        freed_bytes = 0
        retention = 30

        while retention > min_retention_days and current_mb > max_total_mb:
            result = self.cleanup_old_files(retention_days=retention)
            deleted += result["deleted_count"]
            freed_bytes += result["freed_mb"] * 1024 * 1024
            retention -= 5
            stats = self.get_storage_stats()
            current_mb = stats["total_mb"]

        return {
            "enforced": True,
            "freed_mb": round(freed_bytes / (1024 * 1024), 1),
            "deleted_count": deleted,
            "current_mb": round(current_mb, 1),
            "quota_mb": max_total_mb,
        }

    def auto_cleanup_schedule(self, retention_days: int = 30,
                              max_total_mb: int = 10000) -> dict:
        """一键自动清理: 先按保留期清理，再检查配额"""
        result = self.cleanup_old_files(retention_days=retention_days)
        quota_result = self.enforce_quota(max_total_mb=max_total_mb)
        result["quota"] = quota_result
        return result
