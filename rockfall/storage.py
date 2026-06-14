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

from .config import (
    DATA_DIR, RESULTS_DIR,
    FILE_RETENTION_DAYS, THUMBNAIL_RETENTION_DAYS,
    STRICT_RETENTION, ALERT_RETENTION_DAYS,
)


class StorageManager:
    """存储管理器 — 清理过期文件, 配额控制"""

    def __init__(self):
        self._results_dir = RESULTS_DIR
        self._clips_dir = RESULTS_DIR / "clips"
        self._alerts_dir = RESULTS_DIR / "alerts"
        self._uploads_dir = DATA_DIR / "uploads"

    def cleanup_old_files(self, retention_days: int | None = None,
                          dry_run: bool = False) -> dict:
        """
        清理超过保留期的文件。

        参数:
            retention_days: 保留天数 (默认 FILE_RETENTION_DAYS, 即 365 天)
            dry_run: True 则只统计不实际删除

        返回: {"deleted_count": int, "freed_mb": float, "errors": [...]}
        """
        if retention_days is None:
            retention_days = FILE_RETENTION_DAYS
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
                      min_retention_days: int | None = None) -> dict:
        """
        强制执行存储配额。当总存储超过 max_total_mb 时，从最旧文件开始删除。

        清理优先级: 缩略图 > 临时文件 > 普通帧 > 预警帧 > 视频片段
        严格模式 (STRICT_RETENTION=true): 即使磁盘满也不删除未到期文件。

        返回: {"enforced": True/False, "freed_mb": float, "deleted_count": int}
        """
        if min_retention_days is None:
            min_retention_days = FILE_RETENTION_DAYS

        stats = self.get_storage_stats()
        current_mb = stats["total_mb"]

        if current_mb <= max_total_mb:
            return {"enforced": False, "freed_mb": 0, "deleted_count": 0,
                    "current_mb": current_mb, "quota_mb": max_total_mb}

        # 严格模式: 不删除未到期文件
        if STRICT_RETENTION:
            return {
                "enforced": False,
                "freed_mb": 0,
                "deleted_count": 0,
                "current_mb": round(current_mb, 1),
                "quota_mb": max_total_mb,
                "msg": "严格保留模式已启用, 跳过配额清理。磁盘使用率过高, 请扩容或手动归档。",
            }

        # 超配额, 逐步缩短保留期
        deleted = 0
        freed_bytes = 0
        retention = FILE_RETENTION_DAYS

        while retention > min_retention_days and current_mb > max_total_mb:
            result = self.cleanup_old_files(retention_days=retention)
            deleted += result["deleted_count"]
            freed_bytes += result["freed_mb"] * 1024 * 1024
            retention -= 30  # 每次缩短 30 天
            stats = self.get_storage_stats()
            current_mb = stats["total_mb"]

        return {
            "enforced": True,
            "freed_mb": round(freed_bytes / (1024 * 1024), 1),
            "deleted_count": deleted,
            "current_mb": round(current_mb, 1),
            "quota_mb": max_total_mb,
        }

    def auto_cleanup_schedule(self, retention_days: int | None = None,
                              max_total_mb: int = 10000) -> dict:
        """一键自动清理: 先清理缩略图 → 再按保留期清理 → 最后检查配额"""
        # 1. 优先清理缩略图 (低价值, 短保留期)
        thumb_result = self.cleanup_thumbnails()
        # 2. 清理过期文件
        result = self.cleanup_old_files(retention_days=retention_days)
        result["thumbnails"] = thumb_result
        # 3. 配额管理
        quota_result = self.enforce_quota(max_total_mb=max_total_mb)
        result["quota"] = quota_result
        return result

    def cleanup_thumbnails(self,
                           retention_days: int | None = None) -> dict:
        """
        清理低质量缩略图 (320x240 级别)。

        缩略图保留期更短 (默认 7 天)，优先于原始帧清理。

        返回: {"deleted_count": int, "freed_mb": float}
        """
        if retention_days is None:
            retention_days = THUMBNAIL_RETENTION_DAYS

        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        freed_bytes = 0

        # 主要清理 results 目录中的缩略图 (thumb_*.jpg)
        thumbnail_patterns = ["thumb_*.jpg", "*_thumb.jpg", "*_lowres.jpg"]
        dirs = [self._results_dir, self._alerts_dir]

        for d in dirs:
            if not d.exists():
                continue
            try:
                for pattern in thumbnail_patterns:
                    for f in d.glob(pattern):
                        if not f.is_file():
                            continue
                        try:
                            if f.stat().st_mtime < cutoff:
                                size = f.stat().st_size
                                f.unlink()
                                deleted += 1
                                freed_bytes += size
                        except Exception:
                            pass
            except Exception:
                pass

        return {
            "deleted_count": deleted,
            "freed_mb": round(freed_bytes / (1024 * 1024), 1),
        }

    def get_retention_policy(self) -> dict:
        """返回当前保留策略配置和存储统计。"""
        stats = self.get_storage_stats()
        return {
            "policy": {
                "alert_retention_days": ALERT_RETENTION_DAYS,
                "file_retention_days": FILE_RETENTION_DAYS,
                "thumbnail_retention_days": THUMBNAIL_RETENTION_DAYS,
                "strict_retention": STRICT_RETENTION,
            },
            "stats": stats,
        }
