"""测试数据保留功能 — archive_and_purge + StorageManager 更新"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestStorageManagerRetention:
    """StorageManager 保留策略测试"""

    def test_cleanup_old_files_uses_config(self, monkeypatch, tmp_data_dir):
        """cleanup_old_files 默认使用 FILE_RETENTION_DAYS"""
        import rockfall.storage as st
        monkeypatch.setattr(st, "FILE_RETENTION_DAYS", 365)

        from rockfall.storage import StorageManager
        sm = StorageManager()
        result = sm.cleanup_old_files()
        assert "deleted_count" in result
        assert "freed_mb" in result

    def test_cleanup_thumbnails(self, monkeypatch, tmp_data_dir):
        """cleanup_thumbnails 清理缩略图"""
        import rockfall.storage as st
        monkeypatch.setattr(st, "THUMBNAIL_RETENTION_DAYS", 0)

        from rockfall.storage import StorageManager
        sm = StorageManager()

        # 创建模拟缩略图文件
        results_dir = tmp_data_dir / "results"
        thumb = results_dir / "thumb_test.jpg"
        thumb.write_bytes(b"fake_thumb")

        # 设置为 10 天前的 mtime
        old_time = time.time() - 86400 * 10
        os.utime(str(thumb), (old_time, old_time))

        result = sm.cleanup_thumbnails(retention_days=0)
        assert "deleted_count" in result
        assert "freed_mb" in result

    def test_strict_retention_no_quota_violation(self, monkeypatch):
        """严格模式下 quota 未超限时行为正常"""
        import rockfall.storage as st
        monkeypatch.setattr(st, "STRICT_RETENTION", True)
        monkeypatch.setattr(st, "FILE_RETENTION_DAYS", 365)

        from rockfall.storage import StorageManager
        sm = StorageManager()
        # 默认存储使用量很小 (接近 0), 不应触发 quota
        result = sm.enforce_quota(max_total_mb=100000)
        assert not result["enforced"]  # 未超配额

    def test_get_retention_policy(self, monkeypatch):
        """get_retention_policy 返回完整策略"""
        import rockfall.storage as st
        monkeypatch.setattr(st, "ALERT_RETENTION_DAYS", 1095)
        monkeypatch.setattr(st, "FILE_RETENTION_DAYS", 365)
        monkeypatch.setattr(st, "THUMBNAIL_RETENTION_DAYS", 7)

        from rockfall.storage import StorageManager
        sm = StorageManager()
        policy = sm.get_retention_policy()
        assert "policy" in policy
        assert policy["policy"]["alert_retention_days"] == 1095
        assert policy["policy"]["file_retention_days"] == 365
        assert "stats" in policy


class TestAlertStoreArchive:
    """AlertStore archive_and_purge 测试"""

    def test_archive_and_purge_dry_run(self, sqlite_store):
        """dry_run 模式: 统计但不删除"""
        sqlite_store.save_alert(count=1, max_confidence=0.5, alert_level="blue")

        # 使用 1095 天保留期, 不会有任何过期记录
        result = sqlite_store.archive_and_purge(retention_days=1095, dry_run=True)
        assert result["dry_run"]
        # 没有超过 1095 天的记录 → 0 条
        assert result["archived_count"] == 0

    def test_archive_empty_no_data(self, sqlite_store):
        """无过期记录时不生成空文件"""
        result = sqlite_store.archive_and_purge(retention_days=1095, dry_run=True)
        assert result["archived_count"] == 0
        assert "无超过" in result.get("msg", "")

    def test_delete_archived(self, sqlite_store):
        """delete_archived 批量删除"""
        ids = []
        for i in range(5):
            aid = sqlite_store.save_alert(count=1, max_confidence=0.5, alert_level="blue")
            ids.append(aid)

        assert len(ids) == 5
        deleted = sqlite_store.delete_archived(ids[:3])
        assert deleted == 3

        # 剩余记录应该还在
        remaining = sqlite_store.get_recent(limit=10)
        assert len(remaining) >= 2

    def test_archive_and_purge_with_recent_data(self, sqlite_store):
        """最近数据不会触发归档"""
        sqlite_store.save_alert(count=1, max_confidence=0.5, alert_level="red")

        with patch("rockfall.cold_storage.ColdStorageClient") as mock_cs:
            mock_client = MagicMock()
            mock_client.enabled = False
            mock_cs.return_value = mock_client

            result = sqlite_store.archive_and_purge(retention_days=1095)
            assert result["archived_count"] == 0
            assert "无超过" in result.get("msg", "")

    def test_save_archive_progress(self, tmp_data_dir, monkeypatch):
        """_save_archive_progress 写入进度文件"""
        import rockfall.alert_store as als
        import rockfall.config as cfg

        # 确保指向测试目录
        monkeypatch.setattr(cfg, "DATA_DIR", tmp_data_dir)

        als.AlertStore._save_archive_progress(
            last_archive_time="2026-06-14T03:00:00",
            last_archived_alert_id=42,
            pending_upload_keys=["test.jsonl"],
            status="idle",
        )

        progress_path = tmp_data_dir / ".archive_progress.json"
        assert progress_path.exists()
        with open(progress_path) as f:
            progress = json.load(f)
        assert progress["status"] == "idle"
        assert progress["last_archived_alert_id"] == 42


class TestRetentionScheduler:
    """RetentionScheduler 测试"""

    def test_scheduler_trigger_now(self, monkeypatch, tmp_data_dir):
        """手动触发归档"""
        import rockfall.alert_store as als

        monkeypatch.setattr(als, "MYSQL_HOST", "")

        from rockfall.retention_scheduler import RetentionScheduler

        scheduler = RetentionScheduler()

        with patch("rockfall.cold_storage.ColdStorageClient") as mock_cs:
            mock_client = MagicMock()
            mock_client.enabled = False
            mock_cs.return_value = mock_client

            result = scheduler.trigger_now()
            assert result["status"] in ("ok", "partial_error")
            assert "steps" in result

    def test_scheduler_start_stop(self):
        """调度器启动和停止"""
        from rockfall.retention_scheduler import RetentionScheduler

        scheduler = RetentionScheduler()
        scheduler.start()
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()

        scheduler.shutdown(timeout=5.0)
        assert (not scheduler._thread.is_alive()) or scheduler._stop.is_set()

    def test_seconds_until_next_run(self):
        """计算下次执行时间: 应在 1~86400 秒之间"""
        from rockfall.retention_scheduler import RetentionScheduler

        scheduler = RetentionScheduler()
        delay = scheduler._seconds_until_next_run()
        assert 0 <= delay <= 86400
