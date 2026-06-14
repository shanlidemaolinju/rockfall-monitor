"""
定时归档调度器 — 后台线程 + 健康检查集成
=========================================
每天在指定时间自动执行:
  1. 清理低质量缩略图 (THUMBNAIL_RETENTION_DAYS)
  2. 清理过期检测帧/片段 (FILE_RETENTION_DAYS)
  3. 配额管理 (enforce_quota)
  4. 归档并删除 DB 旧记录 (archive_and_purge)

用法:
    from rockfall.retention_scheduler import RetentionScheduler
    scheduler = RetentionScheduler()
    scheduler.start()
    # ...
    scheduler.shutdown()
"""

import logging
import threading
import time
from datetime import datetime, timedelta

from .config import ARCHIVE_SCHEDULE_HOUR

logger = logging.getLogger(__name__)


class RetentionScheduler:
    """定时归档调度器 — daemon 线程, 每天执行一次。

    特性:
      - 支持手动触发 (trigger_now)
      - 支持优雅停止 (shutdown)
      - 自动记录审计日志
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_run_time: str = ""
        self._last_result: dict | None = None
        self._running = False

    # ----------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------

    def start(self):
        """启动后台调度线程。"""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="retention-scheduler",
        )
        self._thread.start()
        logger.info("归档调度器已启动 (每天 %d:00 执行)", ARCHIVE_SCHEDULE_HOUR)

    def shutdown(self, timeout: float = 30.0):
        """优雅停止调度器。"""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("归档调度器已停止")

    # ----------------------------------------------------------------
    # 运行循环
    # ----------------------------------------------------------------

    def _run_loop(self):
        """后台主循环: 计算距离下次执行时间的延迟, 等待, 然后执行。"""
        while not self._stop.is_set():
            delay = self._seconds_until_next_run()
            logger.debug("归档调度器: 下次执行在 %d 秒后", delay)
            # 分段等待以便响应 shutdown
            while delay > 0 and not self._stop.is_set():
                sleep_chunk = min(delay, 60)
                self._stop.wait(sleep_chunk)
                delay -= sleep_chunk

            if self._stop.is_set():
                break

            self._execute()

    def _seconds_until_next_run(self) -> int:
        """计算距离下次 ARCHIVE_SCHEDULE_HOUR 的秒数。"""
        now = datetime.now()
        target = now.replace(
            hour=ARCHIVE_SCHEDULE_HOUR, minute=7, second=0, microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        return int((target - now).total_seconds())

    # ----------------------------------------------------------------
    # 执行
    # ----------------------------------------------------------------

    def trigger_now(self) -> dict:
        """手动触发一次归档 (同步, 返回结果)。"""
        return self._execute()

    def _execute(self) -> dict:
        """执行完整的归档流程。"""
        if self._running:
            logger.warning("归档流程正在运行中, 跳过本次触发")
            return {"status": "skipped", "msg": "已有归档流程在运行"}

        self._running = True
        result = {
            "status": "ok",
            "time": datetime.now().isoformat(),
            "steps": {},
            "errors": [],
        }

        sm = None  # StorageManager 惰性初始化

        try:
            # 1. 清理缩略图
            try:
                from .storage import StorageManager
                sm = StorageManager()
                thumb_result = sm.cleanup_thumbnails()
                result["steps"]["thumbnails"] = thumb_result
            except Exception as e:
                result["errors"].append(f"缩略图清理失败: {e}")

            # 2. 清理过期文件
            try:
                if sm is None:
                    from .storage import StorageManager
                    sm = StorageManager()
                file_result = sm.cleanup_old_files()
                result["steps"]["files"] = file_result
            except Exception as e:
                result["errors"].append(f"文件清理失败: {e}")

            # 3. 配额管理
            try:
                if sm is None:
                    from .storage import StorageManager
                    sm = StorageManager()
                quota_result = sm.enforce_quota()
                result["steps"]["quota"] = quota_result
            except Exception as e:
                result["errors"].append(f"配额管理失败: {e}")

            # 4. 归档 DB 旧记录
            try:
                from .alert_store import get_alert_store
                store = get_alert_store()
                archive_result = store.archive_and_purge()
                result["steps"]["archive"] = archive_result
            except Exception as e:
                result["errors"].append(f"归档失败: {e}")

            # 5. 审计日志
            if not result["errors"]:
                result["status"] = "ok"
                logger.info("归档调度完成: %s", _summarize(result))
            else:
                result["status"] = "partial_error"
                logger.warning("归档调度部分失败: %s", result["errors"])

            # 记录审计日志
            try:
                from .audit import audit_log
                audit_log(
                    "retention_archive",
                    operator="scheduler",
                    detail=_summarize(result),
                    result=result["status"],
                )
            except Exception:
                pass

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error("归档调度异常: %s", e)
        finally:
            self._last_run_time = datetime.now().isoformat()
            self._last_result = result
            self._running = False

        return result

    # ----------------------------------------------------------------
    # 查询
    # ----------------------------------------------------------------

    @property
    def last_run_time(self) -> str:
        return self._last_run_time

    @property
    def last_result(self) -> dict | None:
        return self._last_result

    @property
    def is_running(self) -> bool:
        return self._running


def _summarize(result: dict) -> str:
    """生成归档结果摘要文本。"""
    steps = result.get("steps", {})
    parts = []
    if "thumbnails" in steps:
        parts.append(f"缩略图={steps['thumbnails'].get('deleted_count', 0)}个")
    if "files" in steps:
        parts.append(f"文件={steps['files'].get('deleted_count', 0)}个")
    if "archive" in steps:
        parts.append(f"归档={steps['archive'].get('archived_count', 0)}条")
    if "quota" in steps:
        parts.append(f"配额={'已执行' if steps['quota'].get('enforced') else '未触发'}")
    return ", ".join(parts)
