"""
模型版本轮询线程 — 定期检查远程新版本并下载
============================================
daemon 线程，每隔 MODEL_REGISTRY_POLL_INTERVAL_SEC 秒检查 S3/OSS
是否有新模型版本，有则自动下载并通知运维。

用法:
    from rockfall.model_poller import ModelPoller
    poller = ModelPoller()
    poller.start()
    # ...
    poller.shutdown()
"""

import logging
import threading
import time
from datetime import datetime, timedelta

from .config import MODEL_REGISTRY_POLL_INTERVAL_SEC, MODEL_REGISTRY_ENABLED
from .logger import log_event

logger = logging.getLogger(__name__)


class ModelPoller:
    """后台轮询线程 — 检查远程新版本并自动下载。

    特性:
      - 支持手动触发 (poll_now)
      - 支持优雅停止 (shutdown)
      - 远程版本列表缓存 (避免频繁 S3 list 操作)
      - 下载完成后日志通知 (可选推送告警)
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_poll_time: str = ""
        self._last_poll_result: dict | None = None
        self._running = False
        self._version_cache: list[dict] = []
        self._registry = None  # 惰性初始化

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self):
        """启动后台轮询线程。"""
        if not MODEL_REGISTRY_ENABLED:
            logger.info("模型注册表未启用, 跳过轮询线程启动")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="model-poller",
        )
        self._thread.start()
        logger.info("模型轮询线程已启动 (间隔 %ds)", MODEL_REGISTRY_POLL_INTERVAL_SEC)

    def shutdown(self, timeout: float = 10.0):
        """优雅停止轮询线程。"""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("模型轮询线程已停止")

    # ── 运行循环 ──────────────────────────────────────────────

    def _run_loop(self):
        """后台主循环"""
        # 启动后立即执行一次
        self._execute()

        while not self._stop.is_set():
            interval = max(60, MODEL_REGISTRY_POLL_INTERVAL_SEC)
            # 分段等待以便响应 shutdown
            waited = 0
            while waited < interval and not self._stop.is_set():
                sleep_chunk = min(interval - waited, 60)
                self._stop.wait(sleep_chunk)
                waited += sleep_chunk

            if self._stop.is_set():
                break

            self._execute()

    # ── 执行 ──────────────────────────────────────────────────

    def poll_now(self) -> dict:
        """手动触发一次远程版本检查 (同步, 返回结果)。"""
        return self._execute()

    def _execute(self) -> dict:
        """执行一次远程版本检查 + 自动下载新版本。"""
        if self._running:
            logger.warning("模型轮询正在执行中, 跳过本次触发")
            return {"status": "skipped", "msg": "已有轮询在执行"}

        self._running = True
        result = {
            "status": "ok",
            "time": datetime.now().isoformat(),
            "remote_versions": [],
            "downloaded": [],
            "errors": [],
        }

        try:
            from .model_registry import get_registry
            if self._registry is None:
                self._registry = get_registry()

            if not self._registry.enabled:
                result["status"] = "disabled"
                return result

            # 1. 列出远程版本
            try:
                remote = self._registry.check_remote_versions()
                result["remote_versions"] = remote
                self._version_cache = remote
            except Exception as e:
                result["errors"].append(f"远程版本检查失败: {e}")
                result["status"] = "error"
                return result

            # 2. 下载本地缺失的新版本 (通过快照避免直接迭代 _versions)
            local_names = set(v["name"] for v in self._registry.get_status().get("versions", []))
            for rv in remote:
                name = rv.get("name", "")
                if name and name not in local_names and name.endswith(".pt"):
                    try:
                        logger.info("发现新模型版本: %s (%.1fMB)", name, rv.get("size_mb", 0))
                        downloaded_path = self._registry.download_model(name)
                        result["downloaded"].append({
                            "name": name,
                            "path": str(downloaded_path),
                        })
                        log_event("model_poller", level="INFO",
                                  msg=f"新模型已自动下载: {name}")
                    except Exception as e:
                        err_msg = f"下载模型失败 {name}: {e}"
                        result["errors"].append(err_msg)
                        logger.warning(err_msg)

            if result["downloaded"]:
                result["status"] = "new_versions_downloaded"
                # 通知运维 (可选)
                try:
                    from .notifier import send_alert_async
                    names = ", ".join(d["name"] for d in result["downloaded"])
                    send_alert_async(
                        alert_type="model_update",
                        title=f"📦 新模型版本已下载: {names}",
                        content=f"从对象存储下载了 {len(result['downloaded'])} 个新模型版本，"
                                f"请通过管理 API 确认是否激活。",
                    )
                except Exception:
                    pass

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error("模型轮询异常: %s", e)
        finally:
            self._last_poll_time = datetime.now().isoformat()
            self._last_poll_result = result
            self._running = False

        return result

    # ── 查询 ──────────────────────────────────────────────────

    @property
    def last_poll_time(self) -> str:
        return self._last_poll_time

    @property
    def last_poll_result(self) -> dict | None:
        return self._last_poll_result

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def version_cache(self) -> list[dict]:
        return list(self._version_cache)


# ══════════════════════════════════════════════════════════════
# 模块级单例
# ══════════════════════════════════════════════════════════════

_poller: ModelPoller | None = None


def get_poller() -> ModelPoller:
    """获取或创建 ModelPoller 单例。"""
    global _poller
    if _poller is not None:
        return _poller
    _poller = ModelPoller()
    return _poller
