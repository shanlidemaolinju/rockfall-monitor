"""
预警确认与自动升级调度器 — 时间驱动的自动升级机制
==================================================
当预警发出后超过指定时间无人确认, 自动将预警等级提升一级,
并重新推送给更高级别负责人, 形成"系统发现→推送→追责"的双向闭环。

升级规则:
  blue   → yellow  (Ⅳ级→Ⅲ级: 从静默记录升级为弹窗提示)
  yellow → orange  (Ⅲ级→Ⅱ级: 从弹窗升级为多通道推送)
  orange → red     (Ⅱ级→Ⅰ级: 从多通道升级为全通道+声光报警)
  red 不升级 (已是最高级)

用法:
    from rockfall.alert_escalation import AlertEscalationScheduler
    scheduler = AlertEscalationScheduler()
    scheduler.start()
    # ...
    scheduler.shutdown()

设计原则:
  1. 每条预警仅自动升级一次 (升级后 push_status 变为 pending,
     下次扫描时已升级过的预警不在"未确认"范围内——因为 push_status
     已不是原始状态, 且有 workflow_history 记录)
  2. 仅对"已推送成功"的预警执行自动升级 (push_status in sent/popup/recorded/retry_ok)
  3. 每次升级记录审计日志和 workflow_history
"""

import logging
import threading
import time
from datetime import datetime, timedelta

from .alert_classifier import LEVEL_ORDER

logger = logging.getLogger(__name__)


class AlertEscalationScheduler:
    """预警自动升级调度器 — daemon 线程, 定期扫描未确认预警。

    特性:
      - 支持手动触发 (trigger_now)
      - 支持优雅停止 (shutdown)
      - 每次升级自动记录审计日志
      - 统计信息可查询
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_run_time: str = ""
        self._last_result: dict | None = None
        self._running = False

        # 统计
        self._total_escalations: int = 0
        self._stats_lock = threading.Lock()

    # ----------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------

    def start(self):
        """启动后台调度线程。"""
        from .config import (
            ALERT_ESCALATION_ENABLED,
            ALERT_ESCALATION_CHECK_INTERVAL_SEC,
        )

        if not ALERT_ESCALATION_ENABLED:
            logger.info("预警自动升级已禁用 (ALERT_ESCALATION_ENABLED=false)")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="alert-escalation",
        )
        self._thread.start()
        logger.info(
            "预警自动升级调度器已启动 (每 %d 秒扫描, 超时阈值 %d 分钟)",
            ALERT_ESCALATION_CHECK_INTERVAL_SEC,
            self._get_timeout_minutes(),
        )

    def shutdown(self, timeout: float = 30.0):
        """优雅停止调度器。"""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("预警自动升级调度器已停止")

    # ----------------------------------------------------------------
    # 运行循环
    # ----------------------------------------------------------------

    def _run_loop(self):
        """后台主循环: 按固定间隔扫描未确认预警。"""
        from .config import ALERT_ESCALATION_CHECK_INTERVAL_SEC

        while not self._stop.is_set():
            self._execute()

            # 分段等待以便响应 shutdown
            interval = ALERT_ESCALATION_CHECK_INTERVAL_SEC
            waited = 0
            while waited < interval and not self._stop.is_set():
                sleep_chunk = min(10, interval - waited)
                self._stop.wait(sleep_chunk)
                waited += sleep_chunk

    # ----------------------------------------------------------------
    # 执行
    # ----------------------------------------------------------------

    def trigger_now(self) -> dict:
        """手动触发一次自动升级扫描 (同步, 返回结果)。"""
        return self._execute()

    def _execute(self) -> dict:
        """执行一次完整的升级扫描流程。"""
        from .config import ALERT_ESCALATION_ENABLED

        if not ALERT_ESCALATION_ENABLED:
            return {"status": "disabled", "msg": "自动升级已禁用"}

        if self._running:
            logger.debug("自动升级扫描正在运行中, 跳过本次触发")
            return {"status": "skipped", "msg": "已有扫描在运行"}

        self._running = True
        result = {
            "status": "ok",
            "time": datetime.now().isoformat(),
            "scanned": 0,
            "escalated": 0,
            "details": [],
            "errors": [],
        }

        try:
            from .alert_store import get_alert_store
            from .alert_classifier import LEVEL_ORDER, LEVEL_LABELS

            store = get_alert_store()
            timeout_minutes = self._get_timeout_minutes()
            min_level = self._get_min_level()

            # 扫描未确认预警
            unconfirmed = store.get_unconfirmed_alerts(
                timeout_minutes=timeout_minutes,
                min_level=min_level,
            )
            result["scanned"] = len(unconfirmed)

            if not unconfirmed:
                result["msg"] = "无超时未确认的预警"
            else:
                # 逐条处理升级
                for alert in unconfirmed:
                    alert_id = alert.get("id", 0)
                    old_level = alert.get("alert_level", "")

                    # 查找升级目标等级
                    new_level = store.ESCALATION_RULES.get(old_level)
                    if new_level is None:
                        # red → 无法再升级, 跳过
                        result["details"].append({
                            "alert_id": alert_id,
                            "old_level": old_level,
                            "new_level": old_level,
                            "action": "skip",
                            "reason": "已是最高等级, 无法升级",
                        })
                        continue

                    # 仅对已推送成功的预警执行升级
                    push_status = alert.get("push_status", "")
                    if push_status not in ("sent", "popup", "recorded",
                                           "retry_ok", "pending"):
                        # 跳过推送失败或未知状态的预警
                        result["details"].append({
                            "alert_id": alert_id,
                            "old_level": old_level,
                            "new_level": new_level,
                            "action": "skip",
                            "reason": f"推送状态为 {push_status}, 不执行升级",
                        })
                        continue

                    # 检查是否已被升级过 (workflow_history 中包含 auto_escalation)
                    import json as _json
                    history_raw = alert.get("workflow_history", "[]")
                    if isinstance(history_raw, str):
                        try:
                            history = _json.loads(history_raw)
                        except (_json.JSONDecodeError, TypeError):
                            history = []
                    else:
                        history = history_raw or []

                    already_escalated = any(
                        h.get("action") == "auto_escalation"
                        for h in history
                    )
                    if already_escalated:
                        # 已经升级过, 不再重复升级
                        result["details"].append({
                            "alert_id": alert_id,
                            "old_level": old_level,
                            "new_level": old_level,
                            "action": "skip",
                            "reason": "已执行过自动升级, 跳过",
                        })
                        continue

                    # 执行升级
                    old_label = LEVEL_LABELS.get(old_level, old_level)
                    new_label = LEVEL_LABELS.get(new_level, new_level)
                    reason = (
                        f"预警 #{alert_id} ({old_label}) 发出超过 "
                        f"{timeout_minutes} 分钟无人确认, "
                        f"自动升级至 {new_label}"
                    )

                    esc_result = store.escalate_alert(
                        alert_id=alert_id,
                        new_level=new_level,
                        reason=reason,
                        operator="system",
                    )

                    if esc_result.get("ok"):
                        # 重新推送升级后的预警
                        self._re_push_escalated_alert(alert, new_level)

                        result["escalated"] += 1
                        result["details"].append({
                            "alert_id": alert_id,
                            "old_level": old_level,
                            "new_level": new_level,
                            "action": "escalated",
                            "reason": reason,
                        })

                        # 更新统计
                        with self._stats_lock:
                            self._total_escalations += 1

                        # 记录审计日志
                        try:
                            from .audit import audit_log
                            audit_log(
                                "alert_auto_escalation",
                                operator="system",
                                detail=reason,
                                alert_id=alert_id,
                                result="ok",
                                before={"alert_level": old_level},
                                after={"alert_level": new_level},
                            )
                        except Exception:
                            pass

                        logger.info(
                            "预警 #%d: %s → %s (自动升级, %d 分钟未确认)",
                            alert_id, old_level, new_level, timeout_minutes,
                        )
                    else:
                        result["errors"].append(
                            f"升级失败 #{alert_id}: {esc_result.get('msg')}"
                        )
                        logger.warning(
                            "预警 #%d 升级失败: %s",
                            alert_id, esc_result.get("msg"),
                        )

                if result["escalated"] > 0:
                    result["msg"] = (
                        f"扫描 {result['scanned']} 条未确认预警, "
                        f"自动升级 {result['escalated']} 条"
                    )

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error("预警自动升级扫描异常: %s", e)
        finally:
            self._last_run_time = datetime.now().isoformat()
            self._last_result = result
            self._running = False

        return result

    # ----------------------------------------------------------------
    # 重新推送
    # ----------------------------------------------------------------

    @staticmethod
    def _re_push_escalated_alert(alert: dict, new_level: str):
        """
        对升级后的预警重新推送到对应等级的通知渠道。

        推送策略 (统一使用 dispatch_alert, 由其内部根据等级决定渠道):
          blue   → yellow: 弹窗提示 + 持久化记录
          yellow → orange: PushPlus + 多通道推送
          orange → red:    全通道推送 + 声光报警
        """
        try:
            from .notifier import dispatch_alert

            count = alert.get("count", 0)
            max_conf = alert.get("max_confidence", 0)
            rock_diameter = alert.get("rock_diameter_cm", 0)
            class_summary = alert.get("class_summary", "") or "落石"
            saved_frame = alert.get("saved_frame", "")
            track_ids_raw = alert.get("track_ids", "[]")
            import json as _json
            if isinstance(track_ids_raw, str):
                try:
                    track_ids = _json.loads(track_ids_raw)
                except (_json.JSONDecodeError, TypeError):
                    track_ids = []
            else:
                track_ids = track_ids_raw or []

            alert_id = alert.get("id", 0)

            # 统一使用 dispatch_alert: 内部根据 alert_level 自动选择渠道
            #   yellow → 弹窗 (不推送微信)
            #   orange → 多通道推送
            #   red    → 全通道 + 声光报警
            dispatch_alert(
                count=count,
                max_confidence=max_conf,
                alert_level=new_level,
                frame_bgr=None,  # 无原始帧, 依赖已有截图
                tracks=None,
                rock_diameter_cm=rock_diameter,
            )
            logger.info(
                "预警 #%d 重新推送: %s (自动升级)", alert_id, new_level,
            )

        except Exception as e:
            logger.error("预警 #%d 重新推送失败: %s", alert.get("id", 0), e)

    # ----------------------------------------------------------------
    # 配置读取
    # ----------------------------------------------------------------

    @staticmethod
    def _get_timeout_minutes() -> int:
        from .config import ALERT_ESCALATION_TIMEOUT_MINUTES
        return ALERT_ESCALATION_TIMEOUT_MINUTES

    @staticmethod
    def _get_min_level() -> str:
        from .config import ALERT_ESCALATION_MIN_LEVEL
        return ALERT_ESCALATION_MIN_LEVEL

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

    @property
    def total_escalations(self) -> int:
        with self._stats_lock:
            return self._total_escalations


# 模块级单例
_escalation_scheduler: AlertEscalationScheduler | None = None
_scheduler_lock = threading.Lock()


def get_escalation_scheduler() -> AlertEscalationScheduler:
    """获取自动升级调度器单例。"""
    global _escalation_scheduler
    if _escalation_scheduler is None:
        with _scheduler_lock:
            if _escalation_scheduler is None:
                _escalation_scheduler = AlertEscalationScheduler()
    return _escalation_scheduler
