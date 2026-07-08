"""
阈值自动调参 — 基于误报率的数据闭环 (v2.4+)
============================================
周期性读取各点位复核后的误报率, 用带滞回的 PID 式控制自动微调
detection_confidence, 实现"检测→复核→统计→自动调参"的完整闭环。

调整策略:
  误报率 > target + hysteresis → 提高阈值 (变严格, 减少误报)
  误报率 < target - hysteresis → 降低阈值 (变宽松, 减少漏报)
  在容忍带内 → 不调整

用法:
    from rockfall.threshold_tuner import get_tuner
    tuner = get_tuner()
    tuner.start()
    # ...
    tuner.shutdown()

设计原则:
  1. 下坡慢、上坡快: step_down < step_up (漏报代价高于误报)
  2. 不跨点位影响: 每个 site 独立调整 detection_confidence
  3. 仅调 detection_confidence: 不动四级预警分界阈值
  4. 双写生效: SiteStore (持久化) + RuntimeConfig (即时热更新)
  5. 有界调整: conf 永远在 [MIN, MAX] 区间内
"""

import logging
import threading
import time
from datetime import datetime

from .logger import log_event

logger = logging.getLogger(__name__)


class ThresholdAutoTuner:
    """阈值自动调参器 — daemon 线程, 定期扫描各点位误报率并微调阈值。

    特性:
      - 支持手动触发 (trigger_now)
      - 支持优雅停止 (shutdown)
      - 低流量点位自动扩展回看窗口
      - 每次调整记录审计日志 (含 direction 字段)
      - 统计信息可查询
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_run_time: str = ""
        self._last_result: dict | None = None
        self._running = False

        # 统计
        self._total_adjustments: int = 0
        self._total_tighten: int = 0
        self._total_relax: int = 0
        self._stats_lock = threading.Lock()

    # ----------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------

    def start(self):
        """启动后台调参线程。"""
        from .config import THRESHOLD_AUTO_TUNE_ENABLED, THRESHOLD_AUTO_TUNE_INTERVAL_HOURS

        if not THRESHOLD_AUTO_TUNE_ENABLED:
            logger.info("阈值自动调参已禁用 (THRESHOLD_AUTO_TUNE_ENABLED=false)")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="threshold-tuner",
        )
        self._thread.start()
        logger.info(
            "阈值自动调参已启动 (每 %d 小时扫描, 目标误报率 %.0f%%)",
            THRESHOLD_AUTO_TUNE_INTERVAL_HOURS,
            self._get_target_fp_rate() * 100,
        )

    def shutdown(self, timeout: float = 30.0):
        """优雅停止调参器。"""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("阈值自动调参已停止 "
                    "(累计调整 %d 次: tighten=%d, relax=%d)",
                    self._total_adjustments, self._total_tighten,
                    self._total_relax)

    # ----------------------------------------------------------------
    # 运行循环
    # ----------------------------------------------------------------

    def _run_loop(self):
        """后台主循环: 按固定间隔扫描各点位误报率。"""
        from .config import THRESHOLD_AUTO_TUNE_INTERVAL_HOURS

        # 首次延迟 5 分钟启动 (给系统预热时间, 避免刚启动时样本不足)
        initial_delay = 300
        logger.info("阈值自动调参将在 %d 秒后首次扫描", initial_delay)
        self._stop.wait(initial_delay)
        if self._stop.is_set():
            return

        while not self._stop.is_set():
            self._execute()

            # 分段等待以便响应 shutdown
            interval = THRESHOLD_AUTO_TUNE_INTERVAL_HOURS * 3600
            waited = 0
            while waited < interval and not self._stop.is_set():
                sleep_chunk = min(60, interval - waited)
                self._stop.wait(sleep_chunk)
                waited += sleep_chunk

    # ----------------------------------------------------------------
    # 执行
    # ----------------------------------------------------------------

    def trigger_now(self) -> dict:
        """手动触发一次调参扫描 (同步, 返回结果)。"""
        return self._execute()

    def _execute(self) -> dict:
        """执行一次完整的调参扫描。"""
        from .config import THRESHOLD_AUTO_TUNE_ENABLED

        if not THRESHOLD_AUTO_TUNE_ENABLED:
            return {"status": "disabled", "msg": "阈值自动调参已禁用"}

        if self._running:
            logger.debug("阈值调参扫描正在运行中, 跳过本次触发")
            return {"status": "skipped", "msg": "已有扫描在运行"}

        self._running = True
        result = {
            "status": "ok",
            "time": datetime.now().isoformat(),
            "sites_scanned": 0,
            "sites_adjusted": 0,
            "details": [],
            "errors": [],
        }

        try:
            from .site_config import list_sites, get_site_store
            from .alert_store import get_alert_store

            store = get_site_store()
            alert_store = get_alert_store()
            sites = list_sites()

            if not sites:
                result["msg"] = "无活跃监测点位"
                return result

            result["sites_scanned"] = len(sites)

            for site in sites:
                try:
                    detail = self._tune_site(site, store, alert_store)
                    result["details"].append(detail)
                    if detail.get("adjusted"):
                        result["sites_adjusted"] += 1
                except Exception as e:
                    site_id = getattr(site, 'site_id', 'unknown')
                    err_msg = f"site={site_id}: {e}"
                    result["errors"].append(err_msg)
                    logger.warning("调参异常 — %s", err_msg)

            if result["sites_adjusted"] > 0:
                result["msg"] = (
                    f"扫描 {result['sites_scanned']} 个点位, "
                    f"调整 {result['sites_adjusted']} 个"
                )
            else:
                result["msg"] = (
                    f"扫描 {result['sites_scanned']} 个点位, "
                    f"无需调整"
                )

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error("阈值自动调参扫描异常: %s", e)
        finally:
            self._last_run_time = datetime.now().isoformat()
            self._last_result = result
            self._running = False

        return result

    # ----------------------------------------------------------------
    # 单点位调参逻辑
    # ----------------------------------------------------------------

    def _tune_site(self, site, store, alert_store) -> dict:
        """
        对单个点位执行阈值调参。

        返回:
            {
                "site_id": str,
                "adjusted": bool,
                "direction": "tighten" | "relax" | "",
                "old_conf": float,
                "new_conf": float,
                "fp_rate": float,
                "reviewed_count": int,
                "lookback_days": int,
                "reason": str,
            }
        """
        from .config import (
            THRESHOLD_AUTO_TUNE_LOOKBACK_DAYS,
            THRESHOLD_AUTO_TUNE_TARGET_FP_RATE,
            THRESHOLD_AUTO_TUNE_HYSTERESIS,
            THRESHOLD_AUTO_TUNE_MIN_SAMPLES,
            THRESHOLD_AUTO_TUNE_STEP_UP,
            THRESHOLD_AUTO_TUNE_STEP_DOWN,
            THRESHOLD_AUTO_TUNE_CONF_MIN,
            THRESHOLD_AUTO_TUNE_CONF_MAX,
        )

        site_id = site.site_id

        # 1. 获取当前有效阈值
        thresholds = site.get_thresholds()
        old_conf = thresholds["detection_confidence"]

        # 2. 获取该点位的误报率统计
        lookback_days = THRESHOLD_AUTO_TUNE_LOOKBACK_DAYS
        samples = alert_store.get_false_alarm_stats(days=lookback_days)

        # 两级回退: 样本不足 → 扩展到 30 天 → 降至实际样本数(≥10)
        actual_lookback = lookback_days
        min_samples = THRESHOLD_AUTO_TUNE_MIN_SAMPLES

        if samples["total_reviewed"] < min_samples:
            # 回退1: 扩展到 30 天
            samples = alert_store.get_false_alarm_stats(days=30)
            actual_lookback = 30

        if samples["total_reviewed"] < min_samples:
            # 回退2: 降至实际样本数 (至少 10 条)
            if samples["total_reviewed"] >= 10:
                logger.warning(
                    "site=%s 复核样本不足 (n=%d < MIN=%d), 使用可用数据",
                    site_id, samples["total_reviewed"], min_samples,
                )
            else:
                logger.info(
                    "site=%s 复核样本严重不足 (n=%d < 10), 跳过",
                    site_id, samples["total_reviewed"],
                )
                return {
                    "site_id": site_id,
                    "adjusted": False,
                    "direction": "",
                    "old_conf": old_conf,
                    "new_conf": old_conf,
                    "fp_rate": samples["false_alarm_rate"],
                    "reviewed_count": samples["total_reviewed"],
                    "lookback_days": actual_lookback,
                    "reason": f"样本严重不足 ({samples['total_reviewed']} < 10)",
                }

        fp_rate = samples["false_alarm_rate"]
        target = THRESHOLD_AUTO_TUNE_TARGET_FP_RATE
        hysteresis = THRESHOLD_AUTO_TUNE_HYSTERESIS

        # 3. 滞回判断
        if fp_rate > target + hysteresis:
            direction = "tighten"
            new_conf = min(old_conf + THRESHOLD_AUTO_TUNE_STEP_UP,
                          THRESHOLD_AUTO_TUNE_CONF_MAX)
        elif fp_rate < target - hysteresis:
            direction = "relax"
            new_conf = max(old_conf - THRESHOLD_AUTO_TUNE_STEP_DOWN,
                          THRESHOLD_AUTO_TUNE_CONF_MIN)
        else:
            # 在容忍带内, 不调整
            return {
                "site_id": site_id,
                "adjusted": False,
                "direction": "",
                "old_conf": old_conf,
                "new_conf": old_conf,
                "fp_rate": fp_rate,
                "reviewed_count": samples["total_reviewed"],
                "lookback_days": actual_lookback,
                "reason": f"FP rate {fp_rate:.3f} 在容忍带内 "
                          f"({target - hysteresis:.2f}~{target + hysteresis:.2f})",
            }

        # 4. 检查是否已达边界
        if abs(new_conf - old_conf) < 0.001:
            boundary = "上限" if direction == "tighten" else "下限"
            return {
                "site_id": site_id,
                "adjusted": False,
                "direction": direction,
                "old_conf": old_conf,
                "new_conf": old_conf,
                "fp_rate": fp_rate,
                "reviewed_count": samples["total_reviewed"],
                "lookback_days": actual_lookback,
                "reason": f"已达置信度{boundary} ({old_conf:.3f}), 无法继续调整",
            }

        # 5. 持久化: SiteStore (重启后生效)
        site.detection_confidence = new_conf
        site.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store.update(site)

        # 6. 热更新: RuntimeConfig (即时生效, detector 每帧读取)
        try:
            from .config import RuntimeConfig
            RuntimeConfig.set("DETECTION_CONFIDENCE", new_conf)
        except Exception as e:
            logger.warning("RuntimeConfig 热更新失败 (持久化已成功): %s", e)

        # 7. 审计日志
        direction_label = "收紧" if direction == "tighten" else "放松"
        detail = (
            f"site={site_id} fp_rate={fp_rate:.3f} ({samples['total_reviewed']}条复核) "
            f"→ 置信度 {old_conf:.3f}→{new_conf:.3f} ({direction_label})"
        )
        try:
            from .audit import audit_log
            audit_log(
                "threshold_auto_tune",
                operator="system",
                detail=detail,
                before={
                    "detection_confidence": old_conf,
                    "fp_rate": round(fp_rate, 4),
                },
                after={
                    "detection_confidence": new_conf,
                    "direction": direction,
                    "fp_rate": round(fp_rate, 4),
                    "reviewed_count": samples["total_reviewed"],
                    "false_alarm_count": samples["false_alarm"],
                    "lookback_days": actual_lookback,
                },
            )
        except Exception:
            pass

        # 8. 更新统计
        with self._stats_lock:
            self._total_adjustments += 1
            if direction == "tighten":
                self._total_tighten += 1
            else:
                self._total_relax += 1

        direction_cn = "收紧" if direction == "tighten" else "放松"
        logger.info(
            "site=%s: FP rate=%.3f → %s 置信度 %.3f→%.3f",
            site_id, fp_rate, direction_cn, old_conf, new_conf,
        )
        log_event("threshold_auto_tune", level="INFO",
                  site_id=site_id, direction=direction,
                  old_conf=round(old_conf, 4), new_conf=round(new_conf, 4),
                  fp_rate=round(fp_rate, 4),
                  reviewed_count=samples["total_reviewed"],
                  lookback_days=actual_lookback)

        return {
            "site_id": site_id,
            "adjusted": True,
            "direction": direction,
            "old_conf": old_conf,
            "new_conf": new_conf,
            "fp_rate": fp_rate,
            "reviewed_count": samples["total_reviewed"],
            "lookback_days": actual_lookback,
            "reason": detail,
        }

    # ----------------------------------------------------------------
    # 配置读取
    # ----------------------------------------------------------------

    @staticmethod
    def _get_target_fp_rate() -> float:
        from .config import THRESHOLD_AUTO_TUNE_TARGET_FP_RATE
        return THRESHOLD_AUTO_TUNE_TARGET_FP_RATE

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
    def total_adjustments(self) -> int:
        with self._stats_lock:
            return self._total_adjustments

    @property
    def total_tighten(self) -> int:
        with self._stats_lock:
            return self._total_tighten

    @property
    def total_relax(self) -> int:
        with self._stats_lock:
            return self._total_relax


# ══════════════════════════════════════════════════════════════
# 模块级单例
# ══════════════════════════════════════════════════════════════

_tuner: ThresholdAutoTuner | None = None
_tuner_lock = threading.Lock()


def get_tuner() -> ThresholdAutoTuner:
    """获取 ThresholdAutoTuner 单例。"""
    global _tuner
    if _tuner is None:
        with _tuner_lock:
            if _tuner is None:
                _tuner = ThresholdAutoTuner()
    return _tuner
