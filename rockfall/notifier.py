"""
通知层 — PushPlus 微信报警推送 + 四级分级预警调度
====================================================
四级预警 (对齐《公路自然灾害监测预警系统技术指南》):
  Ⅰ 级 (特别严重，红色):   → 微信推送 + 声音报警 + 红色弹窗
  Ⅱ 级 (严重，橙色):       → 微信推送通知
  Ⅲ 级 (较重，黄色):       → 界面弹窗提示 (不推送微信)
  Ⅳ 级 (一般，蓝色):       → 仅本地记录 (不推送、不弹窗)

调度逻辑由 dispatch_alert() 统一入口处理。

依赖: rockfall.config, rockfall.logger, rockfall.alert_store
"""

import base64
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import requests

from .config import (
    PUSHPLUS_TOKEN,
    PUSHPLUS_TOPIC,
    PUSHPLUS_URL,
    ALERT_COOLDOWN_SECONDS,
    get_location,
    RESULTS_DIR,
    PUSH_EXECUTOR_WORKERS,
)
from .logger import log_event


class AlertManager:
    """报警管理器 — 封装冷却计时器和连续帧确认状态 (实例级, 线程安全)"""

    def __init__(self):
        self._last_alert_time: float = 0
        self._confirm_buffer: dict[int, list] = {}
        self._executor = ThreadPoolExecutor(max_workers=PUSH_EXECUTOR_WORKERS, thread_name_prefix="alert")

    # ---- 公共 API ----

    def send(
        self, count: int, max_confidence: float, image_url: str = "",
        frame_bgr: np.ndarray | None = None, tracks: list[dict] | None = None,
        confirm_frames: int = 1, alert_level: str = "yellow",
        rock_diameter_cm: float = 0, monitoring_location: str = "",
    ) -> dict:
        """发送落石/滑坡报警到微信 (同步)。返回 {"code": ..., "msg": ...}"""
        gate = self._check_gates(tracks, confirm_frames)
        if gate:
            return gate

        self._last_alert_time = time.time()
        alert_path = self._save_frame(frame_bgr) if frame_bgr is not None else ""
        class_summary = _build_class_summary(tracks)
        title, content = _build_message(
            count, max_confidence, image_url, frame_bgr,
            tracks, alert_level, class_summary,
        )
        result = _push_with_retry(title, content)

        # 持久化
        push_status = "sent" if result.get("code") == 200 else "pending"
        track_ids = [t["id"] for t in (tracks or [])]
        try:
            from .alert_store import get_alert_store
            get_alert_store().save_alert(
                count=count, max_confidence=max_confidence,
                track_ids=track_ids, alert_level=alert_level,
                saved_frame=alert_path, push_status=push_status,
                class_summary=class_summary,
                rock_diameter_cm=rock_diameter_cm,
                monitoring_location=monitoring_location or get_location(),
            )
        except Exception:
            pass

        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, push_result=result,
                  saved_to=alert_path or None)
        return result

    def send_async(
        self,
        count: int,
        max_confidence: float,
        image_url: str = "",
        frame_bgr: np.ndarray | None = None,
        tracks: list[dict] | None = None,
        confirm_frames: int = 1,
        alert_level: str = "yellow",
        rock_diameter_cm: float = 0,
        monitoring_location: str = "",
    ):
        """异步发送落石报警 (不阻塞检测流水线)"""
        self._executor.submit(
            self.send, count, max_confidence,
            image_url=image_url, frame_bgr=frame_bgr,
            tracks=tracks, confirm_frames=confirm_frames,
            alert_level=alert_level,
            rock_diameter_cm=rock_diameter_cm,
            monitoring_location=monitoring_location,
        )

    # ---- 内部 ----

    def _check_gates(self, tracks, confirm_frames: int) -> dict | None:
        """返回 None 表示通过, 返回 dict 表示被拦截 (冷却/未确认/未配置)"""
        if confirm_frames > 1 and tracks:
            if not self._check_confirm(tracks, confirm_frames):
                return {"code": 0, "msg": f"等待连续{confirm_frames}帧确认"}
        now = time.time()
        if now - self._last_alert_time < ALERT_COOLDOWN_SECONDS:
            return {"code": 0, "msg": "冷却中"}
        if not PUSHPLUS_TOKEN or PUSHPLUS_TOKEN == "your_token_here":
            return {"code": -1, "msg": "未配置 PUSHPLUS_TOKEN"}
        return None

    @staticmethod
    def _save_frame(frame_bgr: np.ndarray) -> str:
        """保存检测帧到 alerts 目录, 返回路径或空字符串"""
        try:
            alerts_dir = RESULTS_DIR / "alerts"
            alerts_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = str(alerts_dir / f"alert_{ts}.jpg")
            cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return path
        except Exception:
            return ""

    def _check_confirm(self, tracks: list[dict], threshold: int) -> bool:
        """检查是否有目标已连续出现 threshold 帧"""
        for t in tracks:
            tid = t["id"]
            if tid in self._confirm_buffer and t.get("age", 0) <= 1:
                del self._confirm_buffer[tid]
            if tid not in self._confirm_buffer:
                self._confirm_buffer[tid] = []
            self._confirm_buffer[tid].append(t)
            if len(self._confirm_buffer[tid]) > threshold:
                self._confirm_buffer[tid] = self._confirm_buffer[tid][-threshold:]
            if len(self._confirm_buffer[tid]) >= threshold:
                return True
        active_ids = {t["id"] for t in tracks}
        for tid in list(self._confirm_buffer.keys()):
            if tid not in active_ids:
                del self._confirm_buffer[tid]
        return False


# ---- 四级分级预警调度 (核心) ----

def dispatch_alert(
    count: int,
    max_confidence: float,
    alert_level: str,
    image_url: str = "",
    frame_bgr: np.ndarray | None = None,
    tracks: list[dict] | None = None,
    confirm_frames: int = 1,
    rock_diameter_cm: float = 0,
) -> dict:
    """
    四级分级预警统一调度入口 (对齐《公路自然灾害监测预警系统技术指南》强制要求)。

    调度逻辑:
      Ⅳ 级 (蓝色): 仅本地写入报警数据库，不弹窗不推送
      Ⅲ 级 (黄色): 写入数据库 + 触发界面弹窗 (通过 SSE 通知前端)
      Ⅱ 级 (橙色): 写入数据库 + 调用 PushPlus 推送微信通知
      Ⅰ 级 (红色): 写入数据库 + 微信推送 + 触发声光报警

    返回: {"code": ..., "msg": ..., "alert_level": ..., "action": ...}
    """
    from .alert_store import get_alert_store

    store = get_alert_store()
    track_ids = [t["id"] for t in (tracks or [])]
    class_summary = _build_class_summary(tracks)
    alert_path = ""

    # 保存报警帧截图 (所有等级都保存)
    if frame_bgr is not None:
        try:
            from datetime import datetime
            import cv2
            from .config import RESULTS_DIR
            alerts_dir = RESULTS_DIR / "alerts"
            alerts_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            alert_path = str(alerts_dir / f"alert_{alert_level}_{ts}.jpg")
            cv2.imwrite(alert_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception:
            alert_path = ""

    # ---- Ⅳ 级 (蓝色): 仅本地记录 ----
    if alert_level == "blue":
        store.save_alert(
            count=count, max_confidence=max_confidence,
            track_ids=track_ids, alert_level=alert_level,
            saved_frame=alert_path, push_status="recorded",
            class_summary=class_summary,
            rock_diameter_cm=rock_diameter_cm,
            monitoring_location=get_location(),
        )
        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, alert_level="blue",
                  action="record_only", saved_to=alert_path or None,
                  rock_diameter_cm=rock_diameter_cm)
        return {"code": 0, "msg": "Ⅳ级·仅本地记录", "alert_level": "blue", "action": "record_only"}

    # ---- Ⅲ 级 (黄色): 本地记录 + 界面弹窗提示 ----
    if alert_level == "yellow":
        store.save_alert(
            count=count, max_confidence=max_confidence,
            track_ids=track_ids, alert_level=alert_level,
            saved_frame=alert_path, push_status="popup",
            class_summary=class_summary,
            rock_diameter_cm=rock_diameter_cm,
            monitoring_location=get_location(),
        )
        # 写入共享状态供 SSE 推送
        _set_latest_popup_alert(alert_level, count, max_confidence,
                                class_summary, alert_path, track_ids,
                                rock_diameter_cm)
        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, alert_level="yellow",
                  action="popup", saved_to=alert_path or None,
                  rock_diameter_cm=rock_diameter_cm)
        return {"code": 0, "msg": "Ⅲ级·界面弹窗提示", "alert_level": "yellow", "action": "popup"}

    # ---- Ⅱ 级 (橙色): 微信推送通知 ----
    if alert_level == "orange":
        push_result = _default_manager.send(
            count=count, max_confidence=max_confidence,
            image_url=image_url, frame_bgr=frame_bgr,
            tracks=tracks, confirm_frames=confirm_frames,
            alert_level=alert_level,
            rock_diameter_cm=rock_diameter_cm,
            monitoring_location=get_location(),
        )
        # 同时触发界面弹窗 (橙色也要弹窗)
        _set_latest_popup_alert(alert_level, count, max_confidence,
                                class_summary, alert_path, track_ids,
                                rock_diameter_cm)
        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, alert_level="orange",
                  action="wechat_push", push_result=push_result,
                  saved_to=alert_path or None,
                  rock_diameter_cm=rock_diameter_cm)
        return {"code": push_result.get("code", 0),
                "msg": "Ⅱ级·微信推送通知",
                "alert_level": "orange", "action": "wechat_push"}

    # ---- Ⅰ 级 (红色): 微信推送 + 声光报警 ----
    if alert_level == "red":
        push_result = _default_manager.send(
            count=count, max_confidence=max_confidence,
            image_url=image_url, frame_bgr=frame_bgr,
            tracks=tracks, confirm_frames=confirm_frames,
            alert_level=alert_level,
            rock_diameter_cm=rock_diameter_cm,
            monitoring_location=get_location(),
        )
        # 触发声光报警 + 红色弹窗 (通过 SSE 通知前端)
        _set_latest_popup_alert(alert_level, count, max_confidence,
                                class_summary, alert_path, track_ids,
                                rock_diameter_cm, sound_alarm=True)
        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, alert_level="red",
                  action="wechat_push+sound_alarm", push_result=push_result,
                  saved_to=alert_path or None,
                  rock_diameter_cm=rock_diameter_cm)
        return {"code": push_result.get("code", 0),
                "msg": "Ⅰ级·微信推送+声光报警",
                "alert_level": "red", "action": "wechat_push+sound_alarm"}

    # fallback
    return {"code": -1, "msg": "未知预警等级", "alert_level": alert_level, "action": "none"}


def dispatch_alert_async(
    count: int,
    max_confidence: float,
    alert_level: str,
    image_url: str = "",
    frame_bgr: np.ndarray | None = None,
    tracks: list[dict] | None = None,
    confirm_frames: int = 1,
    rock_diameter_cm: float = 0,
):
    """四级分级预警异步调度 (不阻塞检测流水线)"""
    _default_manager._executor.submit(
        dispatch_alert, count, max_confidence, alert_level,
        image_url=image_url, frame_bgr=frame_bgr,
        tracks=tracks, confirm_frames=confirm_frames,
        rock_diameter_cm=rock_diameter_cm,
    )


# ---- 共享弹窗状态 (线程安全, 供 SSE 端点读取) ----

import threading as _threading
_latest_popup: dict | None = None
_popup_lock = _threading.Lock()
_popup_event = _threading.Event()


def _set_latest_popup_alert(alert_level: str, count: int, max_confidence: float,
                            class_summary: str, saved_frame: str,
                            track_ids: list, rock_diameter_cm: float = 0,
                            sound_alarm: bool = False):
    """写入最新弹窗预警到共享状态, 唤醒所有等待的 SSE 连接"""
    global _latest_popup
    with _popup_lock:
        _latest_popup = {
            "alert_level": alert_level,
            "count": count,
            "max_confidence": max_confidence,
            "class_summary": class_summary,
            "saved_frame": saved_frame,
            "track_ids": track_ids,
            "rock_diameter_cm": rock_diameter_cm,
            "sound_alarm": sound_alarm,
            "timestamp": datetime.now().isoformat(),
        }
    _popup_event.set()   # 唤醒所有等待的 SSE 连接
    _popup_event.clear()  # 重置供下次使用


def get_and_clear_popup_alert() -> dict | None:
    """SSE 端点调用: 获取最新弹窗预警并清除 (消费语义)"""
    global _latest_popup
    with _popup_lock:
        alert = _latest_popup
        _latest_popup = None
    return alert


def wait_for_popup_alert(timeout: float = 30.0) -> dict | None:
    """SSE 端点调用: 阻塞等待新的弹窗预警 (最多 timeout 秒)"""
    if _popup_event.wait(timeout=timeout):
        return get_and_clear_popup_alert()
    return None


# ---- 模块级默认实例 (向后兼容) ----
_default_manager = AlertManager()


def send_alert(
    count: int,
    max_confidence: float,
    image_url: str = "",
    frame_bgr: np.ndarray | None = None,
    tracks: list[dict] | None = None,
    confirm_frames: int = 1,
    alert_level: str = "yellow",
) -> dict:
    """向后兼容的同步报警接口, 委托给默认 AlertManager"""
    return _default_manager.send(
        count, max_confidence, image_url=image_url,
        frame_bgr=frame_bgr, tracks=tracks, confirm_frames=confirm_frames,
        alert_level=alert_level,
    )


def send_alert_async(
    count: int,
    max_confidence: float,
    image_url: str = "",
    frame_bgr: np.ndarray | None = None,
    tracks: list[dict] | None = None,
    confirm_frames: int = 1,
    alert_level: str = "yellow",
):
    """向后兼容的异步报警接口, 委托给默认 AlertManager"""
    _default_manager.send_async(
        count, max_confidence, image_url=image_url,
        frame_bgr=frame_bgr, tracks=tracks, confirm_frames=confirm_frames,
        alert_level=alert_level,
    )


def _frame_to_base64(frame_bgr: np.ndarray, quality: int = 60) -> str | None:
    """BGR numpy 数组 → JPEG base64 字符串"""
    try:
        _, buffer = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        from .logger import log_event
        log_event("system", msg=f"帧Base64编码失败: {e}")
        return None


# ---- 推送内容构建 (模块级, 无状态) ----

def _build_class_summary(tracks: list[dict] | None) -> str:
    """统计已确认轨迹的类别分布, 如 "落石:2, 滑坡:1" """
    if not tracks:
        return ""
    from collections import Counter
    cls_counts = Counter(
        t.get("class_name", "落石") for t in tracks if t.get("confirmed")
    )
    return ", ".join(f"{name}:{cnt}" for name, cnt in cls_counts.items()) if cls_counts else ""


def _build_message(count: int, max_confidence: float, image_url: str,
                   frame_bgr: np.ndarray | None, tracks: list[dict] | None,
                   alert_level: str, class_summary: str) -> tuple[str, str]:
    """构建 PushPlus 标题和 HTML 内容"""
    # 四级预警标签 (对齐交通部标准)
    level_labels = {
        "red":    "🔴 Ⅰ级·特别严重",
        "orange": "🟠 Ⅱ级·严重",
        "yellow": "🟡 Ⅲ级·较重",
        "blue":   "🔵 Ⅳ级·一般",
    }
    level_label = level_labels.get(alert_level, "⚠️ 预警")
    if "滑坡" in class_summary:
        event_type = "滑坡+落石"
    elif class_summary:
        event_type = class_summary.replace(":", "")
    else:
        event_type = "落石"
    loc = get_location()
    title = f"{level_label} {event_type}报警：{loc}"

    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"<p>📍 位置：{loc}</p>"
        f"<p>🕒 检测时间：{detection_time}</p>"
        f"<p>📦 目标数量：{count}</p>"
        f"<p>🎯 最大置信度：{max_confidence:.2f}</p>"
    )

    if tracks:
        confirmed = [t for t in tracks if t.get("confirmed")]
        if confirmed:
            parts = []
            for t in confirmed:
                cls_label = f"[{t.get('class_name', '')}] " if t.get("class_id") else ""
                parts.append(f"{cls_label}#{t['id']} (置信度{t['confidence']:.2f}, 存活{t['age']}帧)")
            content += "<p>🏷️ 稳定跟踪目标：" + " ".join(parts) + "</p>"

    if image_url:
        content += f'<img src="{image_url}" width="100%">'
    elif frame_bgr is not None:
        b64 = _frame_to_base64(frame_bgr)
        if b64:
            content += f'<img src="data:image/jpeg;base64,{b64}" width="100%">'

    return title, content


def _push_with_retry(title: str, content: str) -> dict:
    """向 PushPlus 发送推送, 最多 3 次重试"""
    if not PUSHPLUS_TOKEN or PUSHPLUS_TOKEN == "your_token_here":
        return {"code": -1, "msg": "未配置 PUSHPLUS_TOKEN"}

    data = {"token": PUSHPLUS_TOKEN, "title": title, "content": content,
            "topic": PUSHPLUS_TOPIC, "template": "html"}
    result = {"code": -1, "msg": ""}
    for attempt in range(3):
        try:
            res = requests.post(PUSHPLUS_URL, json=data, timeout=10).json()
            result = {"code": res.get("code"), "msg": res.get("msg")}
            break
        except Exception as e:
            result = {"code": -1, "msg": str(e)}
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
    return result
