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
            rock_diameter_cm=rock_diameter_cm,
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
      Ⅱ 级 (橙色): 写入数据库 + 多通道推送 (PushPlus + 邮件)
      Ⅰ 级 (红色): 写入数据库 + 全通道推送 + 触发声光报警

    推送通道由环境变量 ALERT_CHANNEL_MAP 配置，格式:
      red=pushplus,smtp,wecom,dingtalk;orange=pushplus,smtp;yellow=;blue=
    未配置时使用内置默认值。

    返回: {"code": ..., "msg": ..., "alert_level": ..., "action": ...}
    """
    from .alert_store import get_alert_store
    from .metrics import record_alert

    # 递增 Prometheus 告警计数器（按等级）
    record_alert(alert_level)

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
        _set_latest_popup_alert(alert_level, count, max_confidence,
                                class_summary, alert_path, track_ids,
                                rock_diameter_cm)
        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, alert_level="yellow",
                  action="popup", saved_to=alert_path or None,
                  rock_diameter_cm=rock_diameter_cm)
        return {"code": 0, "msg": "Ⅲ级·界面弹窗提示", "alert_level": "yellow", "action": "popup"}

    # ---- 构建推送消息 (orange/red 共用) ----
    title, content = _build_message(
        count, max_confidence, image_url, frame_bgr,
        tracks, alert_level, class_summary,
        rock_diameter_cm=rock_diameter_cm,
    )

    # ---- Ⅱ 级 (橙色): PushPlus + 多通道推送 ----
    if alert_level == "orange":
        push_result = _default_manager.send(
            count=count, max_confidence=max_confidence,
            image_url=image_url, frame_bgr=frame_bgr,
            tracks=tracks, confirm_frames=confirm_frames,
            alert_level=alert_level,
            rock_diameter_cm=rock_diameter_cm,
            monitoring_location=get_location(),
        )
        multich_result = _push_via_registry(title, content, alert_level)
        _set_latest_popup_alert(alert_level, count, max_confidence,
                                class_summary, alert_path, track_ids,
                                rock_diameter_cm)
        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, alert_level="orange",
                  action="multichannel", push_result=push_result,
                  multichannel=multich_result,
                  saved_to=alert_path or None,
                  rock_diameter_cm=rock_diameter_cm)
        return {"code": push_result.get("code", 0),
                "msg": "Ⅱ级·多通道推送",
                "alert_level": "orange", "action": "multichannel"}

    # ---- Ⅰ 级 (红色): 全通道推送 + 声光报警 ----
    if alert_level == "red":
        push_result = _default_manager.send(
            count=count, max_confidence=max_confidence,
            image_url=image_url, frame_bgr=frame_bgr,
            tracks=tracks, confirm_frames=confirm_frames,
            alert_level=alert_level,
            rock_diameter_cm=rock_diameter_cm,
            monitoring_location=get_location(),
        )
        multich_result = _push_via_registry(title, content, alert_level)
        _set_latest_popup_alert(alert_level, count, max_confidence,
                                class_summary, alert_path, track_ids,
                                rock_diameter_cm, sound_alarm=True)
        log_event("alert", count=count, max_confidence=max_confidence,
                  track_ids=track_ids, alert_level="red",
                  action="all_channels+sound_alarm", push_result=push_result,
                  multichannel=multich_result,
                  saved_to=alert_path or None,
                  rock_diameter_cm=rock_diameter_cm)
        return {"code": push_result.get("code", 0),
                "msg": "Ⅰ级·全通道推送+声光报警",
                "alert_level": "red", "action": "all_channels+sound_alarm"}

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

def _build_disposal_suggestions(alert_level: str, rock_diameter_cm: float = 0) -> str:
    """根据预警等级生成处置建议 (委派给 alert_classifier.get_response_workflow)"""
    from .alert_classifier import get_response_workflow

    workflow = get_response_workflow(alert_level)
    lines = [f"{i+1}. {step}" for i, step in enumerate(workflow["disposal_steps"])]

    if rock_diameter_cm > 0:
        lines.insert(1, f"落石估算直径: {rock_diameter_cm:.0f}cm")
    return "\n".join(lines)


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
                   alert_level: str, class_summary: str,
                   rock_diameter_cm: float = 0) -> tuple[str, str]:
    """构建推送标题和 HTML 内容 (含处置建议)"""
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
        f"<p>📍 <b>位置</b>：{loc}</p>"
        f"<p>🕒 <b>检测时间</b>：{detection_time}</p>"
        f"<p>📦 <b>目标数量</b>：{count}</p>"
        f"<p>🎯 <b>最大置信度</b>：{max_confidence:.2f}</p>"
    )
    if rock_diameter_cm > 0:
        content += f"<p>📏 <b>落石估算直径</b>：{rock_diameter_cm:.0f} cm</p>"

    if tracks:
        confirmed = [t for t in tracks if t.get("confirmed")]
        if confirmed:
            parts = []
            for t in confirmed:
                cls_label = f"[{t.get('class_name', '')}] " if t.get("class_id") else ""
                parts.append(f"{cls_label}#{t['id']} (置信度{t['confidence']:.2f}, 存活{t['age']}帧)")
            content += "<p>🏷️ 稳定跟踪目标：" + " ".join(parts) + "</p>"

    # 处置建议
    suggestions = _build_disposal_suggestions(alert_level, rock_diameter_cm)
    content += f"<hr><p><b>📋 处置建议</b>：</p><pre style='font-size:0.9em;'>{suggestions}</pre>"

    if image_url:
        content += f'<img src="{image_url}" width="100%">'
    elif frame_bgr is not None:
        b64 = _frame_to_base64(frame_bgr)
        if b64:
            content += f'<img src="data:image/jpeg;base64,{b64}" width="100%">'

    return title, content


# ---- 多通道推送 (邮件 / 企业微信 / 短信) ----


def _send_email(subject: str, body_html: str, to_emails: list[str] | None = None) -> dict:
    """
    通过 SMTP 发送预警邮件。

    环境变量配置:
      SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD
      ALERT_EMAIL_TO (逗号分隔的收件人列表)

    返回: {"code": 200/0, "msg": ...}
    """
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = os.getenv("SMTP_HOST", "")
    if not smtp_host:
        return {"code": 0, "msg": "SMTP 未配置"}

    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    recipients = to_emails or [
        e.strip() for e in os.getenv("ALERT_EMAIL_TO", "").split(",") if e.strip()
    ]
    if not recipients:
        return {"code": 0, "msg": "邮件收件人未配置 (ALERT_EMAIL_TO)"}

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(body_html, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipients, msg.as_string())

        return {"code": 200, "msg": f"邮件已发送至 {len(recipients)} 位收件人"}
    except Exception as e:
        return {"code": -1, "msg": f"邮件发送失败: {e}"}


def _send_wecom(title: str, content: str) -> dict:
    """
    通过企业微信机器人 Webhook 发送 Markdown 预警消息。

    环境变量:
      WECOM_WEBHOOK_URL — 企业微信群机器人 Webhook 地址

    返回: {"code": 200/0, "msg": ...}
    """
    import os

    webhook_url = os.getenv("WECOM_WEBHOOK_URL", "")
    if not webhook_url:
        return {"code": 0, "msg": "企业微信 Webhook 未配置 (WECOM_WEBHOOK_URL)"}

    # 将 HTML 简化为纯文本 (企业微信仅支持有限的 Markdown)
    import re
    plain = re.sub(r"<[^>]+>", "", content)
    plain = re.sub(r"\n\s*\n", "\n", plain).strip()[:2000]  # 限制长度

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "content": f"## {title}\n\n{plain}\n\n> 系统自动发送 · RockGuard v2.0"
        }
    }
    try:
        r = requests.post(webhook_url, json=payload, timeout=10)
        if r.status_code == 200 and r.json().get("errcode") == 0:
            return {"code": 200, "msg": "企业微信推送成功"}
        return {"code": 0, "msg": f"企业微信返回: {r.text[:100]}"}
    except Exception as e:
        return {"code": -1, "msg": f"企业微信推送失败: {e}"}


def _send_sms_via_email(phone_numbers: list[str], subject: str, body: str) -> dict:
    """
    通过邮件转短信网关发送短信 (中国移动/联通/电信均支持)。

    环境变量:
      SMS_GATEWAY_EMAIL — 短信网关邮箱域名 (如 @139.com 移动)
      SMTP_* — 复用 SMTP 配置

    每个运营商的邮箱网关:
      移动: number@139.com
      联通: number@wo.cn
      电信: number@189.cn

    返回: {"code": 200/0, "msg": ...}
    """
    import os

    gateway = os.getenv("SMS_GATEWAY_EMAIL", "")
    if not gateway:
        return {"code": 0, "msg": "短信网关未配置 (SMS_GATEWAY_EMAIL)"}

    sms_emails = [f"{p}{gateway}" for p in phone_numbers]
    body_short = body[:500]  # 短信长度限制
    return _send_email(subject=subject, body_html=f"<pre>{body_short}</pre>",
                       to_emails=sms_emails)


# ---- 多通道推送 (基于插件注册表, 替代原 _push_multichannel) ----

# 默认通道映射: 按预警等级指定启用哪些通道 (逗号分隔的通道名)
# PushPlus 由 AlertManager 独立处理 (含冷却逻辑)，不在此处重复配置
# 可通过环境变量 ALERT_CHANNEL_MAP 覆盖
# 格式: "red=smtp,wecom,dingtalk;orange=smtp;yellow=;blue="
_DEFAULT_CHANNEL_MAP = {
    "red":    ["smtp", "wecom"],
    "orange": ["smtp"],
    "yellow": [],
    "blue":   [],
}


def _parse_channel_map(env_val: str) -> dict[str, list[str]]:
    """解析环境变量 ALERT_CHANNEL_MAP。

    格式: "red=pushplus,smtp,wecom;orange=pushplus,smtp"
    返回: {"red": ["pushplus","smtp","wecom"], "orange": ["pushplus","smtp"]}
    """
    result: dict[str, list[str]] = {}
    for segment in env_val.split(";"):
        segment = segment.strip()
        if "=" not in segment:
            continue
        level, names = segment.split("=", 1)
        level = level.strip()
        result[level] = [n.strip() for n in names.split(",") if n.strip()]
    return result


def get_alert_channels(alert_level: str) -> list[str]:
    """
    获取指定预警等级应使用的推送通道名称列表。

    优先级: 环境变量 ALERT_CHANNEL_MAP > 默认值
    """
    import os
    env_map = os.getenv("ALERT_CHANNEL_MAP", "")
    if env_map:
        parsed = _parse_channel_map(env_map)
        return parsed.get(alert_level, _DEFAULT_CHANNEL_MAP.get(alert_level, []))
    return _DEFAULT_CHANNEL_MAP.get(alert_level, [])


def _push_via_registry(title: str, content: str,
                       alert_level: str) -> dict[str, dict]:
    """
    通过通道注册表向指定等级的所有就绪通道并行推送。

    返回: {channel_name: {"success": bool, "message": str, ...}, ...}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from .push_channels.registry import get_registry
        registry = get_registry()
        channel_names = get_alert_channels(alert_level)

        # 收集就绪通道
        ready: list[tuple[str, object]] = []
        unready: dict[str, dict] = {}
        for ch_name in channel_names:
            channel = registry.get(ch_name)
            if channel is None:
                unready[ch_name] = {"success": False, "message": f"通道未注册: {ch_name}"}
            elif not channel.validate_config():
                unready[ch_name] = {"success": False, "message": f"通道未配置: {ch_name}"}
            else:
                ready.append((ch_name, channel))

        if not ready:
            return unready

        # 并行发送
        results = dict(unready)
        with ThreadPoolExecutor(max_workers=min(len(ready), 6),
                                thread_name_prefix="push") as ex:
            futures = {
                ex.submit(ch.send, title, content, alert_level): name
                for name, ch in ready
            }
            for future in as_completed(futures, timeout=30):
                name = futures[future]
                try:
                    r = future.result(timeout=15)
                    results[name] = {
                        "success": r.success, "message": r.message, "code": r.code,
                    }
                except Exception as e:
                    results[name] = {"success": False, "message": str(e), "code": -1}

            # 超时未完成的 future: 取消并标记为失败 (防止线程泄漏)
            for future, name in list(futures.items()):
                if name not in results:
                    future.cancel()
                    results[name] = {"success": False, "message": "发送超时", "code": -1}

        return results
    except Exception as e:
        return {"registry_error": {"success": False, "message": str(e)}}


# ══════════════════════════════════════════════════════════════
# 以下为旧版硬编码推送函数 (保留用于向后兼容，不推荐直接调用)
# 推荐使用 push_channels 包的 registry.send_all()
# ══════════════════════════════════════════════════════════════


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
