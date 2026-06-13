"""
Sentry 错误监控初始化模块
==========================
懒加载初始化 Sentry SDK，无 SENTRY_DSN 时静默跳过（零侵入）。

特性:
  - 仅在 SENTRY_DSN 环境变量存在时启用
  - before_send 回调自动脱敏敏感字段（request body、cookies、auth headers）
  - capture_exception() 为 Sentry SDK 未初始化时的安全包装
  - 线程安全，可在多线程任务中安全调用

使用:
    from rockfall.sentry_init import init_sentry, capture_exception

    # 应用启动时调用一次
    init_sentry()

    # 在 except 块中手动上报
    try:
        ...
    except Exception as e:
        capture_exception(e)
"""

import os
import sys

from . import __version__

_sentry_initialized: bool = False
_sentry_available: bool = False


def _get_sentry_sdk():
    """延迟导入 sentry_sdk，避免未安装时崩溃。"""
    try:
        import sentry_sdk  # noqa: F811
        return sentry_sdk
    except ImportError:
        return None


def before_send(event: dict, hint: dict) -> dict | None:
    """Sentry before_send 回调 — 脱敏敏感字段。

    移除:
      - request.data（POST body 可能含密码/token）
      - request.cookies
      - request.headers 中的 Authorization 和 Cookie
      - request.env 中的 SECRETS_KEY 等敏感环境变量
    """
    if "request" in event:
        req = event["request"]
        # 脱敏 request body
        if "data" in req:
            req["data"] = "<redacted>"
        # 脱敏 cookies
        if "cookies" in req:
            req["cookies"] = "<redacted>"
        # 脱敏敏感 headers
        if "headers" in req:
            sensitive_headers = {"Authorization", "Cookie", "X-Api-Key", "X-Auth-Token"}
            req["headers"] = {
                k: ("<redacted>" if k in sensitive_headers else v)
                for k, v in req["headers"].items()
            }
        # 脱敏敏感环境变量
        if "env" in req:
            sensitive_keys = {"SECRETS_KEY", "AUTH_JWT_SECRET", "PUSHPLUS_TOKEN",
                              "MYSQL_PASSWORD", "API_KEY"}
            req["env"] = {
                k: ("<redacted>" if k in sensitive_keys else v)
                for k, v in req["env"].items()
            }

    # 脱敏 user 信息中的 IP 精确值（保留网段）
    if "user" in event and "ip_address" in event["user"]:
        ip = event["user"]["ip_address"]
        if ip and ip != "{{auto}}":
            # 保留前两段，后两段脱敏
            parts = ip.split(".")
            if len(parts) == 4:
                event["user"]["ip_address"] = f"{parts[0]}.{parts[1]}.x.x"

    return event


def init_sentry() -> bool:
    """初始化 Sentry SDK（仅在 SENTRY_DSN 存在时启用）。

    在应用启动时调用一次。线程安全（重复调用无副作用）。

    返回:
        True  如果 Sentry 已启用
        False 如果未配置 SENTRY_DSN 或初始化失败
    """
    global _sentry_initialized, _sentry_available

    if _sentry_initialized:
        return _sentry_available

    _sentry_initialized = True

    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    sdk = _get_sentry_sdk()
    if sdk is None:
        print("[sentry] sentry-sdk 未安装，跳过初始化", file=sys.stderr)
        return False

    try:
        environment = os.getenv("SENTRY_ENVIRONMENT", "production")
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))

        sdk.init(
            dsn=dsn,
            environment=environment,
            traces_sample_rate=traces_sample_rate,
            release=f"rockfall@{__version__}",
            before_send=before_send,
            # 默认不发送个人身份信息
            send_default_pii=False,
        )
        _sentry_available = True
        return True
    except Exception as e:
        print(f"[sentry] 初始化失败: {e}", file=sys.stderr)
        return False


def capture_exception(exc: BaseException) -> str | None:
    """安全捕获异常并上报 Sentry。

    若 Sentry 未启用，静默跳过并返回 None。
    返回 Sentry event_id 字符串，或 None。

    使用:
        try:
            ...
        except Exception as e:
            capture_exception(e)
    """
    if not _sentry_available:
        return None

    try:
        sdk = _get_sentry_sdk()
        if sdk is None:
            return None
        return sdk.capture_exception(exc)
    except Exception:
        return None


def capture_message(message: str, level: str = "error") -> str | None:
    """安全捕获消息并上报 Sentry。

    若 Sentry 未启用，静默跳过并返回 None。

    参数:
        message: 消息文本
        level:   严重级别 ("fatal" | "error" | "warning" | "info" | "debug")
    """
    if not _sentry_available:
        return None

    try:
        sdk = _get_sentry_sdk()
        if sdk is None:
            return None
        with sdk.push_scope() as scope:
            scope.set_level(level)
            return sdk.capture_message(message)
    except Exception:
        return None


def is_enabled() -> bool:
    """返回 Sentry 是否已成功启用。"""
    return _sentry_available
