"""
分布式追踪 — contextvars 实现的请求/会话追踪
============================================
基于 contextvars 提供线程安全 + asyncio 安全的 trace 上下文。
所有日志、数据库记录、预警推送均可携带 request_id / session_id，
实现端到端关联。

使用方式:
    # FastAPI 中间件（自动）
    from rockfall.trace import set_request_id
    set_request_id(request.headers.get("X-Request-ID", ""))

    # 桌面/Streamlit 应用（启动时）
    from rockfall.trace import set_session_id
    set_session_id()

    # 日志自动携带（logger.py JSONFormatter 已集成）
    # 数据库记录手动携带（alert_store.save_alert 自动提取）
"""

import contextvars
import uuid

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default=""
)
_client_ip: contextvars.ContextVar[str] = contextvars.ContextVar(
    "client_ip", default=""
)


def set_request_id(rid: str = "") -> str:
    """设置当前请求 ID（API 中间件调用）。

    若未提供，自动生成 12 字符 hex ID。
    返回最终使用的 request_id。
    """
    if not rid:
        rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    """获取当前请求 ID，无则返回空字符串。"""
    return _request_id.get()


def set_session_id(sid: str = "") -> str:
    """设置当前会话 ID（桌面/Streamlit 启动时调用）。

    若未提供，自动生成 8 字符 hex ID。
    返回最终使用的 session_id。
    """
    if not sid:
        sid = uuid.uuid4().hex[:8]
    _session_id.set(sid)
    return sid


def get_session_id() -> str:
    """获取当前会话 ID，无则返回空字符串。"""
    return _session_id.get()


def get_trace_context() -> dict | None:
    """获取当前 trace 上下文，供日志/数据库使用。

    返回:
        dict 包含 non-empty 的 request_id / session_id
        若两者皆空则返回 None
    """
    ctx = {}
    rid = _request_id.get()
    if rid:
        ctx["request_id"] = rid
    sid = _session_id.get()
    if sid:
        ctx["session_id"] = sid
    return ctx if ctx else None


def clear_trace():
    """清除当前 trace 上下文（请求结束时调用）。"""
    _request_id.set("")
    _session_id.set("")
    _client_ip.set("")


def set_client_ip(ip: str = "") -> str:
    """设置当前请求的客户端 IP（中间件调用）。"""
    _client_ip.set(ip)
    return ip


def get_client_ip() -> str:
    """获取当前请求的客户端 IP，无则返回空字符串。"""
    return _client_ip.get()
