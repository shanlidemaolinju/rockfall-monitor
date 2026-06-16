"""
API层 — FastAPI 路由定义
=========================
所有 HTTP 端点在此定义，逻辑委托给 service.py。

端点一览:
  GET  /                    — Web 看板页面
  GET  /health              — 健康检查
  GET  /api/stream.mjpeg    — MJPEG 实时视频流
  GET  /api/stats           — 检测统计
  GET  /api/alerts          — 最近预警列表
  POST /api/auth/login      — 认证登录 (获取 JWT / API Key)
  POST /api/auth/refresh    — 刷新 Token
  GET  /api/auth/clients    — 列出客户端 Key 状态
  POST /detect/image        — 上传图片检测
  POST /detect/video        — 上传视频检测
  POST /detect/video/local  — 本地视频路径检测

安全特性 (v2.1+):
  - HTTPS 强制 (生产环境自动检测 X-Forwarded-Proto)
  - JWT + 多客户端 API Key 认证 (auth.py)
  - 审计日志全量覆盖所有 POST/PUT/DELETE (audit.py)
  - 文件上传安全校验 (upload_security.py)
  - 敏感配置自动脱敏 (secrets.py)
"""

import os
import sys
from pathlib import Path

# 确保项目根目录可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import io
import json

from fastapi import FastAPI, File, UploadFile, Form, Query, Header, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from server.service import (
    detect_image_file, detect_video_file, detect_video_local,
    detect_video_file_async, detect_video_local_async, get_task_status,
    get_dashboard_stats, get_recent_alerts, query_alerts_page,
    export_alerts_excel, get_export_summary,
    get_sites_data, switch_active_site,
    create_site, update_site, delete_site,
    update_runtime_config, get_runtime_config,
    mark_alert_review, get_alert_statistics, get_alert_image_info,
    get_geo_alerts,
    get_roi_for_site, save_roi_for_site, get_roi_heatmap,
)
from server.schemas import (
    HealthResponse, DashboardStats, AlertItem,
    ImageDetectResponse, VideoDetectResponse, ErrorResponse,
    TaskResponse, TaskStatusResponse,
)
# ── ML 推理 (可选 — Railway等轻量部署时可跳过) ──
try:
    from rockfall.detector import get_latest_frame
except ImportError:
    get_latest_frame = None  # type: ignore[assignment]

from rockfall import __version__
app = FastAPI(title="落石检测系统 API", version=__version__)

# 启动时配置验证 + 设备检测
from rockfall.config import validate_config, get_device, LOG_LEVEL
from rockfall.logger import log_event, setup_logging

# 应用配置的日志级别（DEBUG/INFO/WARN/ERROR），默认 INFO
setup_logging(level=LOG_LEVEL)

# Sentry 错误监控（仅 SENTRY_DSN 配置后启用）
try:
    from rockfall.sentry_init import init_sentry
    _sentry_enabled = init_sentry()
    if _sentry_enabled:
        log_event("system", msg="Sentry 错误监控已启用")
    else:
        log_event("system", msg="Sentry 未配置 (SENTRY_DSN 为空), 跳过错误监控")
except Exception as _sentry_err:
    log_event("system", level="WARN", msg=f"Sentry 初始化异常: {_sentry_err}")

_config_warnings = validate_config()
_device_str, _device_name = get_device()
log_event("system", msg=f"推理设备: {_device_name} ({_device_str})")
for w in _config_warnings:
    log_event("system", level="WARN", msg=f"配置警告: {w}")

# ── 演示凭据自动种子 ──
# 当 API_KEY 环境变量已设置时，AuthManager 会自动创建 master 客户端密钥。
# 此处输出清晰的启动提示，方便比赛评委获取登录信息。
_api_key = os.getenv("API_KEY", "")
if _api_key and _api_key != "your_token_here":
    log_event("system", msg="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log_event("system", msg="🔑 演示登录凭据已就绪:")
    log_event("system", msg=f"   账号: admin")
    log_event("system", msg=f"   密码: {_api_key}")
    log_event("system", msg=f"   地址: http://localhost:8000")
    log_event("system", msg="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---- 分布式追踪中间件 ----
# 在 AuthMiddleware 之前注册，确保 401 响应也携带 X-Request-ID


class TraceMiddleware(BaseHTTPMiddleware):
    """为每个 HTTP 请求注入 request_id，实现端到端追踪。

    优先使用请求头中的 X-Request-ID，否则自动生成。
    响应头 X-Request-ID 始终返回，便于客户端关联。

    同时提取客户端真实 IP（支持反向代理）:
      - X-Forwarded-For (取第一个)
      - X-Real-IP
      - request.client.host (兜底)
    """

    async def dispatch(self, request: Request, call_next):
        from rockfall.trace import set_request_id, get_request_id, set_client_ip

        rid = request.headers.get("X-Request-ID", "")
        set_request_id(rid)

        # 提取客户端真实 IP
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP", "")
            or (request.client.host if request.client else "")
        )
        set_client_ip(client_ip)

        response = await call_next(request)
        response.headers["X-Request-ID"] = get_request_id() or rid
        return response


app.add_middleware(TraceMiddleware)


# ---- HTTPS 强制中间件 ----
# 生产环境检测 X-Forwarded-Proto，若非 https 且非本地请求则拒绝
# 开发环境 (127.0.0.1 / localhost) 自动放行

class HttpsEnforceMiddleware(BaseHTTPMiddleware):
    """生产环境强制 HTTPS。

    检测逻辑:
      - X-Forwarded-Proto: https → 放行（通过反向代理 TLS 终结）
      - 直接 HTTP 请求到非本地地址 → 返回 403，提示使用 HTTPS
      - 本地请求 (127.0.0.1 / localhost) → 放行（开发环境）
    """

    async def dispatch(self, request: Request, call_next):
        import os
        # 仅在明确配置为生产模式时启用
        if os.getenv("ENFORCE_HTTPS", "false").lower() != "true":
            return await call_next(request)

        proto = request.headers.get("X-Forwarded-Proto", "")
        if proto == "https":
            return await call_next(request)

        # 检查是否为本地请求
        host = request.client.host if request.client else ""
        if host in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)

        return JSONResponse(
            {
                "detail": "HTTPS required. 请使用 HTTPS 访问，"
                          "或通过反向代理 (Nginx/Traefik) 配置 TLS 终结。"
            },
            status_code=403,
        )


app.add_middleware(HttpsEnforceMiddleware)


# ---- 安全响应头中间件 ----
# 为所有 HTTP 响应添加安全加固头，防止常见 Web 攻击（点击劫持/MIME嗅探/XSS等）
# 成本极低（仅增加若干响应头），是 OWASP Top 10 推荐的基础防护措施

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """为所有响应添加安全相关的 HTTP 头。

    添加的头:
      - X-Content-Type-Options: nosniff     → 禁止 MIME 类型嗅探
      - X-Frame-Options: DENY               → 禁止页面被嵌入 iframe（防点击劫持）
      - X-XSS-Protection: 1; mode=block     → 启用浏览器 XSS 过滤器
      - Referrer-Policy: strict-origin-when-cross-origin → 控制 Referer 泄露
      - Permissions-Policy: camera=(), microphone=(), geolocation=()
                                            → 禁用敏感浏览器 API
      - Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'
                                            → 基础 CSP（允许同源 + 内联脚本/样式，兼容看板页面）

    注意:
      - HSTS (Strict-Transport-Security) 在应用层不添加，应在外层 Nginx/Traefik 配置
      - 本地开发环境 (127.0.0.1) 同样添加，确保行为一致
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # 基础防护头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # 禁用敏感浏览器特性
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "accelerometer=(), autoplay=(), payment=()"
        )

        # 基础 CSP — 允许同源资源 + 内联脚本/样式（Streamlit/看板依赖内联样式）
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "   # SSE / WebSocket
            "font-src 'self' data:; "
            "frame-ancestors 'none'"            # 等效 X-Frame-Options: DENY
        )

        return response


app.add_middleware(SecurityHeadersMiddleware)


# ---- 认证中间件 (JWT + 多客户端 API Key) ----
# 无需认证的路径（支持前缀匹配）:
#   - 指定路径本身
#   - /docs 及其子路径 (OpenAPI 文档)
#   - /api/stream.mjpeg (自有 STREAM_TOKEN 鉴权)
#   - /api/alerts/stream (SSE 实时推送，浏览器直连)
#   - /metrics (Prometheus 采集)
_PUBLIC_PATH_PREFIXES = (
    "/docs",            # FastAPI Swagger UI + OAuth2 回调
    "/redoc",           # FastAPI ReDoc
    "/openapi.json",    # OpenAPI schema
    "/favicon.ico",
    "/favicon.svg",
    "/assets",          # React SPA 静态资源 (JS/CSS/图片)
    "/icons.svg",       # React SPA 图标
    "/ws/",             # WebSocket 端点 (浏览器不支持自定义 Header, task_id 即鉴权)
)
_PUBLIC_PATHS = {
    "/", "/m", "/mobile",
    "/health", "/health/live", "/health/ready",
    "/api/auth/login", "/api/auth/refresh",
    "/api/stream.mjpeg",     # 自有 STREAM_TOKEN 鉴权
    "/api/alerts/stream",    # SSE 实时预警推送（浏览器直连）
    "/metrics",              # Prometheus 监控指标
}


def _is_public_path(path: str) -> bool:
    """判断路径是否为公开路径（支持前缀匹配）。

    SPA 路由（如 /settings, /alerts）为公开，仅 API 路径需要认证。
    """
    if path in _PUBLIC_PATHS:
        return True
    for prefix in _PUBLIC_PATH_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    # React SPA 路由 (非 API 路径): 公开访问，由前端路由守卫控制
    if not path.startswith("/api/") and not path.startswith("/detect/"):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """统一认证中间件 — JWT Bearer Token + 多客户端 API Key。

    认证优先级:
      1. Authorization: Bearer <JWT>
      2. X-API-Key: <client_key>
      3. api_key query param (兼容旧版，不推荐)

    认证成功后，在 request.state 中注入:
      - request.state.client_id:  客户端标识
      - request.state.auth_method: 认证方式 (jwt / api_key / legacy)
      - request.state.operator:   操作人标签 (用于审计日志)
    """

    async def dispatch(self, request: Request, call_next):
        # 公开路径跳过认证
        if _is_public_path(request.url.path):
            return await call_next(request)

        from rockfall.auth import get_auth_manager

        auth = get_auth_manager()
        try:
            auth_header = request.headers.get("Authorization", "")
            api_key = request.headers.get("X-API-Key", "")
            query_key = request.query_params.get("api_key", "")

            result = auth.authenticate(
                auth_header=auth_header,
                api_key=api_key,
                query_key=query_key,
            )

            # 注入认证结果到 request.state
            request.state.client_id = result["client_id"]
            request.state.auth_method = result["auth_method"]
            request.state.operator = (
                result.get("label", "") or result["client_id"]
            )

        except ValueError:
            return JSONResponse(
                {
                    "detail": "认证失败 — 请提供有效的 JWT (Authorization: Bearer <token>) "
                              "或 API Key (X-API-Key: <key>)",
                    "auth_methods": ["jwt", "api_key"],
                },
                status_code=401,
            )

        return await call_next(request)


app.add_middleware(AuthMiddleware)


# ---- 审计日志中间件 ----
# 自动记录所有 POST/PUT/DELETE 请求（不依赖各端点手动调用 audit_log）
# 各端点仍可手动调用 audit_log 记录更详细的信息（变更前后值等）

_AUDIT_MUTATING_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


class AuditMiddleware(BaseHTTPMiddleware):
    """自动审计中间件 — 记录所有变更操作。

    为所有 POST/PUT/DELETE/PATCH 请求自动创建审计记录，
    包含操作人、来源 IP、User-Agent、请求路径。

    注意: 公开路径（如 /api/auth/login）的审计由各自端点处理。
    """

    async def dispatch(self, request: Request, call_next):
        # 仅记录变更操作
        if request.method not in _AUDIT_MUTATING_METHODS:
            return await call_next(request)

        # 跳过公开路径
        if _is_public_path(request.url.path):
            return await call_next(request)

        # 执行请求
        response = await call_next(request)

        # 记录审计日志
        from rockfall.audit import audit_log
        from rockfall.trace import get_client_ip, get_request_id

        operator = getattr(request.state, "operator", "anonymous")
        client_id = getattr(request.state, "client_id", "")
        ip = get_client_ip() or ""
        rid = get_request_id() or ""

        status = "ok" if response.status_code < 400 else f"error: HTTP {response.status_code}"
        # 路径归一化: 将动态参数替换为 _ 占位符 (e.g. /api/alerts/42/review -> api:alerts:_:review)
        import re as _re
        normalized = _re.sub(r'/\d+', '/_', request.url.path)
        action = f"api:{request.method.lower()}:{normalized.lstrip('/').replace('/', ':')}"

        # 尝试读取请求体用于审计（仅记录前 512 字符，且不记录文件上传内容）
        body_summary = ""
        if "multipart/form-data" not in request.headers.get("content-type", ""):
            try:
                # 注意: request.body() 在中间件中可能已被消费
                # 因此这里仅记录路径和查询参数
                pass
            except Exception:
                pass

        audit_log(
            action=action,
            operator=operator or client_id,
            detail=f"{request.method} {request.url.path}",
            ip=ip,
            result=status,
            user_agent=request.headers.get("User-Agent", ""),
            request_id=rid,
        )

        return response


app.add_middleware(AuditMiddleware)


# ============================================================
# 辅助函数
# ============================================================

def _get_operator(request: Request) -> str:
    """从 request.state 提取操作人标识"""
    return getattr(request.state, "operator", "anonymous")


def _get_client_ip() -> str:
    """获取当前请求的客户端 IP"""
    from rockfall.trace import get_client_ip
    return get_client_ip() or ""


# ============================================================
# 认证 API
# ============================================================

@app.post("/api/auth/login")
def auth_login(
    request: Request,
    api_key: str = Form("", description="API Key"),
    username: str = Form("", description="用户名（与 password 配合使用）"),
    password: str = Form("", description="密码（与 username 配合使用）"),
    client: str = Form("web", description="客户端标识 (web/desktop/mobile)"),
    label: str = Form("", description="客户端备注"),
    expires_hours: int = Form(24, ge=1, le=720, description="Token 有效期(小时)"),
    grant_type: str = Form("api_key", description="认证方式: api_key 或 jwt_refresh"),
    refresh_token: str = Form("", description="用于刷新 JWT 的旧 Token"),
):
    """
    认证登录 — 支持两种方式:

    方式 1 (API Key):
      - api_key: 客户端 API Key

    方式 2 (用户名+密码):
      - username: 用户名
      - password: 密码（作为 API Key 验证）

    返回:
      - access_token: JWT Token
      - token_type:   "Bearer"
      - expires_in:   有效期(秒)
      - client_id:    客户端标识
    """
    from rockfall.auth import get_auth_manager
    from rockfall.audit import audit_log

    auth = get_auth_manager()
    ip = _get_client_ip()

    if grant_type == "jwt_refresh":
        # JWT 刷新
        if not refresh_token:
            raise HTTPException(status_code=400, detail="refresh_token 不能为空")
        try:
            new_token = auth.refresh_jwt(refresh_token)
            audit_log("auth:token_refresh", operator="jwt",
                      detail="JWT token 刷新成功", ip=ip, result="ok")
            return {
                "access_token": new_token,
                "token_type": "Bearer",
                "expires_in": expires_hours * 3600,
                "client_id": "jwt",
            }
        except ValueError as e:
            audit_log("auth:token_refresh", operator="jwt",
                      detail=f"JWT token 刷新失败: {e}", ip=ip, result="error")
            raise HTTPException(status_code=401, detail=str(e))

    # grant_type == "api_key": 确定实际使用的 api_key 值
    # 优先级: api_key 参数 > password 参数（兼容用户名+密码登录）
    effective_key = api_key or password
    effective_client = client
    if username and not client:
        effective_client = username

    if not effective_key:
        raise HTTPException(status_code=400, detail="api_key 或 password 不能为空")

    try:
        result = auth.authenticate(api_key=effective_key)
    except ValueError:
        audit_log("auth:login", operator=username or "unknown",
                  detail="认证失败: API Key 无效", ip=ip, result="error")
        raise HTTPException(status_code=401, detail="账号或密码错误")

    # 签发 JWT
    token = auth.create_jwt(
        client=result["client_id"],
        label=label or f"Web Login · {username or result['client_id']}",
        expires_hours=expires_hours,
    )

    audit_log("auth:login", operator=result["client_id"],
              detail=f"认证成功 ({result['auth_method']})", ip=ip, result="ok")

    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": expires_hours * 3600,
        "client_id": result["client_id"],
        "auth_method": result["auth_method"],
    }


@app.post("/api/auth/refresh")
def auth_refresh(
    request: Request,
    refresh_token: str = Form(...),
    expires_hours: int = Form(24, ge=1, le=720),
):
    """刷新 JWT Token — 在过期前 1 小时内可刷新。"""
    from rockfall.auth import get_auth_manager
    from rockfall.audit import audit_log

    auth = get_auth_manager()
    ip = _get_client_ip()

    try:
        new_token = auth.refresh_jwt(refresh_token)
        audit_log("auth:token_refresh", operator="jwt",
                  detail="JWT token 刷新成功", ip=ip, result="ok")
        return {
            "access_token": new_token,
            "token_type": "Bearer",
            "expires_in": expires_hours * 3600,
        }
    except ValueError as e:
        audit_log("auth:token_refresh", operator="jwt",
                  detail=f"JWT token 刷新失败: {e}", ip=ip, result="error")
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/api/auth/clients")
def auth_list_clients(request: Request):
    """列出所有客户端 API Key 状态（管理端点）。"""
    from rockfall.auth import get_auth_manager
    auth = get_auth_manager()
    return {
        "clients": auth.list_clients(),
        "jwt_secret_masked": auth.jwt_secret[:8] + "***",
    }


@app.post("/api/auth/clients")
def auth_create_client(
    request: Request,
    client_id: str = Form(...),
    label: str = Form(""),
    expire_days: int = Form(90, ge=1, le=3650),
):
    """创建新的客户端 API Key（管理端点）。

    返回的 api_key 仅在此时显示，请妥善保管。
    """
    from rockfall.auth import get_auth_manager
    from rockfall.audit import audit_log

    auth = get_auth_manager()
    ip = _get_client_ip()
    operator = _get_operator(request)

    try:
        raw_key = auth.create_client_key(client_id, label, expire_days)
        audit_log("auth:create_client", operator=operator,
                  detail=f"创建客户端 Key: {client_id} ({label}), "
                         f"有效期 {expire_days} 天",
                  ip=ip, result="ok",
                  after={"client_id": client_id, "label": label, "expire_days": expire_days})
        return {
            "client_id": client_id,
            "api_key": raw_key,
            "expires_in_days": expire_days,
            "warning": "API Key 仅在此时显示，请妥善保管。丢失后需重新创建。",
        }
    except Exception as e:
        audit_log("auth:create_client", operator=operator,
                  detail=f"创建客户端 Key 失败: {e}", ip=ip, result="error")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/auth/clients/{client_id}")
def auth_revoke_client(request: Request, client_id: str):
    """吊销客户端 API Key。"""
    from rockfall.auth import get_auth_manager
    from rockfall.audit import audit_log

    auth = get_auth_manager()
    ip = _get_client_ip()
    operator = _get_operator(request)

    ok = auth.revoke_client(client_id)
    audit_log("auth:revoke_client", operator=operator,
              detail=f"吊销客户端 Key: {client_id}", ip=ip,
              result="ok" if ok else "error: client not found")
    if not ok:
        raise HTTPException(status_code=404, detail="客户端不存在或已被吊销")
    return {"status": "ok", "client_id": client_id}


# ============================================================
# Web 看板
# ============================================================

@app.get("/classic")
def classic_dashboard():
    """经典 Web 看板 (仅当 React SPA 未构建时作为主界面使用)"""
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    html = template_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/m")
@app.get("/mobile")
def mobile_dashboard():
    """移动端 H5 看板 — 预警列表 + 现场截图预览 + 监测点位"""
    from fastapi.responses import HTMLResponse
    template_path = Path(__file__).parent / "templates" / "mobile.html"
    html = template_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/stream.mjpeg")
def mjpeg_stream(
    token: str = Query(""),
    x_stream_token: str | None = Header(None, alias="X-Stream-Token"),
    camera_id: str = Query("default"),
):
    """MJPEG 实时视频流 — 从共享帧缓冲读取, 支持多路摄像头。

    鉴权: 支持 query 参数 token 或请求头 X-Stream-Token。
    推荐使用请求头传递 token, 避免被服务器日志记录泄露。
    多路: camera_id 参数区分不同摄像头 (默认 "default")。
    """
    from rockfall.config import STREAM_TOKEN
    effective_token = x_stream_token if x_stream_token is not None else token
    if STREAM_TOKEN and effective_token != STREAM_TOKEN:
        raise HTTPException(status_code=403, detail="无效的 stream token")

    def generate():
        import time
        from rockfall.config import MJPEG_BLANK_WIDTH, MJPEG_BLANK_HEIGHT, MJPEG_FRAME_INTERVAL
        try:
            while True:
                jpg = get_latest_frame(camera_id) if get_latest_frame else None
                if jpg is None:
                    import cv2
                    import numpy as np
                    bw, bh = MJPEG_BLANK_WIDTH, MJPEG_BLANK_HEIGHT
                    blank = np.zeros((bh, bw, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for stream...", (bw // 5, bh // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    _, jpg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    jpg = jpg.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
                time.sleep(MJPEG_FRAME_INTERVAL)
        except GeneratorExit:
            pass

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# 统计 API
# ============================================================

@app.get("/api/stats", response_model=DashboardStats)
def api_stats():
    """返回检测统计数据 (实时看板用)"""
    return get_dashboard_stats()


@app.get("/api/statistics")
def api_statistics(days: int = Query(7, ge=1, le=90, description="统计天数")):
    """
    预警统计看板 — 聚合数据。

    返回:
        - today: 今日各等级预警次数
        - daily_trends: 每日趋势 (近N天)
        - level_distribution: 等级分布 (近30天)
        - false_alarm: 误报率统计 (近30天)
        - grand_total: 近30天预警总数
    """
    return get_alert_statistics(days=days)


@app.get("/api/alerts", response_model=list[AlertItem])
def api_alerts(limit: int = Query(20, ge=1, le=200)):
    """返回最近预警列表 (简洁版, 供看板实时刷新)"""
    return get_recent_alerts(limit)


@app.get("/api/alerts/paged")
def api_alerts_paged(
    page: int = Query(1, ge=1, description="页码 (从1开始)"),
    page_size: int = Query(20, ge=5, le=200, description="每页条数"),
    start_date: str = Query("", description="起始日期 YYYY-MM-DD"),
    end_date: str = Query("", description="结束日期 YYYY-MM-DD"),
    alert_level: str = Query("", description="预警等级筛选 red/orange/yellow/blue (空=全部)"),
):
    """
    分页查询预警记录, 支持日期+等级筛选。
    返回: {total, page, page_size, total_pages, rows: [...]}
    """
    offset = (page - 1) * page_size
    result = query_alerts_page(
        limit=page_size, offset=offset,
        start_date=start_date, end_date=end_date, alert_level=alert_level,
    )
    total = result["total"]
    total_pages = max(1, (total + page_size - 1) // page_size)
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "rows": result["rows"],
    }


@app.get("/api/alerts/geo")
def api_alerts_geo(
    days: int = Query(30, ge=1, le=365, description="查询最近 N 天的预警"),
    alert_level: str = Query("", description="预警等级筛选 (空=全部)"),
):
    """
    返回带经纬度的预警数据（关联站点表），供前端 Leaflet 地图渲染。

    返回: [
      {
        "id": 1, "time": "2026-06-13 15:30:00", "alert_level": "orange",
        "count": 3, "max_confidence": 0.85, "class_summary": "落石:3",
        "saved_frame": "path/to/img.jpg",
        "site_id": "nanning_naan_s1", "site_name": "南宁...",
        "latitude": 22.817, "longitude": 108.366
      }, ...
    ]
    """
    return get_geo_alerts(days=days, alert_level=alert_level)


@app.post("/api/alerts/{alert_id}/review")
def api_alert_review(
    request: Request,
    alert_id: int,
    review_status: str = Form(..., description="confirmed | false_alarm | (空=清除)"),
    note: str = Form("", description="审核备注"),
):
    """标记预警审核状态 (确认真实/误报)"""
    from rockfall.audit import audit_log

    valid = {"confirmed", "false_alarm", ""}
    if review_status not in valid:
        raise HTTPException(status_code=400, detail=f"review_status 必须是: {valid}")

    # 获取变更前的状态
    old_info = get_alert_image_info(alert_id)
    old_status = old_info.get("review_status", "") if old_info else ""

    result = mark_alert_review(alert_id, review_status, note)

    # 误报标记 → 通知模型注册表 (供 A/B 测试和自动回滚使用)
    if review_status == "false_alarm" and result.get("status") == "ok":
        try:
            from rockfall.model_registry import get_registry, MODEL_REGISTRY_AB_SPLIT_ENABLED
            if MODEL_REGISTRY_AB_SPLIT_ENABLED:
                registry = get_registry()
                active = registry.active_version
                if active is not None:
                    registry.record_inference_metrics(
                        active.name, latency_ms=0.0, is_false_alarm=True,
                    )
        except Exception:
            pass

    operator = _get_operator(request)
    audit_log("alert_review", operator=operator,
              detail=f"预警 #{alert_id}: {old_status or '(未标记)'} → {review_status or '(清除)'}",
              alert_id=alert_id, ip=_get_client_ip(),
              result="ok" if result.get("status") == "ok" else "error",
              before={"review_status": old_status},
              after={"review_status": review_status, "note": note})

    return result


@app.get("/api/alerts/{alert_id}/image")
def api_alert_image(alert_id: int):
    """
    获取预警记录的现场截图。

    返回: JPEG 图片文件
    """
    from fastapi.responses import FileResponse
    info = get_alert_image_info(alert_id)
    if info is None:
        raise HTTPException(status_code=404, detail="预警记录不存在")
    if not info["exists"]:
        raise HTTPException(status_code=404, detail="截图文件不存在或已被清理")

    path = info["display_path"]
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"截图文件不存在: {path}")

    return FileResponse(
        path, media_type="image/jpeg",
        headers={"X-Alert-Id": str(alert_id), "X-Alert-Time": info.get("time", "")},
    )


@app.get("/api/frames/recent")
def frames_recent(camera_id: str = "default", n: int = 10):
    """获取最近 N 帧标注帧 (JPEG base64)，供看板实时预览。

    参数:
        camera_id: 摄像头 ID
        n:         返回帧数 (1-50)
    """
    n = max(1, min(50, n))
    from server.service import _get_detector
    detector = _get_detector(camera_id)
    buf = getattr(detector, "_frame_buffer", None)
    if buf is None:
        return {"frames": [], "msg": "环形缓冲未启用"}
    return {"frames": buf.get_recent_jpegs(n), "buffer_size": len(buf)}


@app.get("/api/alerts/stream")
async def alerts_sse(request: Request):
    """
    SSE (Server-Sent Events) 实时预警推送。

    浏览器连接此端点后, 当检测到 Ⅲ 级(黄色)及以上预警时,
    自动推送 JSON 事件到前端, 触发弹窗/声音报警。

    事件格式:
      event: alert
      data: {"alert_level": "red", "count": 3, "max_confidence": 0.95, ...}

    每 30 秒发送一次 heartbeat 保持连接。
    """
    from rockfall.notifier import wait_for_popup_alert

    async def event_generator():
        try:
            while True:
                # 检查客户端是否断开
                if await request.is_disconnected():
                    break

                # 阻塞等待新预警 (最多 30s, 超时后发送 heartbeat)
                alert = await asyncio.to_thread(wait_for_popup_alert, timeout=30.0)

                if alert is not None:
                    yield f"event: alert\ndata: {json.dumps(alert, ensure_ascii=False)}\n\n"
                else:
                    yield f": heartbeat {asyncio.get_event_loop().time()}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 归档导出 (应急管理部门合规要求)
# ============================================================

@app.get("/api/alerts/export/summary")
def api_export_summary(
    start_date: str = Query("", description="起始日期 YYYY-MM-DD"),
    end_date: str = Query("", description="结束日期 YYYY-MM-DD"),
    alert_level: str = Query("", description="预警等级 red/orange/yellow/blue (空=全部)"),
):
    """导出预览: 返回符合条件的记录数和等级分布"""
    return get_export_summary(
        start_date=start_date, end_date=end_date, alert_level=alert_level,
    )


@app.get("/api/alerts/export")
def api_export_excel(
    request: Request,
    start_date: str = Query("", description="起始日期 YYYY-MM-DD"),
    end_date: str = Query("", description="结束日期 YYYY-MM-DD"),
    alert_level: str = Query("", description="预警等级 red/orange/yellow/blue (空=全部)"),
):
    """
    一键导出预警记录为 Excel (.xlsx), 符合应急管理部门归档要求。

    Excel 包含列:
      序号 | 报警时间 | 监测点位 | 预警等级 | 落石数量 | 最高置信度 |
      落石直径(cm) | 检测类别 | 推送状态 | 截图路径 | 入库时间

    使用方式:
      - 浏览器直接访问此 URL 下载文件
      - 或通过看板页面的"导出Excel"按钮
    """
    from rockfall.config import get_location
    from rockfall.audit import audit_log

    try:
        excel_bytes = export_alerts_excel(
            start_date=start_date, end_date=end_date, alert_level=alert_level,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 审计日志：导出操作
    operator = _get_operator(request)
    audit_log("export_excel", operator=operator,
              detail=f"导出预警记录: {start_date or '全部'} ~ {end_date or '全部'}, "
                     f"等级: {alert_level or '全部'}",
              ip=_get_client_ip(), result="ok")

    # 生成文件名
    loc = get_location() or "监测点"
    date_tag = ""
    if start_date or end_date:
        date_tag = f"_{start_date or 'begin'}_{end_date or 'end'}"
    filename = f"落石预警记录_{loc}{date_tag}.xlsx"

    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================
# 哈希链防篡改验证 (v2.6+)
# ============================================================

@app.get("/api/alerts/{alert_id}/verify")
def api_verify_alert(alert_id: int):
    """验证单条预警记录的哈希完整性。

    返回 {valid, stored_hash, computed_hash, prev_hash_match, msg}
    """
    from rockfall.alert_store import get_alert_store
    store = get_alert_store()
    return store.verify_alert(alert_id)


@app.post("/api/alerts/verify-chain")
def api_verify_chain(
    start_id: int = Form(...),
    end_id: int = Form(...),
    max_records: int | None = Form(None),
):
    """批量验证 ID 区间的哈希链完整性。

    返回 {total_checked, valid, invalid, skipped, breaks, truncated}
    """
    from rockfall.alert_store import get_alert_store
    store = get_alert_store()
    return store.verify_chain(start_id, end_id, max_records=max_records)


@app.get("/api/health/hash-chain")
def api_hash_chain_health():
    """哈希链健康检查: 验证最近 100 条记录的完整性。"""
    from rockfall.alert_store import get_alert_store
    from rockfall.config import ALERT_HASH_CHAIN_ENABLED

    if not ALERT_HASH_CHAIN_ENABLED:
        return {"status": "disabled", "msg": "哈希链功能未启用"}

    store = get_alert_store()
    latest_hash = store.get_latest_hash()
    if latest_hash is None:
        return {"status": "no_data", "msg": "尚无带哈希的记录"}

    # 查询最近 100 条记录所在 ID 范围
    recent = store.get_recent(limit=100)
    if not recent:
        return {"status": "no_data", "msg": "无预警记录"}

    ids = [r["id"] for r in recent if r.get("data_hash")]
    if not ids:
        return {"status": "no_data", "msg": "无附带哈希的记录 (功能可能刚启用)"}

    start_id = min(ids)
    end_id = max(ids)
    result = store.verify_chain(start_id, end_id, max_records=len(ids))

    return {
        "status": "healthy" if result["invalid"] == 0 else "breach_detected",
        "latest_hash": latest_hash,
        **result,
    }


# ============================================================
# 摄像头管理
# ============================================================

@app.get("/api/cameras")
def list_cameras():
    """列出所有活跃的摄像头检测器"""
    from server.service import _detectors, _active_cameras, remove_detector
    result = []
    for cam_id in list(_detectors.keys()):
        info = _active_cameras.get(cam_id, {})
        result.append({
            "camera_id": cam_id,
            "source": info.get("source", ""),
            "fps": info.get("fps", 0),
            "resolution": info.get("resolution", ""),
        })
    return {"cameras": result, "total": len(result)}


@app.delete("/api/cameras/{camera_id}")
def delete_camera(request: Request, camera_id: str):
    """释放指定摄像头的检测器资源"""
    from server.service import remove_detector
    from rockfall.audit import audit_log

    remove_detector(camera_id)
    operator = _get_operator(request)
    audit_log("camera_delete", operator=operator,
              detail=f"释放摄像头资源: {camera_id}",
              ip=_get_client_ip(), result="ok")
    return {"status": "ok", "camera_id": camera_id}


# ============================================================
# 监测点位管理
# ============================================================

@app.get("/api/sites")
def api_sites():
    """获取全部监测点位 + 当前激活点位"""
    return get_sites_data()


@app.post("/api/sites/switch")
def api_switch_site(request: Request, site_id: str = Form(...)):
    """切换当前激活的监测点位"""
    from rockfall.audit import audit_log

    # 获取切换前的点位
    old_data = get_sites_data()
    old_site = old_data.get("active_site_id", "")

    try:
        result = switch_active_site(site_id)
        operator = _get_operator(request)
        audit_log("site_switch", operator=operator,
                  detail=f"监测点位切换: {old_site} → {site_id}",
                  ip=_get_client_ip(), result="ok",
                  before={"site_id": old_site},
                  after={"site_id": site_id})
        return result
    except ValueError as e:
        audit_log("site_switch", operator=_get_operator(request),
                  detail=f"监测点位切换失败: {old_site} → {site_id}, 原因: {e}",
                  ip=_get_client_ip(), result="error",
                  before={"site_id": old_site},
                  after={"site_id": site_id})
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sites")
def api_create_site(request: Request, payload: dict):
    """新增监测点位。

    请求体 JSON:
        {
            "site_id": "gl_ys_s5",
            "name": "桂林阳朔高速 5 号边坡",
            "location": "桂林阳朔高速 5 号边坡",
            "region": "广西·桂林",
            "camera_url": "rtsp://...",
            "latitude": 24.78, "longitude": 110.48,
            "highway": "G65 包茂高速 (阳朔段)",
            "stake_mark": "K2480+100",
            "risk_level": "medium",
            "roi_polygon": [[x1,y1],[x2,y2],...],
            "alert_contacts": [{"name":"张三","phone":"138...","email":"..."}]
        }
    """
    from rockfall.audit import audit_log

    try:
        result = create_site(payload)
        operator = _get_operator(request)
        audit_log("site_create", operator=operator,
                  detail=f"新增监测点位: {payload.get('site_id','')}",
                  ip=_get_client_ip(), result="ok")
        return result
    except ValueError as e:
        audit_log("site_create", operator=_get_operator(request),
                  detail=f"新增点位失败: {e}", ip=_get_client_ip(), result="error")
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/sites/{site_id}")
def api_update_site(request: Request, site_id: str, payload: dict):
    """更新监测点位。

    请求体 JSON: 与创建相同结构，但只需传入要更新的字段。
    不可修改 site_id 本身。
    """
    from rockfall.audit import audit_log

    try:
        result = update_site(site_id, payload)
        operator = _get_operator(request)
        audit_log("site_update", operator=operator,
                  detail=f"更新监测点位: {site_id}",
                  ip=_get_client_ip(), result="ok")
        return result
    except ValueError as e:
        audit_log("site_update", operator=_get_operator(request),
                  detail=f"更新点位失败: {e}", ip=_get_client_ip(), result="error")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/sites/{site_id}")
def api_delete_site(request: Request, site_id: str):
    """删除监测点位（不允许删除当前激活点位）。"""
    from rockfall.audit import audit_log

    try:
        result = delete_site(site_id)
        operator = _get_operator(request)
        audit_log("site_delete", operator=operator,
                  detail=f"删除监测点位: {site_id}",
                  ip=_get_client_ip(), result="ok")
        return result
    except ValueError as e:
        audit_log("site_delete", operator=_get_operator(request),
                  detail=f"删除点位失败: {e}", ip=_get_client_ip(), result="error")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# ROI 多边形管理
# ============================================================

@app.get("/api/roi")
def api_get_roi(site_id: str = Query("", description="站点ID，空=当前激活站点")):
    """获取 ROI 多边形坐标。返回 {site_id, roi_polygon, frame_size}"""
    try:
        return get_roi_for_site(site_id or None)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/roi")
def api_save_roi(request: Request, body: dict):
    """
    保存 ROI 多边形并触发 MOG2 背景模型重建。

    请求体: {"site_id": "xxx", "polygon": [[x1,y1], [x2,y2], ...]}
    返回: {"status": "ok", "site_id": "...", "vertices": N}
    """
    from rockfall.audit import audit_log

    site_id = body.get("site_id", "")
    polygon = body.get("polygon", [])

    if not site_id:
        raise HTTPException(status_code=400, detail="缺少 site_id")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise HTTPException(status_code=400, detail="polygon 必须是至少 3 个点的坐标数组")

    try:
        result = save_roi_for_site(site_id, polygon)
        audit_log("roi_save", operator=_get_operator(request),
                  detail=f"保存ROI: site={site_id}, vertices={len(polygon)}",
                  ip=_get_client_ip(), result="ok")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/roi/heatmap")
def api_roi_heatmap(
    site_id: str = Query("", description="站点ID"),
    frame: str = Query("", description="参考图片路径 (可选)"),
):
    """
    生成 ROI 热力图 overlay — base64 PNG 图片。

    优先使用 FastSAM 道路/边坡分割，不可用时返回梯度热力图。
    返回: {"base64": "data:image/png;base64,...", "width": 1280, "height": 720}
    """
    return get_roi_heatmap(site_id or None, frame)


# ============================================================
# 运行时配置热更新
# ============================================================

@app.get("/api/config/runtime")
def api_config_runtime():
    """获取当前运行中的可调参数值"""
    return get_runtime_config()


@app.post("/api/config/update")
def api_config_update(request: Request, payload: dict):
    """
    热更新检测器参数 (当前会话有效)。

    请求体 JSON:
        {"detection_confidence": 0.5, "alert_blue_high": 0.55, ...}

    支持的白名单键:
        detection_confidence, detection_img_size, motion_min_area,
        alert_blue_high, alert_yellow_high, alert_orange_high
    """
    from rockfall.audit import audit_log

    # 获取变更前的值
    old_config = get_runtime_config()

    result = update_runtime_config(payload)

    operator = _get_operator(request)
    audit_log("config_update", operator=operator,
              detail=f"热更新检测参数: {len(result.get('applied', {}))} 项生效, "
                     f"{len(result.get('skipped', {}))} 项跳过",
              ip=_get_client_ip(),
              result="ok" if not result.get("skipped") else "partial",
              before=old_config,
              after={**old_config, **result.get("applied", {})})

    if result["skipped"]:
        return JSONResponse(
            content={"status": "partial", **result},
            status_code=200,
        )
    return {"status": "ok", **result}


# ============================================================
# 预警分级决策树 & 响应流程
# ============================================================

@app.get("/api/alert-classifier/decision-tree")
def api_decision_tree():
    """
    获取预警分级决策树数据结构 (供前端可视化渲染)。

    返回完整的决策树节点列表，每个节点包含:
      - id: 节点唯一标识
      - type: "root" | "decision" | "leaf-red" | "leaf-orange" | "leaf-yellow" | "leaf-blue" | "leaf-green"
      - label: 显示文本
      - children: 子节点列表 (decision 节点)
      - branch_labels: 分支标签 (decision → children 的映射)
    """
    tree = {
        "id": "root",
        "type": "root",
        "label": "输入: 检测帧 + 跟踪轨迹",
        "children": [
            {
                "id": "conf_90",
                "type": "decision",
                "label": "最高置信度 max_conf ?",
                "branches": [
                    {
                        "label": "> 0.90",
                        "result": {"type": "leaf-red", "label": "🔴 I 级 · 特别严重", "level": "red"},
                    },
                    {
                        "label": "0.70 - 0.90",
                        "node": {
                            "id": "diam_30_conf70",
                            "type": "decision",
                            "label": "落石直径 ?",
                            "branches": [
                                {
                                    "label": "> 30cm",
                                    "result": {"type": "leaf-red", "label": "🔴 I 级 (升级)", "level": "red"},
                                },
                                {
                                    "label": "20 - 30cm",
                                    "result": {"type": "leaf-orange", "label": "🟠 II 级 · 严重", "level": "orange"},
                                },
                                {
                                    "label": "< 20cm",
                                    "node": {
                                        "id": "motion_conf70",
                                        "type": "decision",
                                        "label": "运动状态 ?",
                                        "branches": [
                                            {
                                                "label": "坠落",
                                                "result": {"type": "leaf-orange", "label": "🟠 II 级 · 严重", "level": "orange"},
                                            },
                                            {
                                                "label": "滚动",
                                                "result": {"type": "leaf-yellow", "label": "🟡 III 级 · 较重", "level": "yellow"},
                                            },
                                        ],
                                    },
                                },
                            ],
                        },
                    },
                    {
                        "label": "0.50 - 0.70",
                        "node": {
                            "id": "diam_20_conf50",
                            "type": "decision",
                            "label": "落石直径 ?",
                            "branches": [
                                {
                                    "label": "> 20cm",
                                    "result": {"type": "leaf-orange", "label": "🟠 II 级 (升级)", "level": "orange"},
                                },
                                {
                                    "label": "10 - 20cm",
                                    "result": {"type": "leaf-yellow", "label": "🟡 III 级 · 较重", "level": "yellow"},
                                },
                                {
                                    "label": "< 10cm",
                                    "node": {
                                        "id": "frames_conf50",
                                        "type": "decision",
                                        "label": "持续帧数 ?",
                                        "branches": [
                                            {
                                                "label": "> 10 帧",
                                                "result": {"type": "leaf-yellow", "label": "🟡 III 级 · 较重", "level": "yellow"},
                                            },
                                            {
                                                "label": "< 10 帧",
                                                "result": {"type": "leaf-blue", "label": "🔵 IV 级 · 一般", "level": "blue"},
                                            },
                                        ],
                                    },
                                },
                            ],
                        },
                    },
                    {
                        "label": "0.30 - 0.50",
                        "node": {
                            "id": "conf_30_50",
                            "type": "decision",
                            "label": "落石直径 ?",
                            "branches": [
                                {
                                    "label": "> 10cm → 升级至 III 级",
                                    "result": {"type": "leaf-yellow", "label": "🟡 III 级 · 较重", "level": "yellow"},
                                },
                                {
                                    "label": "≤ 10cm",
                                    "result": {"type": "leaf-blue", "label": "🔵 IV 级 · 一般", "level": "blue"},
                                },
                            ],
                        },
                    },
                ],
            },
            {
                "id": "conf_below_30",
                "type": "leaf-green",
                "label": "🟢 正常 (不预警)",
                "level": "green",
            },
        ],
    }
    return tree


@app.get("/api/alert-classifier/response-workflow/{alert_level}")
def api_response_workflow(alert_level: str):
    """
    获取指定预警等级的响应流程配置。

    alert_level: red | orange | yellow | blue | green

    返回:
      - level: 等级标识
      - label: 中文标签
      - trigger_conditions: 触发条件列表
      - disposal_steps: 处置流程步骤
      - push_channels: 推送渠道
      - requires_sound: 是否触发声光报警
    """
    from rockfall.alert_classifier import get_response_workflow
    workflow = get_response_workflow(alert_level)
    return workflow


@app.get("/api/alert-classifier/response-workflows")
def api_all_response_workflows():
    """获取所有预警等级的响应流程配置。"""
    from rockfall.alert_classifier import get_response_workflow, LEVEL_RED, LEVEL_ORANGE, LEVEL_YELLOW, LEVEL_BLUE
    return {
        level: get_response_workflow(level)
        for level in [LEVEL_RED, LEVEL_ORANGE, LEVEL_YELLOW, LEVEL_BLUE]
    }


@app.post("/api/alert-classifier/classify")
def api_classify_alert(payload: dict):
    """
    手动调用决策树进行预警分级 (供前端调试/验证)。

    请求体:
        {"max_conf": 0.85, "rock_diameter_cm": 25, "motion_state": "快速坠落", "track_age": 8}

    返回:
        {"level": "orange", "label": "🟠 Ⅱ级·严重", "classification_path": [...]}
    """
    from rockfall.alert_classifier import classify_alert_level, LEVEL_LABELS
    level = classify_alert_level(
        max_conf=float(payload.get("max_conf", 0)),
        rock_diameter_cm=float(payload.get("rock_diameter_cm", 0)),
        motion_state=str(payload.get("motion_state", "")),
        track_age=int(payload.get("track_age", 0)),
    )
    return {
        "level": level,
        "label": LEVEL_LABELS.get(level, "未知"),
    }


# ============================================================
# 健康检查
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():
    """基础健康检查（兼容旧接口，等同于 /health/live）"""
    return {"status": "ok", "service": "落石检测系统"}


@app.get("/health/live")
def health_live():
    """K8s liveness probe — 仅检查进程是否存活（轻量，无 DB/GPU 检查）"""
    return {"status": "alive", "service": "落石检测系统"}


@app.get("/health/ready")
def health_ready():
    """K8s readiness probe — 检查 GPU、数据库、模型是否就绪"""
    from rockfall.config import get_device, MODEL_PATH, get_active_model_path
    from rockfall.health import get_health

    issues = []

    # 模型文件检查
    model = get_active_model_path()
    if not model.exists():
        issues.append(f"模型文件不存在: {model}")

    # 数据库检查
    try:
        from rockfall.alert_store import get_alert_store
        store = get_alert_store()
        store.count_alerts()  # 快速健康查询 (不设日期过滤)
    except Exception as e:
        issues.append(f"数据库不可用: {e}")

    if issues:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"status": "not_ready", "issues": issues},
            status_code=503,
        )

    return {"status": "ready", "service": "落石检测系统"}


@app.get("/metrics")
def metrics():
    """Prometheus 监控指标端点。

    收集当前性能快照、摄像头数量、任务队列长度和存储统计。
    返回标准 Prometheus 文本格式。
    """
    from fastapi.responses import Response
    from rockfall.metrics import (
        collect_from_perf, set_camera_count, set_task_queue_length,
        set_storage_stats, set_system_info, get_metrics_text,
        set_db_connections,
    )
    from rockfall.performance import get_global_monitor
    from rockfall.config import get_device, MODEL_PATH
    from server.service import _active_cameras, _task_store

    # 系统信息（仅首次设置）
    device_str, device_name = get_device()
    set_system_info(device=device_str, model=str(MODEL_PATH))

    # 性能快照
    perf = get_global_monitor()
    snap = perf.snapshot()
    collect_from_perf(snap)

    # 摄像头 / 队列
    set_camera_count(len(_active_cameras))
    set_task_queue_length(
        sum(1 for t in _task_store.values() if t.get("status") == "processing")
    )

    # 存储统计
    from rockfall.config import RESULTS_DIR, UPLOADS_DIR
    total_size = 0
    file_count = 0
    for d in [RESULTS_DIR, UPLOADS_DIR]:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
                    file_count += 1
    set_storage_stats(total_size / (1024 ** 3), file_count)

    # 数据库连接状态（缓存 60 秒，避免每次 scrape 都建连）
    from rockfall.alert_store import get_alert_store
    import time as _time
    _now = _time.time()
    _cache_key = "db_conn_check"
    if not hasattr(metrics, "_db_cache"):
        metrics._db_cache = {}
    _cached = metrics._db_cache.get(_cache_key)
    if _cached and _now - _cached["ts"] < 60:
        set_db_connections(_cached["backend"], _cached["available"])
    else:
        try:
            store = get_alert_store()
            backend = store._backend  # "mysql" or "sqlite"
            if backend == "mysql":
                conn = None
                try:
                    from rockfall.db_engine import get_mysql_engine
                    engine = get_mysql_engine()
                    if engine is not None:
                        conn = engine.raw_connection()
                        # conn.close() 放到 finally 确保即使异常也归还
                        set_db_connections("mysql", True)
                        metrics._db_cache[_cache_key] = {"backend": "mysql", "available": True, "ts": _now}
                    else:
                        set_db_connections("mysql", False)
                        metrics._db_cache[_cache_key] = {"backend": "mysql", "available": False, "ts": _now}
                except Exception:
                    set_db_connections("mysql", False)
                    metrics._db_cache[_cache_key] = {"backend": "mysql", "available": False, "ts": _now}
                finally:
                    if conn is not None:
                        conn.close()  # 归还到池
            else:
                db_path = store._db_path
                available = db_path.exists()
                set_db_connections("sqlite", available)
                metrics._db_cache[_cache_key] = {"backend": "sqlite", "available": available, "ts": _now}
        except Exception:
            set_db_connections("unknown", False)

    return Response(
        content=get_metrics_text(),
        media_type="text/plain; charset=utf-8",
    )


# ============================================================
# 图片检测 (含上传安全校验)
# ============================================================

@app.get("/detect", response_model=ImageDetectResponse)
def detect_default(camera_id: str = Query("default")):
    """对默认测试图片检测 (兼容旧版接口)"""
    return detect_image_file(camera_id=camera_id)


@app.post("/detect/image", response_model=ImageDetectResponse)
def detect_uploaded_image(
    request: Request,
    file: UploadFile = File(...),
    camera_id: str = Query("default"),
):
    """上传图片进行落石检测（含安全校验）。"""
    from rockfall.upload_security import get_upload_validator
    from rockfall.audit import audit_log

    # 读取文件内容
    file_content = file.file.read()

    # 安全校验
    validator = get_upload_validator()
    validation = validator.validate(
        file_data=file_content,
        filename=file.filename or "unknown.jpg",
    )

    if not validation["valid"]:
        audit_log("upload_rejected", operator=_get_operator(request),
                  detail=f"文件上传被拒: {file.filename}, "
                         f"原因: {validation['error']}, "
                         f"检测类型: {validation.get('mime_type', 'unknown')}",
                  ip=_get_client_ip(), result="error")
        raise HTTPException(
            status_code=400,
            detail=f"文件安全校验失败: {validation['error']}",
        )

    # 保存文件到隔离目录
    safe_path = validator.save_quarantined(file_content, file.filename or "unknown.jpg")

    # 创建带自动清理的临时文件包装
    class _SafeUploadFile:
        """包装隔离文件为 UploadFile 兼容对象，确保文件句柄正确关闭。"""
        def __init__(self, path, original_name):
            self.filename = Path(path).name
            self._original_name = original_name
            self._path = path
            self._fh = open(path, 'rb')
            self.file = self._fh

        def close(self):
            if self._fh and not self._fh.closed:
                self._fh.close()

    safe_file = _SafeUploadFile(safe_path, file.filename or "unknown.jpg")
    try:
        result = detect_image_file(safe_file, camera_id=camera_id)
        audit_log("upload_image_detect", operator=_get_operator(request),
                  detail=f"上传图片检测完成: {file.filename} → {safe_path}",
                  ip=_get_client_ip(), result="ok")
        return result
    except Exception as e:
        audit_log("upload_image_detect", operator=_get_operator(request),
                  detail=f"上传图片检测失败: {file.filename}, 原因: {e}",
                  ip=_get_client_ip(), result="error")
        raise
    finally:
        safe_file.close()


# ============================================================
# 视频检测 (含上传安全校验)
# ============================================================

@app.post("/detect/video", response_model=TaskResponse)
def detect_uploaded_video(
    request: Request,
    file: UploadFile = File(...),
    save_frames: bool = Form(True),
    push_alerts: bool = Form(False),
    sync: bool = Form(False),
    camera_id: str = Form("default"),
):
    """上传视频进行运动检测+YOLO落石检测（含安全校验）。

    默认异步模式 (sync=false): 立即返回 task_id, 通过 GET /api/tasks/{task_id} 轮询结果。
    同步模式 (sync=true): 阻塞等待, 仅适合短视频 (<60s)。
    camera_id: 区分不同摄像头/监测点 (默认 "default")。
    """
    from rockfall.upload_security import get_upload_validator
    from rockfall.audit import audit_log

    # 读取文件内容
    file_content = file.file.read()

    # 安全校验
    validator = get_upload_validator()
    validation = validator.validate(
        file_data=file_content,
        filename=file.filename or "unknown.mp4",
    )

    if not validation["valid"]:
        audit_log("upload_rejected", operator=_get_operator(request),
                  detail=f"视频上传被拒: {file.filename}, "
                         f"原因: {validation['error']}, "
                         f"检测类型: {validation.get('mime_type', 'unknown')}",
                  ip=_get_client_ip(), result="error")
        raise HTTPException(
            status_code=400,
            detail=f"文件安全校验失败: {validation['error']}",
        )

    # 保存文件到隔离目录
    safe_path = validator.save_quarantined(file_content, file.filename or "unknown.mp4")

    # ── FLV 兼容: 尝试用 OpenCV 重新封装为 AVI ──
    _filename_lower = (file.filename or "").lower()
    if _filename_lower.endswith(".flv"):
        import cv2
        _converted = Path(safe_path).with_suffix(".avi")
        _cap = cv2.VideoCapture(str(safe_path))
        if _cap.isOpened():
            _fps = _cap.get(cv2.CAP_PROP_FPS) or 25.0
            _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _writer = cv2.VideoWriter(
                str(_converted),
                cv2.VideoWriter_fourcc(*"XVID"),
                _fps, (_w, _h),
            )
            while True:
                _ok, _frame = _cap.read()
                if not _ok:
                    break
                _writer.write(_frame)
            _writer.release()
            _cap.release()
            if _converted.exists():
                safe_path = str(_converted)
        else:
            _cap.release()

    # 创建带自动清理的临时文件包装
    class _SafeUploadFile:
        """包装隔离文件为 UploadFile 兼容对象，确保文件句柄正确关闭。"""
        def __init__(self, path, original_name):
            self.filename = Path(path).name
            self._original_name = original_name
            self._path = path
            self._fh = open(path, 'rb')
            self.file = self._fh

        def close(self):
            if self._fh and not self._fh.closed:
                self._fh.close()

    safe_file = _SafeUploadFile(safe_path, file.filename or "unknown.mp4")

    try:
        if sync:
            result = detect_video_file(safe_file, save_frames, push_alerts, camera_id=camera_id)
            audit_log("upload_video_detect", operator=_get_operator(request),
                      detail=f"上传视频检测完成(同步): {file.filename} → {safe_path}",
                      ip=_get_client_ip(), result="ok")
            return result

        task_id = detect_video_file_async(safe_file, save_frames, push_alerts, camera_id=camera_id)
        audit_log("upload_video_detect", operator=_get_operator(request),
                  detail=f"上传视频检测(异步): {file.filename} → {safe_path}, task={task_id}",
                  ip=_get_client_ip(), result="ok")
        return {"task_id": task_id, "status": "processing"}
    except Exception as e:
        audit_log("upload_video_detect", operator=_get_operator(request),
                  detail=f"上传视频检测失败: {file.filename}, 原因: {e}",
                  ip=_get_client_ip(), result="error")
        raise
    finally:
        safe_file.close()


@app.post("/detect/video/local", response_model=TaskResponse)
def detect_local_video(
    request: Request,
    path: str = Form(...),
    save_frames: bool = Form(True),
    push_alerts: bool = Form(False),
    sync: bool = Form(False),
    camera_id: str = Form("default"),
):
    """对服务器本地视频文件进行检测 (仅允许 DATA_DIR 下的文件)"""
    from rockfall.config import DATA_DIR
    from rockfall.audit import audit_log

    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="路径不在允许范围内")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    if sync:
        result = detect_video_local(str(resolved), save_frames, push_alerts, camera_id=camera_id)
        audit_log("local_video_detect", operator=_get_operator(request),
                  detail=f"本地视频检测完成(同步): {path}", ip=_get_client_ip(), result="ok")
        return result

    task_id = detect_video_local_async(str(resolved), save_frames, push_alerts, camera_id=camera_id)
    audit_log("local_video_detect", operator=_get_operator(request),
              detail=f"本地视频检测(异步): {path}, task={task_id}",
              ip=_get_client_ip(), result="ok")
    return {"task_id": task_id, "status": "processing"}


# ============================================================
# 边缘-云协同 — 边缘端上传可疑帧，云端二次确认
# ============================================================

@app.post("/api/edge/upload")
async def edge_upload(
    request: Request,
    frame: UploadFile = File(...),
    source_name: str = Form("edge"),
    site_id: str = Form(""),
):
    """
    边缘设备上传可疑帧到云端做完整推理。

    边缘端已通过 MOG2 运动检测 + (可选) Nano 模型预筛选，
    云端用大模型二次确认，返回检测结果给边缘端。

    请求:
        frame:       JPEG 图片文件
        source_name: 边缘设备标识
        site_id:     监测点位 ID

    返回:
        {
            "detected": true/false,
            "count": 0,
            "max_confidence": 0.0,
            "alert_level": "green"/"yellow"/...,
            "source_name": "edge_cam_1",
            "processing_time_ms": 123
        }
    """
    import cv2
    import numpy as np
    import time as _time

    t0 = _time.perf_counter()

    # 读取上传的 JPEG 并解码为 BGR
    jpg_bytes = await frame.read()
    np_arr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码上传的图片")

    fh, fw = img.shape[:2]

    # 获取或创建该边缘设备对应的检测器
    detector_id = f"edge:{source_name}" if source_name else "edge:default"
    from server.service import _get_detector, _inference_semaphore

    try:
        detector = _get_detector(detector_id)

        # 初始化流状态 (如未初始化)
        if not getattr(detector, '_stream_ready', False):
            detector.init_stream_state(fw, fh)

        # MOG2 预处理
        pp = detector.preprocess_frame(img)

        # 运动分数低 → 跳过推理 (边缘端已预筛选，此处二次确认)
        if pp['motion_score'] < 0.001:
            elapsed = (_time.perf_counter() - t0) * 1000
            return {
                "detected": False,
                "count": 0,
                "max_confidence": 0.0,
                "alert_level": "green",
                "source_name": source_name,
                "processing_time_ms": round(elapsed, 1),
                "skipped_reason": "low_motion",
            }

        # 完整 YOLO 推理（单帧检测，不做跨帧跟踪）
        with _inference_semaphore:
            raw_dets = detector.detect_frame(img, pp['box_mask'], pp['fg'])

        # 从原始检测框构建简化版 tracks_info（无跟踪ID，置信度直接用原始值）
        # 边缘上传是单帧，跟踪需要连续帧才有意义
        tracks_info = []
        for i, d in enumerate(raw_dets):
            x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
            conf = d[4]
            tracks_info.append({
                "id": i,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "smoothed_confidence": conf,
                "area": (x2 - x1) * (y2 - y1),
                "age": 1,
                "speed": 0,
                "motion_state": "未知",
                "confirmed": True,  # 边缘已预筛选，直接确认
                "class_id": int(d[5]) if len(d) > 5 else 0,
                "class_name": "落石",
                "trajectory": [],
            })

        # 预警分级
        if tracks_info:
            ctx = detector.build_alert_context(tracks_info, fw, fh)
            alert_level = detector._grade_alert(ctx)
            max_conf = ctx.max_conf
            count = ctx.total_count
        else:
            alert_level = "green"
            max_conf = 0.0
            count = 0
            ctx = None

        elapsed = (_time.perf_counter() - t0) * 1000

        # 告警推送（达到橙色或以上时触发）
        if alert_level in ("red", "orange"):
            from rockfall.notifier import dispatch_alert_async
            dispatch_alert_async(
                count=count, max_confidence=max_conf,
                alert_level=alert_level,
                frame_bgr=img,
                tracks=tracks_info,
                rock_diameter_cm=ctx.rock_diameter_cm if ctx else 0,
            )

        return {
            "detected": alert_level != "green",
            "count": count,
            "max_confidence": round(max_conf, 4),
            "alert_level": alert_level,
            "source_name": source_name,
            "processing_time_ms": round(elapsed, 1),
        }

    except Exception as e:
        elapsed = (_time.perf_counter() - t0) * 1000
        log_event("system", level="ERROR",
                  msg=f"边缘上传处理失败 [{source_name}]: {e}")
        return {
            "detected": False,
            "count": 0,
            "max_confidence": 0.0,
            "alert_level": "error",
            "source_name": source_name,
            "processing_time_ms": round(elapsed, 1),
            "error": str(e)[:200],
        }


# ============================================================
# 异步任务查询
# ============================================================

@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
def api_task_status(task_id: str):
    """查询异步视频检测任务的状态和结果"""
    task = get_task_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"task_id": task_id, **task}


# ============================================================
# WebSocket — 异步任务进度实时推送
# ============================================================

@app.websocket("/ws/tasks/{task_id}")
async def ws_task_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket 端点: 实时推送视频检测任务进度。

    推送 JSON 格式:
      {"status": "processing", "progress": 45.2, "current_frame": 452, "total_frames": 1000}
      {"status": "completed", "progress": 100.0, "result": {...}}
      {"status": "failed", "progress": 0, "error": "..."}

    前端用法:
      const ws = new WebSocket(`ws://${host}/ws/tasks/${taskId}`);
      ws.onmessage = (e) => { const data = JSON.parse(e.data); updateProgress(data); };
    """
    await websocket.accept()

    # 检查任务是否存在
    task = get_task_status(task_id)
    if task is None:
        await websocket.send_json({"status": "not_found", "error": "任务不存在"})
        await websocket.close()
        return

    # 如果任务已完成/失败，立即发送当前状态并关闭
    if task.get("status") in ("completed", "failed"):
        await websocket.send_json({
            "task_id": task_id,
            "status": task["status"],
            "progress": task.get("progress", 100.0 if task["status"] == "completed" else 0),
            "current_frame": task.get("current_frame", 0),
            "total_frames": task.get("total_frames", 0),
            "result": task.get("result"),
            "error": task.get("error"),
        })
        await websocket.close()
        return

    # 轮询推送进度 (100ms 间隔, 变化时才推送)
    last_progress = -1.0
    last_status = "processing"
    try:
        while True:
            task = get_task_status(task_id)
            if task is None:
                await websocket.send_json({"status": "not_found", "error": "任务已被清理"})
                break

            current_status = task.get("status", "processing")
            current_progress = task.get("progress", 0.0)

            # 仅在进度变化或状态变化时推送
            if abs(current_progress - last_progress) > 0.1 or current_status != last_status:
                payload = {
                    "task_id": task_id,
                    "status": current_status,
                    "progress": current_progress,
                    "current_frame": task.get("current_frame", 0),
                    "total_frames": task.get("total_frames", 0),
                }
                if current_status == "completed":
                    payload["result"] = task.get("result")
                elif current_status == "failed":
                    payload["error"] = task.get("error")
                await websocket.send_json(payload)
                last_progress = current_progress
                last_status = current_status

            # 任务终态: 发送最终结果后断开
            if current_status in ("completed", "failed"):
                break

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        # 客户端断开连接 (正常行为, 无需记录)
        pass
    except Exception as e:
        log_event("system", level="WARN", msg=f"WebSocket 异常 task={task_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================
# 配置热更新
# ============================================================

@app.post("/api/config/reload")
def config_reload(request: Request):
    """热重载 .env 配置 (仅调试用, 已运行的检测流水线需重启才生效)"""
    import importlib
    import rockfall.config
    from dotenv import load_dotenv
    from rockfall.audit import audit_log

    load_dotenv(override=True)
    importlib.reload(rockfall.config)

    warnings = rockfall.config.validate_config()

    audit_log("config_reload", operator=_get_operator(request),
              detail=f"重载 .env 配置, 警告数: {len(warnings)}",
              ip=_get_client_ip(),
              result="ok" if not warnings else "partial")

    return {
        "status": "ok",
        "warnings": warnings,
        "note": "检测流水线需重启才能应用新配置",
    }


@app.get("/api/config/current")
def config_current():
    """查看当前运行中的核心配置（敏感值自动脱敏）"""
    import rockfall.config as cfg
    from rockfall.secrets import SecretsManager

    sanitize = SecretsManager.sanitize_config_for_log

    return {
        "detection": sanitize({
            "confidence": cfg.DETECTION_CONFIDENCE,
            "img_size": cfg.DETECTION_IMG_SIZE,
            "model_path": cfg.MODEL_PATH,
            "tensorrt": cfg.TENSORRT_ENABLED,
        }),
        "skip": {
            "idle": cfg.SKIP_IDLE,
            "active": cfg.SKIP_ACTIVE,
            "critical": cfg.SKIP_CRITICAL,
        },
        "mog2": {
            "history": cfg.MOG2_HISTORY,
            "learning_rate": cfg.MOG2_LEARNING_RATE,
            "reset_idle": cfg.MOG2_RESET_IDLE_FRAMES,
        },
        "alert": {
            "four_level": {
                "blue":   f"{cfg.ALERT_BLUE_CONFIDENCE_LOW}-{cfg.ALERT_BLUE_CONFIDENCE_HIGH}",
                "yellow": f"{cfg.ALERT_BLUE_CONFIDENCE_HIGH}-{cfg.ALERT_YELLOW_CONFIDENCE_HIGH}",
                "orange": f"{cfg.ALERT_YELLOW_CONFIDENCE_HIGH}-{cfg.ALERT_ORANGE_CONFIDENCE_HIGH}",
                "red":    f">{cfg.ALERT_ORANGE_CONFIDENCE_HIGH}",
            },
            "rock_size": {
                "small":  f"<{cfg.ROCK_SMALL_HEIGHT_RATIO*100:.0f}% height (<10cm)",
                "medium": f"{cfg.ROCK_SMALL_HEIGHT_RATIO*100:.0f}%-{cfg.ROCK_MEDIUM_HEIGHT_RATIO*100:.0f}% (10-20cm)",
                "large":  f"{cfg.ROCK_MEDIUM_HEIGHT_RATIO*100:.0f}%-{cfg.ROCK_LARGE_HEIGHT_RATIO*100:.0f}% (20-30cm)",
                "xlarge": f">{cfg.ROCK_LARGE_HEIGHT_RATIO*100:.0f}% (>30cm)",
            },
            "falling_min_conf": cfg.ALERT_FALLING_MIN_CONF,
            "multi_count": cfg.ALERT_MULTI_COUNT,
            "cooldown": cfg.ALERT_COOLDOWN_SECONDS,
        },
        "filters": {
            "tfd": cfg.TFD_ENABLED,
            "mog2_filter": cfg.MOG2_FILTER_ENABLED,
            "sahi": cfg.SAHI_ENABLED,
            "fusion": cfg.FUSION_ENABLED,
            "temporal": cfg.TEMPORAL_ENABLED,
            "edge_enhance": cfg.EDGE_ENHANCE_ENABLED,
        },
        "security": {
            "https_enforced": __import__('os').getenv("ENFORCE_HTTPS", "false"),
            "upload_max_mb": __import__('os').getenv("UPLOAD_MAX_SIZE_MB", "500"),
            "auth_jwt_secret": "***configured***" if __import__('os').getenv("AUTH_JWT_SECRET") else "(auto-generated)",
        },
    }


# ============================================================
# 模型版本管理
# ============================================================

@app.get("/api/models")
def list_models():
    """列出所有可用的模型版本（含时段专用模型和点位专用模型）。"""
    from rockfall.config import list_all_models, get_active_model_path
    all_models = list_all_models()
    active = str(get_active_model_path())
    return {
        "models": all_models,
        "active": active,
        "total": len(all_models),
    }


@app.get("/api/models/current")
def current_model():
    """
    获取当前激活点位实际使用的模型路径（含点位专用和时段模型解析结果）。

    返回:
        - resolved_path: 实际推理使用的模型路径
        - global_active: 全局默认激活模型
        - site_id: 当前点位
        - site_override: 点位专用模型 (如有)
        - slot_model: 时段匹配的模型 (如有)
    """
    from rockfall.config import resolve_model_path, get_active_model_path, _get_model_for_hour
    from rockfall.site_config import get_active_site_id, get_active_site, get_site_by_id
    from datetime import datetime

    site_id = get_active_site_id()
    resolved = str(resolve_model_path(site_id))
    global_active = str(get_active_model_path())
    hour = datetime.now().hour

    site = get_site_by_id(site_id)
    site_override = site.model_override if site else ""

    return {
        "resolved_path": resolved,
        "global_active": global_active,
        "site_id": site_id,
        "site_override": site_override or None,
        "slot_model": _get_model_for_hour(hour),
        "current_hour": hour,
    }


@app.post("/api/models/switch")
def switch_model(request: Request, model_path: str = Form(...)):
    """切换模型版本（原子符号链接操作，支持快速回滚）"""
    from rockfall.config import set_active_model, get_active_model_path
    from rockfall.audit import audit_log

    old_model = str(get_active_model_path())

    try:
        set_active_model(model_path)
        new_model = str(get_active_model_path())
        operator = _get_operator(request)
        audit_log("model_switch", operator=operator,
                  detail=f"模型切换: {old_model} → {new_model}",
                  ip=_get_client_ip(), result="ok",
                  before={"model_path": old_model},
                  after={"model_path": new_model})
        return {
            "success": True,
            "active": new_model,
            "msg": f"模型已切换, 新检测任务将加载新模型（热更新）",
        }
    except FileNotFoundError as e:
        audit_log("model_switch", operator=_get_operator(request),
                  detail=f"模型切换失败 (文件不存在): {model_path}",
                  ip=_get_client_ip(), result="error",
                  before={"model_path": old_model})
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        audit_log("model_switch", operator=_get_operator(request),
                  detail=f"模型切换失败: {e}", ip=_get_client_ip(), result="error")
        raise HTTPException(status_code=500, detail=f"模型切换失败: {e}")


# ============================================================
# 模型注册表 API (Model Registry) — A/B 测试 + 自动回滚
# ============================================================

@app.get("/api/models/registry")
def get_model_registry():
    """获取模型注册表完整状态 (版本列表、A/B 分流、回滚历史)。"""
    try:
        from rockfall.model_registry import get_registry
        registry = get_registry()
        status = registry.get_status()
        # 追加远程版本缓存
        try:
            from rockfall.model_poller import get_poller
            poller = get_poller()
            status["remote_versions_cached"] = poller.version_cache
            status["last_poll_time"] = poller.last_poll_time
        except Exception:
            status["remote_versions_cached"] = []
            status["last_poll_time"] = ""
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取注册表状态失败: {e}")


@app.post("/api/models/ab-split")
def set_ab_split(
    request: Request,
    split_pct: float = Form(..., ge=0.0, le=100.0),
):
    """
    设置 A/B 测试流量分割比例。

    参数:
        split_pct: 0=全用稳定版, 50=各50%, 100=全用候选版
    """
    from rockfall.config import RuntimeConfig, MODEL_REGISTRY_AB_SPLIT
    from rockfall.audit import audit_log

    old_split = RuntimeConfig.get("MODEL_REGISTRY_AB_SPLIT", MODEL_REGISTRY_AB_SPLIT)

    try:
        RuntimeConfig.set("MODEL_REGISTRY_AB_SPLIT", split_pct)
        operator = _get_operator(request)
        audit_log(
            "model_ab_split", operator=operator,
            detail=f"A/B 分流比例: {old_split}% → {split_pct}%",
            ip=_get_client_ip(), result="ok",
        )
        return {
            "success": True,
            "ab_split_pct": split_pct,
            "previous": old_split,
            "msg": f"A/B 分流比例已更新 (需新检测任务生效)",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置 A/B 分流失败: {e}")


@app.get("/api/models/rollback-history")
def get_rollback_history(limit: int = 20):
    """获取模型自动回滚历史记录。"""
    try:
        from rockfall.model_registry import get_registry
        registry = get_registry()
        return {
            "history": registry.get_rollback_history(limit=limit),
            "total": len(registry._rollback_history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取回滚历史失败: {e}")


@app.post("/api/models/activate")
def activate_model(request: Request, version_name: str = Form(..., description="模型版本名 (如 rock_best_v3.pt)")):
    """
    激活指定模型版本 (通过模型注册表, 原子切换 + 更新清单状态)。

    与 /api/models/switch 的区别:
      - /api/models/switch: 接受文件路径, 直接操作符号链接 (绕过注册表)
      - /api/models/activate: 接受版本名, 通过注册表切换并同步 manifest.json
    """
    from rockfall.config import get_active_model_path
    from rockfall.audit import audit_log

    try:
        from rockfall.model_registry import get_registry, MODEL_REGISTRY_ENABLED
        if not MODEL_REGISTRY_ENABLED:
            raise HTTPException(status_code=400, detail="模型注册表未启用 (MODEL_REGISTRY_ENABLED=false)")

        registry = get_registry()
        old_active = registry.active_version.name if registry.active_version else str(get_active_model_path())

        registry.activate_model(version_name)
        new_active = registry.active_version.name if registry.active_version else version_name

        operator = _get_operator(request)
        audit_log("model_activate", operator=operator,
                  detail=f"模型激活 (注册表): {old_active} → {new_active}",
                  ip=_get_client_ip(), result="ok",
                  before={"version": old_active},
                  after={"version": new_active})

        return {
            "success": True,
            "active": new_active,
            "previous": old_active,
            "msg": f"模型已激活: {version_name} (通过注册表, manifest 已同步)",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型激活失败: {e}")


@app.post("/api/models/poll-now")
def trigger_model_poll(request: Request):
    """手动触发远程模型版本检查 (同步, 返回结果)。"""
    from rockfall.audit import audit_log

    try:
        from rockfall.model_poller import get_poller
        poller = get_poller()
        result = poller.poll_now()

        operator = _get_operator(request)
        audit_log(
            "model_poll", operator=operator,
            detail=f"手动触发模型版本检查: {result.get('status')}",
            ip=_get_client_ip(), result=result.get("status", "unknown"),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型版本检查失败: {e}")


# ============================================================
# 预警工单流转
# ============================================================

@app.post("/api/alerts/{alert_id}/workflow")
def transition_alert_workflow(
    request: Request,
    alert_id: int,
    state: str = Form(...),
    operator: str = Form(""),
    note: str = Form(""),
):
    """预警工单状态流转"""
    from rockfall.alert_store import get_alert_store
    from rockfall.audit import audit_log

    store = get_alert_store()
    old_state = store._get_workflow_state(alert_id)

    result = store.transition_workflow(alert_id, state, operator, note)

    effective_operator = operator or _get_operator(request)
    audit_log("workflow_transition", operator=effective_operator,
              detail=f"Alert#{alert_id}: {old_state or '(初始)'} → {state}",
              alert_id=alert_id,
              result="ok" if result["ok"] else result["msg"],
              before={"workflow_state": old_state},
              after={"workflow_state": state, "note": note})
    return result


@app.get("/api/alerts/{alert_id}/workflow")
def get_alert_workflow(alert_id: int):
    """获取预警工单流转历史"""
    from rockfall.alert_store import get_alert_store
    store = get_alert_store()
    current = store._get_workflow_state(alert_id)
    history = store.get_workflow_history(alert_id)
    states = store.WORKFLOW_STATES
    return {"alert_id": alert_id, "current_state": current,
            "current_label": states.get(current, current),
            "history": history, "states": states}


# ============================================================
# 系统健康检查
# ============================================================

@app.get("/api/health/full")
def full_health_check():
    """完整系统健康检查"""
    from rockfall.health import get_health
    return get_health().check_all()


@app.get("/api/health/storage")
def storage_stats():
    """存储统计"""
    from rockfall.storage import StorageManager
    sm = StorageManager()
    return sm.get_storage_stats()


@app.post("/api/health/cleanup")
def trigger_cleanup(
    request: Request,
    retention_days: int = Form(30),
    dry_run: bool = Form(False),
    operator: str = Form(""),
):
    """手动触发文件清理"""
    from rockfall.storage import StorageManager
    from rockfall.audit import audit_log

    sm = StorageManager()
    result = sm.cleanup_old_files(retention_days=retention_days, dry_run=dry_run)

    effective_operator = operator or _get_operator(request)
    audit_log("storage_cleanup", operator=effective_operator,
              detail=f"retention={retention_days}d, dry_run={dry_run}, "
                      f"deleted={result['deleted_count']}, freed={result['freed_mb']}MB",
              ip=_get_client_ip(),
              after={"retention_days": retention_days, "dry_run": dry_run,
                     "deleted_count": result.get("deleted_count", 0),
                     "freed_mb": result.get("freed_mb", 0)})
    return result


# ============================================================
# 数据保留 & 冷存储归档
# ============================================================

@app.get("/api/health/retention")
def api_retention_policy():
    """查看当前保留策略和存储统计。"""
    from rockfall.storage import StorageManager
    sm = StorageManager()
    return sm.get_retention_policy()


@app.post("/api/health/archive")
def api_trigger_archive(
    request: Request,
    retention_days: int | None = Form(None),
    dry_run: bool = Form(False),
    operator: str = Form(""),
):
    """手动触发预警记录归档。

    - retention_days: 自定义保留天数 (默认使用 ALERT_RETENTION_DAYS)
    - dry_run: True=仅统计不实际删除
    """
    from rockfall.alert_store import get_alert_store
    from rockfall.audit import audit_log

    store = get_alert_store()
    result = store.archive_and_purge(
        retention_days=retention_days, dry_run=dry_run,
    )

    effective_operator = operator or _get_operator(request)
    audit_log(
        "manual_archive", operator=effective_operator,
        detail=f"retention_days={retention_days}, dry_run={dry_run}, "
               f"archived={result.get('archived_count', 0)}",
        ip=_get_client_ip(),
        result="ok" if not result.get("errors") else "partial_error",
    )
    return result


@app.get("/api/health/archives")
def api_list_archives(
    prefix: str = Query("", description="Key 前缀过滤"),
):
    """列出冷存储中的归档文件。"""
    from rockfall.cold_storage import ColdStorageClient
    client = ColdStorageClient()
    if not client.enabled:
        return {"enabled": False, "archives": [], "msg": "冷存储未配置"}
    archives = client.list_archives(prefix=prefix)
    return {"enabled": True, "archives": archives}


@app.post("/api/health/archive/restore")
def api_restore_archive(
    key: str = Form(..., description="冷存储中的归档 key"),
):
    """从冷存储下载指定归档文件到本地。"""
    from pathlib import Path
    from rockfall.cold_storage import ColdStorageClient
    from rockfall.config import DATA_DIR

    client = ColdStorageClient()
    if not client.enabled:
        raise HTTPException(status_code=400, detail="冷存储未配置")

    restore_dir = DATA_DIR / "archive" / "restored"
    restore_dir.mkdir(parents=True, exist_ok=True)
    local_path = restore_dir / Path(key).name

    ok = client.download_archive(key, local_path)
    if ok:
        return {"ok": True, "local_path": str(local_path)}
    raise HTTPException(status_code=500, detail="下载失败")


# ============================================================
# 审计日志
# ============================================================

@app.get("/api/audit")
def query_audit_log(action: str = "", operator: str = "",
                    start: str = "", end: str = "",
                    limit: int = 50, offset: int = 0):
    """查询审计日志（含变更前后值）"""
    from rockfall.audit import get_audit_logger
    audit = get_audit_logger()
    rows = audit.query(action=action, operator=operator,
                       start=start, end=end, limit=limit, offset=offset)
    total = audit.count(action=action, operator=operator, start=start, end=end)
    return {"rows": rows, "total": total, "limit": limit, "offset": offset}


@app.get("/api/audit/summary")
def audit_actions_summary():
    """审计日志操作类型汇总"""
    from rockfall.audit import get_audit_logger
    return get_audit_logger().get_actions_summary()


# ============================================================
# 预警工单看板统计
# ============================================================

@app.get("/api/workflow/stats")
def workflow_stats():
    """工单状态统计"""
    from rockfall.alert_store import get_alert_store
    store = get_alert_store()
    counts = store.count_by_workflow_state()
    states = store.WORKFLOW_STATES
    result = {}
    for state, label in states.items():
        result[state] = {"label": label, "count": counts.get(state, 0)}
    return result


# ============================================================
# 生产模式: 挂载 React 前端 (npm run build → server/static/)
# ============================================================
_SPA_DIR = Path(__file__).parent / "static"
_SPA_READY = _SPA_DIR.exists() and (_SPA_DIR / "index.html").exists()
if _SPA_READY:
    # 静态资源 (JS/CSS/图片) — 带路径前缀
    _assets_dir = _SPA_DIR / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="spa_assets")

    # SPA catch-all: 所有非 API 路径兜底返回 index.html
    @app.get("/{full_path:path}", name="spa_fallback")
    async def spa_fallback(request: Request, full_path: str):
        from fastapi.responses import FileResponse
        spa_index = _SPA_DIR / "index.html"
        # 不拦截 API/WebSocket/detect 请求
        if full_path.startswith(("api/", "ws/", "detect/")):
            raise HTTPException(status_code=404)
        return FileResponse(spa_index, media_type="text/html")

    log_event("system", msg=f"React SPA 已挂载: {_SPA_DIR}")
else:
    log_event("system", msg="React SPA 未构建 — 使用经典 Web 看板 (npm run build 以启用)")
