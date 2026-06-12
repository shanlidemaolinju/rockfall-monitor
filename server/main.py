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
  GET  /detect              — 对默认图片检测 (兼容旧接口)
  POST /detect/image        — 上传图片检测
  POST /detect/video        — 上传视频检测
  POST /detect/video/local  — 本地视频路径检测
"""

import sys
from pathlib import Path

# 确保项目根目录可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import io
import json

from fastapi import FastAPI, File, UploadFile, Form, Query, Header, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from server.service import (
    detect_image_file, detect_video_file, detect_video_local,
    detect_video_file_async, detect_video_local_async, get_task_status,
    get_dashboard_stats, get_recent_alerts, query_alerts_page,
    export_alerts_excel, get_export_summary,
    get_sites_data, switch_active_site,
    update_runtime_config, get_runtime_config,
    mark_alert_review, get_alert_statistics, get_alert_image_info,
)
from server.schemas import (
    HealthResponse, DashboardStats, AlertItem,
    ImageDetectResponse, VideoDetectResponse, ErrorResponse,
    TaskResponse, TaskStatusResponse,
)
from rockfall.detector import get_latest_frame

app = FastAPI(title="落石检测系统 API", version="2.1.0")

# 启动时配置验证 + 设备检测
from rockfall.config import validate_config, get_device
_config_warnings = validate_config()
_device_str, _device_name = get_device()
print(f"[推理设备] {_device_name} ({_device_str})")
for w in _config_warnings:
    print(f"[配置警告] {w}")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---- API Key 鉴权中间件 ----
_PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/favicon.ico"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        from rockfall.config import API_KEY
        if not API_KEY or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key != API_KEY:
            return JSONResponse({"detail": "无效的 API Key"}, status_code=401)

        return await call_next(request)


app.add_middleware(ApiKeyMiddleware)


# ============================================================
# Web 看板
# ============================================================

@app.get("/")
def dashboard():
    """Web 看板首页 — 纯静态 HTML, 不依赖 Jinja2 模板引擎"""
    from fastapi.responses import HTMLResponse
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
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
                jpg = get_latest_frame(camera_id)
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


@app.post("/api/alerts/{alert_id}/review")
def api_alert_review(
    alert_id: int,
    review_status: str = Form(..., description="confirmed | false_alarm | (空=清除)"),
    note: str = Form("", description="审核备注"),
):
    """标记预警审核状态 (确认真实/误报)"""
    valid = {"confirmed", "false_alarm", ""}
    if review_status not in valid:
        raise HTTPException(status_code=400, detail=f"review_status 必须是: {valid}")
    return mark_alert_review(alert_id, review_status, note)


@app.get("/api/alerts/{alert_id}/image")
def api_alert_image(alert_id: int):
    """
    获取预警记录的现场截图。

    返回: JSON (图片元信息 + base64 编码的图片数据)
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
    try:
        excel_bytes = export_alerts_excel(
            start_date=start_date, end_date=end_date, alert_level=alert_level,
        )
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))

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
def delete_camera(camera_id: str):
    """释放指定摄像头的检测器资源"""
    from server.service import remove_detector
    remove_detector(camera_id)
    return {"status": "ok", "camera_id": camera_id}


# ============================================================
# 监测点位管理
# ============================================================

@app.get("/api/sites")
def api_sites():
    """获取全部监测点位 + 当前激活点位"""
    return get_sites_data()


@app.post("/api/sites/switch")
def api_switch_site(site_id: str = Form(...)):
    """切换当前激活的监测点位"""
    try:
        return switch_active_site(site_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# 运行时配置热更新
# ============================================================

@app.get("/api/config/runtime")
def api_config_runtime():
    """获取当前运行中的可调参数值"""
    return get_runtime_config()


@app.post("/api/config/update")
def api_config_update(payload: dict):
    """
    热更新检测器参数 (当前会话有效)。

    请求体 JSON:
        {"detection_confidence": 0.5, "alert_blue_high": 0.55, ...}

    支持的白名单键:
        detection_confidence, detection_img_size, motion_min_area,
        alert_blue_high, alert_yellow_high, alert_orange_high
    """
    result = update_runtime_config(payload)
    if result["skipped"]:
        return JSONResponse(
            content={"status": "partial", **result},
            status_code=200,
        )
    return {"status": "ok", **result}


# ============================================================
# 健康检查
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "service": "落石检测系统"}


# ============================================================
# 图片检测
# ============================================================

@app.get("/detect", response_model=ImageDetectResponse)
def detect_default(camera_id: str = Query("default")):
    """对默认测试图片检测 (兼容旧版接口)"""
    return detect_image_file(camera_id=camera_id)


@app.post("/detect/image", response_model=ImageDetectResponse)
def detect_uploaded_image(file: UploadFile = File(...),
                          camera_id: str = Query("default")):
    """上传图片进行落石检测"""
    return detect_image_file(file, camera_id=camera_id)


# ============================================================
# 视频检测
# ============================================================

@app.post("/detect/video", response_model=TaskResponse)
def detect_uploaded_video(
    file: UploadFile = File(...),
    save_frames: bool = Form(True),
    push_alerts: bool = Form(False),
    sync: bool = Form(False),
    camera_id: str = Form("default"),
):
    """上传视频进行运动检测+YOLO落石检测。

    默认异步模式 (sync=false): 立即返回 task_id, 通过 GET /api/tasks/{task_id} 轮询结果。
    同步模式 (sync=true): 阻塞等待, 仅适合短视频 (<60s)。
    camera_id: 区分不同摄像头/监测点 (默认 "default")。
    """
    if sync:
        return detect_video_file(file, save_frames, push_alerts, camera_id=camera_id)
    task_id = detect_video_file_async(file, save_frames, push_alerts, camera_id=camera_id)
    return {"task_id": task_id, "status": "processing"}


@app.post("/detect/video/local", response_model=TaskResponse)
def detect_local_video(
    path: str = Form(...),
    save_frames: bool = Form(True),
    push_alerts: bool = Form(False),
    sync: bool = Form(False),
    camera_id: str = Form("default"),
):
    """对服务器本地视频文件进行检测 (仅允许 DATA_DIR 下的文件)"""
    from rockfall.config import DATA_DIR
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="路径不在允许范围内")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    if sync:
        return detect_video_local(str(resolved), save_frames, push_alerts, camera_id=camera_id)
    task_id = detect_video_local_async(str(resolved), save_frames, push_alerts, camera_id=camera_id)
    return {"task_id": task_id, "status": "processing"}


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
# 配置热更新
# ============================================================

@app.post("/api/config/reload")
def config_reload():
    """热重载 .env 配置 (仅调试用, 已运行的检测流水线需重启才生效)"""
    import importlib
    import rockfall.config
    from dotenv import load_dotenv

    load_dotenv(override=True)
    importlib.reload(rockfall.config)

    warnings = rockfall.config.validate_config()
    return {
        "status": "ok",
        "warnings": warnings,
        "note": "检测流水线需重启才能应用新配置",
    }


@app.get("/api/config/current")
def config_current():
    """查看当前运行中的核心配置"""
    import rockfall.config as cfg
    return {
        "detection": {
            "confidence": cfg.DETECTION_CONFIDENCE,
            "img_size": cfg.DETECTION_IMG_SIZE,
            "model_path": cfg.MODEL_PATH,
            "tensorrt": cfg.TENSORRT_ENABLED,
        },
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
    }
