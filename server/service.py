"""
业务逻辑层 — 封装检测调用 + 看板数据聚合 + 异步任务管理
==========================================================
本层是 API 路由 和 核心算法库 之间的桥梁。

职责:
  - 处理上传文件的保存和清理
  - 委托 rockfall.detector.RockDetector 执行实际检测
  - 聚合检测日志为看板统计
  - 视频检测异步任务管理 (避免长视频 HTTP 超时)
  - 返回标准化的 JSON 结果

不负责: 检测算法本身 (那是 rockfall.detector 的职责)
"""

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from rockfall.detector import RockDetector
from rockfall.logger import read_logs, log_event
from rockfall.config import RESULTS_DIR, UPLOADS_DIR
from rockfall.alert_store import get_alert_store

# 摄像头实例池: 每个摄像头独立检测器 (各自的 MOG2/跟踪器状态), YOLO 模型共享
_detectors: dict[str, RockDetector] = {}
_detector_lock = threading.Lock()

# GPU 推理并发控制: 使用信号量替代全局锁，允许多路摄像头并发推理。
# 默认值为 2（与大多数消费级 GPU 的多流能力匹配）。
# 设为 1 等效于原全局锁行为；若模型不支持多流并发，自动降级。
_GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", "2"))
_inference_semaphore = threading.Semaphore(_GPU_CONCURRENCY)

# 异步任务管理 (视频检测)
_task_store: dict[str, dict] = {}
_task_lock = threading.Lock()
from rockfall.config import VIDEO_TASK_WORKERS
_task_executor = ThreadPoolExecutor(max_workers=VIDEO_TASK_WORKERS, thread_name_prefix="video-task")

# 活跃摄像头列表 (供看板展示)
_active_cameras: dict[str, dict] = {}
_active_cameras_lock = threading.Lock()


def _get_detector(camera_id: str = "default") -> RockDetector:
    """获取或创建指定摄像头的检测器实例（按点位自动选择模型）。"""
    if camera_id not in _detectors:
        with _detector_lock:
            if camera_id not in _detectors:
                # 解析 site_id: camera_id 通常就是 site_id
                _detectors[camera_id] = RockDetector(site_id=camera_id)
    return _detectors[camera_id]


def remove_detector(camera_id: str):
    """释放指定摄像头的检测器资源"""
    with _detector_lock:
        if camera_id in _detectors:
            del _detectors[camera_id]
    with _active_cameras_lock:
        _active_cameras.pop(camera_id, None)


# ============================================================
# 图片检测
# ============================================================

def detect_image_file(file=None, camera_id: str = "default") -> dict:
    """
    图片检测。

    file=None → 使用默认测试图片(兼容旧接口)
    file=UploadFile → 检测上传的图片
    """
    detector = _get_detector(camera_id)

    if file is None:
        # 兼容旧版: 使用 yolo/rock.jpg (如果存在)
        default_img = Path(__file__).resolve().parent.parent / "data" / "rock.jpg"
        path = str(default_img) if default_img.exists() else None
        if path is None:
            return {"error": "没有指定图片且默认图片不存在"}
    else:
        # 保存上传文件
        path = str(UPLOADS_DIR / file.filename)
        with open(path, "wb") as f:
            f.write(file.file.read())

    with _inference_semaphore:
        result = detector.detect_image(path, push_alert=True)

    # 清理临时文件
    if file is not None:
        try:
            Path(path).unlink()
        except Exception as e:
            from rockfall.logger import log_event
            log_event("system", msg=f"临时文件清理失败: {e}")

    return result


# ============================================================
# 视频检测
# ============================================================

def detect_video_file_async(file, save_frames: bool, push_alerts: bool,
                            camera_id: str = "default") -> str:
    """
    异步检测上传的视频文件, 返回 task_id。
    通过 GET /api/tasks/{task_id} 轮询结果。
    """
    path = str(UPLOADS_DIR / file.filename)
    with open(path, "wb") as f:
        f.write(file.file.read())

    task_id = str(uuid.uuid4())
    with _task_lock:
        _task_store[task_id] = {"status": "processing", "result": None,
                                "error": None, "created_at": time.time(),
                                "camera_id": camera_id}

    _task_executor.submit(_run_video_task, task_id, path, save_frames, push_alerts,
                          camera_id=camera_id, cleanup=path)
    return task_id


def detect_video_local_async(path: str, save_frames: bool, push_alerts: bool,
                             camera_id: str = "default") -> str:
    """异步检测服务器本地视频, 返回 task_id"""
    task_id = str(uuid.uuid4())
    with _task_lock:
        _task_store[task_id] = {"status": "processing", "result": None,
                                "error": None, "created_at": time.time(),
                                "camera_id": camera_id}

    _task_executor.submit(_run_video_task, task_id, path, save_frames, push_alerts,
                          camera_id=camera_id, cleanup=None)
    return task_id


def get_task_status(task_id: str) -> dict | None:
    """查询异步任务状态, 如果任务不存在则返回 None"""
    _cleanup_expired_tasks()
    with _task_lock:
        return _task_store.get(task_id)


def _run_video_task(task_id: str, path: str, save_frames: bool, push_alerts: bool,
                    camera_id: str = "default", cleanup: str | None = None):
    """后台执行视频检测并存储结果, 实时更新进度供 WebSocket 推送"""
    import cv2

    # ── 获取视频总帧数 (用于进度百分比) ──
    total_frames = 0
    try:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
    except Exception:
        total_frames = 0

    # ── 初始化进度 ──
    with _task_lock:
        if task_id in _task_store:
            _task_store[task_id]["progress"] = 0.0
            _task_store[task_id]["current_frame"] = 0
            _task_store[task_id]["total_frames"] = total_frames

    def _on_progress(current: int, total: int):
        """进度回调: 实时更新 task_store"""
        effective_total = total if total > 0 else total_frames
        pct = round(current / effective_total * 100, 1) if effective_total > 0 else 0
        with _task_lock:
            if task_id in _task_store:
                _task_store[task_id]["progress"] = min(pct, 99.9)
                _task_store[task_id]["current_frame"] = current
                _task_store[task_id]["total_frames"] = effective_total

    try:
        detector = _get_detector(camera_id)
        with _inference_semaphore:
            result = detector.detect_video(
                path, save_frames=save_frames, push_alerts=push_alerts,
                progress_callback=_on_progress,
            )
        with _task_lock:
            if task_id in _task_store:
                _task_store[task_id].update({
                    "status": "completed", "result": result,
                    "error": None, "progress": 100.0,
                })
            else:
                _task_store[task_id] = {
                    "status": "completed", "result": result,
                    "error": None, "created_at": time.time(),
                    "camera_id": camera_id, "progress": 100.0,
                }
    except Exception as e:
        log_event("system", level="ERROR", msg=f"异步视频检测失败 task={task_id}: {e}")
        # Sentry 错误上报（携带任务上下文）
        try:
            from rockfall.sentry_init import capture_exception
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("task_id", task_id)
                scope.set_tag("camera_id", camera_id)
                scope.set_extra("video_path", path)
                capture_exception(e)
        except Exception:
            pass  # Sentry 绝不影响业务逻辑
        with _task_lock:
            if task_id in _task_store:
                _task_store[task_id].update({
                    "status": "failed", "result": None,
                    "error": str(e),
                })
            else:
                _task_store[task_id] = {
                    "status": "failed", "result": None,
                    "error": str(e), "created_at": time.time(),
                    "camera_id": camera_id,
                }
    finally:
        if cleanup:
            try:
                Path(cleanup).unlink()
            except Exception:
                pass


def _cleanup_expired_tasks():
    """删除超过阈值的已完成/失败任务, 以及卡死的处理中任务, 防止内存泄漏"""
    import time
    from rockfall.config import TASK_CLEANUP_SECONDS, TASK_CLEANUP_STUCK_SECONDS
    now = time.time()
    with _task_lock:
        expired = [
            tid for tid, t in _task_store.items()
            if (
                t.get("status") in ("completed", "failed")
                and now - t.get("created_at", now) > TASK_CLEANUP_SECONDS
            ) or (
                t.get("status") == "processing"
                and now - t.get("created_at", now) > TASK_CLEANUP_STUCK_SECONDS
            )
        ]
        for tid in expired:
            del _task_store[tid]


# 同步版本保留 (供测试或短视频使用)
def detect_video_file(file, save_frames: bool, push_alerts: bool,
                      camera_id: str = "default") -> dict:
    """同步检测上传的视频文件 (仅适合短视频)"""
    detector = _get_detector(camera_id)

    path = str(UPLOADS_DIR / file.filename)
    with open(path, "wb") as f:
        f.write(file.file.read())

    with _inference_semaphore:
        result = detector.detect_video(path, save_frames=save_frames, push_alerts=push_alerts)

    try:
        Path(path).unlink()
    except Exception as e:
        log_event("system", msg=f"临时文件清理失败: {e}")

    return result


def detect_video_local(path: str, save_frames: bool, push_alerts: bool,
                       camera_id: str = "default") -> dict:
    """同步检测服务器本地视频 (仅适合短视频)"""
    detector = _get_detector(camera_id)
    with _inference_semaphore:
        return detector.detect_video(path, save_frames=save_frames, push_alerts=push_alerts)


# ============================================================
# 看板统计 (带内存缓存, 每60秒刷新)
# ============================================================

_stats_cache: dict | None = None
_stats_cache_time: float = 0


def get_dashboard_stats() -> dict:
    """从检测日志聚合看板统计 (60s 缓存)"""
    import time
    global _stats_cache, _stats_cache_time

    now = time.time()
    if _stats_cache is not None and (now - _stats_cache_time) < 60:
        return _stats_cache

    logs = read_logs(limit=500)

    today_str = date.today().strftime("%Y-%m-%d")

    today_total = 0
    today_red = 0
    today_orange = 0
    today_yellow = 0
    today_blue = 0
    last_count = None
    last_conf = None
    last_alert_level = None

    for entry in logs:
        entry_time = entry.get("time", "")
        is_today = entry_time.startswith(today_str)
        alert_level = entry.get("alert_level", "green")

        if is_today and entry.get("event") in ("detection",):
            today_total += 1
            if alert_level == "red":
                today_red += 1
            elif alert_level == "orange":
                today_orange += 1
            elif alert_level == "yellow":
                today_yellow += 1
            elif alert_level == "blue":
                today_blue += 1

        if entry.get("event") == "detection" and last_count is None:
            last_count = entry.get("count")
            last_conf = entry.get("max_confidence")
            last_alert_level = alert_level

    # 同时从 AlertStore 获取更准确的今日统计 (DB 比日志更可靠)
    try:
        store = get_alert_store()
        db_counts = store.count_today_by_level()
        today_red = today_red or db_counts.get("red", 0)
        today_orange = today_orange or db_counts.get("orange", 0)
        today_yellow = today_yellow or db_counts.get("yellow", 0)
        today_blue = today_blue or db_counts.get("blue", 0)
    except Exception:
        pass

    _stats_cache = {
        "today_total": today_total,
        "today_red": today_red,
        "today_orange": today_orange,
        "today_yellow": today_yellow,
        "today_blue": today_blue,
        "last_count": last_count,
        "last_conf": last_conf,
        "last_alert_level": last_alert_level,
    }
    _stats_cache_time = now
    return _stats_cache


def get_recent_alerts(limit: int = 20) -> list[dict]:
    """获取最近预警记录 (从 AlertStore 读取, MySQL/SQLite 自适应)"""
    store = get_alert_store()
    rows = store.get_recent(limit)
    return _rows_to_alert_dicts(rows)


def query_alerts_page(
    limit: int = 20,
    offset: int = 0,
    start_date: str = "",
    end_date: str = "",
    alert_level: str = "",
) -> dict:
    """分页查询预警记录, 支持日期+等级筛选。返回 {total, rows}。"""
    store = get_alert_store()
    total = store.count_alerts(
        start_date=start_date, end_date=end_date, alert_level=alert_level,
    )
    rows = store.query_alerts(
        start_date=start_date, end_date=end_date, alert_level=alert_level,
        limit=limit, offset=offset,
    )
    return {"total": total, "rows": _rows_to_alert_dicts(rows)}


def _rows_to_alert_dicts(rows: list) -> list[dict]:
    """将 AlertStore 原始行转为前端友好的 dict 列表"""
    import json as _json
    alerts = []
    for r in rows:
        track_ids = r.get("track_ids", "[]")
        if isinstance(track_ids, str):
            try:
                track_ids = _json.loads(track_ids)
            except (_json.JSONDecodeError, TypeError):
                track_ids = []
        push_status = r.get("push_status", "pending")
        alerts.append({
            "id": r.get("id", 0),
            "time": r.get("time", ""),
            "alert_level": r.get("alert_level", "green"),
            "count": r.get("count", 0),
            "max_confidence": r.get("max_confidence", 0),
            "track_ids": track_ids,
            "class_summary": r.get("class_summary", ""),
            "saved_frame": r.get("saved_frame", ""),
            "push_status": push_status,
            "rock_diameter_cm": r.get("rock_diameter_cm", 0),
            "monitoring_location": r.get("monitoring_location", ""),
            "review_status": r.get("review_status", ""),
            "reviewer_note": r.get("reviewer_note", ""),
        })
    return alerts


# ============================================================
# 多点位管理
# ============================================================

def get_sites_data() -> dict:
    """获取全部点位 + 当前激活点位信息 (管理页含停用点位)"""
    from rockfall.site_config import list_all_sites_admin, get_active_site, get_site_state
    sites = list_all_sites_admin()
    active = get_active_site()
    return {
        "sites": [s.to_dict() for s in sites],
        "active_site_id": active.site_id,
        "active_site": active.to_dict(),
    }


def switch_active_site(site_id: str) -> dict:
    """切换激活的监测点位, 返回新点位信息"""
    from rockfall.site_config import set_active_site
    try:
        site = set_active_site(site_id)
        return {"status": "ok", "active_site": site.to_dict()}
    except ValueError as e:
        raise ValueError(str(e))


def create_site(site_data: dict) -> dict:
    """新增监测点位"""
    from rockfall.site_config import MonitoringSite, get_site_store, get_site_by_id

    site_id = site_data.get("site_id", "").strip()
    if not site_id:
        raise ValueError("site_id 不能为空")

    # 检查重复
    existing = get_site_by_id(site_id)
    if existing is not None:
        raise ValueError(f"点位ID已存在: {site_id}")

    site = MonitoringSite(
        site_id=site_id,
        name=site_data.get("name", "").strip(),
        location=site_data.get("location", site_data.get("name", "")).strip(),
        region=site_data.get("region", "").strip(),
        camera_url=site_data.get("camera_url", "").strip(),
        description=site_data.get("description", "").strip(),
        latitude=float(site_data.get("latitude", 0)),
        longitude=float(site_data.get("longitude", 0)),
        highway=site_data.get("highway", "").strip(),
        stake_mark=site_data.get("stake_mark", "").strip(),
        risk_level=site_data.get("risk_level", "medium").strip(),
        roi_polygon=site_data.get("roi_polygon"),
        alert_contacts=site_data.get("alert_contacts"),
        is_active=site_data.get("is_active", True),
        model_override=site_data.get("model_override", "").strip(),
    )

    store = get_site_store()
    if not store.insert(site):
        raise ValueError(f"写入数据库失败: {site_id}")

    return {"status": "ok", "site": site.to_dict()}


def update_site(site_id: str, site_data: dict) -> dict:
    """更新已有监测点位"""
    from rockfall.site_config import MonitoringSite, get_site_store, get_site_by_id

    existing = get_site_by_id(site_id)
    if existing is None:
        raise ValueError(f"点位不存在: {site_id}")

    # 合并: 用新值覆盖旧值（保留未传入的字段）
    merged = existing.to_dict()
    for key in ("name", "location", "region", "camera_url", "description",
                "highway", "stake_mark", "risk_level", "model_override"):
        if key in site_data and site_data[key] is not None:
            merged[key] = site_data[key]
    for key in ("latitude", "longitude"):
        if key in site_data and site_data[key] is not None:
            merged[key] = float(site_data[key])
    for key in ("roi_polygon", "alert_contacts"):
        if key in site_data and site_data[key] is not None:
            merged[key] = site_data[key]
    if "is_active" in site_data:
        merged["is_active"] = bool(site_data["is_active"])

    site = MonitoringSite.from_dict(merged)
    store = get_site_store()
    if not store.update(site):
        raise ValueError(f"更新数据库失败: {site_id}")

    return {"status": "ok", "site": site.to_dict()}


def delete_site(site_id: str) -> dict:
    """删除监测点位"""
    from rockfall.site_config import get_site_store, get_active_site

    # 不允许删除当前激活的点位
    active = get_active_site()
    if active.site_id == site_id:
        raise ValueError(f"不能删除当前激活的点位，请先切换到其他点位")

    store = get_site_store()
    if not store.delete(site_id):
        raise ValueError(f"删除失败: {site_id}")

    return {"status": "ok", "deleted": site_id}


# ============================================================
# ROI 多边形管理
# ============================================================

def get_roi_for_site(site_id: str | None = None) -> dict:
    """
    获取指定站点的 ROI 多边形坐标。

    返回: {"site_id": "...", "roi_polygon": [[x,y], ...], "frame_size": [w, h]}
    如果未指定 site_id, 使用当前激活站点。
    """
    from rockfall.site_config import get_active_site, get_site_by_id, get_site_store

    if site_id:
        site = get_site_by_id(site_id)
    else:
        site = get_active_site()

    if site is None:
        raise ValueError("未找到指定站点")

    polygon = site.roi_polygon or []

    # 尝试从已运行的检测器获取实际分辨率 (免去加载 YOLO 模型的开销)
    frame_w, frame_h = 1280, 720
    try:
        from rockfall.config import DETECTION_IMG_SIZE
        # DETECTION_IMG_SIZE 是推理尺寸，实际视频流通常为 1280x720
        # 保持 16:9 比例以适配多数摄像头
    except Exception:
        pass

    return {
        "site_id": site.site_id,
        "roi_polygon": polygon,
        "frame_size": [frame_w, frame_h],
    }


def save_roi_for_site(site_id: str, polygon: list) -> dict:
    """
    保存 ROI 多边形坐标到指定站点, 并触发 MOG2 背景模型重建。

    polygon: [[x1, y1], [x2, y2], ...]  至少 3 个点。
    返回: {"status": "ok", "site_id": "...", "vertices": N}
    """
    from rockfall.site_config import get_site_store, get_site_by_id
    from rockfall.logger import log_event

    if len(polygon) < 3:
        raise ValueError("ROI 多边形至少需要 3 个顶点")

    store = get_site_store()
    site = get_site_by_id(site_id)
    if site is None:
        raise ValueError(f"站点不存在: {site_id}")

    # 更新站点 ROI
    site.roi_polygon = polygon
    store.update(site)

    # 尝试重建活跃检测器的 MOG2 背景模型
    try:
        _rebuild_mog2_for_site(site_id, polygon)
    except Exception as e:
        log_event("system", level="WARN",
                  msg=f"ROI 已保存但 MOG2 重建失败: {e}")

    log_event("system", msg=f"ROI 已更新: site={site_id}, vertices={len(polygon)}")
    return {"status": "ok", "site_id": site_id, "vertices": len(polygon)}


def _rebuild_mog2_for_site(site_id: str, polygon: list):
    """重建指定站点的活跃检测器背景模型。使用检测器实际分辨率。"""
    import numpy as np
    import cv2 as _cv2
    poly_arr = np.array(polygon, dtype=np.int32)

    for key in (site_id, "default"):
        if key in _detectors:
            detector = _detectors[key]
            # 从检测器现有状态获取实际分辨率 (避免硬编码)
            fw = getattr(detector, 'frame_w', 1280) or 1280
            fh = getattr(detector, 'frame_h', 720) or 720
            roi_mask = np.zeros((fh, fw), dtype=np.uint8)
            _cv2.fillPoly(roi_mask, [poly_arr], 255)
            detector.init_stream_state(fw, fh, roi_mask)


def get_roi_heatmap(site_id: str | None = None, frame_source: str = "") -> dict:
    """
    生成 ROI 热力图 overlay — 半透明 mask 帮助用户判断最佳 ROI 区域。

    优先使用 FastSAM 道路/边坡分割；不可用时回退到梯度热力图
    (上半红色=边坡候选区 / 下半蓝色=道路候选区)。

    返回: {"base64": "data:image/png;base64,...", "width": W, "height": H}
    """
    import base64
    import cv2
    import numpy as np
    from pathlib import Path

    width, height = 1280, 720
    heatmap_b64 = ""

    # ── 获取参考图像 ──
    frame = None
    if frame_source and Path(frame_source).exists():
        frame = cv2.imread(frame_source)
    else:
        try:
            from rockfall.detector import get_latest_frame
            buf = get_latest_frame(site_id or "default")
            if buf:
                frame = cv2.imdecode(
                    np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR,
                )
        except Exception:
            pass

    if frame is not None:
        h, w = frame.shape[:2]
        width, height = w, h

    # ── 尝试 FastSAM 分割 ──
    try:
        from rockfall.fastsam_road import auto_segment_from_cap
        if frame is not None:
            temp_path = UPLOADS_DIR / "_roi_heatmap_frame.jpg"
            cv2.imwrite(str(temp_path), frame)
            cap = cv2.VideoCapture(str(temp_path))
            if cap.isOpened():
                road_mask, slope_mask = auto_segment_from_cap(cap, w, h, sample_num=3)
                cap.release()
                if road_mask is not None and slope_mask is not None:
                    overlay = np.zeros((height, width, 4), dtype=np.uint8)
                    overlay[road_mask > 0, 2] = 255
                    overlay[road_mask > 0, 3] = 80
                    overlay[slope_mask > 0, 2] = 255
                    overlay[slope_mask > 0, 0] = 255
                    overlay[slope_mask > 0, 3] = 100
                    _, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
                    heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf).decode()
    except Exception:
        pass

    # ── 回退: 简单渐变热力图 ──
    if not heatmap_b64:
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        mid_y = int(height * 0.6)
        overlay[:mid_y, :, 0] = 255
        overlay[:mid_y, :, 3] = 60
        overlay[mid_y:, :, 2] = 255
        overlay[mid_y:, :, 3] = 80
        _, buf = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf).decode()

    return {"base64": heatmap_b64, "width": width, "height": height}

# 可热更新的参数白名单 (全参数, 分为"实时生效"和"重启流生效"两类)
_HOT_UPDATE_KEYS = {
    # ---- 实时生效 (每帧 RuntimeConfig 读取) ----
    "detection_confidence", "detection_img_size", "motion_min_area",
    "alert_blue_high", "alert_yellow_high", "alert_orange_high",
    "skip_idle", "skip_active", "skip_critical",
    # ---- 下次 init_stream_state 生效 (写入 RuntimeConfig, 需触发流重启) ----
    "mog2_history", "mog2_var_threshold", "mog2_learning_rate",
    "mog2_morph_kernel", "mog2_reset_idle_frames",
    "edge_enhance_alpha", "edge_enhance_interval",
    "fusion_motion_weight",
    "light_change_threshold", "light_change_lr_factor",
    "tfd_iou_threshold", "tfd_threshold",
    "temporal_window", "temporal_iou",
}

_HOT_UPDATE_ATTR_MAP = {
    "detection_confidence": "confidence",
    "detection_img_size": "img_size",
    "motion_min_area": "min_area",
    "alert_blue_high": "alert_blue_conf_high",
    "alert_yellow_high": "alert_yellow_conf_high",
    "alert_orange_high": "alert_orange_conf_high",
}

# 写入 RuntimeConfig 的键映射 (大写 key → 运行时覆盖)
_RC_KEY_MAP = {
    "detection_confidence": "DETECTION_CONFIDENCE",
    "detection_img_size": "DETECTION_IMG_SIZE",
    "motion_min_area": "MOTION_MIN_AREA",
    "alert_blue_high": "ALERT_BLUE_CONFIDENCE_HIGH",
    "alert_yellow_high": "ALERT_YELLOW_CONFIDENCE_HIGH",
    "alert_orange_high": "ALERT_ORANGE_CONFIDENCE_HIGH",
    "skip_idle": "SKIP_IDLE",
    "skip_active": "SKIP_ACTIVE",
    "skip_critical": "SKIP_CRITICAL",
    "mog2_history": "MOG2_HISTORY",
    "mog2_var_threshold": "MOG2_VAR_THRESHOLD",
    "mog2_learning_rate": "MOG2_LEARNING_RATE",
    "mog2_morph_kernel": "MOG2_MORPH_KERNEL",
    "mog2_reset_idle_frames": "MOG2_RESET_IDLE_FRAMES",
    "edge_enhance_alpha": "EDGE_ENHANCE_ALPHA",
    "edge_enhance_interval": "EDGE_ENHANCE_INTERVAL",
    "fusion_motion_weight": "FUSION_MOTION_WEIGHT",
    "light_change_threshold": "LIGHT_CHANGE_THRESHOLD",
    "light_change_lr_factor": "LIGHT_CHANGE_LR_FACTOR",
    "tfd_iou_threshold": "TFD_IOU_THRESHOLD",
    "tfd_threshold": "TFD_THRESHOLD",
    "temporal_window": "TEMPORAL_WINDOW",
    "temporal_iou": "TEMPORAL_IOU",
}

# 需要流重启才能生效的参数 (MOG2/滤波器参数重建)
_STREAM_RESTART_REQUIRED = {
    "mog2_history", "mog2_var_threshold", "mog2_learning_rate",
    "mog2_morph_kernel", "mog2_reset_idle_frames",
    "edge_enhance_alpha", "edge_enhance_interval",
    "fusion_motion_weight",
    "light_change_threshold", "light_change_lr_factor",
    "tfd_iou_threshold", "tfd_threshold",
    "temporal_window", "temporal_iou",
}


def update_runtime_config(updates: dict) -> dict:
    """
    热更新所有运行中检测器的参数 (当前会话有效, 重启后恢复 .env 默认值)。

    参数:
        updates: {"detection_confidence": 0.5, "skip_idle": 8, ...}

    返回:
        {"applied": {...}, "skipped": {...}, "stream_restart_needed": [...]}
    """
    from rockfall.config import RuntimeConfig

    applied = {}
    skipped = {}
    stream_restart_needed = []

    for key, value in updates.items():
        if key not in _HOT_UPDATE_KEYS:
            skipped[key] = f"不支持热更新 (白名单: {sorted(_HOT_UPDATE_KEYS)})"
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            skipped[key] = f"值必须是数字: {value}"
            continue

        # 写入 RuntimeConfig 单例 (所有检测器每帧读取)
        rc_key = _RC_KEY_MAP.get(key, key.upper())
        RuntimeConfig.set(rc_key, value)

        # 对于实时生效的参数, 同时更新活跃检测器实例属性
        if key not in _STREAM_RESTART_REQUIRED:
            attr = _HOT_UPDATE_ATTR_MAP.get(key, key)
            with _detector_lock:
                for cam_id, det in _detectors.items():
                    if hasattr(det, attr):
                        setattr(det, attr, value)
        else:
            stream_restart_needed.append(key)

        # 更新 config 模块常量 (新创建的检测器也会使用新值)
        _update_config_module(key, value)

        applied[key] = value

    result = {"applied": applied, "skipped": skipped}
    if stream_restart_needed:
        result["stream_restart_needed"] = stream_restart_needed
        result["hint"] = "部分参数需要重启视频流后生效 (或等待自动重连)"
    return result


def _update_config_module(key: str, value: float):
    """更新 rockfall.config 模块级常量 (影响后续新创建的检测器)"""
    import rockfall.config as cfg

    _cfg_map = {
        "detection_confidence": "DETECTION_CONFIDENCE",
        "detection_img_size": ("DETECTION_IMG_SIZE", int),
        "motion_min_area": ("MOTION_MIN_AREA", int),
        "alert_blue_high": "ALERT_BLUE_CONFIDENCE_HIGH",
        "alert_yellow_high": "ALERT_YELLOW_CONFIDENCE_HIGH",
        "alert_orange_high": "ALERT_ORANGE_CONFIDENCE_HIGH",
        "skip_idle": ("SKIP_IDLE", int),
        "skip_active": ("SKIP_ACTIVE", int),
        "skip_critical": ("SKIP_CRITICAL", int),
        "mog2_history": ("MOG2_HISTORY", int),
        "mog2_var_threshold": ("MOG2_VAR_THRESHOLD", int),
        "mog2_learning_rate": "MOG2_LEARNING_RATE",
        "mog2_morph_kernel": ("MOG2_MORPH_KERNEL", int),
        "mog2_reset_idle_frames": ("MOG2_RESET_IDLE_FRAMES", int),
        "edge_enhance_alpha": "EDGE_ENHANCE_ALPHA",
        "edge_enhance_interval": ("EDGE_ENHANCE_INTERVAL", int),
        "fusion_motion_weight": "FUSION_MOTION_WEIGHT",
        "light_change_threshold": "LIGHT_CHANGE_THRESHOLD",
        "light_change_lr_factor": "LIGHT_CHANGE_LR_FACTOR",
        "tfd_iou_threshold": "TFD_IOU_THRESHOLD",
        "tfd_threshold": ("TFD_THRESHOLD", int),
        "temporal_window": ("TEMPORAL_WINDOW", int),
        "temporal_iou": "TEMPORAL_IOU",
    }

    target = _cfg_map.get(key)
    if target is None:
        return

    if isinstance(target, tuple):
        attr_name, cast = target
        setattr(cfg, attr_name, cast(value))
    else:
        setattr(cfg, target, value)


def get_runtime_config() -> dict:
    """获取当前运行中的核心配置 (供前端参数设置面板展示)"""
    from rockfall.config import RuntimeConfig

    overrides = RuntimeConfig.get_all_overrides()

    # 基础参数: 优先从活跃检测器读取实际值, 否则从 config 模块读取
    base = {}
    try:
        with _detector_lock:
            if _detectors:
                det = next(iter(_detectors.values()))
                base = {
                    "detection_confidence": det.confidence,
                    "detection_img_size": det.img_size,
                    "motion_min_area": det.min_area,
                    "alert_blue_high": det.alert_blue_conf_high,
                    "alert_yellow_high": det.alert_yellow_conf_high,
                    "alert_orange_high": det.alert_orange_conf_high,
                }
    except Exception:
        pass

    import rockfall.config as cfg
    return {
        "detection_confidence": base.get("detection_confidence", cfg.DETECTION_CONFIDENCE),
        "detection_img_size": base.get("detection_img_size", cfg.DETECTION_IMG_SIZE),
        "motion_min_area": base.get("motion_min_area", cfg.MOTION_MIN_AREA),
        "alert_blue_high": base.get("alert_blue_high", cfg.ALERT_BLUE_CONFIDENCE_HIGH),
        "alert_yellow_high": base.get("alert_yellow_high", cfg.ALERT_YELLOW_CONFIDENCE_HIGH),
        "alert_orange_high": base.get("alert_orange_high", cfg.ALERT_ORANGE_CONFIDENCE_HIGH),
        "skip_idle": RuntimeConfig.get("SKIP_IDLE", cfg.SKIP_IDLE),
        "skip_active": RuntimeConfig.get("SKIP_ACTIVE", cfg.SKIP_ACTIVE),
        "skip_critical": RuntimeConfig.get("SKIP_CRITICAL", cfg.SKIP_CRITICAL),
        "mog2_history": RuntimeConfig.get("MOG2_HISTORY", cfg.MOG2_HISTORY),
        "mog2_var_threshold": RuntimeConfig.get("MOG2_VAR_THRESHOLD", cfg.MOG2_VAR_THRESHOLD),
        "mog2_learning_rate": RuntimeConfig.get("MOG2_LEARNING_RATE", cfg.MOG2_LEARNING_RATE),
        "overrides_count": len(overrides),
    }


# ============================================================
# 归档导出 (应急管理部门合规要求)
# ============================================================

def export_alerts_excel(
    start_date: str = "",
    end_date: str = "",
    alert_level: str = "",
    location: str = "",
) -> bytes:
    """
    查询预警记录并导出为格式化 Excel (.xlsx) 字节。

    参数:
        start_date:  起始日期 "2026-06-01"
        end_date:    结束日期 "2026-06-12"
        alert_level: 预警等级筛选 (空=全部)
        location:    监测点位 (当前仅支持单点位, 保留扩展)

    返回:
        .xlsx 文件字节, 可直接作为 HTTP 响应体
    """
    from rockfall.utils import export_alerts_to_excel
    from rockfall.config import get_location

    store = get_alert_store()

    # 查询数据
    rows = store.query_alerts(
        start_date=start_date,
        end_date=end_date,
        alert_level=alert_level,
        limit=100000,
    )

    # 构建标题
    loc = location or get_location() or "监测点"
    date_range = ""
    if start_date and end_date:
        date_range = f" ({start_date} ~ {end_date})"
    elif start_date:
        date_range = f" (自 {start_date})"
    elif end_date:
        date_range = f" (至 {end_date})"
    title = f"落石监测预警记录 — {loc}{date_range}"

    return export_alerts_to_excel(list(rows), sheet_title=title)


def get_export_summary(
    start_date: str = "",
    end_date: str = "",
    alert_level: str = "",
) -> dict:
    """获取导出预览摘要 (记录数 + 等级分布)"""
    store = get_alert_store()
    total = store.count_alerts(
        start_date=start_date, end_date=end_date, alert_level=alert_level,
    )
    # 各等级分别统计
    by_level = {}
    for lv in ["red", "orange", "yellow", "blue"]:
        cnt = store.count_alerts(
            start_date=start_date, end_date=end_date, alert_level=lv,
        )
        if cnt > 0:
            by_level[lv] = cnt
    return {"total": total, "by_level": by_level}


# ============================================================
# 预警审核 (误报标记)
# ============================================================

def mark_alert_review(alert_id: int, review_status: str, note: str = "") -> dict:
    """
    标记预警的审核状态。

    review_status: 'confirmed' (确认真实) | 'false_alarm' (确认误报) | '' (清除标记)
    """
    store = get_alert_store()
    ok = store.mark_review(alert_id, review_status, note)
    if ok:
        return {"status": "ok", "alert_id": alert_id, "review_status": review_status}
    return {"status": "error", "msg": "更新失败"}


# ============================================================
# 预警统计看板
# ============================================================

def get_alert_statistics(days: int = 7) -> dict:
    """
    聚合预警统计数据: 今日摘要 + 每日趋势 + 等级分布 + 误报率。

    返回:
        {
            "today": {"red": N, "orange": N, "yellow": N, "blue": N, "total": N},
            "daily_trends": [{date, red, orange, yellow, blue, total}, ...],
            "level_distribution": {"red": N, "orange": N, "yellow": N, "blue": N},
            "false_alarm": {"total_reviewed": N, "confirmed": N, "false_alarm": N,
                            "false_alarm_rate": 0.XX, "pending_review": N},
            "grand_total": N
        }
    """
    store = get_alert_store()

    # 今日统计
    today = store.count_today_by_level()
    today["total"] = sum(today.values())

    # 每日趋势
    daily_trends = store.get_daily_trends(days=days)

    # 总等级分布 (近30天)
    from datetime import datetime as dt, timedelta
    start30 = (dt.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    distribution = {}
    grand_total = 0
    for lv in ["red", "orange", "yellow", "blue"]:
        cnt = store.count_alerts(start_date=start30, alert_level=lv)
        distribution[lv] = cnt
        grand_total += cnt

    # 误报率
    false_alarm = store.get_false_alarm_stats(days=30)

    return {
        "today": today,
        "daily_trends": daily_trends,
        "level_distribution": distribution,
        "false_alarm": false_alarm,
        "grand_total": grand_total,
    }


# ============================================================
# 报警截图
# ============================================================

def get_alert_image_info(alert_id: int) -> dict | None:
    """
    查询单条预警记录的截图信息。

    返回: {"alert_id": N, "saved_frame": "path/to/img.jpg", "exists": bool, ...}
    不存在时返回 None。
    """
    store = get_alert_store()
    rows = store.query_alerts(limit=1, offset=0)
    # 用原生 SQL 查单条
    if store._backend == "mysql":
        rows = store._mysql_query(
            "SELECT * FROM alerts WHERE id=%s", (alert_id,)
        )
    else:
        rows = store._sqlite_query(
            "SELECT * FROM alerts WHERE id=?", (alert_id,)
        )

    if not rows:
        return None

    r = rows[0]
    saved_frame = r.get("saved_frame", "") or ""
    exists = bool(saved_frame) and Path(saved_frame).exists()

    # 也尝试从 RESULTS_DIR 查找 stream_ 帧
    fallback_path = ""
    if not exists and saved_frame:
        # 尝试从文件名推断
        import re
        match = re.search(r'stream_(\d+)', saved_frame)
        if match:
            from rockfall.config import RESULTS_DIR
            fb = RESULTS_DIR / f"stream_{int(match.group(1)):06d}.jpg"
            if fb.exists():
                fallback_path = str(fb)
                exists = True

    return {
        "alert_id": alert_id,
        "saved_frame": saved_frame,
        "fallback_path": fallback_path,
        "exists": exists,
        "display_path": fallback_path or saved_frame if exists else "",
        "time": r.get("time", ""),
        "alert_level": r.get("alert_level", ""),
        "count": r.get("count", 0),
        "max_confidence": r.get("max_confidence", 0),
    }


# ============================================================
# 地图可视化 — 带经纬度的预警数据
# ============================================================

def get_geo_alerts(days: int = 30, alert_level: str = "") -> list[dict]:
    """
    查询预警记录并关联站点经纬度，返回适合地图渲染的数据。

    匹配逻辑: alerts.monitoring_location ↔ sites.location (字符串匹配)

    返回: [
      {
        "id": 1, "time": "...", "alert_level": "orange",
        "count": 3, "max_confidence": 0.85,
        "class_summary": "落石:3", "saved_frame": "...",
        "site_id": "nanning_naan_s1",  "site_name": "南宁...",
        "latitude": 22.817, "longitude": 108.366,
      }, ...
    ]
    """
    from datetime import datetime, timedelta
    from rockfall.site_config import get_site_store

    store = get_alert_store()
    site_store = get_site_store()

    # ── 获取所有站点 (构建 location→site 映射) ──
    sites = site_store.list_all()
    loc_to_site: dict[str, dict] = {}
    for s in sites:
        if s.location:
            loc_to_site[s.location] = {
                "site_id": s.site_id,
                "name": s.name,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "risk_level": s.risk_level,
                "highway": s.highway,
                "stake_mark": s.stake_mark,
            }

    # ── 查询预警 (SQLite: 时间过滤) ──
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cols = "id, time, alert_level, count, max_confidence, class_summary, saved_frame, monitoring_location"

    if store._backend == "mysql":
        sql = f"SELECT {cols} FROM alerts WHERE time >= %s"
        params = (cutoff,)
        if alert_level:
            sql += " AND alert_level = %s"
            params = (cutoff, alert_level)
        sql += " ORDER BY time DESC"
        rows = store._mysql_query(sql, params)
    else:
        sql = f"SELECT {cols} FROM alerts WHERE time >= ?"
        params = (cutoff,)
        if alert_level:
            sql += " AND alert_level = ?"
            params = (cutoff, alert_level)
        sql += " ORDER BY time DESC LIMIT 2000"
        rows = store._sqlite_query(sql, params)

    # ── 关联站点坐标 ──
    result = []
    for r in rows:
        loc = r.get("monitoring_location", "") or ""
        site = loc_to_site.get(loc)
        item = {
            "id": r["id"],
            "time": r.get("time", ""),
            "alert_level": r.get("alert_level", ""),
            "count": r.get("count", 0),
            "max_confidence": r.get("max_confidence", 0),
            "class_summary": r.get("class_summary", ""),
            "saved_frame": r.get("saved_frame", ""),
            "site_id": site["site_id"] if site else "",
            "site_name": site["name"] if site else loc,
            "latitude": site["latitude"] if site else 0,
            "longitude": site["longitude"] if site else 0,
        }
        result.append(item)

    return result
