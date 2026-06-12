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
_inference_lock = threading.Lock()  # 串行化 GPU 推理 (所有摄像头共用)

# 异步任务管理 (视频检测)
_task_store: dict[str, dict] = {}
_task_lock = threading.Lock()
from rockfall.config import VIDEO_TASK_WORKERS
_task_executor = ThreadPoolExecutor(max_workers=VIDEO_TASK_WORKERS, thread_name_prefix="video-task")

# 活跃摄像头列表 (供看板展示)
_active_cameras: dict[str, dict] = {}
_active_cameras_lock = threading.Lock()


def _get_detector(camera_id: str = "default") -> RockDetector:
    """获取或创建指定摄像头的检测器实例"""
    if camera_id not in _detectors:
        with _detector_lock:
            if camera_id not in _detectors:
                _detectors[camera_id] = RockDetector()
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

    with _inference_lock:
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
    """后台执行视频检测并存储结果"""
    try:
        detector = _get_detector(camera_id)
        with _inference_lock:
            result = detector.detect_video(path, save_frames=save_frames, push_alerts=push_alerts)
        with _task_lock:
            _task_store[task_id] = {"status": "completed", "result": result,
                                    "error": None, "created_at": time.time(),
                                    "camera_id": camera_id}
    except Exception as e:
        log_event("system", level="ERROR", msg=f"异步视频检测失败 task={task_id}: {e}")
        with _task_lock:
            _task_store[task_id] = {"status": "failed", "result": None,
                                    "error": str(e), "created_at": time.time(),
                                    "camera_id": camera_id}
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

    with _inference_lock:
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
    with _inference_lock:
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
    """获取全部点位 + 当前激活点位信息"""
    from rockfall.site_config import list_sites, get_active_site, get_site_state
    sites = list_sites()
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


# ============================================================
# 运行时参数热更新
# ============================================================

# 可热更新的检测器参数白名单
_HOT_UPDATE_KEYS = {
    "detection_confidence", "detection_img_size", "motion_min_area",
    "alert_blue_high", "alert_yellow_high", "alert_orange_high",
}

_HOT_UPDATE_ATTR_MAP = {
    "detection_confidence": "confidence",
    "detection_img_size": "img_size",
    "motion_min_area": "min_area",
    "alert_blue_high": "alert_blue_conf_high",
    "alert_yellow_high": "alert_yellow_conf_high",
    "alert_orange_high": "alert_orange_conf_high",
}


def update_runtime_config(updates: dict) -> dict:
    """
    热更新所有运行中检测器的参数 (当前会话有效, 重启后恢复 .env 默认值)。

    参数:
        updates: {"detection_confidence": 0.5, "alert_blue_high": 0.55, ...}

    返回:
        {"applied": {...}, "skipped": {...}}
    """
    applied = {}
    skipped = {}

    for key, value in updates.items():
        if key not in _HOT_UPDATE_KEYS:
            skipped[key] = f"不支持热更新 (白名单: {sorted(_HOT_UPDATE_KEYS)})"
            continue

        attr = _HOT_UPDATE_ATTR_MAP.get(key, key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            skipped[key] = f"值必须是数字: {value}"
            continue

        # 更新所有摄像头检测器实例
        with _detector_lock:
            for cam_id, det in _detectors.items():
                if hasattr(det, attr):
                    setattr(det, attr, value)

        # 同时更新 config 模块常量 (新创建的检测器也会使用新值)
        _update_config_module(key, value)

        applied[key] = value

    return {"applied": applied, "skipped": skipped}


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
    try:
        # 从第一个活跃检测器读取实际值, 否则从 config 模块读取
        with _detector_lock:
            if _detectors:
                det = next(iter(_detectors.values()))
                return {
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
        "detection_confidence": cfg.DETECTION_CONFIDENCE,
        "detection_img_size": cfg.DETECTION_IMG_SIZE,
        "motion_min_area": cfg.MOTION_MIN_AREA,
        "alert_blue_high": cfg.ALERT_BLUE_CONFIDENCE_HIGH,
        "alert_yellow_high": cfg.ALERT_YELLOW_CONFIDENCE_HIGH,
        "alert_orange_high": cfg.ALERT_ORANGE_CONFIDENCE_HIGH,
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
