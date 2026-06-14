"""
rockfall — 落石检测核心库
===========================
本包是桌面应用(desktop/)和Web服务(server/)的共享基础层。

模块分层:
  config.py          — 配置层: 从 .env 读取所有参数
  detector.py        — 算法层: MOG2运动检测 + YOLO目标检测 + SORT跟踪流水线
  tracker.py         — 跟踪层: Kalman+IoU 多目标跟踪 (SORT算法)
  edge_enhance.py    — 预处理层: Sobel边缘增强
  motion_detect.py   — 预处理层: 三帧差分 & MOG2中心点运动滤波
  sahi.py            — 推理层: SAHI切片辅助推理
  fusion.py          — 后处理层: 概率融合 & 多帧时序确认
  notifier.py        — 通知层: PushPlus微信推送 (含连续帧确认、base64图片)
  alert_store.py     — 存储层: 预警记录持久化 (MySQL/SQLite)
  logger.py          — 日志层: 检测事件持久化为JSONL文件
  site_config.py     — 配置层: 多监测点位管理
  fastsam_road.py    — 分割层: FastSAM道路/边坡分割 (替代原SAM独立进程)
  road_segmentation.py   — 分割层: 传统CV道路-边坡分割
  road_detector.py   — ROI生成: 纯CV边坡区域检测
  road_refine.py     — 分割层: 轻量颜色+纹理道路分割
  scene_filter.py    — 滤波层: 场景干扰抑制 (天空/车辆/阴影)
  roi_confidence.py  — 评估层: ROI质量评估与自适应降级
  utils.py           — 工具层: 公共函数 (IoU计算, Excel导出)
"""

__version__ = "2.2.0"

# ══════════════════════════════════════════════════════════════
# 延迟导入 — 按需加载以避免一次性加载 torch/cv2/ultralytics
# ══════════════════════════════════════════════════════════════

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # ── 检测流水线 (heavy: torch + ultralytics) ──
    "RockDetector":  ("rockfall.detector", "RockDetector"),
    "AlertContext":  ("rockfall.detector", "AlertContext"),
    # ── 多目标跟踪 (heavy: cv2) ──
    "RockTracker":       ("rockfall.tracker", "RockTracker"),
    "KalmanBoxTracker":  ("rockfall.tracker", "KalmanBoxTracker"),
    # ── 预处理 (heavy: cv2) ──
    "EdgeEnhancer":                ("rockfall.edge_enhance", "EdgeEnhancer"),
    "sobel_edge_enhance":          ("rockfall.edge_enhance", "sobel_edge_enhance"),
    "ThreeFrameDiff":              ("rockfall.motion_detect", "ThreeFrameDiff"),
    "filter_detections_by_motion": ("rockfall.motion_detect", "filter_detections_by_motion"),
    "filter_detections_by_mog2_center": ("rockfall.motion_detect", "filter_detections_by_mog2_center"),
    # ── 推理加速 (heavy: torch) ──
    "SAHISlicer":     ("rockfall.sahi", "SAHISlicer"),
    "sahi_inference": ("rockfall.sahi", "sahi_inference"),
    "fuse_confidence": ("rockfall.fusion", "fuse_confidence"),
    "TemporalFilter":  ("rockfall.fusion", "TemporalFilter"),
    # ── 通知层 (light: requests) ──
    "send_alert":          ("rockfall.notifier", "send_alert"),
    "send_alert_async":    ("rockfall.notifier", "send_alert_async"),
    "dispatch_alert_async": ("rockfall.notifier", "dispatch_alert_async"),
    # ── 持久化 (light: sqlite3 stdlib) ──
    "AlertStore":     ("rockfall.alert_store", "AlertStore"),
    "get_alert_store": ("rockfall.alert_store", "get_alert_store"),
    # ── 日志层 (light: stdlib) ──
    "log_event": ("rockfall.logger", "log_event"),
    "flush":     ("rockfall.logger", "flush"),
    "read_logs": ("rockfall.logger", "read_logs"),
    # ── 多监测点位 (light: stdlib) ──
    "list_sites":      ("rockfall.site_config", "list_sites"),
    "get_active_site": ("rockfall.site_config", "get_active_site"),
    "set_active_site": ("rockfall.site_config", "set_active_site"),
    "MonitoringSite":  ("rockfall.site_config", "MonitoringSite"),
    "PRESET_SITES":    ("rockfall.site_config", "PRESET_SITES"),
    # ── 配置 (light: stdlib, except torch import) ──
    "get_device":      ("rockfall.config", "get_device"),
    "validate_config": ("rockfall.config", "validate_config"),
    # ── 工具层 (light: numpy) ──
    "box_iou_batch":       ("rockfall.utils", "box_iou_batch"),
    "export_alerts_to_excel": ("rockfall.utils", "export_alerts_to_excel"),
    # ── v2.2+ 认证与安全 ──
    "AuthManager":       ("rockfall.auth", "AuthManager"),
    "get_auth_manager":  ("rockfall.auth", "get_auth_manager"),
    "resolve_secret":    ("rockfall.secrets", "resolve_secret"),
    # ── v2.2+ 错误监控 ──
    "init_sentry":       ("rockfall.sentry_init", "init_sentry"),
    "capture_exception": ("rockfall.sentry_init", "capture_exception"),
    # ── v2.2+ 数据库工具 ──
    "is_mysql_available": ("rockfall.db_utils", "is_mysql_available"),
    "get_pymysql":        ("rockfall.db_utils", "get_pymysql"),
    # ── v2.6+ 数据库连接池 ──
    "get_mysql_engine":   ("rockfall.db_engine", "get_mysql_engine"),
    "get_pool_status":    ("rockfall.db_engine", "get_pool_status"),
    "mysql_connection":   ("rockfall.db_engine", "mysql_connection"),
    # ── v2.2+ 性能监控 ──
    "get_global_monitor": ("rockfall.performance", "get_global_monitor"),
}


def __getattr__(name: str):
    """延迟导入 — 首次访问时加载对应子模块。

    用法:
        from rockfall import RockDetector  # 懒加载, 仅导入时解析
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        attr = getattr(mod, attr_name)
        # 缓存到模块命名空间 (后续访问无需再次导入)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'rockfall' has no attribute {name!r}")


__all__ = sorted(_LAZY_IMPORTS.keys()) + ["__version__"]
