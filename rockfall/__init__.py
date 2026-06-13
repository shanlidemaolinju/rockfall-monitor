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

__all__ = [
    "RockDetector", "AlertContext",
    "RockTracker", "KalmanBoxTracker",
    "EdgeEnhancer", "sobel_edge_enhance",
    "ThreeFrameDiff", "filter_detections_by_motion", "filter_detections_by_mog2_center",
    "SAHISlicer", "sahi_inference",
    "fuse_confidence", "TemporalFilter",
    "send_alert", "send_alert_async", "dispatch_alert_async",
    "AlertStore", "get_alert_store",
    "log_event", "flush", "read_logs",
    "list_sites", "get_active_site", "set_active_site",
    "get_device", "validate_config",
    "box_iou_batch", "export_alerts_to_excel",
]
