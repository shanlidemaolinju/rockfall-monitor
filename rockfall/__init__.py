"""
rockfall — 落石检测核心库
===========================
本包是桌面应用(desktop/)和Web服务(server/)的共享基础层。

模块分层:
  config.py   — 配置层: 从 .env 读取所有参数
  detector.py — 算法层: MOG2运动检测 + YOLO目标检测 + SORT跟踪流水线
  tracker.py  — 跟踪层: Kalman+IoU 多目标跟踪 (SORT算法)
  notifier.py — 通知层: PushPlus微信推送 (含连续帧确认、base64图片)
  logger.py   — 日志层: 检测事件持久化为JSONL文件
"""
