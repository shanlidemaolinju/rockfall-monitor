"""
边缘-云协同中继模块 (Edge Relay)
=================================
在边缘端（Jetson Nano / RK3588 / 树莓派5）运行轻量预筛选，
仅上传可疑帧到云端做二次确认，大幅节省带宽和云端算力。

架构:
  RTSP → MOG2 运动检测 → [可选: Nano模型] → 上传可疑帧 → 云端确认

用法:
    from rockfall.edge_relay import EdgeRelay
    relay = EdgeRelay(cloud_endpoint="https://cloud.example.com", api_key="xxx")
    relay.run(source="rtsp://camera.local/stream1")

环境变量:
  EDGE_CLOUD_ENDPOINT     — 云端 API 地址 (必填)
  EDGE_API_KEY            — 云端认证 Key
  EDGE_NANO_MODEL_PATH    — Nano 轻量模型路径 (可选, < 5MB)
  EDGE_MOTION_THRESHOLD   — 运动分数阈值 (默认 0.005)
  EDGE_UPLOAD_QUALITY     — 上传 JPEG 质量 (默认 60)
  EDGE_NANO_CONFIDENCE    — Nano 模型最低置信度 (默认 0.15)
  EDGE_MAX_FPS            — 边缘端最大处理帧率 (默认 5)
"""

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import requests

from .config import (
    MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS,
    MOG2_LEARNING_RATE, MOG2_MORPH_KERNEL,
)
from .logger import log_event


class EdgeRelay:
    """
    边缘端轻量中继 — MOG2 运动检测 + 可选 Nano 模型预筛选 → 上传可疑帧到云端。

    适用硬件: Jetson Nano 2GB+, RK3588, 树莓派5, 或任何能跑 OpenCV 的设备。
    """

    def __init__(
        self,
        cloud_endpoint: str = "",
        api_key: str = "",
        nano_model_path: str = "",
        motion_threshold: float = 0.0,
        upload_quality: int = 0,
        nano_confidence: float = 0.0,
        max_fps: int = 0,
    ):
        # 配置: 参数 > 环境变量 > 默认值
        self.cloud_endpoint = cloud_endpoint or os.getenv("EDGE_CLOUD_ENDPOINT", "")
        self.api_key = api_key or os.getenv("EDGE_API_KEY", "")
        self.nano_model_path = nano_model_path or os.getenv("EDGE_NANO_MODEL_PATH", "")
        self.motion_threshold = motion_threshold or float(
            os.getenv("EDGE_MOTION_THRESHOLD", "0.005")
        )
        self.upload_quality = upload_quality or int(
            os.getenv("EDGE_UPLOAD_QUALITY", "60")
        )
        self.nano_confidence = nano_confidence or float(
            os.getenv("EDGE_NANO_CONFIDENCE", "0.15")
        )
        self.max_fps = max_fps or int(os.getenv("EDGE_MAX_FPS", "5"))
        self.upload_cooldown = float(os.getenv("EDGE_UPLOAD_COOLDOWN", "2.0"))
        self.reconnect_base = int(os.getenv("EDGE_RECONNECT_BASE", "5"))
        self.reconnect_max = int(os.getenv("EDGE_RECONNECT_MAX", "30"))

        # 运行时状态
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._fg_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MOG2_MORPH_KERNEL, MOG2_MORPH_KERNEL),
        )
        self._nano_model = None
        self._frame_count = 0
        self._upload_count = 0
        self._last_upload_time = 0.0
        self._running = False
        self._stats_lock = threading.Lock()

        # Nano 模型加载
        if self.nano_model_path and Path(self.nano_model_path).exists():
            try:
                from ultralytics import YOLO
                self._nano_model = YOLO(self.nano_model_path)
                log_event("system", level="INFO",
                          msg=f"EdgeRelay: Nano 模型已加载 ({Path(self.nano_model_path).name})")
            except Exception as e:
                log_event("system", level="WARN",
                          msg=f"EdgeRelay: Nano 模型加载失败 ({e})，仅用 MOG2 预筛选")

    # ══════════════════════════════════════════════════════════
    # 公共 API
    # ══════════════════════════════════════════════════════════

    def run(
        self, source, source_name: str = "edge", site_id: str = "",
    ) -> Generator[dict, None, None]:
        """
        边缘端主循环 — 生成器，逐帧产出预筛选结果。

        用法:
            for event in relay.run("rtsp://..."):
                if event["uploaded"]:
                    print(f"Frame {event['frame']}: uploaded, cloud says {event['cloud_result']}")

        参数:
            source:      RTSP URL / 摄像头 ID / 视频路径
            source_name: 来源标识
            site_id:     监测点位 ID (上传时附带)

        Yields:
            {"frame": int, "motion_score": float, "uploaded": bool,
             "nano_detections": int, "cloud_result": dict|None}
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log_event("system", level="ERROR",
                      msg=f"EdgeRelay: 无法打开视频源 {source_name}")
            return

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(src_fps / self.max_fps))  # 降采样到 max_fps

        self._running = True
        self._frame_count = 0
        log_event("system", level="INFO",
                  msg=f"EdgeRelay: 启动 {source_name} ({fw}x{fh}, "
                      f"max_fps={self.max_fps}, motion_threshold={self.motion_threshold})")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    log_event("system", level="WARN",
                              msg=f"EdgeRelay: 视频源断开 {source_name}，尝试重连...")
                    reconnect_delay = self.reconnect_base
                    max_attempts = 10
                    reconnected = False
                    for attempt in range(max_attempts):
                        time.sleep(reconnect_delay)
                        cap.release()
                        cap.open(source)
                        if cap.isOpened():
                            log_event("system", level="INFO",
                                      msg=f"EdgeRelay: 重连成功 ({source_name})")
                            reconnected = True
                            break
                        reconnect_delay = min(reconnect_delay * 2, self.reconnect_max)
                        log_event("system", level="WARN",
                                  msg=f"EdgeRelay: 重连失败 ({attempt+1}/{max_attempts})")
                    if not reconnected:
                        log_event("system", level="ERROR",
                                  msg=f"EdgeRelay: 重连失败 ({max_attempts}次)，退出 {source_name}")
                        break
                    continue

                self._frame_count += 1

                # 降采样到目标帧率
                if self._frame_count % frame_interval != 0:
                    continue

                # ---- 阶段 1: MOG2 运动检测 ----
                fg = self._bg_sub.apply(frame, learningRate=MOG2_LEARNING_RATE)
                if MOG2_DETECT_SHADOWS:
                    fg[fg == 127] = 0
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._fg_kernel)
                fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._fg_kernel)

                # 运动分数 (降采样加速)
                ds = 4
                fg_small = cv2.resize(fg, (fw // ds, fh // ds),
                                      interpolation=cv2.INTER_NEAREST)
                motion_score = np.count_nonzero(fg_small) / max(fg_small.size, 1)

                event = {
                    "frame": self._frame_count,
                    "motion_score": round(motion_score, 6),
                    "uploaded": False,
                    "nano_detections": 0,
                    "cloud_result": None,
                }

                # 无运动 → 跳过
                if motion_score < self.motion_threshold:
                    yield event
                    continue

                # ---- 阶段 2: Nano 模型预筛选 (可选) ----
                if self._nano_model is not None:
                    results = self._nano_model(
                        frame, conf=self.nano_confidence, imgsz=320,
                        verbose=False,
                    )
                    if results and results[0].boxes is not None:
                        nano_count = len(results[0].boxes)
                        event["nano_detections"] = nano_count
                        if nano_count == 0:
                            yield event
                            continue
                    else:
                        yield event
                        continue

                # ---- 阶段 3: 上传可疑帧到云端 ----
                # 冷却检查: 避免刷爆云端 API
                now = time.time()
                if now - self._last_upload_time < self.upload_cooldown:
                    yield event
                    continue

                cloud_result = self._upload_frame(frame, source_name, site_id)
                event["uploaded"] = cloud_result is not None
                event["cloud_result"] = cloud_result
                if cloud_result is not None:
                    self._last_upload_time = now
                    self._upload_count += 1

                yield event

        finally:
            cap.release()
            self._running = False

    def stop(self):
        """停止边缘中继"""
        self._running = False

    def get_stats(self) -> dict:
        """获取边缘端运行统计"""
        return {
            "frames_processed": self._frame_count,
            "frames_uploaded": self._upload_count,
            "upload_ratio": (
                round(self._upload_count / max(self._frame_count, 1) * 100, 1)
            ),
            "motion_threshold": self.motion_threshold,
            "nano_model": Path(self.nano_model_path).name if self.nano_model_path else None,
            "cloud_endpoint": self.cloud_endpoint,
        }

    # ══════════════════════════════════════════════════════════
    # 内部
    # ══════════════════════════════════════════════════════════

    def _upload_frame(self, frame: np.ndarray, source_name: str,
                      site_id: str) -> dict | None:
        """上传单帧到云端，返回云端检测结果。失败返回 None。"""
        if not self.cloud_endpoint:
            return None

        try:
            _, jpg = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.upload_quality],
            )

            url = f"{self.cloud_endpoint.rstrip('/')}/api/edge/upload"
            files = {"frame": (f"edge_{source_name}.jpg", jpg.tobytes(), "image/jpeg")}
            data = {"source_name": source_name, "site_id": site_id}
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            r = requests.post(url, files=files, data=data,
                              headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            else:
                log_event("system", level="WARN",
                          msg=f"EdgeRelay: 云端返回 HTTP {r.status_code}")
                return None
        except requests.RequestException as e:
            log_event("system", level="WARN",
                      msg=f"EdgeRelay: 上传失败 ({e})")
            return None


# ══════════════════════════════════════════════════════════════
# 便捷函数: 独立运行的边缘端主循环
# ══════════════════════════════════════════════════════════════

def run_edge_relay(
    source: str,
    cloud_endpoint: str = "",
    api_key: str = "",
    site_id: str = "",
) -> None:
    """
    阻塞式运行边缘中继（适合直接部署在边缘设备上）。

    用法:
        python -m rockfall.edge_relay rtsp://camera.local/stream1

    环境变量:
        EDGE_CLOUD_ENDPOINT / EDGE_API_KEY / EDGE_NANO_MODEL_PATH
    """
    relay = EdgeRelay(
        cloud_endpoint=cloud_endpoint,
        api_key=api_key,
    )
    try:
        for event in relay.run(source, site_id=site_id):
            if event["uploaded"]:
                cloud = event["cloud_result"] or {}
                alert = cloud.get("alert_level", "none")
                log_event("edge", level="INFO",
                          frame=event["frame"],
                          motion=event["motion_score"],
                          uploaded=True,
                          alert=alert)
    except KeyboardInterrupt:
        log_event("system", msg="EdgeRelay: 收到中断信号，退出")
    finally:
        stats = relay.get_stats()
        log_event("system", msg=f"EdgeRelay: 已停止 — "
                  f"处理 {stats['frames_processed']} 帧, "
                  f"上传 {stats['frames_uploaded']} 帧 "
                  f"({stats['upload_ratio']}%)")


# 支持直接运行: python -m rockfall.edge_relay <source>
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m rockfall.edge_relay <rtsp_url_or_camera_id>")
        print("环境变量: EDGE_CLOUD_ENDPOINT, EDGE_API_KEY, EDGE_NANO_MODEL_PATH")
        sys.exit(1)
    run_edge_relay(source=sys.argv[1])
