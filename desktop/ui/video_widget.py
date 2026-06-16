"""
视频检测控件 (V11: 后台线程推理 + 主线程显示, 解决 UI 卡顿)
===========================================================
L1: FastSAM + CLIP文本提示 (主力, 替代旧SAM子进程)
    → L2: 传统CV多帧融合 (备选, FastSAM不可用时)
    → L3: 质量守卫拒绝 → 默认多边形 / 手动框选兜底

V11 改动: YOLO 推理移至 QThread 后台线程, 主线程仅负责:
  - 显示标注帧 (通过 QTimer 轮询共享缓冲区)
  - UI 事件响应 (按钮/滑块/ROI)
  - 声光报警
"""

import collections, json, time, threading
from pathlib import Path
import cv2, numpy as np

try: cv2.ocl.setUseOpenCL(False)
except: pass
cv2.setNumThreads(1)

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

from rockfall.detector import RockDetector
from rockfall.road_detector import generate_roi
from rockfall.road_segmentation import RoadSegmentation, ROIParams
from rockfall.tracker import RockTracker
from rockfall.notifier import send_alert, send_alert_async
from rockfall.config import (RESULTS_DIR, ALERT_COOLDOWN_SECONDS,
    CAMERA_RECONNECT_BASE, CAMERA_RECONNECT_MAX, CAMERA_RECONNECT_BACKOFF,
    CAMERA_RECONNECT_MAX_ATTEMPTS, ROI_CONFIG_PATH)
from rockfall.fastsam_road import (
    auto_segment_from_cap,
    release_model as release_fastsam_model,
)


def _safe_release(cap):
    try: cap.release()
    except: pass


# ══════════════════════════════════════════════════════════════
# 后台推理 Worker (QThread)
# ══════════════════════════════════════════════════════════════

class _DetectionWorker(QObject):
    """
    在后台线程中运行完整的检测流水线:
      read → preprocess(MOG2) → detect(YOLO) → track(SORT) → classify → draw

    通过信号将结果发回主线程, 避免 GUI 冻结。
    """
    # 信号 (跨线程, 自动使用 QueuedConnection)
    new_frame = pyqtSignal()
    stats_changed = pyqtSignal(int, float, float, list, str)  # count, max_conf, fps, track_ids, alert_level
    log_msg = pyqtSignal(str)
    loop_finished = pyqtSignal()       # 处理循环结束 (用于触发 QThread.quit)
    video_ended = pyqtSignal()
    video_disconnected = pyqtSignal()
    video_reconnected = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.detector: RockDetector | None = None
        self.tracker: RockTracker | None = None

        # 线程安全的结果缓冲区 (主线程读取, 工作线程写入)
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_stats: dict = {}

        # 控制标志
        self._running = False
        self._paused = False
        self._pause_cond = threading.Condition()  # 暂停/恢复通知
        self._cap: cv2.VideoCapture | None = None

        # 视频源配置 (由主线程在启动前设置)
        self.source_path: str = ""
        self.source_type: str = ""
        self.frame_w: int = 0
        self.frame_h: int = 0
        self._downscale: float = 1.0
        self._road_mask: np.ndarray | None = None
        self.roi_mask: np.ndarray | None = None
        self.polygons: list = []
        self._frame_n: int = 0

    def get_latest(self, last_seen_idx: int = -1) -> tuple[np.ndarray | None, dict, int]:
        """主线程调用 — 获取最新标注帧和统计信息。线程安全。

        Args:
            last_seen_idx: 上次已显示的帧号, 若未变化则跳过 copy 优化性能

        Returns:
            (frame, stats, frame_idx) — frame 为 None 表示无新帧
        """
        with self._lock:
            if self._latest_frame is None:
                return None, dict(self._latest_stats), self._frame_n
            if self._frame_n == last_seen_idx:
                return None, dict(self._latest_stats), self._frame_n
            frame = self._latest_frame.copy()
            stats = dict(self._latest_stats)
            fn = self._frame_n
        return frame, stats, fn

    @QtCore.pyqtSlot()
    def run_loop(self):
        """主处理循环 (在 QThread 中执行)。"""
        self._running = True

        # ── 打开视频源 ──
        self._cap = self._open_source()
        if self._cap is None or not self._cap.isOpened():
            self.log_msg.emit(f"无法打开视频源: {self.source_path}")
            self.video_ended.emit()
            self.loop_finished.emit()
            return

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25

        # ── 初始化检测器流状态 ──
        if self.detector is not None:
            self.detector.init_stream_state(self.frame_w, self.frame_h, self.roi_mask)
            self.detector._road_mask = self._road_mask
        if self.tracker is not None:
            self.tracker.reset()
            self.tracker.set_video_context(fps, self.frame_h)

        pending_dets = collections.deque(maxlen=3)
        frame_n = 0
        fps_n = 0
        fps_t0 = time.time()
        current_fps = 0.0
        last_alert_time = 0.0
        reconnect_delay = CAMERA_RECONNECT_BASE
        reconnect_attempts = 0

        while self._running:
            # ── 暂停检查 (ROI 框选等交互模式下挂起推理) ──
            with self._pause_cond:
                while self._paused and self._running:
                    self._pause_cond.wait(timeout=0.5)

            if not self._running:
                break
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if not ret:
                if self.source_type in ("rtsp", "webcam"):
                    self.video_disconnected.emit()
                    self.log_msg.emit(f"视频源断开, {reconnect_delay}s后重连...")
                    time.sleep(reconnect_delay)

                    _safe_release(self._cap)
                    self._cap = self._open_source()
                    if self._cap is None or not self._cap.isOpened():
                        reconnect_attempts += 1
                        reconnect_delay = min(
                            int(reconnect_delay * CAMERA_RECONNECT_BACKOFF),
                            CAMERA_RECONNECT_MAX,
                        )
                        if reconnect_attempts > CAMERA_RECONNECT_MAX_ATTEMPTS:
                            self.log_msg.emit(
                                f"重连失败({CAMERA_RECONNECT_MAX_ATTEMPTS}次), "
                                f"请运维人员检查视频源: {self.source_path}"
                            )
                            break
                        self.log_msg.emit(
                            f"重连失败, {reconnect_delay}s后重试 "
                            f"(第{reconnect_attempts}次)"
                        )
                        continue

                    # 重连成功 → 重置状态
                    reconnect_attempts = 0
                    reconnect_delay = CAMERA_RECONNECT_BASE
                    if self.detector is not None:
                        self.detector.init_stream_state(self.frame_w, self.frame_h, self.roi_mask)
                        self.detector._road_mask = self._road_mask
                    if self.tracker is not None:
                        self.tracker.reset()
                    pending_dets.clear()
                    self.video_reconnected.emit()
                    self.log_msg.emit("视频源已恢复")
                    continue
                else:
                    break  # 文件播放完毕

            reconnect_delay = CAMERA_RECONNECT_BASE
            reconnect_attempts = 0

            if self._downscale != 1.0:
                frame = cv2.resize(frame, (self.frame_w, self.frame_h))
            frame_n += 1
            frame = frame.copy()

            # ── 预处理 (MOG2 运动检测) ──
            pp = self.detector.preprocess_frame(frame)

            # ── YOLO 推理 (按跳帧策略) ──
            if frame_n % pp['skip'] == 0:
                self.detector._active_skip = pp['skip']
                raw_dets = self.detector.detect_frame(frame, pp['box_mask'], pp['fg'])
                pending_dets.append((frame_n, raw_dets))

            # ── 合并延迟帧 ──
            dets = []
            stale = []
            for di, dl in pending_dets:
                if abs(frame_n - di) <= 1:
                    dets.extend(dl)
                else:
                    stale.append((di, dl))
            pending_dets.clear()
            for s in stale:
                pending_dets.append(s)

            # ── 跟踪 ──
            tracks = self.tracker.update(dets)
            tracks = [t for t in tracks if not _looks_like_vehicle(t)]

            # ── 分级 ──
            alert_level, max_conf, track_ids = self._classify(tracks)

            # ── 预警推送 ──
            if alert_level in ("red", "yellow"):
                now = time.time()
                if now - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                    last_alert_time = now
                    try:
                        send_alert_async(
                            len(tracks), max_conf, frame_bgr=frame.copy(),
                            tracks=tracks, confirm_frames=1,
                            alert_level=alert_level,
                        )
                    except Exception as e:
                        self.log_msg.emit(f"预警推送失败: {e}")

            # ── FPS 计算 ──
            fps_n += 1
            if fps_n >= 30:
                elapsed = time.time() - fps_t0
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_t0 = time.time()
                fps_n = 0

            # ── 绘制标注 ──
            RockDetector.draw_tracks(frame, tracks)
            if self.polygons:
                for poly in self.polygons:
                    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

            # 道路遮罩 (仅前 10 帧显示)
            if self._road_mask is not None and frame_n < 10:
                m = (self._road_mask == 255).squeeze()
                if m.ndim == 2:
                    frame[m] = (frame[m] * 0.7 + np.array([0, 0, 77])).astype(np.uint8)

            # 预警边框
            color_map = {
                "red": (0, 0, 255), "orange": (0, 140, 255),
                "yellow": (0, 215, 255), "blue": (255, 140, 0),
                "green": (0, 200, 0),
            }
            alert_color = color_map.get(alert_level, (0, 200, 0))
            cv2.rectangle(frame, (0, 0), (self.frame_w, self.frame_h), alert_color, 4)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Rocks: {len(tracks)}  Level: {alert_level.upper()}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
            if self.source_type == "rtsp":
                cv2.putText(frame, "LIVE", (self.frame_w - 60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ── 写入共享缓冲区 ──
            with self._lock:
                self._latest_frame = frame  # 不 copy, 工作线程会分配新 frame
                self._latest_stats = {
                    "count": len(tracks), "max_conf": max_conf,
                    "fps": current_fps, "track_ids": track_ids,
                    "alert_level": alert_level,
                }
            self._frame_n = frame_n

            # ── 通知主线程 ──
            self.new_frame.emit()
            self.stats_changed.emit(
                len(tracks), max_conf, current_fps, track_ids, alert_level,
            )

        # 清理
        _safe_release(self._cap)
        self._cap = None
        # 通知主线程: 视频流结束 (先于 loop_finished, 让 UI 先停定时器)
        self.video_ended.emit()
        self.loop_finished.emit()

    def pause(self):
        """暂停处理循环 (用于 ROI 框选等交互模式)。"""
        with self._pause_cond:
            self._paused = True

    def resume(self):
        """恢复处理循环。"""
        with self._pause_cond:
            self._paused = False
            self._pause_cond.notify()

    def stop(self):
        """停止处理循环。"""
        with self._pause_cond:
            self._running = False
            self._paused = False
            self._pause_cond.notify()  # 唤醒可能在等待的线程

    def _open_source(self) -> cv2.VideoCapture | None:
        if self.source_type == "rtsp":
            cap = cv2.VideoCapture(self.source_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.source_type == "webcam":
            cap = cv2.VideoCapture(int(self.source_path) if self.source_path else 0)
        else:
            cap = cv2.VideoCapture(self.source_path)
        return cap if cap.isOpened() else None

    def _classify(self, tracks):
        if not tracks:
            return "green", 0.0, []
        ctx = RockDetector.build_alert_context(tracks, self.frame_w, self.frame_h)
        alert = self.detector._grade_alert(ctx)
        return alert, ctx.max_conf, ctx.track_ids or []


# ══════════════════════════════════════════════════════════════
# 视频控件 (主线程 — UI 显示)
# ══════════════════════════════════════════════════════════════

class VideoCaptureWidget(QtWidgets.QWidget):
    stats_changed = QtCore.pyqtSignal(int, float, float, list, str)
    log_message = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.detector, self.model_loaded = None, False
        try:
            self.detector = RockDetector()
            self.model_loaded = True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "模型加载失败", f"YOLO模型加载失败\n{e}")

        self.tracker = RockTracker()

        # ── 后台推理线程 (惰性创建, load_source 时替换) ──
        self._worker: _DetectionWorker | None = None
        self._worker_thread: QThread | None = None

        # ── 显示刷新: new_frame 信号驱动 + 30fps 兜底定时器 ──
        self._display_timer = QtCore.QTimer()
        self._display_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._display_timer.timeout.connect(self._refresh_display)
        self._display_interval_ms = 33  # ~30 fps

        self._finished = False

        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumSize(640, 360)
        self.label.setStyleSheet("background-color: #1a1a2e;")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.source_type = None
        self.source_path = None
        self.frame_w = 0
        self.frame_h = 0
        self._downscale = 1.0
        self._road_mask = None
        self.polygon = None
        self.roi_mask = None
        self._road_pct = 0
        self.polygons = []
        self.roi_mode = False
        self.roi_points = []
        self.roi_preview = None
        self._last_rgb = None
        self._last_displayed_fn = -1
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # 显示刷新 (主线程, 定时轮询共享缓冲区)
    # ══════════════════════════════════════════════════════════

    def _refresh_display(self):
        """定时器回调: 从共享缓冲区拉取最新帧并显示。"""
        if self._finished or self._worker is None:
            return
        frame, stats, fn = self._worker.get_latest(self._last_displayed_fn)
        if frame is not None:
            self._display_frame(frame)
            self._last_rgb = frame
            self._last_displayed_fn = fn

    def _display_frame(self, frame_bgr):
        """BGR numpy → QPixmap 并显示 (仅在主线程调用)。"""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lw = self.label.width()
            if lw > 0 and lw != rgb.shape[1]:
                scale = lw / rgb.shape[1]
                rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)))
            rgb = np.ascontiguousarray(rgb)
            h, w, ch = rgb.shape
            q_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
            # .copy() 防止 q_img 持有 rgb buffer 的悬空指针
            self.label.setPixmap(QtGui.QPixmap.fromImage(q_img.copy()))
        except Exception as e:
            self.log_message.emit(f"显示异常: {e}")

    # ══════════════════════════════════════════════════════════
    # Worker 信号回调
    # ══════════════════════════════════════════════════════════

    def _forward_stats(self, count, max_conf, fps, track_ids, alert_level):
        """转发统计信号给 MainWindow (不修改接口兼容性)。"""
        self.stats_changed.emit(count, max_conf, fps, track_ids, alert_level)

    def _on_worker_log(self, msg: str):
        self.log_message.emit(msg)

    def _on_video_ended(self):
        self._finished = True
        self._display_timer.stop()
        self.log_message.emit("视频播放完毕")

    def _on_video_disconnected(self):
        self.log_message.emit("视频源断开, 正在重连...")

    def _on_video_reconnected(self):
        self.log_message.emit("视频源已恢复")

    # ══════════════════════════════════════════════════════════
    # Worker 生命周期管理
    # ══════════════════════════════════════════════════════════

    def _setup_worker(self):
        """创建新的 worker + thread 并连接信号 (每次加载视频源时调用)。"""
        self._worker = _DetectionWorker()
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        # 线程启动 → worker 开始处理
        self._worker_thread.started.connect(self._worker.run_loop)
        # worker 循环结束 → 停止线程 → 清理
        self._worker.loop_finished.connect(self._worker_thread.quit)
        self._worker.loop_finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        # 视频状态信号
        self._worker.video_ended.connect(self._on_video_ended)
        self._worker.video_disconnected.connect(self._on_video_disconnected)
        self._worker.video_reconnected.connect(self._on_video_reconnected)
        # 日志 & 统计转发
        self._worker.log_msg.connect(self._on_worker_log)
        self._worker.stats_changed.connect(self._forward_stats)
        # new_frame 信号驱动显示刷新
        self._worker.new_frame.connect(self._refresh_display)

    def _cleanup_worker(self):
        """安全停止并清理 worker + thread。可重复调用。"""
        worker = self._worker
        thread = self._worker_thread
        self._worker = None
        self._worker_thread = None

        if worker is not None:
            worker.stop()
            # 断开所有信号防止悬空回调
            try:
                worker.blockSignals(True)
            except Exception:
                pass

        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(3000):
                thread.terminate()
                thread.wait(1000)

    # ══════════════════════════════════════════════════════════
    # 公开方法 (load / stop)
    # ══════════════════════════════════════════════════════════

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.flv *.mkv *.wmv);;所有文件 (*)",
        )
        if path:
            self.load_source(path, "file")

    def open_camera(self):
        url, ok = QtWidgets.QInputDialog.getText(self, "RTSP摄像头", "输入RTSP地址:")
        if ok and url.strip():
            self.load_source(url.strip(), "rtsp")

    def load_camera(self, camera_id=0):
        self.load_source(camera_id, "webcam")

    def load_source(self, source, source_type="file"):
        self.stop()
        self.source_type = source_type
        self.source_path = source

        # 先用临时 cap 获取视频信息 (分辨率/ROI)
        if source_type == "rtsp":
            tmp_cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            tmp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif source_type == "webcam":
            tmp_cap = cv2.VideoCapture(source)
        else:
            tmp_cap = cv2.VideoCapture(source)

        if not tmp_cap.isOpened():
            self.log_message.emit(f"无法打开视频源: {source}")
            return

        self.frame_h = int(tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_w, orig_h = self.frame_w, self.frame_h
        MAX_PROC_WIDTH = 1920
        self._downscale = 1.0
        if self.frame_w > MAX_PROC_WIDTH:
            self._downscale = MAX_PROC_WIDTH / self.frame_w
            self.frame_w = MAX_PROC_WIDTH
            self.frame_h = int(self.frame_h * self._downscale)
            self.log_message.emit(
                f"4K降采样: {orig_w}x{orig_h} -> {self.frame_w}x{self.frame_h}"
            )

        # ── ROI 加载 / 自动检测 ──
        if self.polygon is None:
            loaded = self._load_roi()
            roi_valid = False
            if loaded is not None:
                saved_w = loaded.get("frame_w", 0)
                saved_h = loaded.get("frame_h", 0)
                if saved_w == orig_w and saved_h == orig_h:
                    self.polygon = np.array(loaded["polygon"], np.int32)
                    if self._downscale != 1.0:
                        self.polygon = (self.polygon * self._downscale).astype(np.int32)
                    mask_file = loaded.get("mask_file")
                    if mask_file and Path(mask_file).exists():
                        self._road_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                        if self._road_mask is not None:
                            self.roi_mask = 255 - self._road_mask
                            road_pct = (self._road_mask > 0).sum() / (self.frame_w * self.frame_h) * 100
                            if road_pct < 90:
                                roi_valid = True
                                self.log_message.emit(
                                    f"已加载缓存ROI ({len(self.polygon)}顶点, 道路{road_pct:.0f}%)")
                            else:
                                self.log_message.emit(
                                    f"缓存ROI异常(道路{road_pct:.0f}%), 重新检测...")
                                self.polygon = None
                                self._road_mask = None
                                self.roi_mask = None
                    else:
                        roi_valid = True
                        self.log_message.emit(f"已加载缓存ROI ({len(self.polygon)}个顶点)")
                else:
                    self.log_message.emit(
                        f"缓存ROI分辨率不匹配({saved_w}x{saved_h}!={orig_w}x{orig_h}), 重新检测...")
            if not roi_valid:
                self._auto_detect_road(tmp_cap, source_type)

        if self._road_mask is None:
            self._build_roi_mask()

        _safe_release(tmp_cap)

        # ── 启动后台推理 (创建新 worker, QThread 不可复用) ──
        self._setup_worker()
        self._worker.detector = self.detector
        self._worker.tracker = self.tracker
        self._worker.source_path = str(self.source_path)
        self._worker.source_type = self.source_type
        self._worker.frame_w = self.frame_w
        self._worker.frame_h = self.frame_h
        self._worker._downscale = self._downscale
        self._worker._road_mask = self._road_mask
        self._worker.roi_mask = self.roi_mask
        self._worker.polygons = self.polygons
        self._worker._frame_n = 0

        self._finished = False
        self._last_displayed_fn = -1
        # 30fps 定时器兜底 (new_frame 信号已驱动刷新, 定时器确保不掉帧)
        self._display_timer.start(self._display_interval_ms)
        self._worker_thread.start()

        name = (
            f"RTSP:{source[:30]}..." if source_type == "rtsp" else
            f"摄像头#{source}" if source_type == "webcam" else
            Path(source).name
        )
        self.log_message.emit(f"已加载: {name}")

    def stop(self):
        """停止检测并清理资源。"""
        self._finished = True
        self._display_timer.stop()

        # 清理后台线程
        self._cleanup_worker()

        self.source_type = None
        self.source_path = None
        self.polygon = None
        self.roi_mask = None
        self._road_mask = None
        # 释放 FastSAM 显存 (CPU 模式, 但防御性保留)
        try:
            release_fastsam_model()
        except Exception:
            pass
        self.log_message.emit("已停止")

    def simulate_alert(self):
        self.log_message.emit("=== 模拟预警 ===")
        result = send_alert(1, 0.95, image_url="")
        self.log_message.emit(f"推送: {result}")

    # ══════════════════════════════════════════════════════════
    # ROI 交互 (主线程)
    # ══════════════════════════════════════════════════════════

    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            self.roi_points = []
            self.roi_preview = None
            if self._worker is not None:
                self._worker.pause()
            self.log_message.emit("ROI 模式已激活 — 推理已暂停")
        else:
            if self._worker is not None:
                self._worker.resume()
            self.log_message.emit("ROI 模式已退出 — 推理已恢复")
        self.setCursor(
            QtCore.Qt.CursorShape.CrossCursor if self.roi_mode
            else QtCore.Qt.CursorShape.ArrowCursor
        )

    def mousePressEvent(self, event):
        if not self.roi_mode:
            return
        pt = self._label_to_video(event.pos().x(), event.pos().y())
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.roi_points.append(pt)
        elif event.button() == QtCore.Qt.MouseButton.RightButton and len(self.roi_points) >= 3:
            self._finalize_roi()
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton and self.roi_points:
            self.roi_points.pop()

    def mouseMoveEvent(self, event):
        if self.roi_mode:
            self.roi_preview = self._label_to_video(event.pos().x(), event.pos().y())

    def _finalize_roi(self):
        self.polygon = np.array(self.roi_points, np.int32)
        self._build_roi_mask()
        self._save_roi()

    def _label_to_video(self, lx, ly):
        if self.frame_w == 0 or self.frame_h == 0 or self.label.width() == 0:
            return (lx, ly)
        scale = self.label.width() / self.frame_w
        vh = int(self.frame_h * scale)
        bh = (self.label.height() - vh) / 2
        return (
            max(0, min(int(lx / scale), self.frame_w - 1)),
            max(0, min(int((ly - bh) / scale), self.frame_h - 1)),
        )

    def _build_roi_mask(self):
        if self.frame_h > 0 and self.frame_w > 0 and self.polygon is not None:
            self.roi_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
            cv2.fillPoly(self.roi_mask, [self.polygon], 255)

    def _roi_key(self) -> str:
        return f"webcam_{self.source_path}" if self.source_type == "webcam" else str(self.source_path)

    def _save_roi(self):
        if self.polygon is None or self.source_type is None:
            return
        key = self._roi_key()
        roi_data = {}
        if ROI_CONFIG_PATH.exists():
            try:
                roi_data = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                roi_data = {}
        if self._downscale != 1.0:
            poly = (self.polygon / self._downscale).astype(int)
            sw, sh = int(self.frame_w / self._downscale), int(self.frame_h / self._downscale)
        else:
            poly, sw, sh = self.polygon, self.frame_w, self.frame_h
        entry = {"polygon": poly.astype(int).tolist(), "frame_w": sw, "frame_h": sh}
        if self._road_mask is not None:
            import hashlib
            sk = hashlib.md5(key.encode()).hexdigest()[:12]
            mask_path = ROI_CONFIG_PATH.parent / f"mask_{sk}.png"
            cv2.imwrite(str(mask_path), self._road_mask)
            entry["mask_file"] = str(mask_path)
        roi_data[key] = entry
        try:
            ROI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            ROI_CONFIG_PATH.write_text(
                json.dumps(roi_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_roi(self) -> dict | None:
        if self.source_type is None or not ROI_CONFIG_PATH.exists():
            return None
        try:
            roi_data = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
        return roi_data.get(self._roi_key())

    # ══════════════════════════════════════════════════════════
    # 道路检测 (FastSAM主力 + 传统CV备选 + 质量守卫)
    # ══════════════════════════════════════════════════════════

    def _auto_detect_road(self, cap, source_type):
        is_live = source_type in ("rtsp", "webcam")
        fw, fh = self.frame_w, self.frame_h

        tmp_cap = self._open_temp_cap(source_type)
        if tmp_cap is None:
            self.log_message.emit("[ROI] 无法打开临时视频源")
            self.polygon = RockDetector._default_polygon(fw, fh)
            self._build_roi_mask()
            return

        try:
            # L1: FastSAM
            road_mask = None
            from rockfall.fastsam_road import is_model_ready
            from rockfall.config import FASTSAM_NUM_SAMPLES

            try:
                self._road_mask, self.roi_mask = auto_segment_from_cap(
                    tmp_cap, fw, fh, sample_num=FASTSAM_NUM_SAMPLES,
                )
                road_pct = (self._road_mask > 0).sum() / (fw * fh) * 100
                self.log_message.emit(f"[FastSAM] 道路{road_pct:.0f}% (mask)")
                road_mask = self._road_mask
            except Exception as e:
                self.log_message.emit(f"[FastSAM] 异常: {e}")

            # L2: 传统CV
            if road_mask is None:
                self.log_message.emit("[ROI] FastSAM不可用, 切传统CV...")
                try:
                    roi_masks = []
                    tmp_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    for _ in range(10):
                        ret, frame = tmp_cap.read()
                        if not ret:
                            break
                        if frame.shape[1] != fw or frame.shape[0] != fh:
                            frame = cv2.resize(frame, (fw, fh))
                        roi_masks.append(generate_roi(frame))
                    if roi_masks:
                        fused = np.median(np.stack(roi_masks, axis=0), axis=0).astype(np.uint8)
                        fused = cv2.morphologyEx(
                            fused, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                        )
                        fused = cv2.morphologyEx(
                            fused, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                        )
                        self.roi_mask = fused
                        self._road_mask = 255 - fused
                        road_mask = self._road_mask
                except Exception as e:
                    self.log_message.emit(f"[ROI] 传统CV异常: {e}")

            # L3: 默认多边形兜底
            if road_mask is None:
                self.log_message.emit("[ROI] 使用默认ROI (手动框选可覆盖)")
                self.polygon = RockDetector._default_polygon(fw, fh)
                self._build_roi_mask()
                return

            # 质量守卫 + 轮廓提取
            road_pct = (self._road_mask > 0).sum() / (fw * fh) * 100
            slope_pct = (self.roi_mask > 0).sum() / (fw * fh) * 100

            if road_pct > 95 or slope_pct < 5:
                self.log_message.emit(
                    f"[ROI] 质量异常(道路{road_pct:.0f}%), 使用默认ROI")
                self.polygon = RockDetector._default_polygon(fw, fh)
                self._build_roi_mask()
                return

            contours, _ = cv2.findContours(
                self.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            self.polygons = []
            poly_area_pct = 0
            if contours:
                for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                    if cv2.contourArea(cnt) < fw * fh * 0.03:
                        break
                    if len(self.polygons) >= 3:
                        break
                    poly_area_pct += cv2.contourArea(cnt) / (fw * fh) * 100
                    eps = 0.003 * cv2.arcLength(cnt, True)
                    poly = cv2.approxPolyDP(cnt, eps, True).squeeze(1)
                    if poly.ndim == 1:
                        poly = poly.reshape(-1, 2)
                    if not np.array_equal(poly[0], poly[-1]):
                        poly = np.vstack([poly, poly[0:1]])
                    self.polygons.append(poly)
            if self.polygons:
                self.polygon = self.polygons[0]
            else:
                self.polygon = RockDetector._default_polygon(fw, fh)
                self.polygons = [self.polygon]

            self._road_pct = road_pct
            self.log_message.emit(
                f"[ROI] {len(self.polygon)}顶点 边坡框{poly_area_pct:.0f}% 道路{road_pct:.0f}%")

            from rockfall.fastsam_road import is_model_ready
            if is_model_ready():
                self._save_roi()
            else:
                self.log_message.emit("[ROI] CV降级结果不缓存, 下次启动将重试FastSAM")
        finally:
            _safe_release(tmp_cap)

    def _open_temp_cap(self, source_type: str) -> cv2.VideoCapture | None:
        if source_type == "rtsp":
            cap = cv2.VideoCapture(self.source_path, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif source_type == "webcam":
            cap = cv2.VideoCapture(self.source_path if self.source_path else 0)
        else:
            cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            _safe_release(cap)
            return None
        return cap

    def redo_detection(self):
        if self.source_type is None or self.source_path is None:
            self.log_message.emit("请先加载视频/RTSP流")
            return
        if self._road_mask is not None:
            reply = QtWidgets.QMessageBox.question(
                self, "确认重新检测",
                "重新检测将覆盖当前ROI, 是否继续?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                self.log_message.emit("已取消重新检测")
                return
        # 使用临时 cap 重新检测
        tmp_cap = self._open_temp_cap(self.source_type)
        if tmp_cap is None:
            self.log_message.emit("[ROI] 无法打开视频源进行重新检测")
            return
        self._remove_roi()
        self.polygon = None
        self._road_mask = None
        self.roi_mask = None
        self._auto_detect_road(tmp_cap, self.source_type)
        _safe_release(tmp_cap)
        if self.roi_mask is not None and self._worker is not None:
            self._worker.roi_mask = self.roi_mask
            self._worker._road_mask = self._road_mask
            self.log_message.emit("边坡自动检测完成, ROI已更新")

    def reset_and_redetect(self):
        if self.source_type is None or self.source_path is None:
            self.log_message.emit("请先加载视频源")
            return
        reply = QtWidgets.QMessageBox.question(
            self, "重置ROI",
            "将删除当前ROI配置并重新自动检测, 是否继续?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        key = self._roi_key()
        if ROI_CONFIG_PATH.exists():
            try:
                d = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
                if key in d:
                    del d[key]
                ROI_CONFIG_PATH.write_text(
                    json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        self.polygon = None
        self._road_mask = None
        self.roi_mask = None
        self.load_source(self.source_path, self.source_type)

    def _remove_roi(self):
        if self.source_type is None:
            return
        if not ROI_CONFIG_PATH.exists():
            return
        try:
            d = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
            key = self._roi_key()
            if key in d:
                del d[key]
            ROI_CONFIG_PATH.write_text(
                json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass


def _looks_like_vehicle(track: dict) -> bool:
    bbox = track.get("bbox", [0, 0, 0, 0])
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if h <= 0:
        return False
    aspect = w / h
    speed = track.get("speed", 0)
    state = track.get("motion_state", "")
    age = track.get("age", 0)
    score = 0
    if aspect > 1.2:
        score += 2
    if state == "横向滚动":
        score += 2
    if 2 < speed < 60:
        score += 1
    if age > 4:
        score += 1
    return score >= 4
