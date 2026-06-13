"""
视频检测控件 (V10: FastSAM主力 + 传统CV备选 + 质量守卫)
==========================================================
L1: FastSAM + CLIP文本提示 (主力, 替代旧SAM子进程)
    → L2: 传统CV多帧融合 (备选, FastSAM不可用时)
    → L3: 质量守卫拒绝 → 默认多边形 / 手动框选兜底
"""

import collections, json, time
from pathlib import Path
import cv2, numpy as np

try: cv2.ocl.setUseOpenCL(False)
except: pass
cv2.setNumThreads(1)

from PyQt6 import QtWidgets, QtCore, QtGui
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


class VideoCaptureWidget(QtWidgets.QWidget):
    stats_changed = QtCore.pyqtSignal(int, float, float, list, str)
    log_message = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.detector, self.model_loaded = None, False
        try: self.detector = RockDetector(); self.model_loaded = True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "模型加载失败", f"YOLO模型加载失败\n{e}")

        self.tracker = RockTracker()
        self._timer = QtCore.QTimer(); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._update_frame)
        self._processing, self._finished = False, False

        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumSize(640, 360)
        self.label.setStyleSheet("background-color: #1a1a2e;")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout = QtWidgets.QVBoxLayout(); layout.addWidget(self.label)
        self.setLayout(layout)

        self.cap = None; self.source_type = None; self.source_path = None
        self.frame_w = 0; self.frame_h = 0; self._downscale = 1.0
        self._road_mask = None; self.polygon = None; self.roi_mask = None; self._road_pct = 0
        self.polygons = []
        self.polygons = []
        self._fps = 0.0; self._frame_n = 0
        self._reconnect_delay = CAMERA_RECONNECT_BASE
        self._reconnect_attempts = 0
        self._fps_n = 0; self._fps_t0 = time.time(); self._current_fps = 0.0
        self._last_alert_time = 0.0
        self._pending_dets = collections.deque(maxlen=3)
        self.roi_mode = False; self.roi_points = []; self.roi_preview = None
        self._last_rgb = None
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 帧循环
    # ============================================================
    def _update_frame(self):
        if self._processing or self.cap is None or self.detector is None: return
        self._processing = True
        try:
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type in ("rtsp","webcam"):
                    self._timer.stop()
                    self.log_message.emit(f"视频源断开, {self._reconnect_delay}s后重连...")
                    QtCore.QTimer.singleShot(int(self._reconnect_delay*1000), self._try_reconnect)
                else: self.log_message.emit("视频播放完毕"); self._finished = True
                return
            if self._downscale != 1.0: frame = cv2.resize(frame, (self.frame_w, self.frame_h))
            self._frame_n += 1
            self._reconnect_delay = CAMERA_RECONNECT_BASE
            self._reconnect_attempts = 0
            frame = frame.copy()

            pp = self.detector.preprocess_frame(frame)

            if self._frame_n % pp['skip'] == 0:
                self.detector._active_skip = pp['skip']
                raw_dets = self.detector.detect_frame(frame, pp['box_mask'], pp['fg'])
                self._pending_dets.append((self._frame_n, raw_dets))

            dets = []; stale = []
            for di, dl in self._pending_dets:
                if abs(self._frame_n-di) <= 1: dets.extend(dl)
                else: stale.append((di, dl))
            self._pending_dets.clear()
            for s in stale: self._pending_dets.append(s)
            tracks = self.tracker.update(dets)
            tracks = [t for t in tracks if not _looks_like_vehicle(t)]

            alert_level, max_conf, track_ids = self._classify(tracks)

            if alert_level in ("red","yellow"):
                now = time.time()
                if now - self._last_alert_time >= ALERT_COOLDOWN_SECONDS:
                    self._last_alert_time = now
                    try: send_alert_async(len(tracks), max_conf, frame_bgr=frame.copy(),
                        tracks=tracks, confirm_frames=1, alert_level=alert_level)
                    except Exception as e: self.log_message.emit(f"预警推送失败: {e}")

            self._fps_n += 1
            if self._fps_n >= 30:
                elapsed = time.time()-self._fps_t0
                self._current_fps = 30/elapsed if elapsed>0 else 0
                self._fps_t0 = time.time(); self._fps_n = 0

            RockDetector.draw_tracks(frame, tracks)
            if hasattr(self, 'polygons') and self.polygons:
                for poly in self.polygons:
                    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
            # 红色遮罩仅首次显示10帧后移除
            if self._road_mask is not None and self._frame_n < 10:
                m = (self._road_mask == 255).squeeze()
                if m.ndim == 2:
                    frame[m] = (frame[m] * 0.7 + np.array([0,0,77])).astype(np.uint8)
            if self.roi_mode and len(self.roi_points) >= 1:
                pts = np.array(self.roi_points, np.int32)
                cv2.polylines(frame, [pts], False, (0,255,255), 1)
                for i,p in enumerate(self.roi_points):
                    cv2.circle(frame, tuple(p), 3, (0,255,255), -1)
                    cv2.putText(frame, str(i+1), (p[0]+5, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                if self.roi_preview is not None:
                    cv2.line(frame, self.roi_points[-1], self.roi_preview, (0,255,255), 1, cv2.LINE_AA)

            color_map = {"red":(0,0,255),"orange":(0,140,255),"yellow":(0,215,255),"blue":(255,140,0),"green":(0,200,0)}
            alert_color = color_map.get(alert_level, (0,200,0))
            cv2.rectangle(frame, (0,0), (self.frame_w,self.frame_h), alert_color, 4)
            cv2.putText(frame, f"FPS: {self._current_fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, f"Rocks: {len(tracks)}  Level: {alert_level.upper()}", (10,55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
            if self.source_type == "rtsp":
                cv2.putText(frame, "LIVE", (self.frame_w-60,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            self._display_frame(frame)
            self._fps = self._current_fps
            self.stats_changed.emit(len(tracks), max_conf, self._current_fps, track_ids, alert_level)
        except Exception as e:
            self.log_message.emit(f"帧处理异常: {e}")
        finally:
            self._processing = False
            if not self._finished and self.cap is not None and self.cap.isOpened():
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
                self._timer.start(int(1000/fps))

    def _classify(self, tracks):
        if not tracks: return "green", 0.0, []
        ctx = RockDetector.build_alert_context(tracks, self.frame_w, self.frame_h)
        alert = self.detector._grade_alert(ctx)
        return alert, ctx.max_conf, ctx.track_ids or []

    def _try_reconnect(self):
        self._pending_dets.clear()
        self._fps_n = 0; self._fps_t0 = time.time(); self._current_fps = 0.0
        self._last_alert_time = 0.0
        # 彻底释放旧 VideoCapture (防止句柄泄漏)
        old_cap = self.cap
        if old_cap is not None:
            _safe_release(old_cap)
            self.cap = None
        # 创建新 VideoCapture 对象 (不复用已释放的对象)
        if self.source_type == "rtsp":
            new_cap = cv2.VideoCapture(self.source_path, cv2.CAP_FFMPEG)
            new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            new_cap = cv2.VideoCapture(self.source_path if self.source_path else 0)
        self.cap = new_cap
        self._reconnect_delay = min(int(self._reconnect_delay*CAMERA_RECONNECT_BACKOFF), CAMERA_RECONNECT_MAX)
        if new_cap.isOpened():
            self._reconnect_attempts = 0; self._reconnect_delay = CAMERA_RECONNECT_BASE
            self._build_roi_mask()
            self.detector.init_stream_state(self.frame_w, self.frame_h, self.roi_mask)
            self.detector._road_mask = self._road_mask
            self.tracker.reset()
            self.log_message.emit("视频源已恢复")
            fps = new_cap.get(cv2.CAP_PROP_FPS) or 25
            self._timer.start(int(1000/fps))
        else:
            self._reconnect_attempts += 1
            if self._reconnect_attempts > CAMERA_RECONNECT_MAX_ATTEMPTS:
                self.log_message.emit(f"重连失败({CAMERA_RECONNECT_MAX_ATTEMPTS}次), 请运维人员检查视频源: {self.source_path}")
                self._finished = True; return
            self.log_message.emit(f"重连失败, {self._reconnect_delay}s后重试 (第{self._reconnect_attempts}次)")
            QtCore.QTimer.singleShot(int(self._reconnect_delay*1000), self._try_reconnect)

    def _display_frame(self, frame_bgr):
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lw = self.label.width()
            if lw > 0 and lw != rgb.shape[1]:
                scale = lw / rgb.shape[1]
                rgb = cv2.resize(rgb, (int(rgb.shape[1]*scale), int(rgb.shape[0]*scale)))
            rgb = np.ascontiguousarray(rgb)
            h, w, ch = rgb.shape
            q_img = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format.Format_RGB888)
            self._last_rgb = rgb
            self.label.setPixmap(QtGui.QPixmap.fromImage(q_img.copy()))
        except Exception as e: self.log_message.emit(f"显示异常: {e}")

    # ============================================================
    # 公开方法
    # ============================================================
    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.flv *.mkv *.wmv);;所有文件 (*)")
        if path: self.load_source(path, "file")

    def open_camera(self):
        url, ok = QtWidgets.QInputDialog.getText(self, "RTSP摄像头", "输入RTSP地址:")
        if ok and url.strip(): self.load_source(url.strip(), "rtsp")

    def load_camera(self, camera_id=0): self.load_source(camera_id, "webcam")

    def load_source(self, source, source_type="file"):
        self.stop()
        self.source_type = source_type; self.source_path = source
        if source_type == "rtsp":
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif source_type == "webcam": cap = cv2.VideoCapture(source)
        else: cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self.log_message.emit(f"无法打开视频源: {source}"); return

        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_w, orig_h = self.frame_w, self.frame_h
        MAX_PROC_WIDTH = 1920; self._downscale = 1.0
        if self.frame_w > MAX_PROC_WIDTH:
            self._downscale = MAX_PROC_WIDTH / self.frame_w
            self.frame_w = MAX_PROC_WIDTH
            self.frame_h = int(self.frame_h * self._downscale)
            self.log_message.emit(f"4K降采样: {orig_w}x{orig_h} -> {self.frame_w}x{self.frame_h}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        if self.polygon is None:
            loaded = self._load_roi()
            roi_valid = False
            if loaded is not None:
                saved_w = loaded.get("frame_w", 0)
                saved_h = loaded.get("frame_h", 0)
                # 校验分辨率: 缓存的ROI必须与当前视频匹配
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if saved_w == orig_w and saved_h == orig_h:
                    self.polygon = np.array(loaded["polygon"], np.int32)
                    if self._downscale != 1.0:
                        self.polygon = (self.polygon * self._downscale).astype(np.int32)
                    mask_file = loaded.get("mask_file")
                    if mask_file and Path(mask_file).exists():
                        self._road_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                        if self._road_mask is not None:
                            self.roi_mask = 255 - self._road_mask
                            # 质量守卫: 道路占比>90%说明缓存已失效
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
                self._auto_detect_road(cap, source_type)

        if self._road_mask is None: self._build_roi_mask()
        self.detector.init_stream_state(self.frame_w, self.frame_h, self.roi_mask)
        self.detector._road_mask = self._road_mask
        self.tracker.reset()
        self.tracker.set_video_context(fps, self.frame_h)

        self.cap = cap; self._frame_n = 0; self._finished = False
        self._reconnect_delay = CAMERA_RECONNECT_BASE; self._reconnect_attempts = 0
        self._pending_dets.clear()
        self._fps_n = 0; self._fps_t0 = time.time(); self._current_fps = 0.0
        self._last_alert_time = 0.0
        self._timer.start(int(1000/fps))

        name = (f"RTSP:{source[:30]}..." if source_type=="rtsp" else
                f"摄像头#{source}" if source_type=="webcam" else Path(source).name)
        self.log_message.emit(f"已加载: {name}")

    def stop(self):
        self._timer.stop(); self._finished = True
        if self.cap is not None: _safe_release(self.cap); self.cap = None
        self.source_type = None; self.source_path = None
        self.polygon = None; self.roi_mask = None; self._road_mask = None
        self._pending_dets.clear()
        # 释放 FastSAM 显存
        try: release_fastsam_model()
        except: pass
        self.log_message.emit("已停止")

    def simulate_alert(self):
        self.log_message.emit("=== 模拟预警 ===")
        result = send_alert(1, 0.95, image_url="")
        self.log_message.emit(f"推送: {result}")

    # ============================================================
    # ROI
    # ============================================================
    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode: self.roi_points = []; self.roi_preview = None
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor if self.roi_mode else QtCore.Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        if not self.roi_mode: return
        pt = self._label_to_video(event.pos().x(), event.pos().y())
        if event.button() == QtCore.Qt.MouseButton.LeftButton: self.roi_points.append(pt)
        elif event.button() == QtCore.Qt.MouseButton.RightButton and len(self.roi_points) >= 3: self._finalize_roi()
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton and self.roi_points: self.roi_points.pop()

    def mouseMoveEvent(self, event):
        if self.roi_mode: self.roi_preview = self._label_to_video(event.pos().x(), event.pos().y())

    def _finalize_roi(self):
        self.polygon = np.array(self.roi_points, np.int32)
        self._build_roi_mask(); self._save_roi()

    def _label_to_video(self, lx, ly):
        if self.frame_w == 0 or self.frame_h == 0 or self.label.width() == 0: return (lx, ly)
        scale = self.label.width()/self.frame_w
        vh = int(self.frame_h*scale); bh = (self.label.height()-vh)/2
        return (max(0, min(int(lx/scale), self.frame_w-1)), max(0, min(int((ly-bh)/scale), self.frame_h-1)))

    def _build_roi_mask(self):
        if self.frame_h > 0 and self.frame_w > 0 and self.polygon is not None:
            self.roi_mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
            cv2.fillPoly(self.roi_mask, [self.polygon], 255)

    def _roi_key(self) -> str:
        return f"webcam_{self.source_path}" if self.source_type == "webcam" else str(self.source_path)

    def _save_roi(self):
        if self.polygon is None or self.source_type is None: return
        key = self._roi_key()
        roi_data = {}
        if ROI_CONFIG_PATH.exists():
            try: roi_data = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
            except: roi_data = {}
        if self._downscale != 1.0:
            poly = (self.polygon/self._downscale).astype(int)
            sw, sh = int(self.frame_w/self._downscale), int(self.frame_h/self._downscale)
        else: poly, sw, sh = self.polygon, self.frame_w, self.frame_h
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
            ROI_CONFIG_PATH.write_text(json.dumps(roi_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except: pass

    def _load_roi(self) -> dict | None:
        if self.source_type is None or not ROI_CONFIG_PATH.exists(): return None
        try: roi_data = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
        except: return None
        return roi_data.get(self._roi_key())

    # ============================================================
    # 道路检测 (FastSAM主力 + 传统CV备选 + 质量守卫)
    # ============================================================
    def _auto_detect_road(self, cap, source_type):
        """
        自动检测公路区域，生成边坡 ROI。

        优先级: FastSAM → 传统CV → 默认多边形兜底

        重要: 使用独立临时 VideoCapture 读取帧, 避免影响主 cap 的位置。
        FLV 等容器不支持 cap.set(CAP_PROP_POS_FRAMES, ...) 回退,
        在主 cap 上读取帧后无法回到第0帧, 导致 MOG2 初始化异常。
        """
        is_live = source_type in ("rtsp", "webcam")
        fw, fh = self.frame_w, self.frame_h

        # 使用临时 cap 读取帧, 主 cap 位置不受影响
        tmp_cap = self._open_temp_cap(source_type)
        if tmp_cap is None:
            self.log_message.emit("[ROI] 无法打开临时视频源")
            self.polygon = RockDetector._default_polygon(fw, fh)
            self._build_roi_mask()
            return

        try:
            # ========== L1: FastSAM分割 ==========
            road_mask = None
            try:
                self._road_mask, self.roi_mask = auto_segment_from_cap(
                    tmp_cap, fw, fh, sample_num=3,
                )
                road_pct = (self._road_mask > 0).sum() / (fw * fh) * 100
                self.log_message.emit(
                    f"[FastSAM] 道路{road_pct:.0f}% (mask)")
                road_mask = self._road_mask
            except Exception as e:
                self.log_message.emit(f"[FastSAM] 异常: {e}")

            # ========== L2: 传统CV备选 ==========
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
                        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
                        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
                        self.roi_mask = fused
                        self._road_mask = 255 - fused
                        road_mask = self._road_mask
                except Exception as e:
                    self.log_message.emit(f"[ROI] 传统CV异常: {e}")

            # ========== L3: 默认多边形兜底 ==========
            if road_mask is None:
                self.log_message.emit("[ROI] 使用默认ROI (手动框选可覆盖)")
                self.polygon = RockDetector._default_polygon(fw, fh)
                self._build_roi_mask()
                return

            # ========== 轮廓提取 + 质量守卫 ==========
            road_pct = (self._road_mask > 0).sum() / (fw * fh) * 100
            slope_pct = (self.roi_mask > 0).sum() / (fw * fh) * 100

            if road_pct > 95 or slope_pct < 5:
                self.log_message.emit(
                    f"[ROI] 质量异常(道路{road_pct:.0f}%), 使用默认ROI")
                self.polygon = RockDetector._default_polygon(fw, fh)
                self._build_roi_mask()
                return

            contours, _ = cv2.findContours(self.roi_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
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
            self._save_roi()
        finally:
            _safe_release(tmp_cap)

    def _open_temp_cap(self, source_type: str) -> cv2.VideoCapture | None:
        """
        打开一个独立的临时 VideoCapture, 用于 ROI 检测读取帧。

        与主 cap 完全隔离, 避免 FLV 等不支持 seeking 的容器影响主播放位置。
        """
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
        if self.cap is None or self.source_type is None:
            self.log_message.emit("请先加载视频/RTSP流"); return
        if self._road_mask is not None:
            reply = QtWidgets.QMessageBox.question(self, "确认重新检测",
                "重新检测将覆盖当前ROI, 是否继续?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                self.log_message.emit("已取消重新检测"); return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._finished = False
        self._remove_roi()
        self.polygon = None; self._road_mask = None; self.roi_mask = None
        self._auto_detect_road(self.cap, self.source_type)
        if self.roi_mask is not None:
            self.detector.init_stream_state(self.frame_w, self.frame_h, self.roi_mask)
            self.detector._road_mask = self._road_mask
            self.log_message.emit("边坡自动检测完成, ROI已更新")
        self._timer.stop()

    def reset_and_redetect(self):
        if self.source_type is None or self.source_path is None:
            self.log_message.emit("请先加载视频源"); return
        reply = QtWidgets.QMessageBox.question(self, "重置ROI",
            "将删除当前ROI配置并重新自动检测, 是否继续?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes: return
        key = self._roi_key()
        if ROI_CONFIG_PATH.exists():
            try:
                d = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
                if key in d: del d[key]
                ROI_CONFIG_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            except: pass
        self.polygon = None; self._road_mask = None; self.roi_mask = None
        self.load_source(self.source_path, self.source_type)

    def _remove_roi(self):
        if self.source_type is None: return
        if not ROI_CONFIG_PATH.exists(): return
        try:
            d = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
            key = self._roi_key()
            if key in d: del d[key]
            ROI_CONFIG_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        except: pass


def _looks_like_vehicle(track: dict) -> bool:
    bbox = track.get("bbox", [0,0,0,0])
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    if h <= 0: return False
    aspect = w/h; speed = track.get("speed",0)
    state = track.get("motion_state",""); age = track.get("age",0)
    score = 0
    if aspect > 1.2: score += 2
    if state == "横向滚动": score += 2
    if 2 < speed < 60: score += 1
    if age > 4: score += 1
    return score >= 4
