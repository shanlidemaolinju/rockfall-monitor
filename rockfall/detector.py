"""
算法层 — MOG2 运动检测 + YOLO 目标检测 + SORT 跟踪流水线
==========================================================
完整流程:
  1. MOG2 背景减法找出运动区域
  2. 运动区域保持清晰, 非运动区域高斯模糊
  3. 送入 YOLO 进行落石检测
  4. SORT (Kalman+IoU) 多目标跟踪, 分配唯一 ID
  5. 三级预警分级 (红/黄/绿) + 运动状态分类 (静止/滚动/坠落)

支持: 视频文件 / RTSP流 / USB摄像头 / 图片

依赖: rockfall.config, rockfall.notifier, rockfall.tracker, rockfall.logger
"""

import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
cv2.setNumThreads(1)

from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning, module="ultralytics.nn.tasks")

from .config import (
    MODEL_PATH, RESULTS_DIR,
    DETECTION_CONFIDENCE, DETECTION_IMG_SIZE,
    MOTION_MIN_AREA, IMAGE_URL_BASE,
    # 四级预警置信度阈值
    ALERT_BLUE_CONFIDENCE_LOW, ALERT_BLUE_CONFIDENCE_HIGH,
    ALERT_YELLOW_CONFIDENCE_HIGH, ALERT_ORANGE_CONFIDENCE_HIGH,
    # 落石尺寸阈值
    ROCK_SMALL_HEIGHT_RATIO, ROCK_MEDIUM_HEIGHT_RATIO, ROCK_LARGE_HEIGHT_RATIO,
    # 面积辅助阈值
    ALERT_RED_AREA_RATIO, ALERT_RED_MIN_AREA,
    ALERT_RED_HEIGHT_RATIO,
    ALERT_YELLOW_AREA_RATIO, ALERT_YELLOW_MIN_AREA,
    ALERT_YELLOW_HEIGHT_RATIO,
    ALERT_FALLING_MIN_CONF, ALERT_MULTI_COUNT, ALERT_MULTI_TOTAL_AREA_RATIO,
    FALLING_Y_SPEED_THRESHOLD,
    CAMERA_RECONNECT_BASE, CAMERA_RECONNECT_MAX, CAMERA_RECONNECT_BACKOFF,
    CAMERA_RECONNECT_MAX_ATTEMPTS,
    SKIP_IDLE, SKIP_ACTIVE, SKIP_CRITICAL,
    MOTION_SCORE_LOW, MOTION_SCORE_HIGH,
    MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS,
    MOG2_LEARNING_RATE, MOG2_MORPH_KERNEL, MOG2_RESET_IDLE_FRAMES,
    LIGHT_CHANGE_THRESHOLD, LIGHT_CHANGE_LR_FACTOR,
    USE_CUDA_PREPROCESS,
    EDGE_ENHANCE_ENABLED, EDGE_ENHANCE_ALPHA, EDGE_ENHANCE_INTERVAL,
    TFD_ENABLED, TFD_IOU_THRESHOLD, TFD_MORPH_KERNEL, TFD_THRESHOLD,
    MOG2_FILTER_ENABLED,
    SAHI_ENABLED, SAHI_SLICE_SIZE, SAHI_OVERLAP_RATIO, SAHI_MERGE_IOU,
    FUSION_ENABLED, FUSION_MOTION_WEIGHT,
    TEMPORAL_ENABLED, TEMPORAL_WINDOW, TEMPORAL_IOU,
    TENSORRT_ENABLED, TENSORRT_MODEL_PATH,
    get_device,
)
from .notifier import send_alert, send_alert_async, dispatch_alert_async
from .tracker import RockTracker
from .edge_enhance import EdgeEnhancer
from .motion_detect import ThreeFrameDiff, filter_detections_by_motion, filter_detections_by_mog2_center
from .sahi import SAHISlicer, sahi_inference
from .fusion import fuse_confidence, TemporalFilter
from .logger import log_event

# ---- 共享帧缓冲 ----
_frame_lock = threading.Lock()
_latest_frames: dict[str, bytes] = {}


def get_latest_frame(camera_id: str = "default") -> bytes | None:
    """Web 看板 MJPEG 接口用: 获取指定摄像头最新检测帧"""
    with _frame_lock:
        return _latest_frames.get(camera_id)


def _set_latest_frame(jpeg_bytes: bytes, camera_id: str = "default"):
    with _frame_lock:
        _latest_frames[camera_id] = jpeg_bytes


@dataclass
class AlertContext:
    """预警分级入参 — 从已确认轨迹聚合, 避免参数爆炸"""
    max_conf: float = 0.0
    max_area: float = 0.0
    max_height: float = 0.0
    total_area: float = 0.0
    total_count: int = 0
    max_speed: float = 0.0
    max_age: int = 0
    is_falling: bool = False
    frame_area: float = 0.0
    frame_height: int = 0
    track_ids: list = None
    rock_diameter_cm: float = 0.0   # 估算落石直径 (cm)

    def __post_init__(self):
        if self.track_ids is None:
            self.track_ids = []


class RockDetector:
    """落石检测器 — 统一流水线 (服务端+桌面端共用)"""

    _model_cache: dict[str, YOLO] = {}

    def __init__(self):
        # 设备检测: CUDA GPU > CPU, 显式传递避免 YOLO 内部 auto 的不确定性
        self._device_str, self._device_name = get_device()

        # TensorRT 优先: 如果启用且 engine 文件存在则加载, 否则回退 .pt
        model_path = str(MODEL_PATH)
        if TENSORRT_ENABLED and Path(TENSORRT_MODEL_PATH).exists():
            model_path = TENSORRT_MODEL_PATH
        elif not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if model_path not in RockDetector._model_cache:
            RockDetector._model_cache[model_path] = YOLO(model_path)
        self.model = RockDetector._model_cache[model_path]

        # SAHI 在 CPU 上自动禁用 (分块推理在 CPU 上极慢, 毫无实时性)
        if SAHI_ENABLED and self._device_str == "cpu":
            self._sahi_enabled = False
            log_event("system", msg=f"SAHI 在 CPU ({self._device_name}) 上已自动禁用")
        else:
            self._sahi_enabled = SAHI_ENABLED

        self.confidence = DETECTION_CONFIDENCE
        self.img_size = DETECTION_IMG_SIZE
        self.min_area = MOTION_MIN_AREA

        # 四级预警阈值 (实例级, 支持桌面端滑块实时调节)
        self.alert_blue_conf_high = ALERT_BLUE_CONFIDENCE_HIGH     # blue/yellow 分界
        self.alert_yellow_conf_high = ALERT_YELLOW_CONFIDENCE_HIGH  # yellow/orange 分界
        self.alert_orange_conf_high = ALERT_ORANGE_CONFIDENCE_HIGH  # orange/red 分界

        # 流水线状态 (init_stream_state 初始化)
        self._stream_ready = False


    def init_stream_state(self, fw: int, fh: int, roi_mask: np.ndarray | None = None):
        """
        初始化流水线状态。每个视频源调用一次。
        调用后 preprocess_frame() 和 detect_frame() 可用。
        """
        self._fw = fw
        self._fh = fh
        self._roi_mask = roi_mask
        self._roi_pixels = np.count_nonzero(roi_mask) if roi_mask is not None else (fw * fh)

        # CUDA 预处理: 尝试启用 GPU MOG2 + GPU Sobel
        self._cuda_preprocess = False
        if USE_CUDA_PREPROCESS:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self._cuda_preprocess = True
                    print("[CUDA预处理] GPU MOG2 + Sobel 已启用")
            except Exception:
                pass

        if self._cuda_preprocess:
            self._bg_sub = cv2.cuda.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
                detectShadows=MOG2_DETECT_SHADOWS,
            )
        else:
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
                detectShadows=MOG2_DETECT_SHADOWS,
            )

        self._edge_enhancer = EdgeEnhancer(
            enabled=EDGE_ENHANCE_ENABLED, alpha=EDGE_ENHANCE_ALPHA,
            interval=EDGE_ENHANCE_INTERVAL,
            cuda_available=self._cuda_preprocess,
        )
        self._tfd = ThreeFrameDiff(
            threshold=TFD_THRESHOLD, morph_kernel=TFD_MORPH_KERNEL,
            enabled=TFD_ENABLED,
        )
        self._sahi_slicer = SAHISlicer(
            slice_size=SAHI_SLICE_SIZE, overlap_ratio=SAHI_OVERLAP_RATIO,
            merge_iou=SAHI_MERGE_IOU, enabled=self._sahi_enabled,
        )
        self._temporal_filter = TemporalFilter(
            window=TEMPORAL_WINDOW, iou_threshold=TEMPORAL_IOU,
            enabled=TEMPORAL_ENABLED,
        )
        self._consecutive_idle = 0
        self._active_skip = 1  # 当前实际跳帧间隔, 1=无跳帧
        self._prev_brightness = -1.0
        self._stream_ready = True

    def preprocess_frame(self, frame: np.ndarray) -> dict:
        """
        MOG2 运动检测 + 自适应跳帧决策。每帧调用 (含跳过的帧)。

        返回:
            {'fg': np.ndarray, 'motion_score': float, 'has_motion': bool,
             'box_mask': np.ndarray, 'skip': int}
        """
        # 光照突变检测: 帧间整体亮度大幅变化 → 云层移动/阳光变化
        # 此时 MOG2 会把整帧当前景, 需要临时降低学习率防止背景模型被污染
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_brightness = float(np.mean(gray))
        light_change = (
            self._prev_brightness >= 0 and
            abs(cur_brightness - self._prev_brightness) > LIGHT_CHANGE_THRESHOLD
        )
        self._prev_brightness = cur_brightness

        # 自适应学习率: 长时间无运动 → 临时提高学习率快速适应环境变化
        # 但若本帧检测到运动, 立即用低学习率重新应用, 避免落石被快速融入背景
        was_high_lr = self._consecutive_idle >= MOG2_RESET_IDLE_FRAMES
        lr = 0.1 if was_high_lr else MOG2_LEARNING_RATE

        if light_change:
            lr *= LIGHT_CHANGE_LR_FACTOR

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MOG2_MORPH_KERNEL, MOG2_MORPH_KERNEL))
        fg = self._mog2_apply(frame, lr)
        self._postprocess_fg(fg, k)

        # 降采样找轮廓 (1/4 分辨率, ~16x 加速), fg 保持全分辨率供下游使用
        ds = 4
        fg_small = cv2.resize(fg, (self._fw // ds, self._fh // ds), interpolation=cv2.INTER_NEAREST)
        motion_score = np.count_nonzero(fg_small) / max((self._fw // ds) * (self._fh // ds), 1)

        contours, _ = cv2.findContours(fg_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_mask = np.zeros((self._fh, self._fw), dtype=np.uint8)
        pad = 50
        min_area_small = self.min_area // (ds * ds)
        has_motion = False
        for c in contours:
            if cv2.contourArea(c) > min_area_small:
                has_motion = True
                x, y, w, h = cv2.boundingRect(c)
                x, y, w, h = x * ds, y * ds, w * ds, h * ds
                x = max(x - pad, 0); y = max(y - pad, 0)
                w = min(w + 2 * pad, self._fw - x); h = min(h + 2 * pad, self._fh - y)
                cv2.rectangle(box_mask, (x, y), (x + w, y + h), 255, -1)

        if has_motion:
            self._consecutive_idle = 0
            # 若本帧在高学习率模式下检出运动, 用低学习率重新应用 MOG2
            # 防止落石被 lr=0.1 快速融入背景导致后续漏检
            if was_high_lr:
                fg = self._mog2_apply(frame, MOG2_LEARNING_RATE)
                self._postprocess_fg(fg, k)
        else:
            self._consecutive_idle += 1

        # 三级跳帧
        if motion_score < MOTION_SCORE_LOW:
            skip = SKIP_IDLE
        elif motion_score < MOTION_SCORE_HIGH:
            skip = SKIP_ACTIVE
        else:
            skip = SKIP_CRITICAL

        return {
            'fg': fg, 'motion_score': motion_score, 'has_motion': has_motion,
            'box_mask': box_mask, 'skip': skip,
        }

    def _mog2_apply(self, frame: np.ndarray, lr: float) -> np.ndarray:
        """MOG2 前景分割"""
        if self._cuda_preprocess:
            return self._bg_sub.apply(cv2.cuda.GpuMat(frame), learningRate=lr).download()
        return self._bg_sub.apply(frame, learningRate=lr)

    def _postprocess_fg(self, fg: np.ndarray, kernel):
        """前景后处理: 阴影去除 + 形态学 + ROI 裁剪"""
        if MOG2_DETECT_SHADOWS:
            fg[fg == 127] = 0
        # 使用 fg[:] = ... 保证修改作用于调用者的数组
        fg[:] = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg[:] = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        if self._roi_mask is not None:
            cv2.bitwise_and(fg, fg, mask=self._roi_mask, dst=fg)

    def detect_frame(
        self, frame: np.ndarray,
        box_mask: np.ndarray | None = None,
        fg_mask: np.ndarray | None = None,
    ) -> list:
        """
        YOLO 推理 + 全部后处理滤波。仅在非跳帧时调用。

        参数:
            frame:    BGR 原始帧
            box_mask: MOG2 运动区域掩膜 (来自 preprocess_frame)
            fg_mask:  MOG2 前景掩膜 (来自 preprocess_frame)

        返回:
            [[x1, y1, x2, y2, conf], ...]  已过滤的检测框列表
        """
        # 边缘增强 (先增强再模糊非运动区, 否则模糊会削弱边缘)
        det_input = self._edge_enhancer.process(frame)

        # ROI 外区域涂黑 — YOLO 不浪费算力在无关区域
        if self._roi_mask is not None:
            det_input = cv2.bitwise_and(det_input, det_input, mask=self._roi_mask)

        # 非运动区域高斯模糊 — 减少背景干扰
        if box_mask is not None and np.any(box_mask):
            blurred = cv2.GaussianBlur(det_input, (15, 15), 0)
            # box_mask 为 2D (H,W), 需扩展为 3D (H,W,1) 以与 BGR 图像广播
            det_input = np.where(box_mask[..., None] == 255, det_input, blurred)

        # YOLO 推理 (SAHI 或 普通)
        try:
            if self._sahi_enabled:
                raw_dets = sahi_inference(
                    self.model, det_input, self._sahi_slicer, conf=self.confidence,
                )
            else:
                # 确保输入连续 (np.where / bitwise_and 可能产生非连续数组)
                if not det_input.flags["C_CONTIGUOUS"]:
                    det_input = np.ascontiguousarray(det_input)
                results = self.model(
                    det_input, stream=False, conf=self.confidence,
                    imgsz=self.img_size, verbose=False,
                )
                raw_dets = []
                for r in results:
                    if r.boxes is not None:
                        for b in r.boxes:
                            x1, y1, x2, y2 = b.xyxy[0].int().tolist()
                            raw_dets.append([x1, y1, x2, y2, b.conf[0].item(), int(b.cls[0].item())])
                # 同步 CUDA 流, 确保推理结果在继续处理前已就绪
                del results
        except Exception as e:
            log_event("system", level="ERROR", msg=f"YOLO推理失败: {e}")
            return []

        # 概率融合 (YOLO + MOG2) — 先提升置信度再滤波, 避免低置信但有强运动证据的目标被误过滤
        if FUSION_ENABLED and raw_dets:
            raw_dets = fuse_confidence(
                raw_dets, fg_mask, motion_weight=FUSION_MOTION_WEIGHT,
            )

        # 三帧差分运动滤波 (苏国韶2025)
        # 跳帧 >1 时暂停: TFD 需要连续帧, 跳帧导致帧间时间间隔过大, 运动检测失效
        if TFD_ENABLED and self._active_skip <= 1:
            _, tfd_contours = self._tfd.compute(frame)
            if tfd_contours:
                raw_dets = filter_detections_by_motion(
                    raw_dets, tfd_contours, TFD_IOU_THRESHOLD,
                )

        # MOG2 中心点运动滤波 (Zhang2024) — 不依赖帧连续性, 始终可用
        if MOG2_FILTER_ENABLED and raw_dets and fg_mask is not None:
            raw_dets = filter_detections_by_mog2_center(raw_dets, fg_mask)

        # 多帧时序确认 — 跳帧时暂停, 原因同 TFD
        if TEMPORAL_ENABLED and self._active_skip <= 1:
            raw_dets = self._temporal_filter.filter(raw_dets)

        # 道路区域最终过滤: 中心点在道路上的检测框丢弃
        if hasattr(self, '_road_mask') and self._road_mask is not None and raw_dets:
            filtered = []
            for d in raw_dets:
                cx = int((d[0] + d[2]) / 2)
                cy = int((d[1] + d[3]) / 2)
                if 0 <= cx < self._fw and 0 <= cy < self._fh and self._road_mask[cy, cx] == 0:
                    filtered.append(d)
            raw_dets = filtered

        return raw_dets

    # ================================================================
    # 图片检测 (不变)
    # ================================================================

    def detect_image(self, image_path: str, push_alert: bool = True) -> dict:
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not Path(image_path).exists():
            return {"error": f"图片不存在: {image_path}", "time": detection_time}

        results = self.model(str(image_path), imgsz=self.img_size)
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return {"detection": "未检测到落石", "time": detection_time, "count": 0}

        count = len(boxes)
        max_confidence = round(max(float(c) for c in boxes.conf), 4)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f"{ts}.jpg"
        result.save(filename=str(RESULTS_DIR / filename))

        push_result = None
        if push_alert:
            image_url = f"{IMAGE_URL_BASE}/{filename}" if IMAGE_URL_BASE else ""
            push_result = send_alert(count, max_confidence, image_url)

        return {
            "detection": "落石检测到", "time": detection_time,
            "count": count, "max_confidence": max_confidence,
            "saved_to": str(RESULTS_DIR / filename),
            "push_status": push_result,
        }

    # ================================================================
    # 视频检测 (文件)
    # ================================================================

    def detect_video(
        self, video_path: str, save_frames: bool = True,
        push_alerts: bool = True, track: bool = True,
        confirm_frames: int = 3, polygon: np.ndarray | None = None,
        max_frames: int | None = None, stride: int = 1,
        progress_callback=None,
    ) -> dict:
        """对视频文件进行检测

        max_frames: 最大处理帧数 (None=全部, 用于演示限制)
        stride:     帧采样步长 (1=每帧, 2=隔帧, ...)
        progress_callback: 进度回调 (current, total) -> None
        """
        source = str(video_path)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {"error": f"无法打开视频文件: {video_path}"}

        # _process_stream 含 yield, 始终返回生成器; 文件模式下不 yield, 返回值在 StopIteration 中
        gen = self._process_stream(
            cap, source=source, source_name=Path(video_path).name,
            save_frames=save_frames, push_alerts=push_alerts,
            track=track, confirm_frames=confirm_frames,
            polygon=polygon, is_live=False,
            max_frames=max_frames, stride=stride,
            progress_callback=progress_callback,
        )
        result = None
        try:
            next(gen)
        except StopIteration as e:
            result = e.value

        cap.release()
        return result if isinstance(result, dict) else {"error": "视频处理失败"}

    # ================================================================
    # 流检测 (RTSP / 摄像头) 
    # ================================================================

    def detect_stream(
        self,
        source,
        source_name: str = "live",
        save_frames: bool = False,
        push_alerts: bool = True,
        track: bool = True,
        confirm_frames: int = 3,
        polygon: np.ndarray | None = None,
        is_live: bool = True,
        render_to_web: bool = False,
    ) -> Generator[dict, None, None]:
        """
        流模式检测器 (生成器, 逐帧产出结果)。

        参数:
            source:        RTSP URL / 摄像头 ID / 视频路径
            source_name:   来源名称
            render_to_web: 是否将检测帧写入共享缓冲(供Web看板MJPEG)

        Yields:
            {"frame_idx": int, "tracks": [...], "alert_level": str, ...}
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            yield {"error": f"无法打开视频源: {source_name}"}
            return

        try:
            gen = self._process_stream(
                cap, source=source, source_name=source_name,
                save_frames=save_frames, push_alerts=push_alerts,
                track=track, confirm_frames=confirm_frames,
                polygon=polygon, is_live=is_live,
                render_to_web=render_to_web,
            )
            for item in gen:
                yield item
        finally:
            cap.release()

    # ================================================================
    # 内部: 统一流处理引擎
    # ================================================================

    def _process_stream(
        self, cap, *, source, source_name: str,
        save_frames: bool, push_alerts: bool,
        track: bool, confirm_frames: int,
        polygon: np.ndarray | None, is_live: bool,
        render_to_web: bool = False,
        max_frames: int | None = None,
        stride: int = 1,
        progress_callback=None,
    ) -> Generator[dict, None, None] | dict:
        """
        统一的视频/流处理引擎。

        文件模式: 收集所有结果后返回 dict
        流模式:   逐帧 yield dict

        max_frames: 最大处理帧数 (None=全部)
        stride:     帧采样步长 (1=每帧, 3=每3帧处理1帧)
        """
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        if polygon is None:
            polygon = self._default_polygon(fw, fh)
        roi_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon], 255)

        self.init_stream_state(fw, fh, roi_mask)
        trk = RockTracker() if track else None
        if trk is not None:
            trk.set_video_context(fps, fh)

        all_detections = []
        raw_dets: list = []
        frame_idx = 0
        processed_count = 0  # 实际推理帧计数 (max_frames 用)
        disconnected = False
        reconnect_delay = CAMERA_RECONNECT_BASE
        reconnect_attempts = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    reconnect_attempts += 1
                    if reconnect_attempts > CAMERA_RECONNECT_MAX_ATTEMPTS:
                        log_event("system", msg=f"摄像头重连失败({CAMERA_RECONNECT_MAX_ATTEMPTS}次): {source_name}")
                        break
                    disconnected = True
                    log_event("system", msg=f"视频源断开: {source_name}, {reconnect_delay}s后重连({reconnect_attempts}/{CAMERA_RECONNECT_MAX_ATTEMPTS})")
                    time.sleep(reconnect_delay)
                    cap.open(source)
                    reconnect_delay = min(int(reconnect_delay * CAMERA_RECONNECT_BACKOFF), CAMERA_RECONNECT_MAX)
                    continue
                else:
                    break

            if disconnected:
                # 重连成功 → 重置流水线状态 (MOG2/TFD/时序滤波器)
                self.init_stream_state(fw, fh, roi_mask)
                if trk is not None:
                    trk.reset()
                log_event("system", msg=f"视频源恢复: {source_name}")
                disconnected = False
            reconnect_delay = CAMERA_RECONNECT_BASE
            reconnect_attempts = 0

            frame_idx += 1

            # ---- 帧采样步长 (stride > 1 时每隔 stride 帧处理一次) ----
            if stride > 1 and frame_idx % stride != 0:
                continue

            # ---- 最大帧数限制 (演示模式) ----
            if max_frames is not None and processed_count >= max_frames:
                break
            processed_count += 1

            # ---- 进度回调 ----
            if progress_callback is not None:
                progress_callback(processed_count, max_frames or 0)

            # ---- 统一预处理: MOG2 + 跳帧决策 ----
            pp = self.preprocess_frame(frame)

            self._active_skip = max(pp['skip'], 1)
            if frame_idx % self._active_skip == 0:
                raw_dets = self.detect_frame(frame, pp['box_mask'], pp['fg'])
            else:
                raw_dets = []  # 跳帧时清空, 跟踪器仅执行卡尔曼预测

            # ---- SORT 跟踪 ----
            tracks_info = trk.update(raw_dets) if trk else []

            # ---- 分级 + 推送 ----
            alert_ctx = self.build_alert_context(tracks_info, fw, fh) if tracks_info else AlertContext()
            frame_alert = self._grade_alert(alert_ctx)

            frame_det = {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 2),
                "alert_level": frame_alert,
                "boxes": [
                    {
                        "track_id": t["id"],
                        "bbox": t["bbox"],
                        "confidence": t["confidence"],
                        "speed": t.get("speed", 0),
                        "motion_state": t.get("motion_state", "未知"),
                        "confirmed": t["confirmed"],
                        "class_id": t.get("class_id", 0),
                        "class_name": t.get("class_name", "落石"),
                    }
                    for t in tracks_info
                ],
            }
            all_detections.append(frame_det)

            # ---- 绘制标注帧 ----
            annotated = frame.copy()
            self.draw_tracks(annotated, tracks_info, polygon=polygon,
                             fw=fw, fh=fh, alert_level=frame_alert,
                             show_panel=True, show_border=True)
            cv2.putText(annotated, f"ALERT: {frame_alert.upper()}", (int(fw - 280), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        {"red": (0, 0, 255), "orange": (0, 140, 255),
                         "yellow": (0, 215, 255), "blue": (255, 140, 0),
                         "green": (0, 200, 0)}[frame_alert], 2)

            if save_frames:
                cv2.imwrite(str(RESULTS_DIR / f"stream_{frame_idx:06d}.jpg"), annotated)

            if push_alerts and frame_alert != "green":
                image_url = f"{IMAGE_URL_BASE}/stream_{frame_idx:06d}.jpg" if (IMAGE_URL_BASE and save_frames) else ""
                dispatch_alert_async(
                    count=len(tracks_info), max_confidence=alert_ctx.max_conf,
                    alert_level=frame_alert,
                    image_url=image_url,
                    frame_bgr=annotated if not IMAGE_URL_BASE else None,
                    tracks=tracks_info, confirm_frames=confirm_frames,
                    rock_diameter_cm=alert_ctx.rock_diameter_cm,
                )

            log_event("detection", frame=frame_idx,
                      count=len(tracks_info), alert_level=frame_alert,
                      max_confidence=alert_ctx.max_conf,
                      track_ids=[t["id"] for t in tracks_info])

            if not tracks_info:
                cv2.putText(annotated, f"F:{frame_idx} 无检测", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # 写共享帧缓冲
            if render_to_web:
                _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 50])
                _set_latest_frame(jpg.tobytes())

            # ---- 产出结果 ----
            frame_result = {
                "frame_idx": frame_idx,
                "tracks": tracks_info,
                "alert_level": frame_alert,
                "timestamp": datetime.now().isoformat(),
            }
            if is_live:
                yield frame_result

        # 文件模式: 返回汇总
        log_event("system", msg=f"检测完成: {source_name}",
                  total_frames=frame_idx, detections=len(all_detections))
        return {
            "source": source_name,
            "resolution": f"{fw}x{fh}",
            "total_frames": frame_idx,
            "fps": round(fps, 2),
            "frames_with_detections": len(all_detections),
            "detections": all_detections,
        }

    # ================================================================
    # 预警分级
    # ================================================================

    @staticmethod
    def build_alert_context(tracks: list, frame_w: int = 0, frame_h: int = 0) -> AlertContext:
        """从已确认轨迹中提取预警分级所需的所有聚合值"""
        valid = [t for t in tracks if t.get("confirmed")]
        if not valid:
            return AlertContext(frame_area=frame_w * frame_h, frame_height=frame_h)

        max_height_px = max(t["bbox"][3] - t["bbox"][1] for t in valid)
        # 估算落石直径: 以 1080p 为基准, 2% 高度比 ≈ 10cm 直径
        # 直径(cm) = (高度比 / ROCK_SMALL_HEIGHT_RATIO) × 10
        height_ratio = max_height_px / frame_h if frame_h > 0 else 0
        rock_diameter_cm = round((height_ratio / ROCK_SMALL_HEIGHT_RATIO) * 10, 1) if ROCK_SMALL_HEIGHT_RATIO > 0 else 0

        return AlertContext(
            max_conf=max(t.get("smoothed_confidence", t["confidence"]) for t in valid),
            max_area=max(t["area"] for t in valid),
            max_height=max_height_px,
            total_area=sum(t["area"] for t in valid),
            total_count=len(valid),
            max_speed=max(t.get("speed", 0) for t in valid),
            max_age=max(t.get("age", 0) for t in valid),
            is_falling=any(t.get("motion_state") == "快速坠落" for t in valid),
            frame_area=frame_w * frame_h,
            frame_height=frame_h,
            track_ids=[t["id"] for t in valid],
            rock_diameter_cm=rock_diameter_cm,
        )

    def _grade_alert(self, ctx) -> str:
        """
        四级预警分级 (对齐《公路自然灾害监测预警系统技术指南》第5.3节强制要求)。

        分级逻辑 (按置信度 + 落石尺寸综合判定, 取较高等级):
          Ⅰ 级 (特别严重，红色):   置信度 > 0.9 或 直径 > 30cm
          Ⅱ 级 (严重，橙色):       置信度 0.7-0.9 或 直径 20-30cm
          Ⅲ 级 (较重，黄色):       置信度 0.5-0.7 或 直径 10-20cm
          Ⅳ 级 (一般，蓝色):       置信度 0.3-0.5 或 直径 < 10cm
          未达阈值:                 "green" (不触发预警)

        增强因子 (提升一级):
          - 坠落状态 + 置信度 >= 0.3 → 最低黄色
          - 长轨迹 (≥8帧) → 置信度 × 1.15
          - 多目标 (≥3) → 最低黄色
          - 高速运动 (>2×坠落阈值) → 最低黄色
        """
        if ctx.total_count == 0:
            return "green"

        # ---- 长轨迹置信度增强 ----
        effective_conf = ctx.max_conf
        if ctx.max_age >= 8 and ctx.max_conf >= ALERT_FALLING_MIN_CONF:
            effective_conf = min(ctx.max_conf * 1.15, 1.0)

        # ---- 置信度等级 (实例阈值支持桌面滑块调节) ----
        conf_level = "green"
        if effective_conf >= ALERT_BLUE_CONFIDENCE_LOW:
            conf_level = "blue"
        if effective_conf >= self.alert_blue_conf_high:
            conf_level = "yellow"
        if effective_conf >= self.alert_yellow_conf_high:
            conf_level = "orange"
        if effective_conf >= self.alert_orange_conf_high:
            conf_level = "red"

        # ---- 尺寸等级 (落石直径) ----
        size_level = "green"
        if ctx.rock_diameter_cm > 0:
            if ctx.rock_diameter_cm < 10:
                size_level = "blue"
            if ctx.rock_diameter_cm >= 10:
                size_level = "yellow"
            if ctx.rock_diameter_cm >= 20:
                size_level = "orange"
            if ctx.rock_diameter_cm >= 30:
                size_level = "red"

        # ---- 综合判定: 取置信度和尺寸中的较高等级 ----
        level_order = ["green", "blue", "yellow", "orange", "red"]
        base_level = conf_level if level_order.index(conf_level) >= level_order.index(size_level) else size_level

        # ---- 增强因子: 提升一级 ----
        enhanced = base_level
        # 坠落状态 → 至少 yellow
        if ctx.is_falling and effective_conf >= ALERT_FALLING_MIN_CONF:
            if level_order.index(enhanced) < level_order.index("yellow"):
                enhanced = "yellow"
        # 多目标群发 → 至少 yellow
        if ctx.total_count >= ALERT_MULTI_COUNT:
            if level_order.index(enhanced) < level_order.index("yellow"):
                enhanced = "yellow"
        # 多目标总面积 → 至少 yellow
        multi_total_area_thresh = ALERT_MULTI_TOTAL_AREA_RATIO * ctx.frame_area if ctx.frame_area > 0 else 0
        if ctx.total_area >= multi_total_area_thresh and ctx.total_area > 0:
            if level_order.index(enhanced) < level_order.index("yellow"):
                enhanced = "yellow"
        # 高速运动 (坠落判定辅助)
        high_speed = ctx.max_speed > (FALLING_Y_SPEED_THRESHOLD * 2) and ctx.max_speed > 0
        if high_speed and effective_conf >= ALERT_FALLING_MIN_CONF and ctx.max_age >= 3:
            if level_order.index(enhanced) < level_order.index("yellow"):
                enhanced = "yellow"

        return enhanced

    # ================================================================
    # 绘制
    # ================================================================

    @staticmethod
    def draw_tracks(frame, tracks, polygon=None, fw=0, fh=0, alert_level="",
                    show_panel=False, show_border=False):
        """绘制检测框、轨迹、状态信息。

        polygon / alert_level / show_panel / show_border 为可选装饰,
        桌面端可仅调用 draw_tracks(frame, tracks) 只画框和标签。
        """
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            color = (0, 255, 0) if t["confirmed"] else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            state = t.get("motion_state", "")
            spd = t.get("speed", 0)
            cls_name = t.get("class_name", "")
            label = f"#{t['id']} {t['confidence']:.2f} {cls_name} {state} {spd:.1f}px/f"
            cv2.putText(frame, label, (x1, max(y1 - 12, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            traj = t.get("trajectory", [])
            if len(traj) > 1:
                pts = np.array(traj, np.int32)
                cv2.polylines(frame, [pts], False, color, 1)

        if polygon is not None:
            cv2.polylines(frame, [polygon.astype(np.int32)], True, (255, 0, 0), 1)

        if show_panel and fw > 0:
            y0 = 80
            cv2.putText(frame, f"Tracks: {len(tracks)}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, t in enumerate(tracks[:8]):
                spd = t.get("speed", 0)
                state = t.get("motion_state", "")
                cls_name = t.get("class_name", "")
                cv2.putText(frame, f"  #{t['id']} {t['confidence']:.2f} {cls_name} {state} {spd:.1f}p/f",
                            (10, y0 + 20 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if show_border and fw > 0:
            level_colors = {"red": (0, 0, 255), "orange": (0, 140, 255),
                           "yellow": (0, 215, 255), "blue": (255, 140, 0),
                           "green": (0, 200, 0)}
            level_color = level_colors.get(alert_level, (0, 200, 0))
            cv2.rectangle(frame, (0, 0), (fw, fh), level_color, 4)

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _default_polygon(w: int, h: int) -> np.ndarray:
        """默认 ROI: 画面上半部分 (排除底部 40% 道路区域)"""
        top_y = int(h * 0.03)
        bottom_y = int(h * 0.90)
        mx = int(w * 0.60)  # 左侧从60%开始, 排除道路
        return np.array(
            [[mx, top_y], [mx, bottom_y], [w - mx, bottom_y], [w - mx, top_y]], np.int32,
        )
