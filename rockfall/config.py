"""
配置层 — 所有参数从这里统一读取
==============================
读取优先级: 环境变量 > .env 文件 > 代码默认值
"""

import os
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 项目路径
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
UPLOADS_DIR = DATA_DIR / "uploads"
TEMPLATES_DIR = ROOT_DIR / "server" / "templates"
ROI_CONFIG_PATH = DATA_DIR / "roi_config.json"

# ============================================================
# YOLO 模型
# ============================================================
MODEL_PATH = os.getenv("ROCK_MODEL_PATH", str(MODELS_DIR / "rock_best.pt"))
MODEL_LATEST_SYMLINK = MODELS_DIR / "rock_best.latest.pt"  # 符号链接 → 当前版本


def set_active_model(model_path: str | Path) -> None:
    """原子切换模型版本: 更新符号链接指向新模型文件。

    用法:
        set_active_model("models/rock_best_v2.pt")  # 升级
        set_active_model("models/rock_best_v1.pt")  # 回滚
    """
    import os as _os
    target = Path(model_path).resolve()
    symlink = MODEL_LATEST_SYMLINK
    tmp_link = symlink.with_suffix(".tmp")

    if not target.exists():
        raise FileNotFoundError(f"模型文件不存在: {target}")

    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(target)
    tmp_link.replace(symlink)  # 原子操作 (POSIX) / rename (Windows)


def get_active_model_path() -> Path:
    """获取当前激活的模型路径（优先使用符号链接）。

    若 rock_best.latest.pt 存在 → 解析符号链接 → 返回目标路径
    否则返回 MODEL_PATH 默认值。
    """
    if MODEL_LATEST_SYMLINK.exists():
        try:
            resolved = MODEL_LATEST_SYMLINK.resolve()
            if resolved.exists():
                return resolved
        except Exception:
            pass
    return Path(MODEL_PATH)


def list_model_versions() -> list[dict]:
    """列出所有可用模型版本及其元数据。"""
    from datetime import datetime

    versions = []
    for f in MODELS_DIR.glob("rock_best_v*.pt"):
        versions.append({
            "path": str(f),
            "name": f.name,
            "size_mb": round(f.stat().st_size / (1024 ** 2), 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "is_active": (
                MODEL_LATEST_SYMLINK.exists()
                and MODEL_LATEST_SYMLINK.resolve() == f.resolve()
            ),
        })
    return sorted(versions, key=lambda v: v["name"], reverse=True)


# ============================================================
# 多模型热切换 — 按点位 + 时段自动选择 (v2.2+)
# ============================================================

# 时段模型映射: 按 start_hour-end_hour 指定模型文件
# 格式: "0-6=models/rock_night.pt;19-23=models/rock_night.pt"
# 多个时段用分号分隔，时段不重叠
MODEL_SLOT_MAP = os.getenv("MODEL_SLOT_MAP", "")

# 夜间模型路径 (快捷方式，等同于 MODEL_SLOT_MAP="19-23=...;0-6=...")
MODEL_NIGHT_PATH = os.getenv("MODEL_NIGHT_PATH", "")
# 雨天模型路径 (需要外部天气 API 触发，此处仅作预留)
_MODEL_RAIN_PATH = os.getenv("MODEL_RAIN_PATH", "")


def _parse_model_slot_map(env_val: str) -> dict[tuple[int, int], str]:
    """解析 MODEL_SLOT_MAP 环境变量。

    格式: "0-6=models/rock_night.pt;19-23=models/rock_night.pt"
    返回: {(0, 6): "models/rock_night.pt", (19, 23): "models/rock_night.pt"}
    """
    result: dict[tuple[int, int], str] = {}
    if not env_val:
        return result
    for segment in env_val.split(";"):
        segment = segment.strip()
        if "=" not in segment:
            continue
        slot, path = segment.split("=", 1)
        slot = slot.strip()
        path = path.strip()
        if "-" in slot:
            parts = slot.split("-")
            try:
                start, end = int(parts[0]), int(parts[1])
                result[(start, end)] = path
            except ValueError:
                pass
    return result


def _get_model_for_hour(hour: int) -> str | None:
    """根据当前小时返回时段模型路径，无匹配返回 None。

    支持跨午夜时段 (如 19-6 表示 19:00-次日6:00)。
    """
    # 1. 解析 MODEL_SLOT_MAP (支持跨午夜)
    slot_map = _parse_model_slot_map(MODEL_SLOT_MAP)
    for (start, end), path in slot_map.items():
        if start <= end:
            # 普通时段: 如 0-6
            if start <= hour <= end:
                p = Path(path)
                if p.exists():
                    return str(p.resolve())
        else:
            # 跨午夜时段: 如 19-6 (19:00-次日6:00)
            if hour >= start or hour <= end:
                p = Path(path)
                if p.exists():
                    return str(p.resolve())

    # 2. 快捷方式: MODEL_NIGHT_PATH (夜间 19-6 点)
    if MODEL_NIGHT_PATH:
        if hour >= 19 or hour < 6:
            p = Path(MODEL_NIGHT_PATH)
            if p.exists():
                return str(p.resolve())

    return None


def resolve_model_path(site_id: str = "") -> Path:
    """
    按优先级解析模型路径 — 支持多模型热切换。

    优先级:
      1. 点位专用模型 (MonitoringSite.model_override, 从 DB 读取)
      2. 时段模型 (MODEL_SLOT_MAP 或 MODEL_NIGHT_PATH)
      3. 全局激活模型 (MODEL_LATEST_SYMLINK → MODEL_PATH)
      4. TensorRT 引擎 (TENSORRT_MODEL_PATH, 仅 CUDA 设备)

    参数:
        site_id: 当前激活的监测点位 ID

    返回:
        模型文件的绝对路径
    """
    from datetime import datetime

    # 1. 点位专用模型
    if site_id:
        try:
            from .site_config import get_site_by_id
            site = get_site_by_id(site_id)
            if site and site.model_override:
                p = Path(site.model_override)
                if p.exists():
                    from .logger import log_event
                    log_event("system", level="INFO",
                              msg=f"模型选择: 点位专用 ({site_id}) → {p.name}")
                    return p.resolve()
        except Exception:
            pass

    # 2. 时段模型
    hour = datetime.now().hour
    slot_model = _get_model_for_hour(hour)
    if slot_model:
        from .logger import log_event
        log_event("system", level="INFO",
                  msg=f"模型选择: 时段模型 (hour={hour}) → {Path(slot_model).name}")
        return Path(slot_model)

    # 3. 全局激活模型
    return get_active_model_path()


def list_all_models() -> list[dict]:
    """
    列出所有可用模型 (版本 + 时段专用)，供管理 API 使用。
    返回每个模型的路径、名称、大小、类型 (version/slot/site_override)。
    """
    from datetime import datetime

    models = list_model_versions()  # 版本模型
    for m in models:
        m["type"] = "version"

    seen = {m["path"] for m in models}

    # 夜间模型
    if MODEL_NIGHT_PATH:
        p = Path(MODEL_NIGHT_PATH)
        if p.exists() and str(p.resolve()) not in seen:
            models.append({
                "path": str(p.resolve()),
                "name": p.name,
                "size_mb": round(p.stat().st_size / (1024 ** 2), 1),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "is_active": False,
                "type": "slot_night",
            })

    # 时段映射中的其他模型
    slot_map = _parse_model_slot_map(MODEL_SLOT_MAP)
    for (start, end), path in slot_map.items():
        p = Path(path).resolve()
        if p.exists() and str(p) not in seen:
            models.append({
                "path": str(p),
                "name": p.name,
                "size_mb": round(p.stat().st_size / (1024 ** 2), 1),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "is_active": False,
                "type": f"slot_{start}_{end}",
            })
            seen.add(str(p))

    return models

# ============================================================
# 模型版本管理 (Model Registry) — A/B 测试 + 自动回滚
# ============================================================
# 从 S3/OSS 自动拉取新模型, 支持 A/B 灰度测试和基于指标的自动回滚。
# 默认关闭, 不影响现有部署。开启后复用 ColdStorageClient 的 S3/OSS 连接。
MODEL_REGISTRY_ENABLED = os.getenv("MODEL_REGISTRY_ENABLED", "false").lower() == "true"
MODEL_REGISTRY_POLL_INTERVAL_SEC = int(os.getenv("MODEL_REGISTRY_POLL_INTERVAL_SEC", "3600"))
MODEL_REGISTRY_S3_PREFIX = os.getenv("MODEL_REGISTRY_S3_PREFIX", "models/")
MODEL_REGISTRY_AB_SPLIT = float(os.getenv("MODEL_REGISTRY_AB_SPLIT", "0.0"))  # 0=全用稳定版, 50=各50%
MODEL_REGISTRY_AB_SPLIT_ENABLED = MODEL_REGISTRY_AB_SPLIT > 0
# 自动回滚: 需要人工审核数据积累到 MODEL_ROLLBACK_MIN_SAMPLE 条后才启用
MODEL_AUTO_ROLLBACK_ENABLED = os.getenv("MODEL_AUTO_ROLLBACK_ENABLED", "false").lower() == "true"
MODEL_ROLLBACK_FP_RATE_INCREASE = float(os.getenv("MODEL_ROLLBACK_FP_RATE_INCREASE", "2.0"))  # 误报率翻倍
MODEL_ROLLBACK_LATENCY_INCREASE = float(os.getenv("MODEL_ROLLBACK_LATENCY_INCREASE", "1.5"))  # 延迟+50%
MODEL_ROLLBACK_MIN_SAMPLE = int(os.getenv("MODEL_ROLLBACK_MIN_SAMPLE", "100"))  # 最少审核样本数

# ============================================================
# 检测参数
# ============================================================
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.3"))
DETECTION_IMG_SIZE = int(os.getenv("DETECTION_IMG_SIZE", "640"))
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "100"))

# ---- 自适应跳帧策略 (三级: 基于运动显著性得分) ----
# motion_score = 前景像素 / ROI总面积
#   无运动 (motion_score < 0.01):  SKIP_IDLE    — 大幅降采样, 节省算力
#   弱运动 (0.01 ≤ score < 0.1):  SKIP_ACTIVE  — 中等密度
#   强运动 (motion_score ≥ 0.1):   SKIP_CRITICAL — 密集推理, 不漏检
# 例如: 25fps 视频 → idle=8 (~3fps), active=5 (5fps), critical=2 (12.5fps)
# idle=5: 无运动时 ~5fps 推理, 平衡算力与响应速度; 8跳帧过大易漏检快速目标
SKIP_IDLE = int(os.getenv("SKIP_IDLE", "5"))
SKIP_ACTIVE = int(os.getenv("SKIP_ACTIVE", "5"))
SKIP_CRITICAL = int(os.getenv("SKIP_CRITICAL", "2"))
MOTION_SCORE_LOW = float(os.getenv("MOTION_SCORE_LOW", "0.01"))
MOTION_SCORE_HIGH = float(os.getenv("MOTION_SCORE_HIGH", "0.1"))

# ---- 深度空闲降频 (Deep Idle) ----
# 连续无运动超过 DEEP_IDLE_TIMEOUT_SEC 秒后, 进入深度空闲模式:
#   GPU 推理降至 DEEP_IDLE_INFERENCE_INTERVAL_SEC 秒一次 (默认 0.1 FPS)
#   仅用 MOG2 守候, 运动恢复后经防抖确认再唤醒
# 唤醒防抖: 连续 DEEP_IDLE_WAKE_UP_DEBOUNCE 帧有运动才退出深度空闲 (防止飞虫/树叶误唤醒)
DEEP_IDLE_ENABLED = os.getenv("DEEP_IDLE_ENABLED", "true").lower() == "true"
DEEP_IDLE_TIMEOUT_SEC = int(os.getenv("DEEP_IDLE_TIMEOUT_SEC", "600"))           # 进入深度空闲前的无运动等待时间
DEEP_IDLE_INFERENCE_INTERVAL_SEC = float(os.getenv("DEEP_IDLE_INFERENCE_INTERVAL_SEC", "10.0"))  # 深度空闲时推理间隔
DEEP_IDLE_WAKE_UP_DEBOUNCE = int(os.getenv("DEEP_IDLE_WAKE_UP_DEBOUNCE", "3"))   # 唤醒防抖帧数
DEEP_IDLE_ROI_ONLY = os.getenv("DEEP_IDLE_ROI_ONLY", "false").lower() == "true"  # 深度空闲时 MOG2 仅处理 ROI

# ---- MOG2 背景建模参数 ----
MOG2_HISTORY = int(os.getenv("MOG2_HISTORY", "500"))
MOG2_VAR_THRESHOLD = int(os.getenv("MOG2_VAR_THRESHOLD", "32"))
MOG2_DETECT_SHADOWS = os.getenv("MOG2_DETECT_SHADOWS", "false").lower() == "true"
MOG2_LEARNING_RATE = float(os.getenv("MOG2_LEARNING_RATE", "0.001"))
MOG2_MORPH_KERNEL = int(os.getenv("MOG2_MORPH_KERNEL", "5"))
MOG2_RESET_IDLE_FRAMES = int(os.getenv("MOG2_RESET_IDLE_FRAMES", "100"))  # 连续无运动帧数后重置
LIGHT_CHANGE_THRESHOLD = float(os.getenv("LIGHT_CHANGE_THRESHOLD", "15.0"))  # 帧间亮度变化阈值 (0-255)
LIGHT_CHANGE_LR_FACTOR = float(os.getenv("LIGHT_CHANGE_LR_FACTOR", "0.1"))  # 光照突变时学习率缩放因子
USE_CUDA_PREPROCESS = os.getenv("USE_CUDA_PREPROCESS", "false").lower() == "true"  # MOG2/Sobel 使用 CUDA 加速
ROI_CROP_ENABLED = os.getenv("ROI_CROP_ENABLED", "false").lower() == "true"  # MOG2 仅处理 ROI 区域（省算力，需重建背景模型）

# ---- 帧环形缓冲 ----
RING_BUFFER_SIZE = int(os.getenv("RING_BUFFER_SIZE", "150"))  # 缓冲帧数 (150 帧 ≈ 1.1GB)
RING_BUFFER_JPEG_QUALITY = int(os.getenv("RING_BUFFER_JPEG_QUALITY", "70"))  # JPEG 质量 0-100

# ---- 非告警帧缩略图定时保存 ----
# 非告警时段每 THUMBNAIL_SAVE_INTERVAL_MIN 分钟保存一张低质量缩略图 (320x240)
# 告警帧已通过 flush_alert 保存全分辨率, 此处仅保存非告警环境快照
# 缩略图保留 THUMBNAIL_RETENTION_DAYS 天后由 StorageManager.cleanup_thumbnails() 清理
THUMBNAIL_ENABLED = os.getenv("THUMBNAIL_ENABLED", "true").lower() == "true"
THUMBNAIL_SAVE_INTERVAL_MIN = int(os.getenv("THUMBNAIL_SAVE_INTERVAL_MIN", "10"))
THUMBNAIL_WIDTH = int(os.getenv("THUMBNAIL_WIDTH", "320"))
THUMBNAIL_HEIGHT = int(os.getenv("THUMBNAIL_HEIGHT", "240"))
THUMBNAIL_JPEG_QUALITY = int(os.getenv("THUMBNAIL_JPEG_QUALITY", "40"))

# ---- Sobel边缘增强  ----
EDGE_ENHANCE_ENABLED = os.getenv("EDGE_ENHANCE_ENABLED", "false").lower() == "true"
EDGE_ENHANCE_ALPHA = float(os.getenv("EDGE_ENHANCE_ALPHA", "0.3"))
EDGE_ENHANCE_INTERVAL = int(os.getenv("EDGE_ENHANCE_INTERVAL", "1"))

# ---- 三帧差分运动滤波  ----
TFD_ENABLED = os.getenv("TFD_ENABLED", "false").lower() == "true"
TFD_IOU_THRESHOLD = float(os.getenv("TFD_IOU_THRESHOLD", "0.30"))
TFD_MORPH_KERNEL = int(os.getenv("TFD_MORPH_KERNEL", "5"))
TFD_THRESHOLD = int(os.getenv("TFD_THRESHOLD", "25"))

# ---- MOG2中心点运动滤波 ----
MOG2_FILTER_ENABLED = os.getenv("MOG2_FILTER_ENABLED", "false").lower() == "true"

# 自适应松弛: 当MOG2前景占比低于阈值时，中心点检查放宽为邻域搜索
# 解决远景小落石运动微弱导致前景稀疏、中心点检查误杀有效检出的问题
MOG2_RELAX_RADIUS = int(os.getenv("MOG2_RELAX_RADIUS", "0"))          # 松弛半径 (px), 0=禁用, 推荐3
MOG2_RELAX_FG_THRESHOLD = float(os.getenv("MOG2_RELAX_FG_THRESHOLD", "0.03"))  # 前景占比阈值

# ---- 几何误报过滤: 利用落石外观特征排除树枝/飞鸟/光影 ----
GEO_FILTER_ENABLED = os.getenv("GEO_FILTER_ENABLED", "false").lower() == "true"
GEO_FILTER_ASPECT_MIN = float(os.getenv("GEO_FILTER_ASPECT_MIN", "0.3"))   # 宽高比下限 (落石近似方形)
GEO_FILTER_ASPECT_MAX = float(os.getenv("GEO_FILTER_ASPECT_MAX", "3.0"))   # 宽高比上限
GEO_FILTER_AREA_MIN = int(os.getenv("GEO_FILTER_AREA_MIN", "25"))          # 最小面积 (px²)

# ---- SAHI 切片辅助推理 ----
SAHI_ENABLED = os.getenv("SAHI_ENABLED", "false").lower() == "true"
SAHI_SLICE_SIZE = int(os.getenv("SAHI_SLICE_SIZE", "640"))
SAHI_OVERLAP_RATIO = float(os.getenv("SAHI_OVERLAP_RATIO", "0.20"))
SAHI_MERGE_IOU = float(os.getenv("SAHI_MERGE_IOU", "0.50"))
SAHI_MAX_SLICES = int(os.getenv("SAHI_MAX_SLICES", "16"))  # 最大切片数, 超限自动降 overlap 或增大 size

# ---- 概率融合 (YOLO置信度 + MOG2前景证据) ----
# P_joint = P_YOLO + (1 - P_YOLO) × motion_weight × P_MOG2
FUSION_ENABLED = os.getenv("FUSION_ENABLED", "false").lower() == "true"
FUSION_MOTION_WEIGHT = float(os.getenv("FUSION_MOTION_WEIGHT", "0.50"))

# ---- 多帧时序确认 (预SORT闪烁抑制) ----
# IoU 阈值 0.20: 快速目标每帧移动可达自身尺寸 70%, 0.30 会导致漏过滤
TEMPORAL_ENABLED = os.getenv("TEMPORAL_ENABLED", "false").lower() == "true"
TEMPORAL_WINDOW = int(os.getenv("TEMPORAL_WINDOW", "2"))
TEMPORAL_IOU = float(os.getenv("TEMPORAL_IOU", "0.20"))

# ---- TensorRT 推理加速 ----
# 启用后优先加载 .engine 文件 (需 CUDA PyTorch + TensorRT)
# 模型导出: python scripts/export_tensorrt.py
TENSORRT_ENABLED = os.getenv("TENSORRT_ENABLED", "false").lower() == "true"
TENSORRT_MODEL_PATH = os.getenv("TENSORRT_MODEL_PATH", str(MODELS_DIR / "rock_best.engine"))

# ---- SORT 跟踪参数 ----
# min_confirm 从 5 降至 3: 落石从出现到离开可能仅 3-5 帧, 5 帧确认会导致漏报
TRACK_MIN_CONFIRM = int(os.getenv("TRACK_MIN_CONFIRM", "3"))
TRACK_MIN_AGE_FOR_ALERT = int(os.getenv("TRACK_MIN_AGE_FOR_ALERT", "2"))
TRACK_MAX_MISSED = int(os.getenv("TRACK_MAX_MISSED", "10"))
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

# ---- 运动状态物理约束 (以 25fps / 1080p 为基准, 运行时按实际 fps+分辨率缩放) ----
_FPS_REF = 25.0
_RES_REF = 1080.0
FALLING_Y_ACCEL_THRESHOLD = float(os.getenv("FALLING_Y_ACCEL_THRESHOLD", "7.5"))
FALLING_Y_SPEED_THRESHOLD = float(os.getenv("FALLING_Y_SPEED_THRESHOLD", "5.0"))


def scale_physics_for_video(fps: float, frame_height: int) -> tuple[float, float]:
    """
    将基准物理阈值缩放到实际视频参数。

    fps=25 / 1080p 为基准:
      速度阈值 ∝ 1/fps × height/1080  (低帧率 → 每帧位移大 → 阈值更高)
      加速度阈值 ∝ 1/fps² × height/1080
    """
    fps_scale = _FPS_REF / max(fps, 1.0)
    res_scale = frame_height / _RES_REF
    accel = FALLING_Y_ACCEL_THRESHOLD * (fps_scale ** 2) * res_scale
    speed = FALLING_Y_SPEED_THRESHOLD * fps_scale * res_scale
    return accel, speed

# ============================================================
# 摄像头 / RTSP 流
# ============================================================
# 注意: CAMERA_URL 可能含明文密码，生产环境请使用 CAMERA_URL_FILE 或 ENC: 前缀
_CAM_URL = os.getenv("CAMERA_URL", "")
_CAM_URL_FILE = os.getenv("CAMERA_URL_FILE", "")
if _CAM_URL_FILE and Path(_CAM_URL_FILE).exists():
    DEFAULT_CAMERA_URL = Path(_CAM_URL_FILE).read_text(encoding="utf-8").strip()
elif _CAM_URL.startswith("ENC:"):
    try:
        from .secrets import resolve_secret
        DEFAULT_CAMERA_URL = resolve_secret("CAMERA_URL", "")
    except Exception:
        DEFAULT_CAMERA_URL = _CAM_URL
else:
    DEFAULT_CAMERA_URL = _CAM_URL
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")       # RTSP 传输协议: tcp (可靠) | udp (低延迟)
FFMPEG_EXTRA_OPTS = os.getenv("FFMPEG_EXTRA_OPTS", "")    # 额外 FFMPEG 选项, 如 hevc/hwaccel
_ffmpeg_opts = f"rtsp_transport;{RTSP_TRANSPORT}"
if FFMPEG_EXTRA_OPTS:
    _ffmpeg_opts += "|" + FFMPEG_EXTRA_OPTS
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_opts
CAMERA_RECONNECT_BASE = int(os.getenv("CAMERA_RECONNECT_BASE", "5"))      # 初始重连间隔(秒)
CAMERA_RECONNECT_MAX = int(os.getenv("CAMERA_RECONNECT_MAX", "30"))       # 最大重连间隔(秒)
CAMERA_RECONNECT_BACKOFF = float(os.getenv("CAMERA_RECONNECT_BACKOFF", "2.0"))  # 退避因子
CAMERA_RECONNECT_MAX_ATTEMPTS = int(os.getenv("CAMERA_RECONNECT_MAX_ATTEMPTS", "30"))  # 最大重试次数

# ============================================================
# PushPlus 微信推送
# ============================================================
# PushPlus Token 支持加密: PUSHPLUS_TOKEN=ENC:<base64> 或 PUSHPLUS_TOKEN_FILE=/run/secrets/...
_PP_TOKEN = os.getenv("PUSHPLUS_TOKEN", "")
_PP_TOKEN_FILE = os.getenv("PUSHPLUS_TOKEN_FILE", "")
if _PP_TOKEN_FILE and Path(_PP_TOKEN_FILE).exists():
    PUSHPLUS_TOKEN = Path(_PP_TOKEN_FILE).read_text(encoding="utf-8").strip()
elif _PP_TOKEN.startswith("ENC:"):
    try:
        from .secrets import resolve_secret
        PUSHPLUS_TOKEN = resolve_secret("PUSHPLUS_TOKEN", "")
    except Exception:
        PUSHPLUS_TOKEN = _PP_TOKEN
else:
    PUSHPLUS_TOKEN = _PP_TOKEN
PUSHPLUS_TOPIC = os.getenv("PUSHPLUS_TOPIC", "")
PUSHPLUS_URL = os.getenv("PUSHPLUS_URL", "http://www.pushplus.plus/send")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "10"))
PUSH_EXECUTOR_WORKERS = int(os.getenv("PUSH_EXECUTOR_WORKERS", "2"))

# ============================================================
# 预警分级阈值 — 四级分级预警 (对齐《公路自然灾害监测预警系统技术指南》)
# ============================================================
# Ⅰ 级（特别严重，红色）: 置信度 > 0.9 或 直径 > 30cm → 微信推送 + 声光报警
# Ⅱ 级（严重，橙色）:     置信度 0.7-0.9 或 直径 20-30cm → 微信推送通知
# Ⅲ 级（较重，黄色）:     置信度 0.5-0.7 或 直径 10-20cm → 界面弹窗提示
# Ⅳ 级（一般，蓝色）:     置信度 0.3-0.5 或 直径 < 10cm → 仅本地记录，不推送
#
# 落石直径由 YOLO 检测框高度 (像素) 结合参考高度 (focal_length × real_height / px_height) 估算。
# 参考基准: 1080p 画面中, 30cm 落石在 5m 距离约覆盖 70px 高度。
# 简化映射 (分辨率无关, 以检测框高度占画面比例作为近似):
#   直径 < 10cm  → 检测框高度比 < 2%   (1080p: ~22px)
#   直径 10-20cm → 检测框高度比 2%-5%  (1080p: ~22-54px)
#   直径 20-30cm → 检测框高度比 5%-8%  (1080p: ~54-86px)
#   直径 > 30cm  → 检测框高度比 > 8%   (1080p: >86px)
#
# ---- 四级预警置信度阈值 (优先判定) ----
ALERT_BLUE_CONFIDENCE_LOW = float(os.getenv("ALERT_BLUE_CONFIDENCE_LOW", "0.3"))    # Ⅳ 级下限
ALERT_BLUE_CONFIDENCE_HIGH = float(os.getenv("ALERT_BLUE_CONFIDENCE_HIGH", "0.5"))  # Ⅳ 级上限 (= Ⅲ 级下限)
ALERT_YELLOW_CONFIDENCE_HIGH = float(os.getenv("ALERT_YELLOW_CONFIDENCE_HIGH", "0.7"))  # Ⅲ 级上限 (= Ⅱ 级下限)
ALERT_ORANGE_CONFIDENCE_HIGH = float(os.getenv("ALERT_ORANGE_CONFIDENCE_HIGH", "0.9"))  # Ⅱ 级上限 (= Ⅰ 级下限)
# 兼容旧配置: ALERT_RED_CONFIDENCE / ALERT_YELLOW_CONFIDENCE 仍可用, 新配置优先
_legacy_red = float(os.getenv("ALERT_RED_CONFIDENCE", "0.0"))
_legacy_yellow = float(os.getenv("ALERT_YELLOW_CONFIDENCE", "0.0"))
if _legacy_red > 0:
    ALERT_ORANGE_CONFIDENCE_HIGH = _legacy_red
if _legacy_yellow > 0:
    ALERT_BLUE_CONFIDENCE_HIGH = _legacy_yellow

# ---- 落石尺寸阈值 (高度比, 用于辅助判定) ----
ROCK_SMALL_HEIGHT_RATIO = float(os.getenv("ROCK_SMALL_HEIGHT_RATIO", "0.02"))    # < 2%   → 小型 (< 10cm)
ROCK_MEDIUM_HEIGHT_RATIO = float(os.getenv("ROCK_MEDIUM_HEIGHT_RATIO", "0.05"))  # 2%-5%  → 中型 (10-20cm)
ROCK_LARGE_HEIGHT_RATIO = float(os.getenv("ROCK_LARGE_HEIGHT_RATIO", "0.08"))    # 5%-8%  → 大型 (20-30cm)
# > 8% → 特大型 (> 30cm)

# ---- 面积辅助阈值 (兼容旧面积逻辑) ----
ALERT_RED_AREA_RATIO = float(os.getenv("ALERT_RED_AREA_RATIO", "0.02"))
ALERT_RED_MIN_AREA = int(os.getenv("ALERT_RED_MIN_AREA", "5000"))
ALERT_YELLOW_AREA_RATIO = float(os.getenv("ALERT_YELLOW_AREA_RATIO", "0.008"))
ALERT_YELLOW_MIN_AREA = int(os.getenv("ALERT_YELLOW_MIN_AREA", "2000"))
ALERT_RED_HEIGHT_RATIO = float(os.getenv("ALERT_RED_HEIGHT_RATIO", "0.10"))
ALERT_YELLOW_HEIGHT_RATIO = float(os.getenv("ALERT_YELLOW_HEIGHT_RATIO", "0.05"))
ALERT_FALLING_MIN_CONF = float(os.getenv("ALERT_FALLING_MIN_CONF", "0.3"))
ALERT_MULTI_COUNT = int(os.getenv("ALERT_MULTI_COUNT", "3"))
ALERT_MULTI_TOTAL_AREA_RATIO = float(os.getenv("ALERT_MULTI_TOTAL_AREA_RATIO", "0.01"))

# ---- 前兆升压: 漏桶式风险累积 → 预警等级逐步升级 ----
# 原理: 边坡累积损伤不可逆，检测间隙只是检出波动，不代表风险归零
# 告警帧 → 累积风险; 无告警帧 → 风险缓慢消退
PRECURSOR_ESCALATION_ENABLED = os.getenv("PRECURSOR_ESCALATION_ENABLED", "true").lower() == "true"
PRECURSOR_ESCALATION_PERSIST_SEC = float(os.getenv("PRECURSOR_ESCALATION_PERSIST_SEC", "15"))    # 累计风险达此值→升一级
PRECURSOR_ESCALATION_RED_SEC = float(os.getenv("PRECURSOR_ESCALATION_RED_SEC", "30"))           # 累计风险达此值→直冲红色
PRECURSOR_ESCALATION_DECAY_RATE = float(os.getenv("PRECURSOR_ESCALATION_DECAY_RATE", "0.3"))    # 间隙期风险消退速率 (0.3=30%速度)

# ---- 数据库连接池 (MySQL 专用, 高并发稳定) ----
# pool_size: 常驻连接数, max_overflow: 峰值额外连接数, pre_ping: 每次检出前 ping 检测有效性
# recycle: 连接最大存活秒数 (超时自动回收, 避免 MySQL wait_timeout 断开)
# connect_timeout: MySQL 连接超时秒数, read/write_timeout: 读写超时秒数
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_PRE_PING = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
DB_READ_TIMEOUT = int(os.getenv("DB_READ_TIMEOUT", "30"))
DB_WRITE_TIMEOUT = int(os.getenv("DB_WRITE_TIMEOUT", "30"))

# ---- MySQL 数据库 (可选, 不配置则使用 SQLite) ----
# 密码支持加密: MYSQL_PASSWORD=ENC:<base64> 或 MYSQL_PASSWORD_FILE=/run/secrets/db_password
_DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
_DB_PASSWORD_FILE = os.getenv("MYSQL_PASSWORD_FILE", "")
if _DB_PASSWORD_FILE and Path(_DB_PASSWORD_FILE).exists():
    MYSQL_PASSWORD = Path(_DB_PASSWORD_FILE).read_text(encoding="utf-8").strip()
elif _DB_PASSWORD.startswith("ENC:"):
    try:
        from .secrets import resolve_secret
        MYSQL_PASSWORD = resolve_secret("MYSQL_PASSWORD", "")
    except Exception:
        MYSQL_PASSWORD = _DB_PASSWORD
else:
    MYSQL_PASSWORD = _DB_PASSWORD
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rock")

# ---- FastSAM 边坡-公路分割 ----
# 替代旧 SAM 独立进程 + 传统 CV, 利用 FastSAM + CLIP 文本提示精准分割
FASTSAM_ENABLED = os.getenv("FASTSAM_ENABLED", "true").lower() == "true"
FASTSAM_MODEL_NAME = os.getenv("FASTSAM_MODEL_NAME", str(MODELS_DIR / "FastSAM-x.pt"))
FASTSAM_CONFIDENCE = float(os.getenv("FASTSAM_CONFIDENCE", "0.25"))
FASTSAM_IOU = float(os.getenv("FASTSAM_IOU", "0.7"))
FASTSAM_NUM_SAMPLES = int(os.getenv("FASTSAM_NUM_SAMPLES", "7"))    # 初始化多帧采样数(更多=更稳定)
FASTSAM_LIVE_SAMPLE_INTERVAL = float(os.getenv("FASTSAM_LIVE_SAMPLE_INTERVAL", "1.0"))  # RTSP 流采样间隔(秒)
FASTSAM_MIN_QUALITY_SCORE = float(os.getenv("FASTSAM_MIN_QUALITY_SCORE", "0.6"))  # 采样质量最低分
FASTSAM_USE_TEXT_PROMPT = os.getenv("FASTSAM_USE_TEXT_PROMPT", "true").lower() == "true"
# 降级策略: FastSAM 失败时是否回退到传统 CV (road_detector.py)
FASTSAM_FALLBACK_CV = os.getenv("FASTSAM_FALLBACK_CV", "true").lower() == "true"

# ---- 检测类别 ----
CLASS_NAMES = {0: "落石", 1: "滑坡"}

# ============================================================
# 监测站信息 — 多点位管理 (广西+东盟大赛场景)
# ============================================================
# LOCATION 为默认/兜底值; 运行时通过 site_config 获取当前激活点位地理位置
LOCATION = os.getenv("LOCATION", "南宁那安快速路 1 号边坡")
# ACTIVE_SITE_ID: 启动时默认激活的监测点位 ID
#   可选值: nanning_naan_s1 / chongzuo_hena_s2 / fangchenggang_lanhai_s3 / pingxiang_crossborder_s4
#   留空则自动从 site_state.json 恢复, 或使用第一个预设点位
ACTIVE_SITE_ID = os.getenv("ACTIVE_SITE_ID", "")
IMAGE_URL_BASE = os.getenv("IMAGE_URL_BASE", "")
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
# API Key / Stream Token 支持加密和 _FILE 后缀
_API_KEY = os.getenv("API_KEY", "")
_API_KEY_FILE = os.getenv("API_KEY_FILE", "")
if _API_KEY_FILE and Path(_API_KEY_FILE).exists():
    API_KEY = Path(_API_KEY_FILE).read_text(encoding="utf-8").strip()
elif _API_KEY.startswith("ENC:"):
    try:
        from .secrets import resolve_secret
        API_KEY = resolve_secret("API_KEY", "")
    except Exception:
        API_KEY = _API_KEY
else:
    API_KEY = _API_KEY

_STREAM_TOKEN = os.getenv("STREAM_TOKEN", "")
_STREAM_TOKEN_FILE = os.getenv("STREAM_TOKEN_FILE", "")
if _STREAM_TOKEN_FILE and Path(_STREAM_TOKEN_FILE).exists():
    STREAM_TOKEN = Path(_STREAM_TOKEN_FILE).read_text(encoding="utf-8").strip()
elif _STREAM_TOKEN.startswith("ENC:"):
    try:
        from .secrets import resolve_secret
        STREAM_TOKEN = resolve_secret("STREAM_TOKEN", "")
    except Exception:
        STREAM_TOKEN = _STREAM_TOKEN
else:
    STREAM_TOKEN = _STREAM_TOKEN


def get_location() -> str:
    """
    获取当前监测点位地理位置字符串。

    优先级: site_config 激活点位 > 环境变量 LOCATION > 默认值
    该函数每次调用都动态解析，确保点位切换后立即生效。
    """
    try:
        from .site_config import get_active_location
        return get_active_location()
    except Exception:
        return LOCATION

# ---- 隐私脱敏 (Privacy Blur) ----
# 在标注帧落盘前自动模糊人脸/车牌区域，保护行人隐私。
# 使用 OpenCV 内置 Haar Cascade，无需额外下载模型。
# 生产环境如需更高精度，可替换为深度学习模型（通过 PRIVACY_BLUR_MODEL_PATH）。
PRIVACY_BLUR_ENABLED = os.getenv("PRIVACY_BLUR_ENABLED", "false").lower() == "true"
PRIVACY_BLUR_FACES = os.getenv("PRIVACY_BLUR_FACES", "true").lower() == "true"
PRIVACY_BLUR_PLATES = os.getenv("PRIVACY_BLUR_PLATES", "true").lower() == "true"
PRIVACY_BLUR_METHOD = os.getenv("PRIVACY_BLUR_METHOD", "gaussian")       # gaussian | pixelate
PRIVACY_BLUR_KERNEL = int(os.getenv("PRIVACY_BLUR_KERNEL", "25"))         # 模糊核大小 (奇数)
PRIVACY_BLUR_INTERVAL = int(os.getenv("PRIVACY_BLUR_INTERVAL", "1"))      # 跳帧间隔: 1=每帧
PRIVACY_BLUR_MODEL_PATH = os.getenv("PRIVACY_BLUR_MODEL_PATH", "")        # 预留: 自定义检测模型

# ---- 哈希链防篡改 (Hash Chain) ----
# 为每条预警记录计算 SHA256 链式摘要，提供完整性校验。
# 默认关闭；开启后仅对新记录生成哈希，旧记录（data_hash 为空）视为不可信。
ALERT_HASH_CHAIN_ENABLED = os.getenv("ALERT_HASH_CHAIN_ENABLED", "false").lower() == "true"
ALERT_HASH_GENESIS = os.getenv(
    "ALERT_HASH_GENESIS",
    "0000000000000000000000000000000000000000000000000000000000000000",
)
ALERT_HASH_VERIFY_BATCH_SIZE = int(os.getenv("ALERT_HASH_VERIFY_BATCH_SIZE", "500"))

# ---- 数据保留 & 冷存储归档 (Data Retention & Cold Storage) ----
# DB 预警记录保留 ≥1095 天 (3 年)，符合《公路桥梁隧道结构监测系统标准》。
# 冷存储支持 S3 兼容协议 (MinIO, Ceph) 或 Alibaba OSS。
ALERT_RETENTION_DAYS = int(os.getenv("ALERT_RETENTION_DAYS", "1095"))
FILE_RETENTION_DAYS = int(os.getenv("FILE_RETENTION_DAYS", "365"))
THUMBNAIL_RETENTION_DAYS = int(os.getenv("THUMBNAIL_RETENTION_DAYS", "7"))
STRICT_RETENTION = os.getenv("STRICT_RETENTION", "false").lower() == "true"

COLD_STORAGE_TYPE = os.getenv("COLD_STORAGE_TYPE", "")  # "s3" | "oss" | ""=禁用
COLD_STORAGE_ENDPOINT = os.getenv("COLD_STORAGE_ENDPOINT", "")
COLD_STORAGE_BUCKET = os.getenv("COLD_STORAGE_BUCKET", "rockfall-archive")
COLD_STORAGE_ACCESS_KEY = os.getenv("COLD_STORAGE_ACCESS_KEY", "")
COLD_STORAGE_SECRET_KEY = os.getenv("COLD_STORAGE_SECRET_KEY", "")
COLD_STORAGE_REGION = os.getenv("COLD_STORAGE_REGION", "us-east-1")
COLD_STORAGE_PREFIX = os.getenv("COLD_STORAGE_PREFIX", "alerts-archive/")

ARCHIVE_SCHEDULE_HOUR = int(os.getenv("ARCHIVE_SCHEDULE_HOUR", "3"))
ARCHIVE_BATCH_SIZE = int(os.getenv("ARCHIVE_BATCH_SIZE", "10000"))

# ---- 日志级别 (支持热更新) ----
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG | INFO | WARN | ERROR

# ---- 异步任务 / 流 ----
TASK_CLEANUP_SECONDS = int(os.getenv("TASK_CLEANUP_SECONDS", "3600"))
TASK_CLEANUP_STUCK_SECONDS = int(os.getenv("TASK_CLEANUP_STUCK_SECONDS", "7200"))
VIDEO_TASK_WORKERS = int(os.getenv("VIDEO_TASK_WORKERS", "2"))
MJPEG_BLANK_WIDTH = int(os.getenv("MJPEG_BLANK_WIDTH", "640"))
MJPEG_BLANK_HEIGHT = int(os.getenv("MJPEG_BLANK_HEIGHT", "360"))
MJPEG_FRAME_INTERVAL = float(os.getenv("MJPEG_FRAME_INTERVAL", "0.05"))

# ============================================================
# 运行时配置热更新单例 (RuntimeConfig)
# ============================================================
# 所有检测器实例每帧从此单例读取最新值, 无需重启。
# 使用: RuntimeConfig.get("SKIP_IDLE", SKIP_IDLE) — 返回运行时覆盖值或默认值

class _RuntimeConfig:
    """线程安全的运行时配置单例, 支持全参数热更新"""

    def __init__(self):
        self._overrides: dict[str, float | int | bool] = {}
        self._lock = threading.Lock()

    def get(self, key: str, default: float | int | bool) -> float | int | bool:
        """读取运行时值, 未覆盖时返回默认值"""
        with self._lock:
            return self._overrides.get(key, default)

    def set(self, key: str, value: float | int | bool):
        """设置运行时覆盖值"""
        with self._lock:
            self._overrides[key] = value

    def set_batch(self, updates: dict[str, float | int | bool]):
        """批量设置运行时覆盖值"""
        with self._lock:
            self._overrides.update(updates)

    def get_all_overrides(self) -> dict:
        """获取所有已覆盖的值 (供前端展示)"""
        with self._lock:
            return dict(self._overrides)

    def reset(self, key: str | None = None):
        """重置指定 key 或全部覆盖值"""
        with self._lock:
            if key:
                self._overrides.pop(key, None)
            else:
                self._overrides.clear()


RuntimeConfig = _RuntimeConfig()


# 辅助: 带热更新的参数读取
def _rc(key: str, default: float | int | bool) -> float | int | bool:
    """读取配置: RuntimeConfig 覆盖 > 环境变量/默认值"""
    return RuntimeConfig.get(key, default)


# ---- GPU 并发推理 ----
GPU_CONCURRENCY = int(os.getenv("GPU_CONCURRENCY", "2"))  # 多路摄像头并发推理数, 1=串行

# ============================================================
# 推理设备检测
# ============================================================
_device_cache: tuple[str, str] | None = None


def get_device() -> tuple[str, str]:
    """
    检测推理设备，返回 (device_str, description)。

    示例:
        ('cuda:0', 'NVIDIA GeForce RTX 4060')
        ('cpu', 'CPU (Intel Core i7)')

    结果会被缓存，后续调用无开销。
    """
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) or "CUDA GPU"
            _device_cache = ("cuda:0", name)
            return _device_cache
    except ImportError:
        pass

    import platform
    cpu_name = platform.processor() or "CPU"
    _device_cache = ("cpu", cpu_name)
    return _device_cache


# ============================================================
# 确保运行时目录存在
# ============================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def validate_config() -> list[str]:
    """验证关键配置, 返回警告列表 (空列表 = 全部正常)"""
    warnings: list[str] = []

    # ── 模型文件 ──
    if not Path(MODEL_PATH).exists() and not (TENSORRT_ENABLED and Path(TENSORRT_MODEL_PATH).exists()):
        warnings.append(f"模型文件不存在: {MODEL_PATH}")

    # ── 推送配置 ──
    if not PUSHPLUS_TOKEN or PUSHPLUS_TOKEN == "your_token_here":
        warnings.append("PUSHPLUS_TOKEN 未配置, 预警推送将不会发送")
    if PUSH_EXECUTOR_WORKERS < 1:
        warnings.append(f"PUSH_EXECUTOR_WORKERS({PUSH_EXECUTOR_WORKERS}) 应 >= 1")
    if ALERT_COOLDOWN_SECONDS < 1:
        warnings.append(f"ALERT_COOLDOWN_SECONDS({ALERT_COOLDOWN_SECONDS}) 应 >= 1")

    # ── 检测参数 ──
    if DETECTION_CONFIDENCE < 0 or DETECTION_CONFIDENCE > 1:
        warnings.append(f"DETECTION_CONFIDENCE 应在 0-1 之间, 当前: {DETECTION_CONFIDENCE}")
    if DETECTION_IMG_SIZE % 32 != 0:
        warnings.append(f"DETECTION_IMG_SIZE({DETECTION_IMG_SIZE}) 应为 32 的倍数 (YOLO 要求)")
    if MOTION_MIN_AREA <= 0:
        warnings.append(f"MOTION_MIN_AREA({MOTION_MIN_AREA}) 应 > 0")

    # ── 四级预警阈值 ──
    if not (ALERT_BLUE_CONFIDENCE_LOW < ALERT_BLUE_CONFIDENCE_HIGH < ALERT_YELLOW_CONFIDENCE_HIGH < ALERT_ORANGE_CONFIDENCE_HIGH):
        warnings.append("四级预警阈值应满足: blue_low < blue_high < yellow_high < orange_high")
    if ALERT_BLUE_CONFIDENCE_LOW < 0 or ALERT_ORANGE_CONFIDENCE_HIGH > 1:
        warnings.append(f"预警置信度阈值应在 0-1 之间 (当前: low={ALERT_BLUE_CONFIDENCE_LOW}, orange_high={ALERT_ORANGE_CONFIDENCE_HIGH})")
    if ALERT_RED_AREA_RATIO <= ALERT_YELLOW_AREA_RATIO:
        warnings.append("ALERT_RED_AREA_RATIO 应大于 ALERT_YELLOW_AREA_RATIO")
    if ALERT_MULTI_COUNT < 1:
        warnings.append(f"ALERT_MULTI_COUNT({ALERT_MULTI_COUNT}) 应 >= 1")

    # ── 自适应跳帧 ──
    if not (SKIP_IDLE >= SKIP_ACTIVE >= SKIP_CRITICAL):
        warnings.append(f"跳帧参数应满足 SKIP_IDLE({SKIP_IDLE}) >= SKIP_ACTIVE({SKIP_ACTIVE}) >= SKIP_CRITICAL({SKIP_CRITICAL})")
    if SKIP_IDLE <= 0 or SKIP_ACTIVE <= 0 or SKIP_CRITICAL <= 0:
        warnings.append(f"跳帧参数必须 > 0, 否则会触发除零异常 (当前: idle={SKIP_IDLE} active={SKIP_ACTIVE} critical={SKIP_CRITICAL})")
    if MOTION_SCORE_LOW >= MOTION_SCORE_HIGH:
        warnings.append(f"MOTION_SCORE_LOW({MOTION_SCORE_LOW}) 应 < MOTION_SCORE_HIGH({MOTION_SCORE_HIGH}), 否则三级跳帧部分区间失效")

    # ── 深度空闲降频 ──
    if DEEP_IDLE_ENABLED:
        if DEEP_IDLE_TIMEOUT_SEC < 60:
            warnings.append(f"DEEP_IDLE_TIMEOUT_SEC({DEEP_IDLE_TIMEOUT_SEC}) 过小, 建议 ≥ 60 秒以避免频繁进出深度空闲")
        if DEEP_IDLE_INFERENCE_INTERVAL_SEC < 1.0:
            warnings.append(f"DEEP_IDLE_INFERENCE_INTERVAL_SEC({DEEP_IDLE_INFERENCE_INTERVAL_SEC}) < 1s, 深度空闲效果有限")
        if DEEP_IDLE_WAKE_UP_DEBOUNCE < 1:
            warnings.append("DEEP_IDLE_WAKE_UP_DEBOUNCE 应 ≥ 1, 否则无防抖效果")
        if DEEP_IDLE_WAKE_UP_DEBOUNCE > 10:
            warnings.append(f"DEEP_IDLE_WAKE_UP_DEBOUNCE({DEEP_IDLE_WAKE_UP_DEBOUNCE}) 过大, 可能导致真实落石唤醒延迟过长")

    # ── MOG2 背景建模 ──
    if MOG2_LEARNING_RATE < 0 or MOG2_LEARNING_RATE > 1:
        warnings.append(f"MOG2_LEARNING_RATE 应在 0-1 之间, 当前: {MOG2_LEARNING_RATE}")
    if MOG2_MORPH_KERNEL < 3 or MOG2_MORPH_KERNEL % 2 == 0:
        warnings.append(f"MOG2_MORPH_KERNEL({MOG2_MORPH_KERNEL}) 应为 >= 3 的奇数")
    if MOG2_RESET_IDLE_FRAMES < 10:
        warnings.append(f"MOG2_RESET_IDLE_FRAMES({MOG2_RESET_IDLE_FRAMES}) 过小, 建议 >= 10")
    if LIGHT_CHANGE_THRESHOLD < 0 or LIGHT_CHANGE_THRESHOLD > 255:
        warnings.append(f"LIGHT_CHANGE_THRESHOLD({LIGHT_CHANGE_THRESHOLD}) 应在 0-255 之间")

    # ── CUDA 预处理 ──
    if USE_CUDA_PREPROCESS:
        try:
            import cv2
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                warnings.append("USE_CUDA_PREPROCESS=true 但 OpenCV 未启用 CUDA (需 opencv-contrib-python + CUDA 编译), 已回退 CPU")
        except Exception:
            warnings.append("USE_CUDA_PREPROCESS=true 但 cv2.cuda 不可用, 已回退 CPU")

    # ── SORT 跟踪 ──
    if TRACK_IOU_THRESHOLD < 0 or TRACK_IOU_THRESHOLD > 1:
        warnings.append(f"TRACK_IOU_THRESHOLD 应在 0-1 之间, 当前: {TRACK_IOU_THRESHOLD}")
    if TRACK_MIN_CONFIRM < 1:
        warnings.append(f"TRACK_MIN_CONFIRM({TRACK_MIN_CONFIRM}) 应 >= 1")
    if TRACK_MAX_MISSED < 1:
        warnings.append(f"TRACK_MAX_MISSED({TRACK_MAX_MISSED}) 应 >= 1")

    # ── 融合与滤波 ──
    if FUSION_MOTION_WEIGHT < 0 or FUSION_MOTION_WEIGHT > 1:
        warnings.append(f"FUSION_MOTION_WEIGHT 应在 0-1 之间, 当前: {FUSION_MOTION_WEIGHT}")
    if SAHI_OVERLAP_RATIO < 0 or SAHI_OVERLAP_RATIO >= 1:
        warnings.append(f"SAHI_OVERLAP_RATIO 应在 [0, 1) 之间, 当前: {SAHI_OVERLAP_RATIO}")
    if TEMPORAL_IOU < 0 or TEMPORAL_IOU > 1:
        warnings.append(f"TEMPORAL_IOU 应在 0-1 之间, 当前: {TEMPORAL_IOU}")
    if TEMPORAL_WINDOW < 1:
        warnings.append(f"TEMPORAL_WINDOW({TEMPORAL_WINDOW}) 应 >= 1")

    # ── SAHI (CPU 上性能极差) ──
    if SAHI_ENABLED:
        device_str, device_name = get_device()
        if device_str == "cpu":
            warnings.append(f"SAHI_ENABLED=true 在 CPU ({device_name}) 上性能极差, 已自动禁用")

    # ── 帧缓冲 ──
    if RING_BUFFER_SIZE < 10:
        warnings.append(f"RING_BUFFER_SIZE({RING_BUFFER_SIZE}) 过小, 建议 >= 10 帧 (否则告警上下文不足)")

    # ── 缩略图 ──
    if THUMBNAIL_ENABLED:
        if THUMBNAIL_SAVE_INTERVAL_MIN < 1:
            warnings.append(f"THUMBNAIL_SAVE_INTERVAL_MIN({THUMBNAIL_SAVE_INTERVAL_MIN}) 应 >= 1 分钟")

    # ── GPU 并发 ──
    if GPU_CONCURRENCY < 1:
        warnings.append(f"GPU_CONCURRENCY({GPU_CONCURRENCY}) 应 >= 1, 1=串行推理")

    # ── 摄像头重连 ──
    if CAMERA_RECONNECT_BASE < 1:
        warnings.append(f"CAMERA_RECONNECT_BASE({CAMERA_RECONNECT_BASE}) 应 >= 1")
    if CAMERA_RECONNECT_BASE > CAMERA_RECONNECT_MAX:
        warnings.append(f"CAMERA_RECONNECT_BASE({CAMERA_RECONNECT_BASE}) > CAMERA_RECONNECT_MAX({CAMERA_RECONNECT_MAX}), 重连逻辑将异常")
    if CAMERA_RECONNECT_BACKOFF <= 1.0:
        warnings.append(f"CAMERA_RECONNECT_BACKOFF({CAMERA_RECONNECT_BACKOFF}) 应 > 1.0 (退避因子)")
    if CAMERA_RECONNECT_MAX_ATTEMPTS < 1:
        warnings.append(f"CAMERA_RECONNECT_MAX_ATTEMPTS({CAMERA_RECONNECT_MAX_ATTEMPTS}) 应 >= 1")

    # ── FastSAM ──
    if FASTSAM_ENABLED:
        if not Path(FASTSAM_MODEL_NAME).exists():
            warnings.append(f"FastSAM 模型文件不存在: {FASTSAM_MODEL_NAME}")
        if FASTSAM_CONFIDENCE < 0 or FASTSAM_CONFIDENCE > 1:
            warnings.append(f"FASTSAM_CONFIDENCE 应在 0-1 之间, 当前: {FASTSAM_CONFIDENCE}")

    # ── 数据库连接池 ──
    if DB_POOL_SIZE < 1:
        warnings.append(f"DB_POOL_SIZE({DB_POOL_SIZE}) 应 >= 1")
    if DB_MAX_OVERFLOW < 0:
        warnings.append(f"DB_MAX_OVERFLOW({DB_MAX_OVERFLOW}) 应 >= 0")
    if DB_POOL_RECYCLE < 60:
        warnings.append(f"DB_POOL_RECYCLE({DB_POOL_RECYCLE}) 过小, 建议 >= 60 (避免频繁回收)")
    if DB_CONNECT_TIMEOUT < 1 or DB_CONNECT_TIMEOUT > 60:
        warnings.append(f"DB_CONNECT_TIMEOUT({DB_CONNECT_TIMEOUT}) 应在 1-60 之间")
    if MYSQL_HOST and not MYSQL_USER:
        warnings.append("MYSQL_HOST 已配置但 MYSQL_USER 为空")
    if MYSQL_HOST and not MYSQL_DATABASE:
        warnings.append("MYSQL_HOST 已配置但 MYSQL_DATABASE 为空")
    if MYSQL_PORT < 1 or MYSQL_PORT > 65535:
        warnings.append(f"MYSQL_PORT({MYSQL_PORT}) 应在 1-65535 之间")

    # ── 归档 ──
    if ARCHIVE_SCHEDULE_HOUR < 0 or ARCHIVE_SCHEDULE_HOUR > 23:
        warnings.append(f"ARCHIVE_SCHEDULE_HOUR({ARCHIVE_SCHEDULE_HOUR}) 应在 0-23 之间")

    # ── 冷存储 ──
    if COLD_STORAGE_TYPE and COLD_STORAGE_TYPE not in ("s3", "oss"):
        warnings.append(f"COLD_STORAGE_TYPE({COLD_STORAGE_TYPE}) 无效, 仅支持 s3/oss/空")
    if COLD_STORAGE_TYPE and not COLD_STORAGE_ENDPOINT:
        warnings.append(f"已启用 COLD_STORAGE_TYPE={COLD_STORAGE_TYPE} 但未配置 COLD_STORAGE_ENDPOINT")

    return warnings
