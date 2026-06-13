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
DEFAULT_CAMERA_URL = os.getenv("CAMERA_URL", "")          # RTSP 地址或 0(USB摄像头)
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
FASTSAM_NUM_SAMPLES = int(os.getenv("FASTSAM_NUM_SAMPLES", "5"))    # 初始化多帧采样数
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

    if not Path(MODEL_PATH).exists() and not (TENSORRT_ENABLED and Path(TENSORRT_MODEL_PATH).exists()):
        warnings.append(f"模型文件不存在: {MODEL_PATH}")

    if not PUSHPLUS_TOKEN or PUSHPLUS_TOKEN == "your_token_here":
        warnings.append("PUSHPLUS_TOKEN 未配置, 预警推送将不会发送")

    if DETECTION_CONFIDENCE < 0 or DETECTION_CONFIDENCE > 1:
        warnings.append(f"DETECTION_CONFIDENCE 应在 0-1 之间, 当前: {DETECTION_CONFIDENCE}")

    if not (ALERT_BLUE_CONFIDENCE_LOW < ALERT_BLUE_CONFIDENCE_HIGH < ALERT_YELLOW_CONFIDENCE_HIGH < ALERT_ORANGE_CONFIDENCE_HIGH):
        warnings.append("四级预警阈值应满足: blue_low < blue_high < yellow_high < orange_high")

    if ALERT_RED_AREA_RATIO <= ALERT_YELLOW_AREA_RATIO:
        warnings.append("ALERT_RED_AREA_RATIO 应大于 ALERT_YELLOW_AREA_RATIO")

    if not (SKIP_IDLE >= SKIP_ACTIVE >= SKIP_CRITICAL):
        warnings.append(f"跳帧参数应满足 SKIP_IDLE({SKIP_IDLE}) >= SKIP_ACTIVE({SKIP_ACTIVE}) >= SKIP_CRITICAL({SKIP_CRITICAL})")

    if SKIP_IDLE <= 0 or SKIP_ACTIVE <= 0 or SKIP_CRITICAL <= 0:
        warnings.append(f"跳帧参数必须 > 0, 否则会触发除零异常 (当前: idle={SKIP_IDLE} active={SKIP_ACTIVE} critical={SKIP_CRITICAL})")

    if MOTION_SCORE_LOW >= MOTION_SCORE_HIGH:
        warnings.append(f"MOTION_SCORE_LOW({MOTION_SCORE_LOW}) 应 < MOTION_SCORE_HIGH({MOTION_SCORE_HIGH}), 否则三级跳帧部分区间失效")

    if USE_CUDA_PREPROCESS:
        try:
            import cv2
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                warnings.append("USE_CUDA_PREPROCESS=true 但 OpenCV 未启用 CUDA (需 opencv-contrib-python + CUDA 编译), 已回退 CPU")
        except Exception:
            warnings.append("USE_CUDA_PREPROCESS=true 但 cv2.cuda 不可用, 已回退 CPU")

    if TRACK_IOU_THRESHOLD < 0 or TRACK_IOU_THRESHOLD > 1:
        warnings.append(f"TRACK_IOU_THRESHOLD 应在 0-1 之间, 当前: {TRACK_IOU_THRESHOLD}")

    if MOG2_LEARNING_RATE < 0 or MOG2_LEARNING_RATE > 1:
        warnings.append(f"MOG2_LEARNING_RATE 应在 0-1 之间, 当前: {MOG2_LEARNING_RATE}")

    if FUSION_MOTION_WEIGHT < 0 or FUSION_MOTION_WEIGHT > 1:
        warnings.append(f"FUSION_MOTION_WEIGHT 应在 0-1 之间, 当前: {FUSION_MOTION_WEIGHT}")

    if SAHI_OVERLAP_RATIO < 0 or SAHI_OVERLAP_RATIO >= 1:
        warnings.append(f"SAHI_OVERLAP_RATIO 应在 [0, 1) 之间, 当前: {SAHI_OVERLAP_RATIO}")

    if TEMPORAL_IOU < 0 or TEMPORAL_IOU > 1:
        warnings.append(f"TEMPORAL_IOU 应在 0-1 之间, 当前: {TEMPORAL_IOU}")

    if SAHI_ENABLED:
        device_str, device_name = get_device()
        if device_str == "cpu":
            warnings.append(f"SAHI_ENABLED=true 在 CPU ({device_name}) 上性能极差, 已自动禁用")

    return warnings
