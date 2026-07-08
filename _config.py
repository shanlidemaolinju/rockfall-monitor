"""
落石监测系统 — Streamlit Web 封装
=================================
商用标准形态，参赛核心要求。

直接复用 rockfall 核心库 (零逻辑重写):
  - RockDetector  (detector.py)     — MOG2+YOLO+SORT 检测流水线
  - AlertStore    (alert_store.py)  — 预警记录持久化
  - FastSAM       (fastsam_road.py) — 道路/边坡分割
  - site_config   (site_config.py)  — 多监测点位管理

启动: streamlit run app.py
"""

import sys
import time
import csv
import io
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── 确保 rockfall 包可导入 ──────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
THIS_DIR = _THIS_DIR  # 公开别名，供 views/ 页面模块使用
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ── API 地址收口: 本地开发用 localhost, Streamlit Cloud 用环境变量 ──
def build_api_url(path: str) -> str:
    """构建 API 完整 URL。

    优先级:
      1. API_BASE_URL 环境变量 (Streamlit Cloud / 远程部署)
      2. localhost:API_PORT (本地开发)
    """
    import os
    base = os.getenv("API_BASE_URL", "")
    if base:
        return base.rstrip("/") + path
    port = os.getenv("API_PORT", "8000")
    return f"http://localhost:{port}{path}"

# ── 依赖可用性探测 (Streamlit Cloud 可能缺失 torch/ultralytics) ──
IMPORT_ERRORS: list[str] = []
ROCKFALL_AVAILABLE = True

try:
    from rockfall.detector import RockDetector
except ImportError as e:
    IMPORT_ERRORS.append(f"rockfall.detector: {e}")
    RockDetector = None  # type: ignore

try:
    from rockfall.alert_store import AlertStore, get_alert_store
except ImportError as e:
    IMPORT_ERRORS.append(f"rockfall.alert_store: {e}")
    AlertStore = None  # type: ignore
    get_alert_store = None  # type: ignore

try:
    from rockfall.performance import PerformanceMonitor, get_device_info
except ImportError as e:
    IMPORT_ERRORS.append(f"rockfall.performance: {e}")
    PerformanceMonitor = None  # type: ignore
    get_device_info = None  # type: ignore

try:
    from rockfall.replay import generate_alert_replays, stitch_annotated_clip
except ImportError:
    generate_alert_replays = None  # type: ignore
    stitch_annotated_clip = None  # type: ignore

try:
    from rockfall import __version__ as _core_version
except ImportError:
    _core_version = "0.0.0"

try:
    from rockfall.site_config import (
        list_sites, get_active_site, set_active_site,
        get_site_state, get_active_site_name, get_active_location,
        PRESET_SITES, MonitoringSite,
    )
except ImportError as e:
    IMPORT_ERRORS.append(f"rockfall.site_config: {e}")
    list_sites = lambda: []  # type: ignore
    get_active_site = None  # type: ignore
    set_active_site = None  # type: ignore
    get_site_state = None  # type: ignore
    get_active_site_name = lambda: "未知"  # type: ignore
    get_active_location = lambda: "未知"  # type: ignore
    PRESET_SITES = {}  # type: ignore

try:
    from rockfall.hazard_store import (
        HazardPoint, HazardStore, SlopeStability, HistoricalIncident,
        get_hazard_store, RISK_LEVELS, HAZARD_STATUSES,
        calculate_risk_score, determine_risk_level, generate_hazard_report,
    )
except ImportError as e:
    IMPORT_ERRORS.append(f"rockfall.hazard_store: {e}")
    HazardPoint = None  # type: ignore
    HazardStore = None  # type: ignore
    SlopeStability = None  # type: ignore
    HistoricalIncident = None  # type: ignore
    get_hazard_store = None  # type: ignore
    RISK_LEVELS = {}  # type: ignore
    HAZARD_STATUSES = {}  # type: ignore
    calculate_risk_score = None  # type: ignore
    determine_risk_level = None  # type: ignore
    generate_hazard_report = None  # type: ignore
    MonitoringSite = None  # type: ignore

try:
    from rockfall.alert_classifier import (
        get_response_workflow, classify_alert_level,
        LEVEL_LABELS as _ALERT_CLASSIFIER_LABELS,
    )
except ImportError as e:
    IMPORT_ERRORS.append(f"rockfall.alert_classifier: {e}")
    get_response_workflow = None  # type: ignore
    classify_alert_level = None  # type: ignore
    _ALERT_CLASSIFIER_LABELS = {}  # type: ignore

try:
    from rockfall.config import (
        RESULTS_DIR, DATA_DIR, UPLOADS_DIR,
        DETECTION_CONFIDENCE, DETECTION_IMG_SIZE,
        ALERT_BLUE_CONFIDENCE_LOW, ALERT_BLUE_CONFIDENCE_HIGH,
        ALERT_YELLOW_CONFIDENCE_HIGH, ALERT_ORANGE_CONFIDENCE_HIGH,
        MOTION_MIN_AREA, MOTION_SCORE_LOW, MOTION_SCORE_HIGH,
        SKIP_IDLE, SKIP_ACTIVE, SKIP_CRITICAL,
        MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_LEARNING_RATE,
        DENSITY_ALERT_ENABLED, DENSITY_WINDOW_SEC,
        DENSITY_BURST_ZSCORE, DENSITY_CONF_FLOOR, DENSITY_MIN_SAMPLES,
        validate_config,
        CLASS_NAMES,
        get_device as config_get_device,
    )
except ImportError as e:
    IMPORT_ERRORS.append(f"rockfall.config: {e}")
    # 提供安全的默认值
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parent
    RESULTS_DIR = _ROOT / "data" / "results"
    DATA_DIR = _ROOT / "data"
    UPLOADS_DIR = _ROOT / "data" / "uploads"
    DETECTION_CONFIDENCE = 0.3
    DETECTION_IMG_SIZE = 640
    ALERT_BLUE_CONFIDENCE_LOW = 0.3
    ALERT_BLUE_CONFIDENCE_HIGH = 0.5
    ALERT_YELLOW_CONFIDENCE_HIGH = 0.7
    ALERT_ORANGE_CONFIDENCE_HIGH = 0.9
    MOTION_MIN_AREA = 500
    MOTION_SCORE_LOW = 0.001
    MOTION_SCORE_HIGH = 0.01
    SKIP_IDLE = 3
    SKIP_ACTIVE = 1
    SKIP_CRITICAL = 1
    MOG2_HISTORY = 500
    MOG2_VAR_THRESHOLD = 16
    MOG2_LEARNING_RATE = 0.01
    DENSITY_ALERT_ENABLED = True
    DENSITY_WINDOW_SEC = 15
    DENSITY_BURST_ZSCORE = 2.5
    DENSITY_CONF_FLOOR = 0.10
    DENSITY_MIN_SAMPLES = 300
    validate_config = lambda: []  # type: ignore
    CLASS_NAMES = {0: "rock"}
    config_get_device = lambda: ("cpu", "CPU (后备)")  # type: ignore

if IMPORT_ERRORS:
    ROCKFALL_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# 注意: st.set_page_config() 必须在 app.py 中最先调用
# 此处已移至 app.py 入口文件
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 品牌 & 版本信息
# ══════════════════════════════════════════════════════════════

APP_NAME = "RockGuard"
APP_VERSION = f"v{_core_version}"
APP_SUBTITLE = "公路自然灾害监测预警平台"
TEAM_NAME = "落石卫士团队"
COPYRIGHT = "© 2026 RockGuard. 保留所有权利。"

# ══════════════════════════════════════════════════════════════
# 样式 & 配色 (科技蓝主色调)
# ══════════════════════════════════════════════════════════════

PRIMARY_BLUE = "#1565C0"
PRIMARY_BLUE_LIGHT = "#E3F2FD"
DARK_BG = "#0D1B2A"
SURFACE_BG = "#F5F7FA"
TEXT_PRIMARY = "#1B2838"
TEXT_SECONDARY = "#5F6B7A"

ALERT_COLORS = {
    "red":    "#D32F2F",
    "orange": "#E65100",
    "yellow": "#F9A825",
    "blue":   "#1565C0",
    "green":  "#2E7D32",
}

ALERT_BG = {
    "red":    "#FFEBEE",
    "orange": "#FFF3E0",
    "yellow": "#FFFDE7",
    "blue":   "#E3F2FD",
    "green":  "#E8F5E9",
}

ALERT_LABELS = {
    "red":    "I 级 · 特别严重",
    "orange": "II 级 · 严重",
    "yellow": "III 级 · 较重",
    "blue":   "IV 级 · 一般",
    "green":  "正常",
}

ALERT_ICONS = {
    "red": "●", "orange": "●", "yellow": "●", "blue": "●", "green": "●",
}

ALERT_ORDER = {"green": 0, "blue": 1, "yellow": 2, "orange": 3, "red": 4}

RISK_LABELS = {"high": "高风险", "medium": "中风险", "low": "低风险"}

# ══════════════════════════════════════════════════════════════
# 会话状态默认值 & 初始化
# ══════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    "detection_confidence": DETECTION_CONFIDENCE,
    "detection_img_size": DETECTION_IMG_SIZE,
    "motion_min_area": MOTION_MIN_AREA,
    "alert_blue_low": ALERT_BLUE_CONFIDENCE_LOW,
    "alert_blue_high": ALERT_BLUE_CONFIDENCE_HIGH,
    "alert_yellow_high": ALERT_YELLOW_CONFIDENCE_HIGH,
    "alert_orange_high": ALERT_ORANGE_CONFIDENCE_HIGH,
    "motion_score_low": MOTION_SCORE_LOW,
    "motion_score_high": MOTION_SCORE_HIGH,
    "skip_idle": SKIP_IDLE,
    "skip_active": SKIP_ACTIVE,
    "skip_critical": SKIP_CRITICAL,
    "mog2_history": MOG2_HISTORY,
    "mog2_var_threshold": MOG2_VAR_THRESHOLD,
    "mog2_learning_rate": MOG2_LEARNING_RATE,
    "active_site_id": "",
}
