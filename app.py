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
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ── 确保 rockfall 包可导入 ──────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from rockfall.detector import RockDetector
from rockfall.alert_store import AlertStore, get_alert_store
from rockfall.site_config import (
    list_sites, get_active_site, set_active_site,
    get_site_state, get_active_site_name, get_active_location,
    PRESET_SITES, MonitoringSite,
)
from rockfall.config import (
    RESULTS_DIR, DATA_DIR, UPLOADS_DIR,
    DETECTION_CONFIDENCE, DETECTION_IMG_SIZE,
    ALERT_BLUE_CONFIDENCE_LOW, ALERT_BLUE_CONFIDENCE_HIGH,
    ALERT_YELLOW_CONFIDENCE_HIGH, ALERT_ORANGE_CONFIDENCE_HIGH,
    MOTION_MIN_AREA, MOTION_SCORE_LOW, MOTION_SCORE_HIGH,
    SKIP_IDLE, SKIP_ACTIVE, SKIP_CRITICAL,
    MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_LEARNING_RATE,
    validate_config,
    CLASS_NAMES,
    get_device as config_get_device,
)

# ══════════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RockGuard — 公路落石灾害监测预警系统",
    page_icon="::rock::",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 品牌 & 版本信息
# ══════════════════════════════════════════════════════════════

APP_NAME = "RockGuard"
APP_VERSION = "v2.0.0"
APP_SUBTITLE = "公路自然灾害监测预警平台"
TEAM_NAME = "RockGuard Team"
COPYRIGHT = "© 2026 RockGuard. All rights reserved."

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

st.markdown(f"""
<style>
    /* === 全局 === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; color: {TEXT_PRIMARY}; }}

    /* === 品牌顶栏 === */
    .brand-header {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.75rem 1.25rem;
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, #0D47A1 100%);
        border-radius: 10px; color: #fff; margin-bottom: 1rem;
    }}
    .brand-header .logo {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.5px; }}
    .brand-header .meta {{ font-size: 0.75rem; opacity: 0.85; text-align: right; }}
    .brand-header .meta span {{ margin-left: 1rem; }}

    /* === 预警等级标签 === */
    .alert-badge {{
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
        font-size: 0.78rem; font-weight: 600;
    }}
    .alert-badge.red    {{ background: {ALERT_BG['red']};    color: {ALERT_COLORS['red']}; }}
    .alert-badge.orange {{ background: {ALERT_BG['orange']}; color: {ALERT_COLORS['orange']}; }}
    .alert-badge.yellow {{ background: {ALERT_BG['yellow']}; color: #F57F17; }}
    .alert-badge.blue   {{ background: {ALERT_BG['blue']};   color: {ALERT_COLORS['blue']}; }}
    .alert-badge.green  {{ background: {ALERT_BG['green']};  color: {ALERT_COLORS['green']}; }}

    /* === 卡片容器 === */
    .card {{
        background: #fff; border: 1px solid #E3E8EF; border-radius: 10px;
        padding: 1.25rem; margin-bottom: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .card.active {{ border-color: {PRIMARY_BLUE}; box-shadow: 0 0 0 1px {PRIMARY_BLUE}20; }}

    /* === KPI 指标卡 === */
    .kpi-card {{
        background: #fff; border: 1px solid #E3E8EF; border-radius: 10px;
        padding: 1rem 1.25rem; text-align: center;
    }}
    .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {PRIMARY_BLUE}; line-height: 1.2; }}
    .kpi-value.danger {{ color: {ALERT_COLORS['red']}; }}
    .kpi-value.warning {{ color: {ALERT_COLORS['orange']}; }}
    .kpi-label {{ font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 0.25rem; }}

    /* === 场景选择卡 === */
    .scene-card {{
        padding: 1rem; border-radius: 10px; border: 2px solid #E3E8EF;
        background: #fff; margin-bottom: 0.5rem; transition: all 0.15s;
    }}
    .scene-card:hover {{ border-color: {PRIMARY_BLUE}60; }}
    .scene-card.selected {{ border-color: {PRIMARY_BLUE}; background: {PRIMARY_BLUE_LIGHT}; }}

    /* === 状态指示器 === */
    .status-dot {{
        display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;
    }}
    .status-dot.live {{ background: #4CAF50; animation: pulse 2s infinite; }}
    .status-dot.idle {{ background: #9E9E9E; }}
    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

    /* === 数据表格 === */
    .dataframe-container {{ border-radius: 8px; overflow: hidden; }}

    /* === 分割线 === */
    hr.divider {{ border: none; border-top: 1px solid #E3E8EF; margin: 1.5rem 0; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 资源缓存 (Streamlit 全局单例)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_detector() -> RockDetector | None:
    """加载 YOLO 模型, 返回 RockDetector 实例。模型不存在时返回 None。"""
    try:
        return RockDetector()
    except FileNotFoundError as e:
        st.error(f"模型加载失败: {e}")
        return None
    except Exception as e:
        st.error(f"检测器初始化失败: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_store() -> AlertStore:
    """获取 AlertStore 单例 (自动探测 MySQL/SQLite 后端)。"""
    return get_alert_store()


def get_detector_or_stop() -> RockDetector:
    """获取检测器, 若不可用则 st.stop()。"""
    d = get_detector()
    if d is None:
        st.error("检测器未就绪, 请检查模型文件后刷新页面。")
        st.stop()
    return d


def _cleanup_stream_frames():
    """清理上一轮检测的标注帧文件, 避免与新结果混淆。"""
    try:
        for f in RESULTS_DIR.glob("stream_*.jpg"):
            f.unlink(missing_ok=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# 会话状态初始化
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

for k, v in DEFAULT_PARAMS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False
if "last_detection_source" not in st.session_state:
    st.session_state.last_detection_source = ""


# ══════════════════════════════════════════════════════════════
# 侧边栏 — 系统信息
# ══════════════════════════════════════════════════════════════

def render_sidebar():
    """渲染侧边栏: 品牌标识 + 系统状态 + 导航"""
    with st.sidebar:
        # ── 品牌标识 ──
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:0.25rem 0 0.75rem 0;">
            <div style="width:36px;height:36px;border-radius:8px;
                        background:linear-gradient(135deg,{PRIMARY_BLUE},#0D47A1);
                        display:flex;align-items:center;justify-content:center;
                        color:#fff;font-weight:700;font-size:1.1rem;">R</div>
            <div>
                <div style="font-weight:700;font-size:1.1rem;color:{TEXT_PRIMARY};">{APP_NAME}</div>
                <div style="font-size:0.7rem;color:{TEXT_SECONDARY};">{APP_SUBTITLE}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 设备状态指示 ──
        device_str, device_name = config_get_device()
        is_gpu = device_str.startswith("cuda")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:0.5rem 0.75rem;
                    background:{SURFACE_BG};border-radius:8px;margin-bottom:0.75rem;
                    font-size:0.8rem;">
            <span class="status-dot {'live' if is_gpu else 'idle'}"></span>
            <span style="color:{TEXT_SECONDARY};">推理设备</span>
            <span style="font-weight:600;color:{TEXT_PRIMARY};">{device_name[:24]}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── 当前点位 ──
        try:
            site = get_active_site()
            st.markdown(f"""
            <div style="padding:0.5rem 0.75rem;background:{PRIMARY_BLUE_LIGHT};border-radius:8px;
                        border-left:3px solid {PRIMARY_BLUE};margin-bottom:0.75rem;">
                <div style="font-size:0.7rem;color:{TEXT_SECONDARY};">监测点位</div>
                <div style="font-weight:600;font-size:0.9rem;color:{PRIMARY_BLUE};">{site.name}</div>
                <div style="font-size:0.72rem;color:{TEXT_SECONDARY};">{site.region}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

        # ── 今日统计 ──
        try:
            store = get_store()
            today = store.count_today_by_level()
            total_today = sum(today.values())
            if total_today > 0:
                st.markdown(f"""<div style="font-size:0.7rem;color:{TEXT_SECONDARY};
                    text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.25rem;">
                    今日预警统计 &middot; {total_today} 条</div>""", unsafe_allow_html=True)
                cols = st.columns(4)
                for i, (lvl, color) in enumerate([
                    ("red", ALERT_COLORS["red"]), ("orange", ALERT_COLORS["orange"]),
                    ("yellow", ALERT_COLORS["yellow"]), ("blue", ALERT_COLORS["blue"]),
                ]):
                    count = today.get(lvl, 0)
                    cols[i].markdown(f"""
                    <div style="text-align:center;">
                        <div style="font-size:1.1rem;font-weight:700;color:{color};">{count}</div>
                        <div style="font-size:0.6rem;color:{TEXT_SECONDARY};">{lvl.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass

        st.divider()

        # ── 导航 ──
        page = st.radio(
            "",
            ["Preset Demo", "Live Detection", "Alert Records", "Site Manager", "Settings"],
            label_visibility="collapsed",
            format_func=lambda x: {
                "Preset Demo": "    Preset Demo",
                "Live Detection": "    Live Detection",
                "Alert Records": "    Alert Records",
                "Site Manager": "    Site Manager",
                "Settings": "    Settings",
            }[x],
        )

        st.divider()

        # ── 底部信息 ──
        st.markdown(f"""
        <div style="font-size:0.7rem;color:{TEXT_SECONDARY};">
            {APP_NAME} {APP_VERSION}<br>
            {TEAM_NAME}<br>
            {COPYRIGHT}
        </div>
        """, unsafe_allow_html=True)

        # 页面映射 (英文 → 中文 key)
        page_map = {
            "Preset Demo": "预设演示",
            "Live Detection": "实时监测",
            "Alert Records": "预警记录",
            "Site Manager": "点位管理",
            "Settings": "参数设置",
        }

    return page_map[page]


# ══════════════════════════════════════════════════════════════
# 模块 0: 预设演示 (零等待)
# ══════════════════════════════════════════════════════════════

# 预定义演示场景 (与站点配置对应)
DEMO_SCENES = {
    "nanning_naan_s1": {
        "title": "南宁那安快速路 1 号边坡",
        "subtitle": "广西首府核心路段 — 晴天日间落石检测",
        "icon": "🏙️",
        "data_dir": "demo_data/nanning_naan_s1",
        "site_id": "nanning_naan_s1",
    },
}


def _load_demo_summary(scene_id: str) -> dict | None:
    """加载预生成的演示摘要数据"""
    import json as _json
    scene = DEMO_SCENES.get(scene_id)
    if not scene:
        return None
    summary_path = _THIS_DIR / scene["data_dir"] / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def page_demo_showcase():
    """预设演示页面: 预计算结果零等待加载"""
    # ── 品牌顶栏 ──
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">{APP_NAME}</div>
            <div style="font-size:0.8rem;opacity:0.85;">Preset Demo &middot; GPU Pre-computed</div>
        </div>
        <div class="meta">
            <span>{APP_VERSION}</span><span>{TEAM_NAME}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 加载演示数据 ──
    available = []
    for sid, scene in DEMO_SCENES.items():
        summary = _load_demo_summary(sid)
        if summary is not None:
            available.append((sid, scene, summary))

    if not available:
        st.warning("Demo data not found. Run: python scripts/generate_demo.py")
        return

    if "demo_scene" not in st.session_state:
        st.session_state.demo_scene = available[0][0]

    active_sid = st.session_state.demo_scene
    active_scene = DEMO_SCENES.get(active_sid)
    active_summary = _load_demo_summary(active_sid)

    if not active_scene or not active_summary:
        return

    video = active_summary.get("video", {})
    detection = active_summary.get("detection", {})
    alerts = active_summary.get("alerts", {})
    key_frames = active_summary.get("key_frames", [])
    total_alerts = max(alerts.get("total_alert_frames", 1), 1)

    # ── 第一行: 场景信息 + KPI 仪表盘 ──
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.75rem;">
        <div style="width:4px;height:24px;background:{PRIMARY_BLUE};border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">Scenario</div>
        <div class="alert-badge green" style="font-size:0.7rem;">READY</div>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns([2, 3])

    with c_left:
        st.markdown(f"""
        <div class="scene-card selected">
            <div style="font-weight:600;font-size:0.95rem;color:{TEXT_PRIMARY};">{active_scene['title']}</div>
            <div style="font-size:0.78rem;color:{TEXT_SECONDARY};margin-top:0.25rem;">{active_scene['subtitle']}</div>
            <div style="margin-top:0.5rem;font-size:0.75rem;color:{TEXT_SECONDARY};">
                Video: {video.get('file','')} &middot; {video.get('resolution','')} &middot; {video.get('fps',0)} fps
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{video.get('total_frames', 0):,}</div>
                <div class="kpi-label">Total Frames</div></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{detection.get('elapsed_sec', 0):.1f}s</div>
                <div class="kpi-label">Inference Time</div></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{total_alerts}</div>
                <div class="kpi-label">Alert Frames</div></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{ALERT_COLORS['red']};">{alerts.get('red', 0)}</div>
                <div class="kpi-label">Level I (Red)</div></div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{detection.get('device', 'GPU')[:16]}</div>
                <div class="kpi-label">Device</div></div>""", unsafe_allow_html=True)

    # ── 第二行: 预警等级分布 ──
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1.25rem 0 0.75rem 0;">
        <div style="width:4px;height:24px;background:{PRIMARY_BLUE};border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">Alert Distribution</div>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_legend = st.columns([3, 1])

    with col_chart:
        chart_data = pd.DataFrame({
            "Level": ["I · Red", "II · Orange", "III · Yellow", "IV · Blue"],
            "Frames": [
                alerts.get("red", 0), alerts.get("orange", 0),
                alerts.get("yellow", 0), alerts.get("blue", 0),
            ],
        })
        chart_data = chart_data[chart_data["Frames"] > 0]
        st.bar_chart(
            chart_data.set_index("Level"),
            use_container_width=True,
            color=PRIMARY_BLUE,
        )

    with col_legend:
        for lvl, color in [("red", ALERT_COLORS["red"]), ("orange", ALERT_COLORS["orange"]),
                            ("yellow", ALERT_COLORS["yellow"]), ("blue", ALERT_COLORS["blue"])]:
            count = alerts.get(lvl, 0)
            pct = count / total_alerts * 100 if total_alerts > 0 else 0
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;">
                <div style="width:12px;height:12px;border-radius:3px;background:{color};"></div>
                <div style="flex:1;font-size:0.82rem;">{ALERT_LABELS[lvl]}</div>
                <div style="font-weight:600;font-size:0.9rem;">{count}</div>
                <div style="font-size:0.72rem;color:{TEXT_SECONDARY};">{pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ── 第三行: 关键帧查看器 ──
    if key_frames:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin:1.25rem 0 0.75rem 0;">
            <div style="width:4px;height:24px;background:{PRIMARY_BLUE};border-radius:2px;"></div>
            <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">Key Frame Viewer</div>
            <div style="font-size:0.78rem;color:{TEXT_SECONDARY};">{len(key_frames)} frames</div>
        </div>
        """, unsafe_allow_html=True)

        if "demo_frame_idx" not in st.session_state:
            st.session_state.demo_frame_idx = 0

        # 主图 + 控制
        kf = key_frames[st.session_state.demo_frame_idx]
        frame_path = _THIS_DIR / active_scene["data_dir"] / kf["thumbnail"]
        lvl = kf["alert_level"]

        c_left, c_right = st.columns([4, 1])

        with c_left:
            if frame_path.exists():
                st.image(str(frame_path), use_container_width=True)

            # 缩略图条
            cols = st.columns(min(len(key_frames), 15))
            for i, kf_th in enumerate(key_frames[:15]):
                fp_th = _THIS_DIR / active_scene["data_dir"] / kf_th["thumbnail"]
                with cols[i]:
                    is_current = i == st.session_state.demo_frame_idx
                    if fp_th.exists():
                        st.image(str(fp_th), use_container_width=True)
                        if is_current:
                            st.markdown(f"""<div style="height:2px;background:{PRIMARY_BLUE};
                                border-radius:1px;margin-top:-8px;"></div>""", unsafe_allow_html=True)

        with c_right:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:0.7rem;color:{TEXT_SECONDARY};text-transform:uppercase;">Frame Info</div>
                <div style="font-size:1.5rem;font-weight:700;color:{TEXT_PRIMARY};margin:0.25rem 0;">#{kf['frame_idx']}</div>
                <div><span class="alert-badge {lvl}">{ALERT_LABELS.get(lvl, lvl)}</span></div>
                <div style="margin-top:0.75rem;font-size:0.82rem;">
                    <div>Confidence <b style="float:right;">{kf['max_confidence']:.3f}</b></div>
                    <div>Targets <b style="float:right;">{kf['track_count']}</b></div>
                    <div>Timestamp <b style="float:right;">{kf['time_sec']:.1f}s</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.button("Previous", key="demo_prev", use_container_width=True,
                      disabled=st.session_state.demo_frame_idx == 0,
                      on_click=lambda: st.session_state.update(
                          demo_frame_idx=max(0, st.session_state.demo_frame_idx - 1)))
            st.button("Next", key="demo_next", use_container_width=True,
                      disabled=st.session_state.demo_frame_idx >= len(key_frames) - 1,
                      on_click=lambda: st.session_state.update(
                          demo_frame_idx=min(len(key_frames) - 1, st.session_state.demo_frame_idx + 1)))

        # 滑块
        st.slider("", 0, len(key_frames) - 1, st.session_state.demo_frame_idx,
                  key="demo_slider", label_visibility="collapsed",
                  on_change=lambda: st.session_state.update(demo_frame_idx=st.session_state.demo_slider))


# ══════════════════════════════════════════════════════════════
# 模块 1: 实时监测
# ══════════════════════════════════════════════════════════════

def page_realtime_monitor():
    """实时监测页面: 上传视频 → 检测 → 结果显示"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">Live Detection</div>
            <div style="font-size:0.8rem;opacity:0.85;">Upload video &middot; CPU inference &middot; Real-time results</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    detector = get_detector_or_stop()
    store = get_store()

    # ── 输入区域 ──
    col1, col2 = st.columns([3, 2])

    with col1:
        video_file = st.file_uploader(
            "上传视频文件",
            type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
            help="支持常见视频格式, 建议分辨率 ≥ 720p",
        )

    with col2:
        camera_url = st.text_input(
            "或输入摄像头/RTSP地址",
            value="",
            placeholder="rtsp://... 或 0 (USB摄像头)",
            help="留空则使用上传的视频文件; 摄像头模式支持自动重连",
        )

    is_file_mode = bool(video_file)
    is_live_mode = bool(camera_url)

    # ── 检测控制 ──
    c1, c2 = st.columns([1, 3])

    with c1:
        save_frames_flag = st.checkbox("保存标注帧", value=True,
                                       help="将标注后的帧保存到 results 目录")
    with c2:
        push_alerts_flag = st.checkbox("推送预警", value=True,
                                       help="触发预警时通过 PushPlus 推送微信消息; "
                                            "需在 .env 中配置 PUSHPLUS_TOKEN")

    # ── 演示模式参数 (CPU 优化) ──
    with st.expander("⚡ 演示模式 (CPU 加速)", expanded=True):
        st.caption("Streamlit Cloud 为纯 CPU 环境, 限制帧数保证演示速度。")
        c1, c2, c3 = st.columns(3)
        with c1:
            demo_max_frames = st.slider(
                "最大推理帧数", min_value=30, max_value=500, value=150, step=10,
                help="最多处理的帧数, 越小越快。演示建议 100-200 帧",
            )
        with c2:
            demo_stride = st.slider(
                "帧采样步长", min_value=1, max_value=10, value=3, step=1,
                help="每隔 N 帧处理 1 帧 (1=全部处理, 3=隔3取1)。值越大越快",
            )
        with c3:
            demo_img_size = st.selectbox(
                "推理分辨率", options=[320, 416, 640], index=0,
                help="320 最快, 640 最精确。CPU 建议 320",
            )

    start_btn = st.button("▶ 开始检测", type="primary", use_container_width=True,
                          disabled=(not video_file and not camera_url))

    # ── 执行检测 ──
    if start_btn:
        source_path = ""
        source_name = ""

        if is_file_mode:
            # 保存上传视频到 uploads 目录
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            source_path = str(UPLOADS_DIR / video_file.name)
            with open(source_path, "wb") as f:
                f.write(video_file.read())
            source_name = video_file.name

            # 清理旧的标注帧 (避免与上一轮结果混淆)
            _cleanup_stream_frames()
        else:
            source_path = camera_url
            source_name = camera_url

        st.session_state.last_detection_source = source_name

        start_time = time.time()

        if is_file_mode:
            # ── 临时覆盖推理尺寸 (演示加速) ──
            _orig_img_size = detector.img_size
            detector.img_size = demo_img_size

            # ── 进度条 ──
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _progress_cb(current: int, total: int):
                if total > 0:
                    progress_bar.progress(min(current / total, 1.0))
                status_text.text(f"🔍 推理中... 第 {current} 帧" + (f" / {total}" if total else ""))

            # ── 文件模式: detect_video() 一次性处理 ──
            with st.spinner(f"🔍 正在检测 `{source_name}` ..."):
                result = detector.detect_video(
                    source_path,
                    save_frames=save_frames_flag,
                    push_alerts=push_alerts_flag,
                    track=True,
                    max_frames=demo_max_frames,
                    stride=demo_stride,
                    progress_callback=_progress_cb,
                )

            # 恢复原始设置
            detector.img_size = _orig_img_size
            progress_bar.progress(1.0)
            status_text.text("✅ 检测完成")

            elapsed = time.time() - start_time

            if isinstance(result, dict) and "error" not in result:
                all_frame_results = result.get("detections", [])
                alert_frames = [
                    fr for fr in all_frame_results
                    if fr.get("alert_level", "green") != "green"
                ]
                total_frames = result.get("total_frames", len(all_frame_results))
                fps = result.get("fps", 25.0)

                st.session_state.detection_results = {
                    "source": source_name,
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "elapsed_seconds": round(elapsed, 1),
                    "alert_frames": alert_frames,
                    "all_frames": all_frame_results,
                    "mode": "file",
                }
                st.success(f"✅ 检测完成 — 耗时 {elapsed:.1f}s, "
                          f"共 {total_frames} 帧, "
                          f"{len(alert_frames)} 帧触发预警")
            else:
                st.error(f"检测失败: {result}")
                st.session_state.detection_results = None

        else:
            # ── 摄像头模式: detect_stream() 逐帧产出 ──
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            frame_placeholder = st.empty()

            all_frame_results = []
            alert_frames = []
            frame_idx = 0

            try:
                gen = detector.detect_stream(
                    source=source_path,
                    source_name=source_name,
                    save_frames=save_frames_flag,
                    push_alerts=push_alerts_flag,
                    track=True,
                    is_live=True,
                )

                for frame_result in gen:
                    all_frame_results.append(frame_result)
                    frame_idx = frame_result["frame_idx"]
                    alert_level = frame_result.get("alert_level", "green")
                    tracks = frame_result.get("tracks", [])

                    # 渐进进度 (摄像头无总帧数, 用伪进度)
                    pct = min(frame_idx / max(frame_idx + 50, 1), 0.95)
                    progress_bar.progress(pct)

                    n_tracks = len(tracks)
                    status_text = f"📍 帧 {frame_idx}"
                    if n_tracks > 0:
                        status_text += f" | 🪨 {n_tracks} 目标"
                    if alert_level != "green":
                        status_text += f" | {ALERT_LABELS.get(alert_level, alert_level)}"
                        alert_frames.append(frame_result)
                    status_placeholder.text(status_text)

                    # 显示最新标注帧
                    if save_frames_flag:
                        frame_path = RESULTS_DIR / f"stream_{frame_idx:06d}.jpg"
                        if frame_path.exists():
                            frame_placeholder.image(
                                str(frame_path),
                                caption=f"F{frame_idx} | {ALERT_LABELS.get(alert_level, alert_level)}",
                                use_container_width=True,
                            )

            except Exception as e:
                st.error(f"检测过程出错: {e}")
                import traceback
                st.code(traceback.format_exc())

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_placeholder.success(
                f"✅ 检测完成 — 耗时 {elapsed:.1f}s, 共 {len(all_frame_results)} 帧"
            )

            st.session_state.detection_results = {
                "source": source_name,
                "total_frames": len(all_frame_results),
                "fps": 25.0,
                "elapsed_seconds": round(elapsed, 1),
                "alert_frames": alert_frames,
                "all_frames": all_frame_results,
                "mode": "live",
            }

    # ── 显示检测结果 ──
    results = st.session_state.detection_results
    if results is None:
        st.info("👆 请上传视频文件或输入摄像头地址, 然后点击「开始检测」。")
        return

    st.divider()
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <div style="width:4px;height:24px;background:{PRIMARY_BLUE};border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">Detection Report</div>
    </div>
    """, unsafe_allow_html=True)

    # 统计卡片
    total = results["total_frames"]
    alert_count = len(results["alert_frames"])
    alert_ratio = (alert_count / total * 100) if total > 0 else 0

    level_counts = {"red": 0, "orange": 0, "yellow": 0, "blue": 0, "green": 0}
    for fr in results["all_frames"]:
        lvl = fr.get("alert_level", "green")
        if lvl in level_counts:
            level_counts[lvl] += 1

    # ── KPI 行 ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Frames", total)
    c2.metric("Alert Frames", alert_count, delta=f"{alert_ratio:.1f}%" if alert_count > 0 else None)
    c3.metric("Level I (Red)", level_counts["red"])
    c4.metric("Level II (Orange)", level_counts["orange"])
    c5.metric("Level III (Yellow)", level_counts["yellow"])

    if alert_count > 0:
        st.divider()
        chart_data = pd.DataFrame({
            "Level": ["I · Red", "II · Orange", "III · Yellow", "IV · Blue"],
            "Frames": [
                level_counts["red"], level_counts["orange"],
                level_counts["yellow"], level_counts["blue"],
            ],
        })
        chart_data = chart_data[chart_data["Frames"] > 0]
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.bar_chart(chart_data.set_index("Level"), use_container_width=True)
        with col_b:
            if results["alert_frames"]:
                tl_data = []
                for fr in results["alert_frames"]:
                    tl_data.append({
                        "Frame": fr["frame_idx"],
                        "Time (s)": fr.get("time_sec", fr["frame_idx"] / max(results.get("fps", 25), 1)),
                        "Level": fr.get("alert_level", "yellow"),
                        "Targets": len(fr.get("tracks", [])),
                    })
                tl_df = pd.DataFrame(tl_data)
                st.scatter_chart(tl_df.set_index("Time (s)")[["Targets"]], use_container_width=True)
                st.caption("Alert timeline: X = time (s), Y = detected targets")

    # 预警帧图库
    if alert_count > 0 and save_frames_flag:
        st.divider()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
            <div style="width:4px;height:24px;background:{PRIMARY_BLUE};border-radius:2px;"></div>
            <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">Alert Frame Gallery</div>
        </div>
        """, unsafe_allow_html=True)

        # 只显示有预警的帧, 最多 20 张
        show_frames = results["alert_frames"][:20]
        cols_per_row = 4
        for i in range(0, len(show_frames), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, fr in enumerate(show_frames[i:i + cols_per_row]):
                frame_idx = fr["frame_idx"]
                frame_path = RESULTS_DIR / f"stream_{frame_idx:06d}.jpg"
                if frame_path.exists():
                    lvl = fr.get("alert_level", "yellow")
                    n_tracks = len(fr.get("tracks", []))
                    cols[j].image(
                        str(frame_path),
                        caption=f"F{frame_idx} | {ALERT_LABELS.get(lvl, lvl)} | {n_tracks}目标",
                        use_container_width=True,
                    )

    # 导出
    st.divider()
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("📥 导出检测报告 (CSV)", use_container_width=True):
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["帧号", "时间(秒)", "预警等级", "目标数", "Track IDs", "最高置信度"])
            for fr in results["all_frames"]:
                tracks = fr.get("tracks", [])
                track_ids = ",".join(str(t["id"]) for t in tracks)
                max_conf = max((t.get("confidence", 0) for t in tracks), default=0)
                writer.writerow([
                    fr["frame_idx"],
                    fr.get("time_sec", 0),
                    fr.get("alert_level", "green"),
                    len(tracks),
                    track_ids,
                    round(max_conf, 4),
                ])
            st.download_button(
                "💾 下载 CSV",
                csv_buffer.getvalue(),
                file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 预警记录摘要
    with c2:
        try:
            recent = store.get_recent(limit=20)
            if recent:
                st.caption(f"📋 最近 {len(recent)} 条预警记录:")
                summary = []
                for r in recent[:10]:
                    lvl = r.get("alert_level", "yellow")
                    summary.append({
                        "时间": r.get("time", ""),
                        "等级": ALERT_LABELS.get(lvl, lvl),
                        "数量": r.get("count", 0),
                        "置信度": r.get("max_confidence", 0),
                    })
                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# 模块 2: 预警记录
# ══════════════════════════════════════════════════════════════

def page_alert_records():
    """预警记录页面: 查询、筛选、导出历史预警"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">Alert Records</div>
            <div style="font-size:0.8rem;opacity:0.85;">History &middot; Filter &middot; Export</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    store = get_store()

    # ── 筛选条件 ──
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])

    with col1:
        today = datetime.now().date()
        date_range = st.date_input(
            "日期范围",
            value=(today - timedelta(days=7), today),
            help="选择起止日期 (含首尾)",
        )

    with col2:
        alert_filter = st.multiselect(
            "预警等级",
            options=["red", "orange", "yellow", "blue"],
            default=["red", "orange", "yellow", "blue"],
            format_func=lambda x: ALERT_LABELS.get(x, x),
            help="留空 = 全部等级",
        )

    with col3:
        page_size = st.selectbox("每页条数", [20, 50, 100, 200], index=1)

    with col4:
        st.write("")  # 对齐
        export_btn = st.button("📥 导出当前筛选结果", use_container_width=True)

    # ── 今日统计卡片 ──
    today_counts = store.count_today_by_level()
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    tc1.metric("Level I (Red)", today_counts.get("red", 0))
    tc2.metric("Level II (Orange)", today_counts.get("orange", 0))
    tc3.metric("Level III (Yellow)", today_counts.get("yellow", 0))
    tc4.metric("Level IV (Blue)", today_counts.get("blue", 0))
    tc5.metric("Total Today", sum(today_counts.values()))

    # ── 查询 ──
    start_str = date_range[0].strftime("%Y-%m-%d") if len(date_range) > 0 else ""
    end_str = date_range[1].strftime("%Y-%m-%d") if len(date_range) > 1 else ""

    # 分页
    if "alert_page" not in st.session_state:
        st.session_state.alert_page = 0

    offset = st.session_state.alert_page * page_size

    all_rows = []
    total_count = 0

    if len(alert_filter) == 0:
        # 无筛选 → 空结果
        pass
    elif len(alert_filter) == 1:
        rows = store.query_alerts(
            start_date=start_str, end_date=end_str,
            alert_level=alert_filter[0], limit=page_size, offset=offset,
        )
        total_count = store.count_alerts(start_date=start_str, end_date=end_str, alert_level=alert_filter[0])
        all_rows = rows
    else:
        # 多个等级 → 分别查询并合并
        for lvl in alert_filter:
            rows = store.query_alerts(
                start_date=start_str, end_date=end_str,
                alert_level=lvl, limit=page_size * 2, offset=0,
            )
            all_rows.extend(rows)
        # 按时间降序排列
        all_rows.sort(key=lambda r: r.get("time", ""), reverse=True)
        total_count = len(all_rows)
        # 手动分页
        all_rows = all_rows[offset:offset + page_size]

    # ── 数据表格 ──
    if all_rows:
        df_data = []
        for r in all_rows:
            lvl = r.get("alert_level", "")
            df_data.append({
                "ID": r.get("id", ""),
                "时间": r.get("time", ""),
                "预警等级": ALERT_LABELS.get(lvl, lvl),
                "数量": r.get("count", 0),
                "最高置信度": round(r.get("max_confidence", 0), 4),
                "落石直径(cm)": r.get("rock_diameter_cm", 0),
                "监测点位": r.get("monitoring_location", ""),
                "推送状态": r.get("push_status", ""),
            })

        st.dataframe(
            pd.DataFrame(df_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "最高置信度": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        # 分页控制
        total_pages = max(1, (total_count + page_size - 1) // page_size)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅ 上一页", disabled=(st.session_state.alert_page == 0)):
                st.session_state.alert_page = max(0, st.session_state.alert_page - 1)
                st.rerun()
        with c2:
            st.caption(f"第 {st.session_state.alert_page + 1} / {total_pages} 页 (共 {total_count} 条)")
        with c3:
            if st.button("下一页 ➡", disabled=(st.session_state.alert_page >= total_pages - 1)):
                st.session_state.alert_page = min(total_pages - 1, st.session_state.alert_page + 1)
                st.rerun()

        # ── 导出 ──
        if export_btn:
            # 导出全部筛选结果 (不受分页限制)
            export_rows = []
            if len(alert_filter) == 1:
                export_rows = store.query_alerts(
                    start_date=start_str, end_date=end_str,
                    alert_level=alert_filter[0], limit=100000,
                )
            elif len(alert_filter) > 1:
                for lvl in alert_filter:
                    export_rows.extend(store.query_alerts(
                        start_date=start_str, end_date=end_str,
                        alert_level=lvl, limit=100000,
                    ))
                export_rows.sort(key=lambda r: r.get("time", ""), reverse=True)

            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow([
                "ID", "时间", "预警等级", "数量", "最高置信度",
                "Track IDs", "类别摘要", "保存帧", "推送状态",
                "落石直径(cm)", "监测点位", "创建时间",
            ])
            for r in export_rows:
                writer.writerow([
                    r.get("id", ""),
                    r.get("time", ""),
                    r.get("alert_level", ""),
                    r.get("count", 0),
                    r.get("max_confidence", 0),
                    r.get("track_ids", ""),
                    r.get("class_summary", ""),
                    r.get("saved_frame", ""),
                    r.get("push_status", ""),
                    r.get("rock_diameter_cm", 0),
                    r.get("monitoring_location", ""),
                    r.get("created_at", ""),
                ])

            st.download_button(
                "💾 下载 CSV",
                csv_buffer.getvalue(),
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("📭 当前筛选条件下无预警记录。")


# ══════════════════════════════════════════════════════════════
# 模块 3: 点位管理
# ══════════════════════════════════════════════════════════════

def page_site_management():
    """点位管理页面: 查看/切换监测点位"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">Site Manager</div>
            <div style="font-size:0.8rem;opacity:0.85;">4 preset monitoring sites &middot; Guangxi + ASEAN region</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    active_site = get_active_site()
    all_sites = list_sites()

    # ── 当前激活点位 ──
    st.subheader("Active Site")
    with st.container():
        _render_site_card(active_site, is_active=True, show_detail=True)

    # ── 系统配置验证 ──
    st.divider()
    with st.expander("🔍 系统配置检查", expanded=False):
        warnings = validate_config()
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("✅ 所有配置项正常")

    # ── 全部预设点位 ──
    st.divider()
    st.subheader("Preset Sites")
    st.caption(f"{len(all_sites)} sites available. Click 'Activate' to switch.")

    cols = st.columns(2)
    for i, site in enumerate(all_sites):
        is_active = site.site_id == active_site.site_id
        with cols[i % 2]:
            _render_site_card(site, is_active=is_active, show_detail=True)

            if not is_active:
                if st.button(
                    "Activate This Site",
                    key=f"switch_{site.site_id}",
                    use_container_width=True,
                ):
                    try:
                        new_site = set_active_site(site.site_id)
                        st.success(f"Activated: {new_site.name}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    # ── ROI 配置 ──
    st.divider()
    st.subheader("ROI Calibration")

    from rockfall.site_config import load_site_config
    roi_params, polygon, road_mask = load_site_config(active_site.site_id)

    if roi_params is not None:
        st.success(f"✅ 已有 ROI 标定数据 (最近校准: {active_site.site_id})")
        st.json({
            "sat_max": roi_params.sat_max,
            "val_min": roi_params.val_min,
            "val_max": roi_params.val_max,
            "morph_close": roi_params.morph_close,
            "morph_open": roi_params.morph_open,
            "min_area_ratio": roi_params.min_area_ratio,
        })
        if polygon is not None:
            st.caption(f"ROI 多边形顶点数: {len(polygon)}")
        if road_mask is not None:
            st.caption(f"道路掩膜尺寸: {road_mask.shape}")
    else:
        st.info("📝 该点位尚未进行 ROI 标定, 将使用默认 ROI 区域。")


def _render_site_card(site: MonitoringSite, is_active: bool = False, show_detail: bool = False):
    """渲染单个点位卡片"""
    border_style = "2px solid #0d6efd" if is_active else "1px solid #dee2e6"
    bg_style = "#f0f7ff" if is_active else "#ffffff"

    with st.container():
        st.markdown(f"""
        <div style="padding:1rem; border-radius:8px; border:{border_style}; background:{bg_style}; margin-bottom:0.5rem;">
            <b>{'✅ ' if is_active else ''}{site.name}</b>
            <span style="float:right;">{RISK_LABELS.get(site.risk_level, site.risk_level)}</span>
            <br><small>📍 {site.region} | 🛣️ {site.highway} | 🏷️ {site.stake_mark}</small>
            <br><small>📝 {site.description}</small>
            <br><small>🌐 经纬度: {site.latitude:.3f}, {site.longitude:.3f}</small>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 模块 4: 参数设置
# ══════════════════════════════════════════════════════════════

def page_settings():
    """参数设置页面: 调整检测和预警阈值"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">Settings</div>
            <div style="font-size:0.8rem;opacity:0.85;">Detection thresholds &middot; Alert levels &middot; Frame strategy</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    detector = get_detector_or_stop()

    # ── 检测参数 ──
    st.subheader("🎯 检测参数")

    c1, c2 = st.columns(2)
    with c1:
        new_conf = st.slider(
            "检测置信度阈值",
            min_value=0.1, max_value=0.9, value=float(DETECTION_CONFIDENCE), step=0.05,
            help="YOLO 检测的最低置信度, 低于此值的检测框被丢弃",
        )
        new_img_size = st.selectbox(
            "推理图像尺寸",
            options=[320, 416, 480, 640, 800, 960, 1280],
            index=3,  # 640
            help="YOLO 推理时的输入尺寸, 越大越精确但越慢",
        )
    with c2:
        new_min_area = st.slider(
            "最小运动区域 (像素)",
            min_value=50, max_value=2000, value=MOTION_MIN_AREA, step=50,
            help="MOG2 运动检测的最小连通区域面积",
        )
        new_mog2_lr = st.slider(
            "MOG2 学习率",
            min_value=0.0001, max_value=0.1, value=MOG2_LEARNING_RATE, step=0.0005, format="%.4f",
            help="背景模型更新速度, 越小越稳定但适应变化越慢",
        )

    # ── 四级预警阈值 ──
    st.divider()
    st.subheader("🚨 四级预警置信度阈值")
    st.caption("对齐《公路自然灾害监测预警系统技术指南》第5.3节强制要求。")

    c1, c2 = st.columns(2)
    with c1:
        blue_low = st.slider(
            "🔵 Ⅳ级(蓝色)下限",
            min_value=0.1, max_value=0.5, value=float(ALERT_BLUE_CONFIDENCE_LOW), step=0.05,
            help="置信度 ≥ 此值 → Ⅳ级预警",
        )
        yellow_high = st.slider(
            "🟡 Ⅲ级(黄色)上限",
            min_value=0.4, max_value=0.8, value=float(ALERT_YELLOW_CONFIDENCE_HIGH), step=0.05,
            help="置信度 ≥ 此值 → Ⅱ级预警",
        )
    with c2:
        blue_high = st.slider(
            "🔵→🟡 蓝黄分界",
            min_value=0.2, max_value=0.6, value=float(ALERT_BLUE_CONFIDENCE_HIGH), step=0.05,
            help="置信度 ≥ 此值 → Ⅲ级预警",
        )
        orange_high = st.slider(
            "🟠 Ⅱ级(橙色)上限",
            min_value=0.7, max_value=0.99, value=float(ALERT_ORANGE_CONFIDENCE_HIGH), step=0.05,
            help="置信度 ≥ 此值 → Ⅰ级预警",
        )

    # ── 跳帧策略 ──
    st.divider()
    st.subheader("⏩ 自适应跳帧策略")
    st.caption("基于运动强度自动调整推理频率, 平衡实时性与算力消耗。")

    c1, c2, c3 = st.columns(3)
    with c1:
        new_skip_idle = st.slider(
            "静止跳帧 (每N帧推理1次)",
            min_value=1, max_value=30, value=SKIP_IDLE, step=1,
            help="无运动时的跳帧间隔, 越大越省算力",
        )
    with c2:
        new_skip_active = st.slider(
            "弱运动跳帧",
            min_value=1, max_value=15, value=SKIP_ACTIVE, step=1,
            help="弱运动时的跳帧间隔",
        )
    with c3:
        new_skip_critical = st.slider(
            "强运动跳帧",
            min_value=1, max_value=5, value=SKIP_CRITICAL, step=1,
            help="强运动时的跳帧间隔, 越小检测越密集",
        )

    new_motion_low = st.slider(
        "运动得分阈值",
        min_value=0.001, max_value=0.2, value=(float(MOTION_SCORE_LOW), float(MOTION_SCORE_HIGH)),
        step=0.005, format="%.3f",
        help="(低阈值, 高阈值): 低=静止→弱运动分界, 高=弱运动→强运动分界",
    )

    # ── 应用按钮 ──
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("✅ 应用参数", type="primary", use_container_width=True):
            # 更新检测器实例参数
            detector.confidence = new_conf
            detector.img_size = new_img_size
            detector.min_area = new_min_area
            detector.alert_blue_conf_high = blue_high
            detector.alert_yellow_conf_high = yellow_high
            detector.alert_orange_conf_high = orange_high

            # 更新会话状态
            st.session_state.detection_confidence = new_conf
            st.session_state.detection_img_size = new_img_size
            st.session_state.motion_min_area = new_min_area
            st.session_state.alert_blue_low = blue_low
            st.session_state.alert_blue_high = blue_high
            st.session_state.alert_yellow_high = yellow_high
            st.session_state.alert_orange_high = orange_high
            st.session_state.skip_idle = new_skip_idle
            st.session_state.skip_active = new_skip_active
            st.session_state.skip_critical = new_skip_critical
            st.session_state.motion_score_low = new_motion_low[0]
            st.session_state.motion_score_high = new_motion_low[1]
            st.session_state.mog2_learning_rate = new_mog2_lr

            st.success("✅ 参数已应用 (当前会话有效)")

    with c2:
        if st.button("🔄 恢复默认", use_container_width=True):
            for k, v in DEFAULT_PARAMS.items():
                st.session_state[k] = v
            # 恢复检测器参数
            detector.confidence = DEFAULT_PARAMS["detection_confidence"]
            detector.img_size = DEFAULT_PARAMS["detection_img_size"]
            detector.min_area = DEFAULT_PARAMS["motion_min_area"]
            detector.alert_blue_conf_high = DEFAULT_PARAMS["alert_blue_high"]
            detector.alert_yellow_conf_high = DEFAULT_PARAMS["alert_yellow_high"]
            detector.alert_orange_conf_high = DEFAULT_PARAMS["alert_orange_high"]
            st.rerun()

    # ── 当前配置状态 ──
    with st.expander("📋 当前完整配置", expanded=False):
        st.json({
            "detection": {
                "confidence": detector.confidence,
                "img_size": detector.img_size,
                "min_area": detector.min_area,
            },
            "alert_thresholds": {
                "blue_low": blue_low,
                "blue_high": detector.alert_blue_conf_high,
                "yellow_high": detector.alert_yellow_conf_high,
                "orange_high": detector.alert_orange_conf_high,
            },
            "skip_strategy": {
                "idle": st.session_state.get("skip_idle", SKIP_IDLE),
                "active": st.session_state.get("skip_active", SKIP_ACTIVE),
                "critical": st.session_state.get("skip_critical", SKIP_CRITICAL),
            },
            "device": config_get_device(),
        })


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def main():
    """Streamlit 主入口 — 按侧边栏选择渲染对应页面。"""

    # 启动时检查配置
    warnings = validate_config()
    if warnings:
        with st.sidebar:
            with st.expander("⚠️ 配置警告", expanded=True):
                for w in warnings:
                    st.warning(w)

    # 渲染侧边栏 + 获取当前页面
    page = render_sidebar()

    # 路由到各页面
    if "预设演示" in page:
        page_demo_showcase()
    elif "实时监测" in page:
        page_realtime_monitor()
    elif "预警记录" in page:
        page_alert_records()
    elif "点位管理" in page:
        page_site_management()
    elif "参数设置" in page:
        page_settings()


if __name__ == "__main__":
    main()
