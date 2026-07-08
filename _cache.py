"""RockGuard — 资源缓存 & 会话初始化"""
import streamlit as st
from _config import *

# ══════════════════════════════════════════════════════════════
# 资源缓存 (Streamlit 全局单例)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_detector(site_id: str = "") -> RockDetector | None:
    """加载 YOLO 模型, 返回 RockDetector 实例。模型不存在时返回 None。

    site_id: 监测点位 ID，用于加载点位级阈值配置。
             空字符串 = 使用全局默认阈值。
    """
    if RockDetector is None:
        st.error("RockDetector 未能导入 — 请检查依赖安装: `pip install ultralytics torch`")
        return None
    try:
        return RockDetector(site_id=site_id)
    except FileNotFoundError as e:
        st.error(f"模型加载失败: {e}")
        return None
    except Exception as e:
        st.error(f"检测器初始化失败: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_store() -> AlertStore | None:
    """获取 AlertStore 单例 (自动探测 MySQL/SQLite 后端)。"""
    if get_alert_store is None:
        st.error("AlertStore 未能导入 — 请检查依赖安装")
        return None
    return get_alert_store()

def _get_active_site_id() -> str:
    """安全地获取当前活跃点位 ID，失败时返回空字符串（使用全局默认）。"""
    try:
        site = get_active_site()
        return site.site_id if site else ""
    except Exception:
        return ""

def get_detector_or_stop() -> RockDetector:
    """获取检测器, 若不可用则 st.stop()。自动加载当前点位阈值。"""
    d = get_detector(site_id=_get_active_site_id())
    if d is None:
        st.error("检测器未就绪, 请检查模型文件后刷新页面。")
        st.stop()
    return d

def cleanup_stream_frames():
    """清理上一轮检测的标注帧文件, 避免与新结果混淆。"""
    try:
        for f in RESULTS_DIR.glob("stream_*.jpg"):
            f.unlink(missing_ok=True)
    except Exception:
        pass


def init_session_state():
    """初始化 Streamlit session_state 默认值。在 app.py 中 st.set_page_config() 之后调用。"""
    for k, v in DEFAULT_PARAMS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "detection_results" not in st.session_state:
        st.session_state.detection_results = None
    if "detection_running" not in st.session_state:
        st.session_state.detection_running = False
    if "last_detection_source" not in st.session_state:
        st.session_state.last_detection_source = ""

    # 生成 Streamlit 会话 ID（所有日志/告警可溯源，注意：不暴露给外部 API）
    if "session_id" not in st.session_state:
        from rockfall.trace import set_session_id, get_session_id
        st.session_state.session_id = set_session_id()

