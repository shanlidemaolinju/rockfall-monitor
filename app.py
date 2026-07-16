"""
落石监测系统 — Streamlit Web 封装
=================================
RockGuard v2.3.0 · 公路落石灾害监测预警系统

启动: streamlit run app.py

架构:
  app.py            — 入口 + 路由 (本文件)
  _shared.py        — 共享基础设施 (CSS, 会话, 缓存, 侧边栏)
  views/            — 11 个功能页面模块
    demo_showcase.py       — 预设演示
    realtime_monitor.py    — 实时监测
    multi_camera.py        — 多路监控
    algorithm_showcase.py  — 算法亮点
    extreme_scenarios.py   — 极端场景
    alert_standards.py     — 预警标准
    alert_records.py       — 预警记录
    hazard_investigation.py— 隐患点排查
    site_management.py     — 点位管理
    settings.py            — 参数设置
    system.py              — 系统管理
"""

import sys
from pathlib import Path

# ── 确保 rockfall 包可导入 ──
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import streamlit as st

# ══════════════════════════════════════════════════════════════
# 页面配置 — 必须是第一个 Streamlit 命令
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RockGuard v2.3 — 公路落石灾害监测预警系统",
    page_icon=":rock:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 导入共享基础设施
# ══════════════════════════════════════════════════════════════
from _shared import (
    # 版本 & 品牌
    APP_NAME, APP_VERSION, APP_SUBTITLE, TEAM_NAME, COPYRIGHT,
    # 依赖状态
    IMPORT_ERRORS, ROCKFALL_AVAILABLE,
    # 初始化函数
    inject_css, init_session_state,
    # 侧边栏
    render_sidebar,
    # 配置验证
    validate_config,
    # 配置变量 (供页面使用)
    DEFAULT_PARAMS,
)

# ══════════════════════════════════════════════════════════════
# 导入页面模块
# ══════════════════════════════════════════════════════════════
from views.demo_showcase import page_demo_showcase
from views.realtime_monitor import page_realtime_monitor
from views.multi_camera import page_multi_camera
from views.algorithm_showcase import page_algorithm_showcase
from views.extreme_scenarios import page_extreme_scenarios
from views.alert_standards import page_alert_standards
from views.alert_records import page_alert_records
from views.hazard_investigation import page_hazard_investigation
from views.site_management import page_site_management
from views.settings import page_settings
from views.system import page_system

# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════


def main():
    """Streamlit 主入口 — 按侧边栏选择渲染对应页面。"""

    # ── 初始化共享资源 ──
    inject_css()
    init_session_state()

    # ── 启动时检查依赖可用性 ──
    if IMPORT_ERRORS:
        st.error("## 依赖缺失 — 应用无法正常启动")
        st.markdown("""
        以下 Python 包未能成功导入。请检查 `requirements.txt` 是否完整安装：
        """)
        for err in IMPORT_ERRORS:
            st.code(err, language=None)
        st.info("""
        **如何修复 (Streamlit Cloud):**
        1. 确保 `requirements.txt` 包含 `torch`, `ultralytics`, `opencv-python-headless`
        2. 确保 `packages.txt` 包含 `ffmpeg`, `libgomp1`, `libgl1`, `libglib2.0-0t64`
        3. 重新部署 (Streamlit Cloud 会在每次 push 后自动重新安装依赖)

        **如何修复 (本地):**
        ```bash
        pip install -r requirements.txt
        ```
        """)
        st.stop()

    # 启动时检查配置
    warnings = validate_config()
    if warnings:
        with st.sidebar:
            with st.expander("配置警告", expanded=True):
                for w in warnings:
                    st.warning(w)

    # 渲染侧边栏 + 获取当前页面
    page = render_sidebar()

    # 路由到各页面
    if "预设演示" in page:
        page_demo_showcase()
    elif "实时监测" in page:
        page_realtime_monitor()
    elif "多路监控" in page:
        page_multi_camera()
    elif "算法亮点" in page:
        page_algorithm_showcase()
    elif "极端场景" in page:
        page_extreme_scenarios()
    elif "预警标准" in page:
        page_alert_standards()
    elif "预警记录" in page:
        page_alert_records()
    elif "隐患点排查" in page:
        page_hazard_investigation()
    elif "点位管理" in page:
        page_site_management()
    elif "参数设置" in page:
        page_settings()
    elif "系统管理" in page:
        page_system()


if __name__ == "__main__":
    main()
