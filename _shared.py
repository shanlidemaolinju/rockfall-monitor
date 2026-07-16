"""
RockGuard — Streamlit 共享基础设施
=====================================

本模块从子模块 re-export 所有符号:
  _config.py — 配置常量 / 依赖探测 / 品牌配色
  _styles.py — CSS 样式注入
  _cache.py — 资源缓存 / 会话初始化

并保留: render_sidebar / update_perf_dashboard / DEMO_SCENES / demo 辅助函数
"""

import streamlit as st
from pathlib import Path

# ── 模块自身路径 (不依赖 _config 的导出) ──
_THIS_DIR = Path(__file__).resolve().parent
THIS_DIR = _THIS_DIR  # 公开别名，供 views/ 页面模块使用

# ── Re-export 所有子模块符号 ──
from _config import *
from _styles import *
from _cache import *

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
            "导航",
            ["预设演示", "实时监测", "多路监控", "算法亮点", "极端场景", "预警标准", "预警记录", "隐患点排查", "点位管理", "参数设置", "系统管理"],
            label_visibility="collapsed",
            format_func=lambda x: f"    {x}",
        )

        st.divider()

        # ── 主题切换 ──
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False
        dark = st.checkbox("🌙 暗色模式", value=st.session_state.dark_mode,
                          help="切换亮色/暗色主题")

        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        if st.session_state.dark_mode:
            st.markdown("""
            <style>
            /* 暗色主题 — 对齐 FastAPI dashboard.html 配色 */
            .stApp { background: #0d1117; }
            .stSidebar { background: #161b22; }
            .stSidebar [data-testid="stMarkdownContainer"] * { color: #c9d1d9 !important; }
            .stSidebar .stRadio label, .stSidebar .stCheckbox label { color: #c9d1d9 !important; }
            .stMain h1, .stMain h2, .stMain h3, .stMain h4 { color: #c9d1d9; }
            .stMain p, .stMain span, .stMain div { color: #c9d1d9; }
            .stMetric label { color: #8b949e !important; }
            .stMetric [data-testid="stMetricValue"] { color: #c9d1d9 !important; }
            [data-testid="stExpander"] { background: #161b22; border-color: #30363d; }
            .stButton button { background: #21262d; color: #c9d1d9; border-color: #30363d; }
            .stDataFrame { background: #161b22; }
            </style>
            """, unsafe_allow_html=True)

        st.divider()

        # ── 底部信息 ──
        st.markdown(f"""
        <div style="font-size:0.7rem;color:{TEXT_SECONDARY};">
            {APP_NAME} {APP_VERSION}<br>
            {TEAM_NAME}<br>
            {COPYRIGHT}
        </div>
        """, unsafe_allow_html=True)

    return page

# ══════════════════════════════════════════════════════════════
# 性能仪表盘渲染
# ══════════════════════════════════════════════════════════════

def update_perf_dashboard(placeholders: dict, detail_placeholder, snap) -> None:
    """更新实时性能仪表盘 (由进度回调触发)"""
    # FPS
    fps_color = "#2E7D32" if snap.fps >= 15 else ("#E65100" if snap.fps >= 5 else "#D32F2F")
    placeholders["fps"].markdown(f"""
    <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
        <div style="font-size:1.4rem;font-weight:700;color:{fps_color};">{snap.fps:.1f}</div>
        <div style="font-size:0.65rem;color:#5F6B7A;">FPS</div>
    </div>
    """, unsafe_allow_html=True)

    # 推理耗时
    placeholders["inference"].markdown(f"""
    <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
        <div style="font-size:1.4rem;font-weight:700;color:#1B2838;">{snap.inference_ms_avg:.0f}<span style="font-size:0.7rem;">ms</span></div>
        <div style="font-size:0.65rem;color:#5F6B7A;">推理耗时 (平均)</div>
    </div>
    """, unsafe_allow_html=True)

    # GPU 利用率
    if snap.gpu_available:
        gpu_color = "#2E7D32" if snap.gpu_utilization < 80 else ("#E65100" if snap.gpu_utilization < 95 else "#D32F2F")
        placeholders["gpu_util"].markdown(f"""
        <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
            <div style="font-size:1.4rem;font-weight:700;color:{gpu_color};">{snap.gpu_utilization:.0f}<span style="font-size:0.7rem;">%</span></div>
            <div style="font-size:0.65rem;color:#5F6B7A;">GPU 利用率</div>
        </div>
        """, unsafe_allow_html=True)
    elif snap.torch_gpu_available:
        placeholders["gpu_util"].markdown(f"""
        <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
            <div style="font-size:1.1rem;font-weight:700;color:#1565C0;">{snap.torch_gpu_name[:8]}</div>
            <div style="font-size:0.65rem;color:#5F6B7A;">GPU (Torch)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        placeholders["gpu_util"].markdown(f"""
        <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
            <div style="font-size:1.1rem;font-weight:700;color:#9E9E9E;">N/A</div>
            <div style="font-size:0.65rem;color:#5F6B7A;">GPU (none)</div>
        </div>
        """, unsafe_allow_html=True)

    # GPU 显存
    if snap.gpu_available:
        placeholders["gpu_mem"].markdown(f"""
        <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
            <div style="font-size:1.4rem;font-weight:700;color:#1B2838;">{snap.gpu_memory_used_mb:.0f}<span style="font-size:0.7rem;">MB</span></div>
            <div style="font-size:0.65rem;color:#5F6B7A;">显存占用 / {snap.gpu_memory_total_mb:.0f}MB</div>
        </div>
        """, unsafe_allow_html=True)
    elif snap.torch_gpu_available:
        placeholders["gpu_mem"].markdown(f"""
        <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
            <div style="font-size:1.4rem;font-weight:700;color:#1B2838;">{snap.torch_memory_allocated_mb:.0f}<span style="font-size:0.7rem;">MB</span></div>
            <div style="font-size:0.65rem;color:#5F6B7A;">Torch 显存</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        placeholders["gpu_mem"].markdown(f"""
        <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
            <div style="font-size:1.1rem;font-weight:700;color:#9E9E9E;">--</div>
            <div style="font-size:0.65rem;color:#5F6B7A;">显存 (N/A)</div>
        </div>
        """, unsafe_allow_html=True)

    # CPU
    cpu_color = "#2E7D32" if snap.cpu_percent < 60 else ("#E65100" if snap.cpu_percent < 85 else "#D32F2F")
    placeholders["cpu"].markdown(f"""
    <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
        <div style="font-size:1.4rem;font-weight:700;color:{cpu_color};">{snap.cpu_percent:.0f}<span style="font-size:0.7rem;">%</span></div>
        <div style="font-size:0.65rem;color:#5F6B7A;">CPU</div>
    </div>
    """, unsafe_allow_html=True)

    # 内存
    ram_color = "#2E7D32" if snap.memory_percent < 60 else ("#E65100" if snap.memory_percent < 85 else "#D32F2F")
    placeholders["ram"].markdown(f"""
    <div style="text-align:center;padding:0.5rem 0.3rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
        <div style="font-size:1.4rem;font-weight:700;color:{ram_color};">{snap.memory_percent:.0f}<span style="font-size:0.7rem;">%</span></div>
        <div style="font-size:0.65rem;color:#5F6B7A;">内存</div>
    </div>
    """, unsafe_allow_html=True)

    # 详情行
    detail_placeholder.markdown(f"""
    <div style="padding:0.4rem 0.75rem;background:#F5F7FA;border-radius:6px;margin-top:0.3rem;
                font-size:0.72rem;color:#5F6B7A;display:flex;gap:1.5rem;flex-wrap:wrap;">
        <span>已处理: <b style="color:#1B2838;">{snap.total_frames_processed}</b> 帧</span>
        <span>已用时间: <b style="color:#1B2838;">{snap.elapsed_seconds:.1f}s</b></span>
        <span>预警: <b style="color:#D32F2F;">{snap.total_alerts}</b></span>
        <span>进程内存: <b style="color:#1B2838;">{snap.process_memory_mb:.0f}MB</b></span>
        <span style="font-size:0.65rem;">监控开销: {snap.monitor_overhead_ms:.1f}ms</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 模块 0: 预设演示 (零等待)
# ══════════════════════════════════════════════════════════════

# 预定义演示场景 (与站点配置对应)
# 源视频全部来自钦州落石试验现场 + 宜宾滑坡新闻视频
# 添加新场景: python scripts/generate_all_demos.py <video> --scene <scene_id>
DEMO_SCENES = {
    "qinzhou_demo": {
        "title": "钦州落石试验演示",
        "subtitle": "钦州公路边坡现场落石试验 — 4K 高速摄像机",
        "icon": "City",
        "data_dir": "demo_data/nanning_naan_s1",
        "site_id": "qinzhou_s0",
        "source_video": "VID_20230304_153140.mp4",
        "tags": ["钦州", "落石试验", "4K", "示范"],
    },
    "qinzhou_cam_3": {
        "title": "钦州监测视频 3",
        "subtitle": "钦州公路边坡实时监测 — 3.flv",
        "icon": "Camera",
        "data_dir": "demo_data/qinzhou_cam_3",
        "site_id": "qinzhou_s0",
        "source_video": "3.flv",
        "tags": ["钦州", "监测摄像头", "实时"],
    },
    "qinzhou_test_a": {
        "title": "钦州落石试验 A",
        "subtitle": "钦州现场落石试验 — VID_20230304_114247.mp4",
        "icon": "Experiment",
        "data_dir": "demo_data/baise_s1",
        "site_id": "qinzhou_s0",
        "source_video": "VID_20230304_114247.mp4",
        "tags": ["钦州", "落石试验", "现场"],
    },
    "qinzhou_test_b": {
        "title": "钦州落石试验 B",
        "subtitle": "钦州现场落石试验 — VID_20230304_153140.mp4",
        "icon": "Experiment",
        "data_dir": "demo_data/qinzhou_s1",
        "site_id": "qinzhou_s0",
        "source_video": "VID_20230304_153140.mp4",
        "tags": ["钦州", "落石试验", "现场"],
    },
    # ── 以下场景已有视频源，待 GPU 生成 demo 数据 ──
    "qinzhou_cam_1": {
        "title": "钦州监测视频 1",
        "subtitle": "钦州公路边坡实时监测 — 1.flv",
        "icon": "Camera",
        "data_dir": "demo_data/qinzhou_cam_1",
        "site_id": "qinzhou_s0",
        "source_video": "1.flv",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_cam_4": {
        "title": "钦州监测视频 4",
        "subtitle": "钦州公路边坡实时监测 — 4.flv",
        "icon": "Camera",
        "data_dir": "demo_data/qinzhou_cam_4",
        "site_id": "qinzhou_s0",
        "source_video": "4.flv",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_cam_5": {
        "title": "钦州监测视频 5",
        "subtitle": "钦州公路边坡实时监测 — 5.flv",
        "icon": "Camera",
        "data_dir": "demo_data/qinzhou_cam_5",
        "site_id": "qinzhou_s0",
        "source_video": "5.flv",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_cam_6": {
        "title": "钦州监测视频 6",
        "subtitle": "钦州公路边坡实时监测 — 6.flv",
        "icon": "Camera",
        "data_dir": "demo_data/qinzhou_cam_6",
        "site_id": "qinzhou_s0",
        "source_video": "6.flv",
        "tags": ["钦州", "监测摄像头", "待生成"],
    },
    "qinzhou_test_c": {
        "title": "钦州落石试验 C",
        "subtitle": "钦州现场落石试验 — VID_20230304131810.mp4",
        "icon": "Experiment",
        "data_dir": "demo_data/qinzhou_test_c",
        "site_id": "qinzhou_s0",
        "source_video": "VID_20230304131810.mp4",
        "tags": ["钦州", "落石试验", "待生成"],
    },
    "qinzhou_test_d": {
        "title": "钦州落石试验 D",
        "subtitle": "钦州现场落石试验 — VID_20230304132003.mp4",
        "icon": "Experiment",
        "data_dir": "demo_data/qinzhou_test_d",
        "site_id": "qinzhou_s0",
        "source_video": "VID_20230304132003.mp4",
        "tags": ["钦州", "落石试验", "待生成"],
    },
    "qinzhou_test_e": {
        "title": "钦州落石试验 E",
        "subtitle": "钦州现场落石试验 — VID_20230304132126.mp4",
        "icon": "Experiment",
        "data_dir": "demo_data/qinzhou_test_e",
        "site_id": "qinzhou_s0",
        "source_video": "VID_20230304132126.mp4",
        "tags": ["钦州", "落石试验", "待生成"],
    },
    "yibin_s1": {
        "title": "宜宾 G85 渝昆高速滑坡",
        "subtitle": "四川盆地南缘 — 前兆小落石→红色预警→大规模崩塌 (43秒)",
        "icon": "Mountain",
        "data_dir": "demo_data/yibin_s1",
        "site_id": "yibin_s1",
        "source_video": "3.7日，四川宜宾一高速路段发生山体滑坡.mp4",
        "tags": ["宜宾", "滑坡", "前兆预警", "红色升级"],
    },
}

def load_demo_summary(scene_id: str) -> dict | None:
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

def load_demo_result(scene_id: str) -> dict | None:
    """加载预生成的演示检测结果 (含轨迹数据)"""
    import json as _json
    scene = DEMO_SCENES.get(scene_id)
    if not scene:
        return None
    result_path = _THIS_DIR / scene["data_dir"] / "result.json"
    if not result_path.exists():
        return None
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None

def find_demo_for_site(active_site) -> dict | None:
    """将 MonitoringSite 映射到 DEMO_SCENES 中的真实 demo 数据。

    优先精确匹配 site_id，失败时按前缀匹配（如 qinzhou_s0 → qinzhou_test_a）。
    返回 (scene_id, summary_dict) 或 None。
    """
    if active_site is None:
        return None
    site_id = getattr(active_site, "site_id", "")
    # 1) 精确匹配
    if site_id in DEMO_SCENES:
        summary = load_demo_summary(site_id)
        if summary:
            return (site_id, summary)
    # 2) 前缀匹配（去掉尾部 _sN 后缀）
    prefix = site_id.rsplit("_", 1)[0] if "_" in site_id else site_id
    for scene_id in DEMO_SCENES:
        if scene_id == prefix or scene_id.startswith(prefix + "_"):
            summary = load_demo_summary(scene_id)
            if summary:
                return (scene_id, summary)
    return None
