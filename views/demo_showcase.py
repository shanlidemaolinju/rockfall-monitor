"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from pathlib import Path
from _shared import *

# 直接计算项目根路径，不依赖 _shared 的 _ROOT 导出
_ROOT = Path(__file__).resolve().parent.parent

def page_demo_showcase():
    """预设演示页面: 预计算结果零等待加载"""
    # ── 品牌顶栏 ──
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">{APP_NAME}</div>
            <div style="font-size:0.8rem;opacity:0.85;">预设演示 &middot; FastSAM ROI v2 &middot; 2026-07-11 03:25 UTC</div>
        </div>
        <div class="meta">
            <span>{APP_VERSION}</span><span>{TEAM_NAME}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 加载演示数据 ──
    available = []
    unavailable = []
    for sid, scene in DEMO_SCENES.items():
        summary = load_demo_summary(sid)
        if summary is not None:
            available.append((sid, scene, summary))
        else:
            unavailable.append((sid, scene))

    if not available:
        st.warning("未找到演示数据，请用以下命令生成:")
        st.code("python scripts/generate_all_demos.py /path/to/video.mp4 --scene qinzhou_demo")
        st.markdown("""
        **已注册但未生成的场景** (上传对应视频后运行 `scripts/generate_all_demos.py`):
        """)
        if unavailable:
            for sid, scene in unavailable:
                st.markdown(f"- **{scene['title']}** — _{scene['subtitle']}_ → `{scene['data_dir']}/`")
        else:
            st.info("请在场景配置中注册场景，然后运行生成脚本。")
        return

    # ── 场景选择器 (多场景时显示) ──
    if "demo_scene" not in st.session_state:
        st.session_state.demo_scene = available[0][0]

    if len(available) > 1:
        scene_options = {sid: f"{sc['title']} — {sc['subtitle']}" for sid, sc, _ in available}
        selected = st.selectbox(
            "选择演示场景",
            options=list(scene_options.keys()),
            format_func=lambda x: scene_options[x],
            index=list(scene_options.keys()).index(st.session_state.demo_scene)
            if st.session_state.demo_scene in scene_options else 0,
        )
        st.session_state.demo_scene = selected
    else:
        # 单场景时显示标签
        sid, sc, _ = available[0]
        tags_html = " ".join(
            f'<span style="display:inline-block;padding:0.1rem 0.5rem;background:{PRIMARY_BLUE_LIGHT};'
            f'border-radius:4px;font-size:0.7rem;color:{PRIMARY_BLUE};">{t}</span>'
            for t in sc.get("tags", [])
        )
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;">
            <span style="font-size:0.78rem;color:{TEXT_SECONDARY};">当前场景:</span>
            <span style="font-weight:600;color:{TEXT_PRIMARY};">{sc['title']}</span>
            {tags_html}
        </div>
        """, unsafe_allow_html=True)

    # ── 显示未生成场景的提示 ──
    if unavailable:
        with st.expander(f"+ {len(unavailable)} 个场景待生成数据", expanded=False):
            for sid, sc in unavailable:
                st.markdown(
                    f"- **{sc['title']}**: `python scripts/generate_all_demos.py <video> --scene {sid}`"
                )

    active_sid = st.session_state.demo_scene
    active_scene = DEMO_SCENES.get(active_sid)
    active_summary = load_demo_summary(active_sid)

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
        <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">场景</div>
        <div class="alert-badge green" style="font-size:0.7rem;">就绪</div>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns([2, 3])

    with c_left:
        st.markdown(f"""
        <div class="scene-card selected">
            <div style="font-weight:600;font-size:0.95rem;color:{TEXT_PRIMARY};">{active_scene['title']}</div>
            <div style="font-size:0.78rem;color:{TEXT_SECONDARY};margin-top:0.25rem;">{active_scene['subtitle']}</div>
            <div style="margin-top:0.5rem;font-size:0.75rem;color:{TEXT_SECONDARY};">
                视频: {video.get('file','')} &middot; {video.get('resolution','')} &middot; {video.get('fps',0)} fps
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{video.get('total_frames', 0):,}</div>
                <div class="kpi-label">总帧数</div></div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{detection.get('elapsed_sec', 0):.1f}s</div>
                <div class="kpi-label">推理时间</div></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{total_alerts}</div>
                <div class="kpi-label">预警帧数</div></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value" style="color:{ALERT_COLORS['red']};">{alerts.get('red', 0)}</div>
                <div class="kpi-label">I级 (红色)</div></div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-value">{detection.get('device', 'GPU')[:16]}</div>
                <div class="kpi-label">设备</div></div>""", unsafe_allow_html=True)

    # ── 第二行: 预警等级分布 ──
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1.25rem 0 0.75rem 0;">
        <div style="width:4px;height:24px;background:{PRIMARY_BLUE};border-radius:2px;"></div>
        <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">预警分布</div>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_legend = st.columns([3, 1])

    with col_chart:
        chart_data = pd.DataFrame({
            "预警等级": ["I级 · 红色", "II级 · 橙色", "III级 · 黄色", "IV级 · 蓝色"],
            "帧数": [
                alerts.get("red", 0), alerts.get("orange", 0),
                alerts.get("yellow", 0), alerts.get("blue", 0),
            ],
        })
        chart_data = chart_data[chart_data["帧数"] > 0]
        st.bar_chart(
            chart_data.set_index("预警等级"),
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
            <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">关键帧查看器</div>
            <div style="font-size:0.78rem;color:{TEXT_SECONDARY};">{len(key_frames)} 帧</div>
        </div>
        """, unsafe_allow_html=True)

        if "demo_frame_idx" not in st.session_state:
            st.session_state.demo_frame_idx = 0

        # 主图 + 控制
        kf = key_frames[st.session_state.demo_frame_idx]
        frame_path = _ROOT / active_scene["data_dir"] / kf["thumbnail"]
        lvl = kf["alert_level"]

        c_left, c_right = st.columns([4, 1])

        with c_left:
            if frame_path.exists():
                st.image(frame_path.read_bytes(), use_container_width=True)

            # 缩略图条
            cols = st.columns(min(len(key_frames), 15))
            for i, kf_th in enumerate(key_frames[:15]):
                fp_th = _ROOT / active_scene["data_dir"] / kf_th["thumbnail"]
                with cols[i]:
                    is_current = i == st.session_state.demo_frame_idx
                    if fp_th.exists():
                        st.image(fp_th.read_bytes(), use_container_width=True)
                        if is_current:
                            st.markdown(f"""<div style="height:2px;background:{PRIMARY_BLUE};
                                border-radius:1px;margin-top:-8px;"></div>""", unsafe_allow_html=True)

        with c_right:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:0.7rem;color:{TEXT_SECONDARY};text-transform:uppercase;">帧信息</div>
                <div style="font-size:1.5rem;font-weight:700;color:{TEXT_PRIMARY};margin:0.25rem 0;">#{kf['frame_idx']}</div>
                <div><span class="alert-badge {lvl}">{ALERT_LABELS.get(lvl, lvl)}</span></div>
                <div style="margin-top:0.75rem;font-size:0.82rem;">
                    <div>置信度 <b style="float:right;">{kf['max_confidence']:.3f}</b></div>
                    <div>目标数 <b style="float:right;">{kf['track_count']}</b></div>
                    <div>时间戳 <b style="float:right;">{kf['time_sec']:.1f}s</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.button("上一帧", key="demo_prev", use_container_width=True,
                      disabled=st.session_state.demo_frame_idx == 0,
                      on_click=lambda: st.session_state.update(
                          demo_frame_idx=max(0, st.session_state.demo_frame_idx - 1)))
            st.button("下一帧", key="demo_next", use_container_width=True,
                      disabled=st.session_state.demo_frame_idx >= len(key_frames) - 1,
                      on_click=lambda: st.session_state.update(
                          demo_frame_idx=min(len(key_frames) - 1, st.session_state.demo_frame_idx + 1)))

        # 滑块
        st.slider("", 0, len(key_frames) - 1, st.session_state.demo_frame_idx,
                  key="demo_slider", label_visibility="collapsed",
                  on_change=lambda: st.session_state.update(demo_frame_idx=st.session_state.demo_slider))
