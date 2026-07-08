"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

MAX_CAMERAS = 4
CAMERA_COLORS = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A"]  # 蓝/橙/绿/紫

def page_multi_camera():
    """多路监控页面: 同时展示多路视频的检测结果"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">多路监控</div>
            <div style="font-size:0.8rem;opacity:0.85;">最多4路 &middot; 同步视图 &middot; 聚合预警</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    detector = get_detector_or_stop()

    # ── 会话状态: 多路摄像头配置 ──
    if "mc_configs" not in st.session_state:
        st.session_state.mc_configs = []  # [{"name": str, "source": str, "type": "file"|"url"}]
    if "mc_results" not in st.session_state:
        st.session_state.mc_results = {}   # {cam_idx: {"frames": [...], "alerts": [...], "fps": float}}
    if "mc_processing" not in st.session_state:
        st.session_state.mc_processing = False
    if "mc_active_view" not in st.session_state:
        st.session_state.mc_active_view = -1  # -1 = grid, 0-3 = single cam enlarged

    # ── 配置面板 ──
    with st.expander("摄像头配置", expanded=not st.session_state.mc_results):
        st.caption(f"添加 2-{MAX_CAMERAS} 路视频源，然后点击「开始多路检测」")

        num_cams = st.slider(
            "摄像头数量", min_value=2, max_value=MAX_CAMERAS, value=max(2, len(st.session_state.mc_configs)),
            help="选择要同时监控的摄像头数量"
        )

        configs = []
        for i in range(num_cams):
            col1, col2, col3 = st.columns([2, 4, 2])
            with col1:
                cam_name = st.text_input(
                    f"名称", value=f"摄像头 {i+1}",
                    key=f"mc_name_{i}", placeholder=f"摄像头 {i+1}"
                )
            with col2:
                cam_source = st.text_input(
                    f"视频路径或 URL",
                    value=st.session_state.mc_configs[i]["source"] if i < len(st.session_state.mc_configs) else "",
                    key=f"mc_source_{i}",
                    placeholder="上传视频文件 或 输入 RTSP URL"
                )
            with col3:
                cam_type = st.selectbox(
                    f"类型", options=["video_file", "rtsp_url"],
                    key=f"mc_type_{i}",
                    format_func=lambda x: "视频文件" if x == "video_file" else "RTSP流"
                )
            configs.append({"name": cam_name, "source": cam_source, "type": cam_type})

            # 视频文件上传
            if cam_type == "video_file":
                uploaded = st.file_uploader(
                    f"上传视频 {i+1}", type=["mp4", "avi", "mov", "mkv"],
                    key=f"mc_upload_{i}",
                    help=f"为摄像头 {i+1} 上传视频文件"
                )
                if uploaded:
                    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                    save_path = UPLOADS_DIR / f"multi_cam_{i}_{uploaded.name}"
                    with open(save_path, "wb") as f:
                        f.write(uploaded.read())
                    configs[-1]["source"] = str(save_path)
                    st.success(f"已保存: {uploaded.name}")

        st.session_state.mc_configs = configs

        # ── 检测参数 (多路共用) ──
        c1, c2, c3 = st.columns(3)
        with c1:
            mc_max_frames = st.slider(
                "每路最大帧数", 30, 300, 100, 10,
                key="mc_max_frames",
                help="每路视频最多处理的帧数"
            )
        with c2:
            mc_stride = st.slider(
                "帧采样步长", 1, 8, 2, 1,
                key="mc_stride",
                help="每隔 N 帧推理一次"
            )
        with c3:
            mc_img_size = st.selectbox(
                "推理分辨率", [320, 416, 640], 0,
                key="mc_img_size"
            )

        start_mc = st.button(
            "▶ 开始多路检测", type="primary", use_container_width=True,
            disabled=not any(c["source"] for c in configs)
        )

    # ── 执行多路检测 ──
    if start_mc:
        st.session_state.mc_processing = True
        st.session_state.mc_results = {}
        cleanup_stream_frames()

        valid_configs = [(i, c) for i, c in enumerate(configs) if c["source"]]

        for cam_idx, cfg in valid_configs:
            source_path = cfg["source"]
            if not Path(source_path).exists() and cfg["type"] == "video_file":
                st.warning(f"{cfg['name']}: 文件不存在，跳过")
                continue

            st.markdown(f"---")
            status_col, progress_col = st.columns([1, 4])
            with status_col:
                st.markdown(f"""
                <div style="padding:0.5rem;background:#F5F7FA;border-radius:6px;text-align:center;">
                    <div style="font-weight:600;color:{CAMERA_COLORS[cam_idx % len(CAMERA_COLORS)]};">
                        {cfg['name']}
                    </div>
                    <div style="font-size:0.7rem;color:#5F6B7A;">处理中...</div>
                </div>
                """, unsafe_allow_html=True)

            with progress_col:
                prog_bar = st.progress(0.0)
                status_text = st.empty()

            _orig_size = detector.img_size
            detector.img_size = mc_img_size

            try:
                result = detector.detect_video(
                    source_path,
                    save_frames=True,
                    push_alerts=False,
                    track=True,
                    max_frames=mc_max_frames,
                    stride=mc_stride,
                    progress_callback=lambda c, t: (
                        prog_bar.progress(min(c / max(t, 1), 1.0)),
                        status_text.text(f"Frame {c}" + (f"/{t}" if t else ""))
                    ),
                )
            finally:
                detector.img_size = _orig_size

            if isinstance(result, dict) and "error" not in result:
                frames = result.get("detections", [])
                fps = result.get("fps", 25.0)
                alert_frames = [f for f in frames if f.get("alert_level", "green") != "green"]

                st.session_state.mc_results[cam_idx] = {
                    "name": cfg["name"],
                    "total_frames": len(frames),
                    "fps": round(fps, 2),
                    "alert_frames": alert_frames,
                    "all_frames": frames,
                    "source": source_path,
                    "color": CAMERA_COLORS[cam_idx % len(CAMERA_COLORS)],
                }
                prog_bar.progress(1.0)
                status_text.text(f"{len(frames)} 帧, {len(alert_frames)} 预警")
            else:
                status_text.text(f"失败: {result}")

        st.session_state.mc_processing = False

    # ── 显示多路监控结果 ──
    mc_results = st.session_state.mc_results
    if not mc_results:
        st.info("👆 请配置至少 2 路视频源，然后点击「开始多路检测」。")
        return

    # ── 聚合统计 ──
    st.divider()
    all_alerts_agg = []
    for cam_idx, res in mc_results.items():
        for fr in res.get("alert_frames", []):
            all_alerts_agg.append({
                "camera": res["name"],
                "cam_idx": cam_idx,
                "color": res.get("color", "#1565C0"),
                **fr,
            })

    total_cams = len(mc_results)
    total_frames = sum(r["total_frames"] for r in mc_results.values())
    total_alerts = len(all_alerts_agg)

    # KPI 行
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("活跃摄像头", total_cams)
    c2.metric("总帧数", total_frames)
    c3.metric("总预警", total_alerts)
    c4.metric("平均帧率", round(sum(r["fps"] for r in mc_results.values()) / max(total_cams, 1), 1))

    # 按等级统计聚合
    level_counts = {"red": 0, "orange": 0, "yellow": 0, "blue": 0}
    for a in all_alerts_agg:
        lvl = a.get("alert_level", "green")
        if lvl in level_counts:
            level_counts[lvl] += 1

    if total_alerts > 0:
        st.markdown("#### 聚合预警分布")
        chart_cols = st.columns(4)
        for i, (lvl, color_key) in enumerate([
            ("red", ALERT_COLORS["red"]), ("orange", ALERT_COLORS["orange"]),
            ("yellow", ALERT_COLORS["yellow"]), ("blue", ALERT_COLORS["blue"]),
        ]):
            count = level_counts[lvl]
            chart_cols[i].markdown(f"""
            <div style="text-align:center;padding:0.5rem;background:#fff;border:1px solid #E3E8EF;border-radius:8px;">
                <div style="font-size:1.6rem;font-weight:700;color:{color_key};">{count}</div>
                <div style="font-size:0.7rem;color:#5F6B7A;">{ALERT_LABELS.get(lvl, lvl)}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── 视图模式切换 ──
    st.divider()
    view_c1, view_c2 = st.columns([3, 1])
    with view_c1:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:4px;height:24px;background:#1565C0;border-radius:2px;"></div>
            <div style="font-weight:600;font-size:1rem;color:#1B2838;">
                {"网格视图" if st.session_state.mc_active_view < 0 else f"摄像头: {mc_results.get(st.session_state.mc_active_view, {}).get('name', '')}"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with view_c2:
        if st.session_state.mc_active_view >= 0:
            if st.button("↩ 返回网格视图", use_container_width=True):
                st.session_state.mc_active_view = -1
                st.rerun()

    # ── 网格视图 ──
    if st.session_state.mc_active_view < 0:
        cam_indices = sorted(mc_results.keys())
        # 2x2 布局
        rows = (len(cam_indices) + 1) // 2
        for row in range(rows):
            row_cams = cam_indices[row * 2: row * 2 + 2]
            cols = st.columns(2)
            for j, cam_idx in enumerate(row_cams):
                res = mc_results[cam_idx]
                with cols[j]:
                    color = res.get("color", "#1565C0")
                    alert_count = len(res.get("alert_frames", []))

                    # 卡片头部
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;justify-content:space-between;
                                padding:0.5rem 0.75rem;background:#fff;border:1px solid #E3E8EF;
                                border-left:3px solid {color};border-radius:6px;margin-bottom:0.3rem;">
                        <div>
                            <span style="font-weight:600;font-size:0.85rem;color:#1B2838;">{res['name']}</span>
                            <span style="font-size:0.7rem;color:#5F6B7A;margin-left:8px;">
                                {res['total_frames']}帧 | {res['fps']}fps
                            </span>
                        </div>
                        <div>
                            <span class="alert-badge red" style="font-size:0.7rem;">{alert_count} 条预警</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 显示最近的标注帧
                    all_frames = res.get("all_frames", [])
                    if all_frames:
                        # 优先显示最近预警帧
                        alert_frames_list = res.get("alert_frames", [])
                        display_frame_idx = alert_frames_list[-1]["frame_idx"] if alert_frames_list else all_frames[-1]["frame_idx"]
                        frame_path = RESULTS_DIR / f"stream_{display_frame_idx:06d}.jpg"

                        col_img, col_info = st.columns([3, 1])
                        with col_img:
                            if frame_path.exists():
                                st.image(str(frame_path), use_container_width=True)
                            else:
                                st.caption(f"标注帧不可用 (F{display_frame_idx})")
                        with col_info:
                            # 相机统计
                            latest_alert = alert_frames_list[-1] if alert_frames_list else None
                            st.markdown(f"""
                            <div style="font-size:0.7rem;color:#5F6B7A;line-height:1.6;">
                                <div>总帧: <b>{res['total_frames']}</b></div>
                                <div>FPS: <b>{res['fps']}</b></div>
                                <div>预警: <b style="color:#D32F2F;">{alert_count}</b></div>
                                <div>最近等级: <b style="color:{ALERT_COLORS.get(latest_alert.get('alert_level', 'green'), '#2E7D32')};">{ALERT_LABELS.get(latest_alert.get('alert_level', 'green'), '-') if latest_alert else '-'}</b></div>
                            </div>
                            """, unsafe_allow_html=True)

                    # 放大按钮
                    if st.button(f"放大查看", key=f"mc_zoom_{cam_idx}", use_container_width=True):
                        st.session_state.mc_active_view = cam_idx
                        st.rerun()

    # ── 单路放大视图 ──
    else:
        cam_idx = st.session_state.mc_active_view
        if cam_idx in mc_results:
            res = mc_results[cam_idx]
            color = res.get("color", "#1565C0")

            # 帧浏览
            all_frames = res.get("all_frames", [])
            alert_frames_list = res.get("alert_frames", [])
            alert_frame_indices = {f["frame_idx"] for f in alert_frames_list}

            if "mc_frame_pos" not in st.session_state:
                st.session_state.mc_frame_pos = 0

            max_pos = max(len(all_frames) - 1, 0)

            # 导航行
            nav_c1, nav_c2, nav_c3 = st.columns([1, 3, 1])
            with nav_c1:
                if st.button("⬅ 上一帧", use_container_width=True, disabled=st.session_state.mc_frame_pos == 0):
                    st.session_state.mc_frame_pos = max(0, st.session_state.mc_frame_pos - 1)
                    st.rerun()
            with nav_c2:
                st.slider(
                    "帧浏览器", 0, max_pos, st.session_state.mc_frame_pos,
                    key="mc_frame_slider", label_visibility="collapsed",
                    on_change=lambda: st.session_state.update(
                        mc_frame_pos=st.session_state.mc_frame_slider
                    )
                )
            with nav_c3:
                if st.button("下一帧 ➡", use_container_width=True, disabled=st.session_state.mc_frame_pos >= max_pos):
                    st.session_state.mc_frame_pos = min(max_pos, st.session_state.mc_frame_pos + 1)
                    st.rerun()

            current_fr = all_frames[st.session_state.mc_frame_pos] if all_frames else None
            frame_path = RESULTS_DIR / f"stream_{current_fr['frame_idx']:06d}.jpg" if current_fr else None

            # 预警帧快捷跳转
            if alert_frames_list:
                alert_shortcuts = st.columns(min(len(alert_frames_list), 10))
                for i, af in enumerate(alert_frames_list[:10]):
                    lvl = af.get("alert_level", "green")
                    with alert_shortcuts[i]:
                        if st.button(f"{ALERT_LABELS.get(lvl, lvl[:1].upper())}",
                                     key=f"mc_alert_jump_{i}",
                                     help=f"跳转到帧 {af['frame_idx']} ({ALERT_LABELS.get(lvl, lvl)})",
                                     use_container_width=True):
                            # 找到这个frame_idx在all_frames中的位置
                            for pos, f in enumerate(all_frames):
                                if f["frame_idx"] == af["frame_idx"]:
                                    st.session_state.mc_frame_pos = pos
                                    st.rerun()

            # 主画面
            if frame_path and frame_path.exists():
                st.image(str(frame_path), use_container_width=True)
                if current_fr:
                    lvl = current_fr.get("alert_level", "green")
                    is_alert = current_fr["frame_idx"] in alert_frame_indices
                    st.markdown(f"""
                    <div style="display:flex;gap:1rem;padding:0.5rem 0.75rem;background:#F5F7FA;border-radius:6px;
                                font-size:0.8rem;color:#1B2838;flex-wrap:wrap;align-items:center;">
                        <span><b>{res['name']}</b></span>
                        <span>帧号: <b>#{current_fr['frame_idx']}</b></span>
                        <span>预警等级:
                            <span class="alert-badge {lvl}">{ALERT_LABELS.get(lvl, lvl)}</span>
                        </span>
                        <span>目标数: <b>{len(current_fr.get('boxes', []))}</b></span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption(f"标注帧不可用")

    # ── 聚合预警时间线 ──
    if all_alerts_agg:
        st.divider()
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
            <div style="width:4px;height:24px;background:#1565C0;border-radius:2px;"></div>
            <div style="font-weight:600;font-size:1rem;color:#1B2838;">聚合预警时间线</div>
        </div>
        """, unsafe_allow_html=True)

        # 按时间排序
        all_alerts_agg.sort(key=lambda a: a.get("time_sec", 0))

        tl_data = []
        for a in all_alerts_agg:
            tl_data.append({
                "摄像头": a["camera"],
                "时间 (秒)": a.get("time_sec", 0),
                "等级": a.get("alert_level", "yellow"),
                "目标数": len(a.get("boxes", [])),
            })

        if tl_data:
            tl_df = pd.DataFrame(tl_data)
            # 绘制散点图，按摄像头着色
            st.scatter_chart(
                tl_df.set_index("时间 (秒)")[["目标数"]],
                use_container_width=True,
            )
            st.caption("横轴: 时间(秒) | 纵轴: 检出目标数 | 每路摄像头独立显示")
