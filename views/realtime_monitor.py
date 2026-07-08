"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

def page_realtime_monitor():
    """实时监测页面: 上传视频 → 检测 → 结果显示"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">实时监测</div>
            <div style="font-size:0.8rem;opacity:0.85;">上传视频 &middot; CPU 推理 &middot; 实时结果</div>
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
    with st.expander("演示模式 (CPU 加速)", expanded=True):
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

    # ── 性能仪表盘占位 ──
    perf_container = st.container()
    with perf_container:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
            <div style="width:4px;height:24px;background:#1565C0;border-radius:2px;"></div>
            <div style="font-weight:600;font-size:0.95rem;color:#1B2838;">性能仪表盘</div>
            <div style="font-size:0.72rem;color:#5F6B7A;">实时监测</div>
        </div>
        """, unsafe_allow_html=True)
        perf_cols = st.columns(6)
        perf_placeholders = {
            "fps": perf_cols[0].empty(),
            "inference": perf_cols[1].empty(),
            "gpu_util": perf_cols[2].empty(),
            "gpu_mem": perf_cols[3].empty(),
            "cpu": perf_cols[4].empty(),
            "ram": perf_cols[5].empty(),
        }
        perf_detail = st.empty()

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
            cleanup_stream_frames()
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

            # ── 性能监控 ──
            if PerformanceMonitor is not None:
                monitor = PerformanceMonitor()
                monitor.start()
            else:
                monitor = None
            _last_cb_time = [time.time()]  # 用列表避免闭包问题

            def _progress_cb(current: int, total: int):
                if total > 0:
                    progress_bar.progress(min(current / total, 1.0))
                status_text.text(f"推理中... 第 {current} 帧" + (f" / {total}" if total else ""))
                # 计算帧间推理耗时
                now = time.time()
                delta_ms = (now - _last_cb_time[0]) * 1000
                _last_cb_time[0] = now
                if monitor is not None:
                    monitor.record_frame(inference_ms=delta_ms)
                # 每 5 帧更新一次仪表盘
                if current % 5 == 0 and monitor is not None:
                    snap = monitor.snapshot()
                    update_perf_dashboard(perf_placeholders, perf_detail, snap)

            # ── 文件模式: detect_video() 一次性处理 ──
            with st.spinner(f"正在检测 `{source_name}` ..."):
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
            status_text.text("检测完成")
            if monitor is not None:
                monitor.stop()

            elapsed = time.time() - start_time

            # ── 性能摘要 ──
            if monitor is not None:
                final_snap = monitor.snapshot()
                update_perf_dashboard(perf_placeholders, perf_detail, final_snap)

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
                    "video_path": source_path,  # 保存原始视频路径供回放
                    "clips": {},  # 回放片段将在下面生成
                }

                # ── 生成预警回放片段 ──
                if alert_frames and save_frames_flag:
                    with st.spinner("🎬 正在生成预警回放片段..."):
                        try:
                            clips = generate_alert_replays(
                                alert_frames=alert_frames,
                                fps=fps,
                                context_frames=50,
                                max_per_level=5,
                            )
                            st.session_state.detection_results["clips"] = clips
                            total_clips = sum(len(v) for v in clips.values())
                            if total_clips > 0:
                                st.info(f"已生成 {total_clips} 个预警回放片段")
                        except Exception as e:
                            st.warning(f"回放片段生成失败: {e}")
                st.success(f"检测完成 — 耗时 {elapsed:.1f}s, "
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
                    status_text = f"帧 {frame_idx}"
                    if n_tracks > 0:
                        status_text += f" | {n_tracks} 目标"
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
                f"检测完成 — 耗时 {elapsed:.1f}s, 共 {len(all_frame_results)} 帧"
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
        <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">检测报告</div>
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
    c1.metric("总帧数", total)
    c2.metric("预警帧数", alert_count, delta=f"{alert_ratio:.1f}%" if alert_count > 0 else None)
    c3.metric("I级 (红色)", level_counts["red"])
    c4.metric("II级 (橙色)", level_counts["orange"])
    c5.metric("III级 (黄色)", level_counts["yellow"])

    if alert_count > 0:
        st.divider()
        chart_data = pd.DataFrame({
            "预警等级": ["I级 · 红色", "II级 · 橙色", "III级 · 黄色", "IV级 · 蓝色"],
            "帧数": [
                level_counts["red"], level_counts["orange"],
                level_counts["yellow"], level_counts["blue"],
            ],
        })
        chart_data = chart_data[chart_data["帧数"] > 0]
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.bar_chart(chart_data.set_index("预警等级"), use_container_width=True)
        with col_b:
            if results["alert_frames"]:
                tl_data = []
                for fr in results["alert_frames"]:
                    tl_data.append({
                        "帧号": fr["frame_idx"],
                        "时间 (秒)": fr.get("time_sec", fr["frame_idx"] / max(results.get("fps", 25), 1)),
                        "等级": fr.get("alert_level", "yellow"),
                        "目标数": len(fr.get("tracks", [])),
                    })
                tl_df = pd.DataFrame(tl_data)
                st.scatter_chart(tl_df.set_index("时间 (秒)")[["目标数"]], use_container_width=True)
                st.caption("预警时间线: X轴 = 时间(秒), Y轴 = 检出目标数")

        # ── 密度爆发监测状态 ──
        if DENSITY_ALERT_ENABLED:
            st.divider()
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.3rem;">
                <div style="font-size:0.85rem;font-weight:600;color:{TEXT_PRIMARY};">🔬 密度爆发监测</div>
                <span style="font-size:0.7rem;padding:2px 8px;border-radius:8px;
                    background-color:#E3F2FD;color:#1565C0;">v3 · 宜宾滑坡验证</span>
            </div>
            """, unsafe_allow_html=True)

            dc1, dc2, dc3, dc4 = st.columns(4)
            detector = get_detector_or_stop()
            dm = detector._density_monitor if detector is not None else None

            if dm is not None and dm.is_ready:
                stats = dm.last_stats
                dc1.metric("窗口大小", f"{dm.window_frames}帧",
                          delta=f"{dm.window_sec}s")
                dc2.metric("基线均值", f"{stats.window_mean:.1f}",
                          help="滚动窗口内平均每帧检测数")
                dc3.metric("当前 z-score",
                          f"{stats.zscore:.1f}",
                          delta="爆发!" if stats.is_burst else "正常",
                          delta_color="inverse" if not stats.is_burst else "normal")
                dc4.metric("z-score 阈值", f"{dm.burst_zscore}",
                          help="超过此值触发密度告警")
            else:
                dc1.metric("窗口大小", f"{DENSITY_WINDOW_SEC}s × {results.get('fps', 30):.0f}fps",
                          help="滚动统计窗口")
                dc2.metric("置信度下限", f"{DENSITY_CONF_FLOOR:.2f}",
                          help="纳入密度统计的最低YOLO置信度")
                dc3.metric("爆发阈值", f"z > {DENSITY_BURST_ZSCORE}",
                          help="检测数超过基线N倍标准差")
                dc4.metric("最少样本", f"{DENSITY_MIN_SAMPLES}帧",
                          help="需收集足够样本后才判定")
                if dm is not None:
                    st.caption(f"📊 已收集 {dm.last_stats.window_size}/{dm.window_frames} 帧样本 (就绪后开始判定)")

    # 预警帧图库
    if alert_count > 0 and save_frames_flag:
        st.divider()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
            <div style="width:4px;height:24px;background:{PRIMARY_BLUE};border-radius:2px;"></div>
            <div style="font-weight:600;font-size:1rem;color:{TEXT_PRIMARY};">预警帧图库</div>
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
            writer.writerow(["帧号", "时间(秒)", "预警等级", "目标数", "跟踪ID", "最高置信度"])
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

    # ── 预警回放 ──
    clips = results.get("clips", {})
    if clips:
        total_clips = sum(len(v) for v in clips.values())
        if total_clips > 0:
            st.divider()
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.75rem;">
                <div style="width:4px;height:24px;background:#1565C0;border-radius:2px;"></div>
                <div style="font-weight:600;font-size:1rem;color:#1B2838;">预警回放</div>
                <div style="font-size:0.78rem;color:#5F6B7A;">{total_clips} 个片段</div>
            </div>
            """, unsafe_allow_html=True)

            # 按级别选择
            level_tabs = st.tabs([
                f"🔴 I 级 ({len(clips.get('red', []))})",
                f"🟠 II 级 ({len(clips.get('orange', []))})",
                f"🟡 III 级 ({len(clips.get('yellow', []))})",
                f"🔵 IV 级 ({len(clips.get('blue', []))})",
            ])

            level_keys = ["red", "orange", "yellow", "blue"]
            for tab, lvl in zip(level_tabs, level_keys):
                with tab:
                    lvl_clips = clips.get(lvl, [])
                    if not lvl_clips:
                        st.info(f"无 {ALERT_LABELS.get(lvl, lvl)} 等级回放片段")
                        continue
                    # 每个片段显示为可播放的视频卡片
                    cols_per_row = 2
                    for i in range(0, len(lvl_clips), cols_per_row):
                        row_cols = st.columns(cols_per_row)
                        for j, clip in enumerate(lvl_clips[i:i + cols_per_row]):
                            with row_cols[j]:
                                clip_path = Path(clip["clip_path"])
                                frame_idx = clip["frame_idx"]
                                duration = clip.get("duration_sec", 0)
                                if clip_path.exists():
                                    st.markdown(f"""
                                    <div style="padding:0.3rem 0.5rem;background:#F5F7FA;
                                                border-radius:4px;margin-bottom:0.3rem;
                                                font-size:0.78rem;color:#1B2838;">
                                        <b>帧 #{frame_idx}</b>
                                        <span style="float:right;color:#5F6B7A;">{duration}s</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.video(str(clip_path), format="video/mp4")
                                else:
                                    st.caption(f"片段不可用: 帧 #{frame_idx}")

    # 预警记录摘要
    with c2:
        try:
            recent = store.get_recent(limit=20)
            if recent:
                st.caption(f"最近 {len(recent)} 条预警记录:")
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
