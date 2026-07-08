"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

def page_alert_records():
    """预警记录页面: 查询、筛选、导出历史预警"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">预警记录</div>
            <div style="font-size:0.8rem;opacity:0.85;">历史记录 &middot; 筛选 &middot; 导出</div>
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
    tc1.metric("I级 (红色)", today_counts.get("red", 0))
    tc2.metric("II级 (橙色)", today_counts.get("orange", 0))
    tc3.metric("III级 (黄色)", today_counts.get("yellow", 0))
    tc4.metric("IV级 (蓝色)", today_counts.get("blue", 0))
    tc5.metric("今日合计", sum(today_counts.values()))

    # ── 自动升级状态卡片 ──
    unconfirmed_count = 0
    try:
        from rockfall.config import ALERT_ESCALATION_ENABLED, ALERT_ESCALATION_TIMEOUT_MINUTES
        from rockfall.alert_escalation import get_escalation_scheduler

        esc_scheduler = get_escalation_scheduler()
        unconfirmed = store.get_unconfirmed_alerts(
            timeout_minutes=ALERT_ESCALATION_TIMEOUT_MINUTES, limit=500,
        )
        unconfirmed_count = len(unconfirmed)

        if ALERT_ESCALATION_ENABLED:
            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                color = "#D32F2F" if unconfirmed_count > 0 else "#2E7D32"
                st.markdown(f"""
                <div style="text-align:center;padding:0.6rem 0.4rem;background:#fff;
                            border:1px solid #E3E8EF;border-radius:8px;">
                    <div style="font-size:1.3rem;font-weight:700;color:{color};">{unconfirmed_count}</div>
                    <div style="font-size:0.65rem;color:#5F6B7A;">超时未确认</div>
                    <div style="font-size:0.55rem;color:#9E9E9E;">&gt;{ALERT_ESCALATION_TIMEOUT_MINUTES}min</div>
                </div>
                """, unsafe_allow_html=True)
            with ec2:
                st.markdown(f"""
                <div style="text-align:center;padding:0.6rem 0.4rem;background:#fff;
                            border:1px solid #E3E8EF;border-radius:8px;">
                    <div style="font-size:1.3rem;font-weight:700;color:#1565C0;">{esc_scheduler.total_escalations}</div>
                    <div style="font-size:0.65rem;color:#5F6B7A;">累计自动升级</div>
                    <div style="font-size:0.55rem;color:#9E9E9E;">blue→yellow→orange→red</div>
                </div>
                """, unsafe_allow_html=True)
            with ec3:
                last_run = esc_scheduler.last_run_time
                last_display = last_run[11:19] if last_run and len(last_run) > 11 else "N/A"
                st.markdown(f"""
                <div style="text-align:center;padding:0.6rem 0.4rem;background:#fff;
                            border:1px solid #E3E8EF;border-radius:8px;">
                    <div style="font-size:1.0rem;font-weight:700;color:#1B2838;">{last_display}</div>
                    <div style="font-size:0.65rem;color:#5F6B7A;">上次扫描</div>
                    <div style="font-size:0.55rem;color:#9E9E9E;">每60s自动扫描</div>
                </div>
                """, unsafe_allow_html=True)
            with ec4:
                st.markdown(f"""
                <div style="text-align:center;padding:0.6rem 0.4rem;background:#fff;
                            border:1px solid #E3E8EF;border-radius:8px;">
                    <div style="font-size:1.0rem;font-weight:700;color:#1B2838;">
                        {"🟢 运行中" if esc_scheduler.is_running or not ALERT_ESCALATION_ENABLED else "⚫ 已停止"}
                    </div>
                    <div style="font-size:0.65rem;color:#5F6B7A;">调度器状态</div>
                    <div style="font-size:0.55rem;color:#9E9E9E;">双向闭环</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("💡 预警自动升级已禁用。设置 `ALERT_ESCALATION_ENABLED=true` 以启用在未确认时自动升级预警等级。")
    except Exception:
        pass  # 后端未部署时静默跳过

    # ── 超时未确认预警 · 快速确认 ──
    if unconfirmed_count > 0:
        with st.expander(f"⚠️ {unconfirmed_count} 条预警超时未确认 — 点击展开快速确认", expanded=True):
            st.markdown(
                '<div style="font-size:0.78rem;color:#D32F2F;margin-bottom:0.5rem;">'
                '以下预警已超过规定时间未被确认，将触发自动升级。请及时确认以停止计时器。</div>',
                unsafe_allow_html=True,
            )
            for ua in unconfirmed[:10]:  # 最多显示10条
                ua_id = ua.get("id", 0)
                ua_time = ua.get("time", "")
                ua_level = ua.get("alert_level", "")
                ua_loc = ua.get("monitoring_location", "") or "未知点位"
                ua_count = ua.get("count", 0)
                ua_conf = ua.get("max_confidence", 0)
                ua_confirmed = ua.get("confirmed_at", "")

                # 计算超时时长
                try:
                    alert_dt = datetime.strptime(ua_time, "%Y-%m-%d %H:%M:%S")
                    elapsed_min = int((datetime.now() - alert_dt).total_seconds() / 60)
                except Exception:
                    elapsed_min = 0

                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    level_icon = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "blue": "🔵"}.get(ua_level, "⚪")
                    st.markdown(
                        f"{level_icon} **#{ua_id}** | {ua_time} | {ALERT_LABELS.get(ua_level, ua_level)} | "
                        f"{ua_loc} | {ua_count}目标 | 置信度{ua_conf:.2f} | "
                        f"<span style='color:#D32F2F;font-weight:600;'>已超时 {elapsed_min} 分钟</span>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    operator_name = st.text_input(
                        "确认人", value="管理员", key=f"confirm_op_{ua_id}",
                        label_visibility="collapsed",
                    )
                    if st.button(f"✅ 确认", key=f"confirm_btn_{ua_id}", use_container_width=True):
                        try:
                            import requests
                            r = requests.post(
                                build_api_url(f"/api/alerts/{ua_id}/confirm"),
                                data={"confirmed_by": operator_name},
                                timeout=5,
                            )
                            if r.status_code == 200:
                                st.success(f"#{ua_id} 已确认，计时器已停止")
                                st.rerun()
                            else:
                                st.error(r.json().get("detail", "确认失败"))
                        except Exception as e:
                            st.error(f"API 不可用: {e}")
            if unconfirmed_count > 10:
                st.caption(f"... 还有 {unconfirmed_count - 10} 条未确认预警")

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
            wf_labels = {"pending": "待审核", "confirmed": "已确认", "false_alarm": "误报",
                         "dispatched": "已派单", "arrived": "已到场", "handled": "已处置", "archived": "已归档"}
            # 确认状态
            confirmed_at = r.get("confirmed_at", "")
            confirmed_by = r.get("confirmed_by", "")
            if confirmed_at:
                confirm_status = f"✅ {confirmed_by or '已确认'} {confirmed_at[5:16] if len(confirmed_at) > 11 else ''}"
            else:
                confirm_status = "⏳ 待确认"
            df_data.append({
                "ID": r.get("id", ""),
                "时间": r.get("time", ""),
                "预警等级": ALERT_LABELS.get(lvl, lvl),
                "数量": r.get("count", 0),
                "最高置信度": round(r.get("max_confidence", 0), 4),
                "落石直径(cm)": r.get("rock_diameter_cm", 0),
                "监测点位": r.get("monitoring_location", ""),
                "工单状态": wf_labels.get(r.get("workflow_state", ""), r.get("workflow_state", "待审核")),
                "确认状态": confirm_status,
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

        # ── 工单流转 ──
        st.divider()
        with st.expander("工单管理", expanded=False):
            wf_col1, wf_col2 = st.columns([2, 3])
            with wf_col1:
                wf_alert_id = st.number_input("预警ID", min_value=1, value=1, key="wf_alert_id")
                wf_operator = st.text_input("操作人", value="管理员", key="wf_operator",
                                           help="操作人员姓名或工号")
                wf_note = st.text_input("备注", placeholder="备注信息 (可选)", key="wf_note")
            with wf_col2:
                wf_state = st.selectbox("目标状态",
                    options=["confirmed", "false_alarm", "dispatched", "arrived", "handled", "archived"],
                    format_func=lambda x: {
                        "confirmed": "确认真实落石",
                        "false_alarm": "标记为误报",
                        "dispatched": "派单给现场人员",
                        "arrived": "现场人员已到场",
                        "handled": "处置完毕",
                        "archived": "归档",
                    }.get(x, x),
                    key="wf_state")
                st.write("")
                if st.button("执行流转", key="wf_execute", use_container_width=True):
                    try:
                        import requests
                        
                        r = requests.post(
                            build_api_url(f"/api/alerts/{wf_alert_id}/workflow"),
                            data={"state": wf_state, "operator": wf_operator, "note": wf_note},
                            timeout=5)
                        result = r.json()
                        if result.get("ok"):
                            st.success(result.get("msg"))
                        else:
                            st.error(result.get("msg"))
                    except Exception as e:
                        st.error(f"API不可用: {e}")

            # 显示当前状态
            if wf_alert_id:
                try:
                    import requests
                    
                    r = requests.get(build_api_url(f"/api/alerts/{wf_alert_id}/workflow"), timeout=5)
                    wf_data = r.json()
                    st.markdown(f"**当前状态**: {wf_data.get('current_label', wf_data.get('current_state', 'N/A'))}")
                    history = wf_data.get("history", [])
                    if history:
                        st.markdown("**历史记录**:")
                        for h in history[-5:]:
                            st.caption(f"{h.get('time','')} | {h.get('operator','')} | "
                                      f"{h.get('from','')} -> {h.get('to','')} | {h.get('note','')}")
                except Exception:
                    pass

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
                "跟踪ID", "类别摘要", "保存帧", "推送状态",
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
