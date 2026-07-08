"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

def page_hazard_investigation():
    """隐患点排查与风险评估页面 — 对应\"第1步：隐患点排查与风险评估\"工作流程"""
    import io as _io

    _RISK_LABELS = {
        "high": "🔴 高风险", "medium": "🟡 中风险", "low": "🟢 一般风险"
    }
    _RISK_COLORS = {
        "high": "#DC3545", "medium": "#FFC107", "low": "#28A745"
    }
    _STATUS_LABELS = {
        "identified": "🔍 已识别", "assessed": "📋 已评估",
        "monitored": "📡 监测中", "remediated": "🛠️ 已治理", "cleared": "✅ 已消除"
    }

    # ── 初始化种子数据 ──
    try:
        store = get_hazard_store()
        seeded = store.seed_demo_data()
        if seeded > 0:
            from rockfall.logger import log_event
            log_event("system", level="INFO",
                      msg=f"隐患点种子数据已写入 DB ({seeded} 条)")
    except Exception:
        store = None

    hazards = store.list_all() if store else []
    level_counts = store.count_by_level() if store else {"high": 0, "medium": 0, "low": 0}

    # ── 页面头部 ──
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">隐患点排查与风险评估</div>
            <div style="font-size:0.8rem;opacity:0.85;">
                {len(hazards)} 个隐患点 &nbsp;|&nbsp;
                🔴 {level_counts.get('high',0)} 高风险 &nbsp;
                🟡 {level_counts.get('medium',0)} 中风险 &nbsp;
                🟢 {level_counts.get('low',0)} 一般风险
            </div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tab 分页 ──
    tab_overview, tab_list, tab_add, tab_report = st.tabs([
        "📊 总览面板", "📋 隐患点清单", "➕ 新增排查", "📄 评估报告"
    ])

    # ═══════════════════ Tab 1: 总览面板 ═══════════════════
    with tab_overview:
        # ── 统计卡片 ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("隐患点总数", len(hazards))
        with col2:
            monitored = sum(1 for h in hazards if h.linked_site_id)
            unmonitored = len(hazards) - monitored
            st.metric("已部署监测", f"{monitored}/{len(hazards)}",
                      delta=f"{unmonitored} 待部署" if unmonitored > 0 else "全覆盖")
        with col3:
            avg_score = round(sum(h.risk_score for h in hazards) / len(hazards), 1) if hazards else 0
            st.metric("平均风险评分", f"{avg_score}/100")
        with col4:
            has_incidents = sum(1 for h in hazards if h.historical_incidents)
            st.metric("有历史灾害", f"{has_incidents} 处")

        # ── 风险分布柱状图 ──
        st.divider()
        st.subheader("风险等级分布")
        import plotly.express as px
        df_level = pd.DataFrame([
            {"等级": "高风险", "数量": level_counts.get("high", 0), "颜色": "#DC3545"},
            {"等级": "中风险", "数量": level_counts.get("medium", 0), "颜色": "#FFC107"},
            {"等级": "一般风险", "数量": level_counts.get("low", 0), "颜色": "#28A745"},
        ])
        fig = px.bar(df_level, x="等级", y="数量", color="等级",
                     color_discrete_map={"高风险": "#DC3545", "中风险": "#FFC107", "一般风险": "#28A745"},
                     text="数量", title="隐患点风险等级分布")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        # ── 工作流程说明 ──
        st.divider()
        st.subheader("排查工作流程")
        st.markdown("""
        | 步骤 | 工作内容 | 参与方 | 输出 |
        |------|---------|--------|------|
        | ① 现场踏勘 | 对目标路段进行全面踏勘，识别落石风险点 | 公路养护 + 地质部门 | 隐患点初步清单 |
        | ② 历史数据梳理 | 标记发生过落石、塌方事件的路段 | 交通局/公路局 | 历史灾害台账 |
        | ③ 边坡稳定性评估 | 地质条件分析、节理裂隙调查、稳定性计算 | 地质勘察单位 | 边坡评估报告 |
        | ④ 风险等级分级 | 按高风险/中风险/一般风险对隐患点分级 | 联合专家组 | 隐患点清单及风险评估报告 |
        """)

    # ═══════════════════ Tab 2: 隐患点清单 ═══════════════════
    with tab_list:
        # ── 筛选栏 ──
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_level = st.selectbox(
                "风险等级筛选", ["全部", "高风险", "中风险", "一般风险"],
                key="hz_filter_level")
        with col_f2:
            filter_status = st.selectbox(
                "状态筛选", ["全部"] + list(_STATUS_LABELS.values()),
                key="hz_filter_status")
        with col_f3:
            search_term = st.text_input("🔍 搜索", placeholder="公路/地点/编号...",
                                        key="hz_search")

        # 应用筛选
        filtered = list(hazards)
        level_map_rev = {"高风险": "high", "中风险": "medium", "一般风险": "low"}
        if filter_level != "全部":
            filtered = [h for h in filtered if h.risk_level == level_map_rev.get(filter_level, "")]
        if filter_status != "全部":
            status_map_rev = {v: k for k, v in _STATUS_LABELS.items()}
            filtered = [h for h in filtered if _STATUS_LABELS.get(h.status, "") == filter_status]
        if search_term:
            q = search_term.lower()
            filtered = [h for h in filtered if
                        q in h.name.lower() or q in h.highway.lower()
                        or q in h.hazard_id.lower() or q in h.region.lower()]

        st.caption(f"共 {len(filtered)} 个隐患点")

        if not filtered:
            st.info("暂无符合条件的隐患点")
        else:
            for i, h in enumerate(filtered):
                _render_hazard_card(h, i)

    # ═══════════════════ Tab 3: 新增排查 ═══════════════════
    with tab_add:
        st.subheader("录入新隐患点排查数据")
        st.caption("包含基础信息、边坡稳定性评估、历史灾害记录。联合地质部门评估后填写。")

        with st.form("add_hazard_form"):
            # ── 基础信息 ──
            st.markdown("#### 📍 基础信息")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                new_hz_id = st.text_input("隐患点编号 *", key="hz_id",
                                          placeholder="如: HZ-G75-003",
                                          help="唯一标识符，建议格式: HZ-公路编号-序号")
                new_hz_name = st.text_input("隐患点名称 *", key="hz_name",
                                            placeholder="如: 桂林阳朔高速某边坡隐患点")
                new_hz_region = st.text_input("所属区域", key="hz_region",
                                              placeholder="如: 广西·桂林")
                new_hz_lat = st.number_input("纬度", key="hz_lat",
                                             min_value=0.0, max_value=90.0,
                                             step=0.001, value=22.817, format="%.3f")
                new_hz_lng = st.number_input("经度", key="hz_lng",
                                             min_value=0.0, max_value=180.0,
                                             step=0.001, value=108.366, format="%.3f")
            with col_a2:
                new_hz_highway = st.text_input("所属公路", key="hz_highway",
                                               placeholder="如: G65 包茂高速")
                new_hz_stake = st.text_input("桩号", key="hz_stake",
                                             placeholder="如: K2480+500")
                new_hz_location = st.text_input("具体位置描述", key="hz_location",
                                                placeholder="报警推送中显示的位置")
                new_hz_responsible = st.text_input("责任单位", key="hz_responsible",
                                                   placeholder="如: XX公路管理局")
                new_hz_contact = st.text_input("联系人/电话", key="hz_contact",
                                               placeholder="如: 张工 / 138-XXXX-XXXX")
                new_hz_surveyed = st.date_input("排查日期", key="hz_surveyed")

            new_hz_desc = st.text_area("隐患点综合描述", key="hz_desc",
                                       placeholder="地形地貌、地质特征、周边环境等综合描述...")

            # ── 已有防护措施 ──
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                new_hz_protection = st.text_input("已有防护措施", key="hz_protection",
                                                  placeholder="如: SNS主动防护网(局部), 挡石墙")
            with col_p2:
                new_hz_recommended = st.text_input("建议防治措施", key="hz_recommended",
                                                   placeholder="如: 增设被动防护网+监测摄像头")

            # ── 边坡稳定性评估 ──
            st.markdown("#### 🔬 边坡稳定性评估")
            st.caption("建议联合地质部门填写以下评估参数")

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                new_slope_angle = st.number_input("坡度 (°)", key="hz_sa",
                                                  min_value=0.0, max_value=90.0,
                                                  step=1.0, value=45.0)
                new_slope_height = st.number_input("坡高 (m)", key="hz_sh",
                                                   min_value=0.0, max_value=500.0,
                                                   step=1.0, value=30.0)
                new_slope_type = st.selectbox("坡型", key="hz_st",
                                              options=["直线坡", "凸坡", "凹坡", "复合坡"])
                new_rock_type = st.selectbox("岩性", key="hz_rt",
                                             options=["石灰岩", "花岗岩", "砂岩", "页岩",
                                                      "砂岩夹页岩", "砂岩泥岩互层", "其他"])
            with col_b2:
                new_weathering = st.selectbox("风化程度", key="hz_wd",
                                              options=["强", "中", "弱", "未风化"])
                new_joint = st.selectbox("节理发育程度", key="hz_jd",
                                         options=["发育", "较发育", "不发育"])
                new_vegetation = st.selectbox("植被覆盖", key="hz_vc",
                                              options=["好", "中", "差", "裸露"])
                new_drainage = st.selectbox("排水条件", key="hz_dc",
                                            options=["好", "中", "差"])
            with col_b3:
                new_geo_score = st.slider("地质稳定性评分 (0-100)", key="hz_gs",
                                          min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                                          help="越低表示越不稳定")
                new_survey_team = st.text_input("勘察单位/人员", key="hz_survey_team",
                                                placeholder="如: XX地质工程勘察院")
                new_survey_remarks = st.text_area("勘察备注", key="hz_survey_remarks",
                                                  placeholder="坡顶裂缝、孤石分布、渗水情况等...",
                                                  height=68)

            # ── 历史灾害记录 ──
            st.markdown("#### 📜 历史灾害记录")
            st.caption("如有多条记录，请逐条添加。至少填写日期和类型。")

            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1:
                new_inc_date = st.date_input("发生日期", key="hz_inc_date")
            with col_c2:
                new_inc_type = st.selectbox("灾害类型", key="hz_inc_type",
                                            options=["落石 (rockfall)", "滑坡 (landslide)", "崩塌 (collapse)"])
            with col_c3:
                new_inc_severity = st.selectbox("严重程度", key="hz_inc_severity",
                                                options=["轻微 (minor)", "中等 (moderate)", "严重 (major)"])
            with col_c4:
                new_inc_casualties = st.text_input("伤亡情况", key="hz_inc_casualties",
                                                   placeholder="如: 无")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                new_inc_closure = st.text_input("道路中断情况", key="hz_inc_closure",
                                                placeholder="如: 半幅通行2小时")
            with col_d2:
                new_inc_source = st.text_input("数据来源", key="hz_inc_source",
                                               placeholder="如: XX市交通局养护记录")
            new_inc_desc = st.text_area("事件描述", key="hz_inc_desc",
                                        placeholder="简要描述事件经过...")

            # ── 提交 ──
            submitted = st.form_submit_button("💾 保存隐患点排查记录", use_container_width=True, type="primary")

            if submitted:
                if not new_hz_id.strip() or not new_hz_name.strip():
                    st.error("隐患点编号和名称不能为空")
                else:
                    try:
                        # 检查重复
                        existing = store.get_by_id(new_hz_id.strip()) if store else None
                        if existing is not None:
                            st.error(f"隐患点编号已存在: {new_hz_id.strip()}")
                        else:
                            # 构建边坡评估记录
                            slope_assessment = SlopeStability(
                                slope_angle=float(new_slope_angle),
                                slope_height=float(new_slope_height),
                                slope_type=new_slope_type,
                                rock_type=new_rock_type,
                                weathering_degree=new_weathering,
                                joint_development=new_joint,
                                vegetation_coverage=new_vegetation,
                                drainage_condition=new_drainage,
                                geological_score=float(new_geo_score),
                                survey_date=str(new_hz_surveyed),
                                survey_team=new_survey_team,
                                remarks=new_survey_remarks,
                            )

                            # 构建历史灾害记录
                            incidents = []
                            inc_type_map = {"落石 (rockfall)": "rockfall",
                                           "滑坡 (landslide)": "landslide",
                                           "崩塌 (collapse)": "collapse"}
                            inc_sev_map = {"轻微 (minor)": "minor",
                                          "中等 (moderate)": "moderate",
                                          "严重 (major)": "major"}
                            # 如果填写了灾害日期才添加
                            inc_date_str = str(new_inc_date) if new_inc_date else ""
                            if inc_date_str:
                                incident = HistoricalIncident(
                                    incident_date=inc_date_str,
                                    incident_type=inc_type_map.get(new_inc_type, "rockfall"),
                                    severity=inc_sev_map.get(new_inc_severity, "minor"),
                                    description=new_inc_desc,
                                    casualties=new_inc_casualties,
                                    road_closure=new_inc_closure,
                                    source=new_inc_source,
                                )
                                incidents.append(incident)

                            # 计算风险评分
                            score = calculate_risk_score(
                                slope_assessment,
                                [HistoricalIncident.from_dict(i.to_dict()) for i in incidents]
                            )
                            risk_lvl = determine_risk_level(score)

                            # 解析联系人
                            contact_parts = new_hz_contact.split("/") if new_hz_contact else ["", ""]
                            contact_person = contact_parts[0].strip() if len(contact_parts) > 0 else ""
                            contact_phone = contact_parts[1].strip() if len(contact_parts) > 1 else new_hz_contact.strip()

                            hazard = HazardPoint(
                                hazard_id=new_hz_id.strip(),
                                name=new_hz_name.strip(),
                                location=(new_hz_location.strip() or new_hz_name.strip()),
                                region=new_hz_region.strip(),
                                highway=new_hz_highway.strip(),
                                stake_mark=new_hz_stake.strip(),
                                latitude=float(new_hz_lat),
                                longitude=float(new_hz_lng),
                                risk_level=risk_lvl,
                                risk_score=score,
                                slope_assessments=[slope_assessment.to_dict()],
                                historical_incidents=[i.to_dict() for i in incidents],
                                status="assessed",
                                description=new_hz_desc.strip(),
                                responsible_unit=new_hz_responsible.strip(),
                                contact_person=contact_person,
                                contact_phone=contact_phone,
                                protection_measures=new_hz_protection.strip(),
                                recommended_measures=new_hz_recommended.strip(),
                                surveyed_at=str(new_hz_surveyed),
                            )

                            if store and store.insert(hazard):
                                st.success(
                                    f"✅ 隐患点 '{new_hz_name}' 已录入！\n\n"
                                    f"自动评估风险等级: {_RISK_LABELS.get(risk_lvl, risk_lvl)} "
                                    f"(综合评分: {score})"
                                )
                                st.rerun()
                            else:
                                st.error("写入数据库失败")
                    except Exception as e:
                        st.error(f"保存失败: {e}")

    # ═══════════════════ Tab 4: 评估报告 ═══════════════════
    with tab_report:
        st.subheader("隐患点清单及风险等级评估报告")
        st.caption("基于排查数据自动生成，可直接导出或打印。")

        if not hazards:
            st.info("暂无隐患点数据，请先录入排查信息。")
        else:
            report_md = generate_hazard_report(hazards)

            # 导出按钮
            col_r1, col_r2, col_r3 = st.columns([1, 1, 4])
            with col_r1:
                st.download_button(
                    label="📥 导出 Markdown",
                    data=report_md,
                    file_name=f"隐患点清单及风险评估报告_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_r2:
                # 导出 CSV 概要
                csv_rows = []
                for h in hazards:
                    csv_rows.append({
                        "隐患点编号": h.hazard_id,
                        "名称": h.name,
                        "区域": h.region,
                        "公路": h.highway,
                        "桩号": h.stake_mark,
                        "风险等级": _RISK_LABELS.get(h.risk_level, h.risk_level),
                        "风险评分": h.risk_score,
                        "历史灾害次数": len(h.historical_incidents),
                        "监测状态": "已部署" if h.linked_site_id else "待部署",
                        "排查日期": h.surveyed_at,
                    })
                csv_df = pd.DataFrame(csv_rows)
                csv_buffer = _io.BytesIO()
                csv_buffer.write(b'\xef\xbb\xbf')  # UTF-8 BOM for Excel compatibility
                csv_content = csv_df.to_csv(index=False)
                csv_buffer.write(csv_content.encode('utf-8'))
                st.download_button(
                    label="📊 导出 CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"隐患点清单_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # ── 渲染报告 ──
            st.divider()
            with st.container(height=600):
                st.markdown(report_md)

def _render_hazard_card(h: "HazardPoint", idx: int):
    """渲染单个隐患点卡片"""
    _RISK_LABELS = {
        "high": "🔴 高风险", "medium": "🟡 中风险", "low": "🟢 一般风险"
    }
    _RISK_COLORS = {
        "high": "#DC3545", "medium": "#FFC107", "low": "#28A745"
    }
    _STATUS_LABELS = {
        "identified": "🔍 已识别", "assessed": "📋 已评估",
        "monitored": "📡 监测中", "remediated": "🛠️ 已治理", "cleared": "✅ 已消除"
    }
    color = _RISK_COLORS.get(h.risk_level, "#999")
    level_label = _RISK_LABELS.get(h.risk_level, h.risk_level)
    status_label = _STATUS_LABELS.get(h.status, h.status)

    # 最新评估信息
    latest_sa = h.slope_assessments[-1] if h.slope_assessments else {}
    slope_info = ""
    if latest_sa:
        slope_info = (f"坡度 {latest_sa.get('slope_angle','?')}° / "
                      f"坡高 {latest_sa.get('slope_height','?')}m / "
                      f"{latest_sa.get('rock_type','?')} / "
                      f"地质评分 {latest_sa.get('geological_score','?')}")

    incident_count = len(h.historical_incidents)
    monitor_badge = "✅ 已部署" if h.linked_site_id else "⚠️ 待部署"

    with st.expander(
        f"{level_label} | {h.hazard_id} | {h.name} | {status_label} | 灾害{incident_count}次",
        expanded=(idx == 0 and h.risk_level == "high")
    ):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            **隐患点**: {h.name}
            **位置**: {h.region} / {h.location}
            **公路桩号**: {h.highway} {h.stake_mark}
            **坐标**: ({h.latitude:.4f}, {h.longitude:.4f})
            **边坡概况**: {slope_info}
            **综合描述**: {h.description or '待补充'}
            """)

            # 历史灾害
            if h.historical_incidents:
                st.markdown("**历史灾害记录**:")
                for inc in h.historical_incidents:
                    type_label = {"rockfall": "落石", "landslide": "滑坡", "collapse": "崩塌"}.get(
                        inc.get("incident_type", ""), inc.get("incident_type", ""))
                    sev_label = {"major": "严重", "moderate": "中等", "minor": "轻微"}.get(
                        inc.get("severity", ""), inc.get("severity", ""))
                    st.caption(
                        f"📅 {inc.get('incident_date','?')} | "
                        f"{type_label} | {sev_label} | "
                        f"{inc.get('description','')[:50]}..."
                    )
            else:
                st.caption("📜 无历史灾害记录")

            # 防治措施
            if h.protection_measures:
                st.caption(f"🛡️ 已有防护: {h.protection_measures}")
            if h.recommended_measures:
                st.caption(f"💡 建议措施: {h.recommended_measures}")

        with col2:
            st.markdown(f"""
            <div style="text-align:center;padding:0.5rem;border-radius:8px;
                        background:{color}15;border:2px solid {color};">
                <div style="font-size:0.7rem;color:#666;">风险评分</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};">{h.risk_score:.0f}</div>
                <div style="font-size:0.7rem;color:#666;">/100</div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(f"**状态**: {status_label}")
            st.caption(f"**监测**: {monitor_badge}")
            if h.linked_site_id:
                st.caption(f"站点: `{h.linked_site_id}`")
            if h.responsible_unit:
                st.caption(f"**责任单位**: {h.responsible_unit}")
            if h.surveyed_at:
                st.caption(f"**排查日期**: {h.surveyed_at}")

        # ── 操作按钮行 ──
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            # 关联监测点位
            if not h.linked_site_id:
                from rockfall.site_config import list_sites
                available_sites = list_sites()
                site_options = {s.site_id: f"{s.name} ({s.site_id})" for s in available_sites}

                link_site = st.selectbox(
                    "关联监测点位",
                    options=[""] + list(site_options.keys()),
                    format_func=lambda x: "选择点位..." if not x else site_options.get(x, x),
                    key=f"link_site_{h.hazard_id}_{idx}"
                )
                if link_site and st.button("🔗 关联", key=f"btn_link_{h.hazard_id}"):
                    h.linked_site_id = link_site
                    h.status = "monitored"
                    try:
                        _store = get_hazard_store()
                        if _store:
                            _store.update(h)
                    except Exception:
                        pass
                    st.success(f"已关联监测点位: {link_site}")
                    st.rerun()
            else:
                if st.button("🔓 解除关联", key=f"btn_unlink_{h.hazard_id}"):
                    h.linked_site_id = ""
                    h.status = "assessed"
                    try:
                        _store = get_hazard_store()
                        if _store:
                            _store.update(h)
                    except Exception:
                        pass
                    st.success("已解除关联")
                    st.rerun()

        with col_act2:
            # 快速更新状态
            new_status = st.selectbox(
                "更新状态",
                options=list(_STATUS_LABELS.keys()),
                format_func=lambda x: _STATUS_LABELS.get(x, x),
                index=list(_STATUS_LABELS.keys()).index(h.status) if h.status in _STATUS_LABELS else 0,
                key=f"status_{h.hazard_id}_{idx}"
            )
            if new_status != h.status:
                if st.button("✏️ 更新", key=f"btn_status_{h.hazard_id}"):
                    h.status = new_status
                    try:
                        _store = get_hazard_store()
                        if _store:
                            _store.update(h)
                    except Exception:
                        pass
                    st.success(f"状态已更新: {_STATUS_LABELS.get(new_status, new_status)}")
                    st.rerun()

        with col_act3:
            # 删除
            if st.button("🗑️ 删除", key=f"btn_del_{h.hazard_id}",
                        type="secondary"):
                # 二次确认
                st.session_state[f"confirm_del_{h.hazard_id}"] = True

            if st.session_state.get(f"confirm_del_{h.hazard_id}"):
                st.error(f"⚠️ 确认删除隐患点 {h.hazard_id}？此操作不可撤销。")
                cf1, cf2 = st.columns(2)
                with cf1:
                    if st.button("✅ 确认删除", key=f"cfm_del_{h.hazard_id}"):
                        try:
                            _store = get_hazard_store()
                            if _store:
                                _store.delete(h.hazard_id)
                        except Exception:
                            pass
                        st.session_state.pop(f"confirm_del_{h.hazard_id}", None)
                        st.success("已删除")
                        st.rerun()
                with cf2:
                    if st.button("❌ 取消", key=f"cancel_del_{h.hazard_id}"):
                        st.session_state.pop(f"confirm_del_{h.hazard_id}", None)
                        st.rerun()
