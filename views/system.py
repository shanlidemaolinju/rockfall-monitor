"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

def page_system():
    """系统管理页面: 健康检查 + 审计日志 + 存储管理 + 工单统计"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">系统管理</div>
            <div style="font-size:0.8rem;opacity:0.85;">健康检查 &middot; 审计日志 &middot; 存储管理 &middot; 工单统计</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["健康检查", "审计日志", "存储管理", "工单统计", "数据完整性", "智能调度"])

    # ── Tab 1: 系统健康检查 ──
    with tab1:
        st.markdown("### 系统健康检查")
        if st.button("运行健康检查", key="sys_health_check", use_container_width=True):
            try:
                import requests
                
                r = requests.get(build_api_url("/api/health/full"), timeout=5)
                health = r.json()
            except Exception:
                health = {"healthy": False, "warnings": ["API 服务不可达，请确认 FastAPI 已启动"]}

            healthy = health.get("healthy", False)
            st.markdown(f"""
            <div style="padding:1rem;border-radius:8px;margin-bottom:0.75rem;
                        background:{'#E8F5E9' if healthy else '#FFEBEE'};
                        border:2px solid {'#2E7D32' if healthy else '#D32F2F'};">
                <div style="font-size:1.2rem;font-weight:700;color:{'#2E7D32' if healthy else '#D32F2F'};">
                    {'正常' if healthy else '异常'}
                </div>
                <div style="font-size:0.75rem;color:#5F6B7A;">运行时间: {health.get('uptime_hours', 'N/A')}h | 故障次数: {health.get('fail_count', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

            if health.get("warnings"):
                for w in health["warnings"]:
                    st.warning(w)

            checks = health.get("checks", {})
            if checks:
                c1, c2, c3 = st.columns(3)
                for i, (key, val) in enumerate(checks.items()):
                    col = [c1, c2, c3][i % 3]
                    if isinstance(val, dict):
                        with col:
                            st.metric(key, val.get("percent", "N/A") if isinstance(val.get("percent"), (int, float)) else str(val.get("exists", val.get("writable", "N/A"))))

    # ── Tab 2: 审计日志 ──
    with tab2:
        st.markdown("### 审计日志")
        try:
            import requests
            
            r = requests.get(build_api_url("/api/audit?limit=50"), timeout=5)
            data = r.json()
            rows = data.get("rows", [])
            total = data.get("total", 0)

            st.caption(f"共 {total} 条记录 (显示最近 {len(rows)} 条)")

            if rows:
                df = pd.DataFrame([{
                    "ID": r["id"], "操作": r["action"], "操作人": r["operator"],
                    "详情": r["detail"][:80], "预警ID": r["alert_id"],
                    "结果": r["result"], "时间": r["created_at"],
                } for r in rows])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("暂无审计记录")
        except Exception as e:
            st.warning(f"审计API不可用: {e}")

    # ── Tab 3: 存储管理 ──
    with tab3:
        st.markdown("### 存储管理")
        try:
            import requests
            
            r = requests.get(build_api_url("/api/health/storage"), timeout=5)
            stats = r.json()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**目录使用情况**")
                for name, info in stats.items():
                    if name == "total_mb":
                        continue
                    st.metric(name, f"{info['size_mb']:.0f} MB", delta=f"{info['file_count']} 个文件")
            with c2:
                total_mb = stats.get("total_mb", 0)
                quota_mb = 10000
                st.metric("总存储量", f"{total_mb:.0f} MB",
                         delta=f"配额: {quota_mb}MB ({total_mb/quota_mb*100:.0f}%)" if quota_mb > 0 else "")

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                retention = st.number_input("保留天数", 7, 365, 30, key="sys_retention")
            with c2:
                st.write("")
                if st.button("运行清理 (试运行)", key="sys_cleanup_dry", use_container_width=True):
                    r = requests.post(build_api_url("/api/health/cleanup"),
                                     data={"retention_days": retention, "dry_run": True})
                    result = r.json()
                    st.info(f"将删除 {result['deleted_count']} 个文件, 释放 {result['freed_mb']} MB")
        except Exception as e:
            st.warning(f"存储API不可用: {e}")

    # ── Tab 4: 工单统计 ──
    with tab4:
        st.markdown("### 工单统计")
        try:
            import requests
            
            r = requests.get(build_api_url("/api/workflow/stats"), timeout=5)
            wf_stats = r.json()

            cols = st.columns(4)
            for i, (state, info) in enumerate(wf_stats.items()):
                with cols[i % 4]:
                    count = info.get("count", 0)
                    st.metric(info.get("label", state), count)

            st.divider()
            st.markdown("**工单状态流转**")
            st.markdown("""
            | 从 | 到 | 说明 |
            |------|-----|-------------|
            | 待确认 | 已确认 | 预警核验为真实落石 |
            | 待确认 | 误报 | 预警标记为误报 |
            | 已确认 | 已派单 | 已派单给现场人员 |
            | 已派单 | 已到场 | 人员到达现场 |
            | 已到场 | 已处理 | 情况已处置 |
            | 已处理 | 已归档 | 工单已归档 |
            """)
        except Exception as e:
            st.warning(f"工单API不可用: {e}")

    # ── Tab 5: 数据完整性校验 (哈希链) ──
    with tab5:
        st.markdown("### 数据完整性 — 哈希链校验")
        st.caption("SHA256 链式校验: 每条预警记录链接上一条的哈希值, 形成防篡改证据链。")

        try:
            from rockfall.hash_chain import build_chain, verify_chain_batch
            from rockfall.alert_store import get_alert_store as _gs
            _store = _gs()
            _recent = _store.query_alerts(limit=50, offset=0)

            if _recent:
                genesis = "0" * 64

                # 批量验证哈希链
                result = verify_chain_batch(_recent, genesis)
                total = result.get("total", len(_recent))
                ok = result.get("valid", 0)
                bad = result.get("invalid", 0)
                breaks = result.get("breaks", [])

                # 状态展示
                if bad == 0:
                    st.success(f"所有 {ok} 条记录哈希链完整, 数据未被篡改。")
                else:
                    st.error(f"发现 {bad} 条记录哈希链断裂！断裂位置: {breaks}")

                # 哈希链详情表
                st.markdown("**哈希链样本 (最近 10 条)**")
                hashes = build_chain(_recent, genesis)
                chain_data = []
                for i, record in enumerate(_recent[:10]):
                    h = hashes[i] if i < len(hashes) else "N/A"
                    chain_data.append({
                        "ID": record.get("id", ""),
                        "时间": str(record.get("time", ""))[:19],
                        "等级": record.get("alert_level", ""),
                        "Hash": h[:16] + "..." if len(h) > 16 else h,
                    })
                st.dataframe(chain_data, use_container_width=True, hide_index=True)

                # 验证原理说明
                with st.expander("哈希链验证原理", expanded=False):
                    st.markdown("""
                    **SHA256 哈希链防篡改机制:**
                    1. 每条预警记录的 `data_hash = SHA256(time|level|count|conf|tracks|...|prev_hash)`
                    2. `prev_hash` 指向上一条记录的 `data_hash`
                    3. 首条记录使用创世哈希 (全零)
                    4. **任何记录被修改** → 其 `data_hash` 不匹配 → 后续所有记录的 `prev_hash` 全部断裂

                    **验证公式:**
                    ```
                    recalculated_hash = SHA256(record_fields + prev_record_hash)
                    if recalculated_hash != stored_hash → 记录被篡改!
                    ```
                    """)
            else:
                st.info("暂无预警记录可用于哈希链验证。检测到落石后会自动生成。")
        except ImportError as e:
            st.warning(f"哈希链模块不可用: {e}")
        except Exception as e:
            st.warning(f"哈希链验证暂不可用: {e}")

    # ── Tab 6: 智能调度 (阈值调参 + 预警升级 + 模型热切换) ──
    with tab6:
        st.markdown("### 智能调度 — 自动运维面板")
        st.caption("阈值自动调参、预警自动升级、夜间模型热切换。Streamlit 直连，无需外部 API。")

        # ── 子面板 A: 阈值自动调参 ──
        st.markdown("#### 阈值自动调参")
        if ROCKFALL_AVAILABLE:
            try:
                from rockfall.threshold_tuner import get_tuner
                from rockfall.config import (
                    THRESHOLD_AUTO_TUNE_ENABLED,
                    THRESHOLD_AUTO_TUNE_INTERVAL_HOURS,
                    THRESHOLD_AUTO_TUNE_TARGET_FP_RATE,
                    THRESHOLD_AUTO_TUNE_CONF_MIN,
                    THRESHOLD_AUTO_TUNE_CONF_MAX,
                )
                tuner = get_tuner()

                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                col_a1.metric("状态", "已启用" if THRESHOLD_AUTO_TUNE_ENABLED else "已禁用")
                col_a2.metric("累计调整", tuner.total_adjustments)
                col_a3.metric("收紧", tuner.total_tighten)
                col_a4.metric("放松", tuner.total_relax)

                with st.expander("配置详情", expanded=False):
                    st.json({
                        "扫描间隔(小时)": THRESHOLD_AUTO_TUNE_INTERVAL_HOURS,
                        "目标误报率": f"{THRESHOLD_AUTO_TUNE_TARGET_FP_RATE:.0%}",
                        "置信度范围": f"[{THRESHOLD_AUTO_TUNE_CONF_MIN}, {THRESHOLD_AUTO_TUNE_CONF_MAX}]",
                        "上次运行": tuner.last_run_time or "N/A",
                        "上次结果": tuner.last_result,
                    })

                if st.button("手动触发调参", key="tuner_trigger", use_container_width=True):
                    result = tuner.trigger_now()
                    st.json(result)
            except Exception as e:
                st.warning(f"调参模块不可用: {e}")
        else:
            st.info("rockfall 核心库未安装。")

        st.divider()

        # ── 子面板 B: 预警自动升级 ──
        st.markdown("#### 预警自动升级")
        if ROCKFALL_AVAILABLE:
            try:
                from rockfall.alert_escalation import get_escalation_scheduler
                from rockfall.config import (
                    ALERT_ESCALATION_ENABLED,
                    ALERT_ESCALATION_TIMEOUT_MINUTES,
                    ALERT_ESCALATION_MIN_LEVEL,
                )
                scheduler = get_escalation_scheduler()

                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("状态", "已启用" if ALERT_ESCALATION_ENABLED else "已禁用")
                col_b2.metric("累计升级", scheduler.total_escalations)
                col_b3.metric("超时阈值", f"{ALERT_ESCALATION_TIMEOUT_MINUTES} 分钟")

                with st.expander("升级规则", expanded=False):
                    st.json({
                        "超时分钟": ALERT_ESCALATION_TIMEOUT_MINUTES,
                        "最低触发等级": ALERT_ESCALATION_MIN_LEVEL,
                        "升级链": "blue → yellow → orange → red",
                        "上次运行": scheduler.last_run_time or "N/A",
                    })

                if st.button("手动触发升级", key="esc_trigger", use_container_width=True):
                    result = scheduler.trigger_now()
                    st.json(result)
            except Exception as e:
                st.warning(f"升级模块不可用: {e}")
        else:
            st.info("rockfall 核心库未安装。")

        st.divider()

        # ── 子面板 C: 模型热切换 ──
        st.markdown("#### 模型热切换状态")
        if ROCKFALL_AVAILABLE:
            try:
                from rockfall.config import (
                    MODEL_NIGHT_PATH, MODEL_SLOT_MAP,
                    get_active_model_path, _get_model_for_hour,
                )
                from datetime import datetime
                from pathlib import Path

                current_hour = datetime.now().hour
                active_path = get_active_model_path()
                night_model = _get_model_for_hour(current_hour)
                is_night = night_model is not None
                night_exists = Path(MODEL_NIGHT_PATH).exists() if MODEL_NIGHT_PATH else False

                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.metric("当前时段", f"{current_hour}:00",
                             delta="夜间模式" if is_night else "白天模式",
                             delta_color="off" if is_night else "normal")
                col_c2.metric("激活模型",
                             active_path.name if active_path.exists() else "unknown")
                col_c3.metric("夜间模型",
                             "就绪" if night_exists else "未部署",
                             delta=MODEL_NIGHT_PATH[:25] if MODEL_NIGHT_PATH else None)

                with st.expander("时段路由详情", expanded=False):
                    st.json({
                        "当前小时": current_hour,
                        "夜间模式启用": bool(MODEL_NIGHT_PATH),
                        "夜间模型路径": MODEL_NIGHT_PATH or "",
                        "夜间模型存在": night_exists,
                        "当前是否夜间": is_night,
                        "当前应使用模型": str(night_model) if night_model else "默认(rock_best.pt)",
                        "时段映射(SLOT_MAP)": MODEL_SLOT_MAP or "未配置",
                    })
            except Exception as e:
                st.warning(f"模型状态不可用: {e}")
        else:
            st.info("rockfall 核心库未安装。")
