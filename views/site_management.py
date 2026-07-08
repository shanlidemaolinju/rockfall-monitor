"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

def page_site_management():
    """点位管理页面: 查看/切换监测点位"""
    active_site = get_active_site() if get_active_site is not None else None
    all_sites = list_sites()

    # 保底：确保 PRESET_SITES 中的所有点位都出现在列表中（即使 DB 同步失败）
    from rockfall.site_config import PRESET_SITES, get_site_store
    db_ids = {s.site_id for s in all_sites}
    missing = [ps for ps in PRESET_SITES if ps.site_id not in db_ids]
    if missing:
        try:
            store = get_site_store()
            store.seed_from_presets(missing)
        except Exception:
            pass
        all_sites.extend(missing)

    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">点位管理</div>
            <div style="font-size:0.8rem;opacity:0.85;">{len(all_sites)} 个监测点位</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── 当前激活点位 ──
    st.subheader("当前点位")
    if active_site is None:
        st.warning("⚠️ 未能加载激活点位，请检查系统配置和数据库连接")
    else:
        with st.container():
            _render_site_card(active_site, is_active=True, show_detail=True)

    # ── 系统配置验证 ──
    st.divider()
    with st.expander("系统配置检查", expanded=False):
        warnings = validate_config()
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("所有配置项正常")

    # ── 全部预设点位 ──
    st.divider()
    st.subheader("预设点位")
    st.caption(f"{len(all_sites)} 个点位可用，点击下方按钮切换。")

    cols = st.columns(2)
    for i, site in enumerate(all_sites):
        is_active = (active_site is not None and site.site_id == active_site.site_id)
        with cols[i % 2]:
            _render_site_card(site, is_active=is_active, show_detail=True)

            if not is_active:
                if st.button(
                    "启用此点位",
                    key=f"switch_{site.site_id}",
                    use_container_width=True,
                ):
                    try:
                        new_site = set_active_site(site.site_id)
                        st.success(f"已启用: {new_site.name}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    # ── 新增点位 ──
    st.divider()
    with st.expander("➕ 新增点位", expanded=False):
        with st.form("add_site_form"):
            st.caption("填写点位信息以添加新的监测点位")
            col1, col2 = st.columns(2)
            with col1:
                new_site_id = st.text_input(
                    "点位ID *", key="new_site_id",
                    placeholder="如: gl_ys_s5",
                    help="唯一标识符，仅允许字母、数字、下划线和连字符")
                new_name = st.text_input(
                    "点位名称 *", key="new_site_name",
                    placeholder="如: 桂林阳朔高速 5 号边坡")
                new_location = st.text_input(
                    "地理位置", key="new_site_location",
                    placeholder="报警推送中显示的位置名称")
                new_region = st.text_input(
                    "所属区域", key="new_site_region",
                    placeholder="如: 广西·桂林")
                new_lat = st.number_input(
                    "纬度", key="new_site_lat",
                    min_value=0.0, max_value=90.0, step=0.001,
                    value=22.817, format="%.3f")
                new_lng = st.number_input(
                    "经度", key="new_site_lng",
                    min_value=0.0, max_value=180.0, step=0.001,
                    value=108.366, format="%.3f")
            with col2:
                new_highway = st.text_input(
                    "所属公路", key="new_site_highway",
                    placeholder="如: G75 兰海高速")
                new_stake = st.text_input(
                    "桩号", key="new_site_stake",
                    placeholder="如: K1952+300")
                new_risk = st.selectbox(
                    "风险等级", key="new_site_risk",
                    options=["high", "medium", "low"],
                    format_func=lambda x: {"high": "⚠️ 高风险", "medium": "🔶 中风险", "low": "🟢 低风险"}.get(x, x))
                new_camera = st.text_input(
                    "摄像头地址 (RTSP/HTTP)", key="new_site_camera",
                    placeholder="rtsp://... 或留空")
                new_is_active = st.selectbox(
                    "是否启用", key="new_site_active",
                    options=[True, False],
                    format_func=lambda x: "✅ 启用" if x else "❌ 停用")
            new_desc = st.text_area(
                "点位描述", key="new_site_desc",
                placeholder="点位描述信息...")

            submitted = st.form_submit_button("💾 保存点位", use_container_width=True)
            if submitted:
                if not new_site_id.strip():
                    st.error("点位ID 不能为空")
                elif not new_name.strip():
                    st.error("点位名称不能为空")
                else:
                    try:
                        from rockfall.site_config import (
                            MonitoringSite, get_site_store, get_site_by_id,
                        )
                        # 检查重复
                        existing = get_site_by_id(new_site_id.strip())
                        if existing is not None:
                            st.error(f"点位ID已存在: {new_site_id.strip()}")
                        else:
                            site = MonitoringSite(
                                site_id=new_site_id.strip(),
                                name=new_name.strip(),
                                location=(new_location.strip() or new_name.strip()),
                                region=new_region.strip(),
                                camera_url=new_camera.strip(),
                                description=new_desc.strip(),
                                latitude=float(new_lat),
                                longitude=float(new_lng),
                                highway=new_highway.strip(),
                                stake_mark=new_stake.strip(),
                                risk_level=new_risk,
                                is_active=bool(new_is_active),
                            )
                            store = get_site_store()
                            if store.insert(site):
                                st.success(f"✅ 点位 '{new_name}' 已添加")
                                st.rerun()
                            else:
                                st.error("写入数据库失败")
                    except Exception as e:
                        st.error(f"保存失败: {e}")

    # ── 点位阈值配置 ──
    st.divider()
    st.subheader("检测灵敏度")
    _render_site_threshold_editor(active_site)

    # ── ROI 配置 ──
    st.divider()
    st.subheader("ROI 标定")

    from rockfall.site_config import load_site_config
    roi_params, polygon, road_mask = load_site_config(active_site.site_id)

    if roi_params is not None:
        st.success(f"已有 ROI 标定数据 (最近校准: {active_site.site_id})")
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
        st.info("该点位尚未进行 ROI 标定, 将使用默认 ROI 区域。")

def _render_site_threshold_editor(site):
    """点位灵敏度配置——快捷预设 + 精细调节"""
    if site is None:
        st.info("请先选择一个点位")
        return

    thresholds = _get_site_thresholds(site)

    # ── 灵敏度预设 ──
    st.caption("快捷预设")
    preset_cols = st.columns(4)
    presets = {
        "🔴 高敏": {"conf": 0.10, "blue_low": 0.10, "blue_high": 0.20, "yellow_high": 0.40, "orange_high": 0.70,
                    "desc": "宜宾滑坡级别，宁误报不漏报"},
        "🟠 中高": {"conf": 0.20, "blue_low": 0.20, "blue_high": 0.35, "yellow_high": 0.55, "orange_high": 0.80,
                    "desc": "山区高风险边坡"},
        "🟡 标准": {"conf": 0.30, "blue_low": 0.30, "blue_high": 0.50, "yellow_high": 0.70, "orange_high": 0.90,
                    "desc": "通用默认"},
        "🟢 低敏": {"conf": 0.45, "blue_low": 0.40, "blue_high": 0.60, "yellow_high": 0.80, "orange_high": 0.95,
                    "desc": "城市道路，减少告警"},
    }

    preset_clicked = None
    for i, (label, cfg) in enumerate(presets.items()):
        with preset_cols[i]:
            if st.button(label, key=f"preset_{label}_{site.site_id}", use_container_width=True,
                        help=cfg["desc"]):
                preset_clicked = cfg

    # ── 精细调节 ──
    with st.expander("精细调节", expanded=False):
        if preset_clicked:
            conf_val = preset_clicked["conf"]
            blue_low_val = preset_clicked["blue_low"]
            blue_high_val = preset_clicked["blue_high"]
            yellow_high_val = preset_clicked["yellow_high"]
            orange_high_val = preset_clicked["orange_high"]
        else:
            conf_val = thresholds["detection_confidence"]
            blue_low_val = thresholds["alert_blue_low"]
            blue_high_val = thresholds["alert_blue_high"]
            yellow_high_val = thresholds["alert_yellow_high"]
            orange_high_val = thresholds["alert_orange_high"]

        c1, c2 = st.columns(2)
        with c1:
            new_conf = st.slider("YOLO检测置信度", 0.05, 0.80, conf_val, 0.01,
                                help="越低检出越多，越高误报越少")
            new_blue_low = st.slider("🔵 蓝色预警下限 (准入)", 0.05, 0.50, blue_low_val, 0.01,
                                    help="置信度≥此值进入IV级预警")
            new_blue_high = st.slider("🔵→🟡 蓝黄分界", 0.10, 0.60, blue_high_val, 0.01,
                                     help="置信度≥此值升为III级")

        with c2:
            new_yellow_high = st.slider("🟡→🟠 黄橙分界", 0.30, 0.80, yellow_high_val, 0.01,
                                       help="置信度≥此值升为II级")
            new_orange_high = st.slider("🟠→🔴 橙红分界", 0.60, 0.99, orange_high_val, 0.01,
                                       help="置信度≥此值升为I级")

        # 当前有效值提示
        if not preset_clicked:
            overrides = []
            if site.detection_confidence > 0:
                overrides.append(f"conf={site.detection_confidence}")
            if site.alert_blue_low > 0:
                overrides.append(f"blue_low={site.alert_blue_low}")
            if overrides:
                st.caption(f"点位覆盖: {', '.join(overrides)}")
            else:
                st.caption("使用全局默认值 (.env)")

        if st.button("💾 保存阈值", key=f"save_thresh_{site.site_id}", type="primary",
                     use_container_width=True):
            site.detection_confidence = new_conf
            site.alert_blue_low = new_blue_low
            site.alert_blue_high = new_blue_high
            site.alert_yellow_high = new_yellow_high
            site.alert_orange_high = new_orange_high

            try:
                from rockfall.site_config import get_site_store
                store = get_site_store()
                if store.get_by_id(site.site_id):
                    store.update(site)
                else:
                    store.insert(site)
                st.success(f"已保存 {site.name} 的阈值配置")
                st.cache_resource.clear()  # 清除检测器缓存，下次使用新阈值
                st.rerun()
            except Exception as e:
                st.error(f"保存失败: {e}")

def _get_site_thresholds(site) -> dict:
    """获取点位的有效检测阈值（兼容新旧版本 MonitoringSite）。

    优先使用点位级配置 > 0 的值，否则回退到全局默认 (config.py)。
    完全不依赖 MonitoringSite.get_thresholds 方法，兼容任何版本的 MonitoringSite。
    """
    from rockfall.config import (
        DETECTION_CONFIDENCE as _DEF_CONF,
        ALERT_BLUE_CONFIDENCE_LOW as _DEF_BLUE_LOW,
        ALERT_BLUE_CONFIDENCE_HIGH as _DEF_BLUE_HIGH,
        ALERT_YELLOW_CONFIDENCE_HIGH as _DEF_YELLOW_HIGH,
        ALERT_ORANGE_CONFIDENCE_HIGH as _DEF_ORANGE_HIGH,
    )

    def _val(field_name: str, default: float) -> float:
        v = getattr(site, field_name, 0)
        if v is None:
            v = 0
        return float(v) if float(v) > 0 else default

    return {
        "detection_confidence": _val("detection_confidence", _DEF_CONF),
        "alert_blue_low": _val("alert_blue_low", _DEF_BLUE_LOW),
        "alert_blue_high": _val("alert_blue_high", _DEF_BLUE_HIGH),
        "alert_yellow_high": _val("alert_yellow_high", _DEF_YELLOW_HIGH),
        "alert_orange_high": _val("alert_orange_high", _DEF_ORANGE_HIGH),
    }

def _render_site_card(site: MonitoringSite, is_active: bool = False, show_detail: bool = False):
    """渲染单个点位卡片"""
    if site is None:
        st.warning("⚠️ 暂无激活的监测点位，请先选择一个点位")
        return

    thresholds = _get_site_thresholds(site)

    border_style = "2px solid #0d6efd" if is_active else "1px solid #dee2e6"
    bg_style = "#f0f7ff" if is_active else "#ffffff"

    with st.container():
        # 阈值信息
        conf = thresholds["detection_confidence"]
        blue_low = thresholds["alert_blue_low"]
        thresh_str = f"🔍检测:{conf:.2f} 🔵≥{blue_low:.2f} 🟡≥{thresholds['alert_blue_high']:.2f} 🟠≥{thresholds['alert_yellow_high']:.2f} 🔴≥{thresholds['alert_orange_high']:.2f}"

        st.markdown(f"""
        <div style="padding:1rem; border-radius:8px; border:{border_style}; background:{bg_style}; margin-bottom:0.5rem;">
            <b>{'' if is_active else ''}{site.name}</b>
            <span style="float:right;">{RISK_LABELS.get(site.risk_level, site.risk_level)}</span>
            <br><small>{site.region} | 🛣️ {site.highway} | 🏷️ {site.stake_mark}</small>
            <br><small>{site.description}</small>
            <br><small>🌐 经纬度: {site.latitude:.3f}, {site.longitude:.3f}</small>
            <br><small style="font-family:monospace;color:#1565C0;">{thresh_str}</small>
        </div>
        """, unsafe_allow_html=True)
