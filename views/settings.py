"""RockGuard Streamlit — 页面模块"""
import streamlit as st
from _shared import *

def page_settings():
    """参数设置页面: 调整检测和预警阈值"""
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="logo">参数设置</div>
            <div style="font-size:0.8rem;opacity:0.85;">检测阈值 &middot; 预警等级 &middot; 跳帧策略</div>
        </div>
        <div class="meta"><span>{APP_VERSION}</span></div>
    </div>
    """, unsafe_allow_html=True)

    detector = get_detector_or_stop()

    # ── 检测参数 ──
    st.subheader("检测参数")

    c1, c2 = st.columns(2)
    with c1:
        new_conf = st.slider(
            "检测置信度阈值",
            min_value=0.1, max_value=0.9, value=float(DETECTION_CONFIDENCE), step=0.05,
            help="YOLO 检测的最低置信度, 低于此值的检测框被丢弃",
        )
        new_img_size = st.selectbox(
            "推理图像尺寸",
            options=[320, 416, 480, 640, 800, 960, 1280],
            index=3,  # 640
            help="YOLO 推理时的输入尺寸, 越大越精确但越慢",
        )
    with c2:
        new_min_area = st.slider(
            "最小运动区域 (像素)",
            min_value=50, max_value=2000, value=MOTION_MIN_AREA, step=50,
            help="MOG2 运动检测的最小连通区域面积",
        )
        new_mog2_lr = st.slider(
            "MOG2 学习率",
            min_value=0.0001, max_value=0.1, value=MOG2_LEARNING_RATE, step=0.0005, format="%.4f",
            help="背景模型更新速度, 越小越稳定但适应变化越慢",
        )

    # ── 四级预警阈值 ──
    st.divider()
    st.subheader("四级预警置信度阈值")
    st.caption("对齐《公路自然灾害监测预警系统技术指南》第5.3节强制要求。")

    c1, c2 = st.columns(2)
    with c1:
        blue_low = st.slider(
            "🔵 Ⅳ级(蓝色)下限",
            min_value=0.1, max_value=0.5, value=float(ALERT_BLUE_CONFIDENCE_LOW), step=0.05,
            help="置信度 ≥ 此值 → Ⅳ级预警",
        )
        yellow_high = st.slider(
            "🟡 Ⅲ级(黄色)上限",
            min_value=0.4, max_value=0.8, value=float(ALERT_YELLOW_CONFIDENCE_HIGH), step=0.05,
            help="置信度 ≥ 此值 → Ⅱ级预警",
        )
    with c2:
        blue_high = st.slider(
            "🔵→🟡 蓝黄分界",
            min_value=0.2, max_value=0.6, value=float(ALERT_BLUE_CONFIDENCE_HIGH), step=0.05,
            help="置信度 ≥ 此值 → Ⅲ级预警",
        )
        orange_high = st.slider(
            "🟠 Ⅱ级(橙色)上限",
            min_value=0.7, max_value=0.99, value=float(ALERT_ORANGE_CONFIDENCE_HIGH), step=0.05,
            help="置信度 ≥ 此值 → Ⅰ级预警",
        )

    # ── 跳帧策略 ──
    st.divider()
    st.subheader("自适应跳帧策略")
    st.caption("基于运动强度自动调整推理频率, 平衡实时性与算力消耗。")

    c1, c2, c3 = st.columns(3)
    with c1:
        new_skip_idle = st.slider(
            "静止跳帧 (每N帧推理1次)",
            min_value=1, max_value=30, value=SKIP_IDLE, step=1,
            help="无运动时的跳帧间隔, 越大越省算力",
        )
    with c2:
        new_skip_active = st.slider(
            "弱运动跳帧",
            min_value=1, max_value=15, value=SKIP_ACTIVE, step=1,
            help="弱运动时的跳帧间隔",
        )
    with c3:
        new_skip_critical = st.slider(
            "强运动跳帧",
            min_value=1, max_value=5, value=SKIP_CRITICAL, step=1,
            help="强运动时的跳帧间隔, 越小检测越密集",
        )

    new_motion_low = st.slider(
        "运动得分阈值",
        min_value=0.001, max_value=0.2, value=(float(MOTION_SCORE_LOW), float(MOTION_SCORE_HIGH)),
        step=0.005, format="%.3f",
        help="(低阈值, 高阈值): 低=静止→弱运动分界, 高=弱运动→强运动分界",
    )

    # ── 早期预警：检测密度爆发 ──
    st.divider()
    st.subheader("🔬 早期预警 — 检测密度爆发")
    st.caption("远景小落石置信度低 (0.10-0.30), 但短时间内检出数量骤增即是前兆信号。"
               "开启后系统自动统计滚动窗口内的检测密度, 异常飙升时触发预警升级。")

    c1, c2, c3 = st.columns(3)
    with c1:
        density_enabled = st.checkbox(
            "启用密度爆发监测",
            value=DENSITY_ALERT_ENABLED,
            help="开启后检测密度异常飙升时自动升级预警等级 (宜宾滑坡验证有效)",
        )
    with c2:
        density_window = st.slider(
            "滚动窗口 (秒)",
            min_value=5, max_value=60, value=DENSITY_WINDOW_SEC, step=5,
            help="统计过去N秒内的检测框数量作为基线",
        )
    with c3:
        density_zscore = st.slider(
            "爆发 z-score 阈值",
            min_value=1.5, max_value=5.0, value=float(DENSITY_BURST_ZSCORE), step=0.5,
            help="当前帧检测数超过基线均值N倍标准差时触发爆发告警",
        )

    c1, c2 = st.columns(2)
    with c1:
        density_conf_floor = st.slider(
            "密度统计置信度下限",
            min_value=0.05, max_value=0.30, value=float(DENSITY_CONF_FLOOR), step=0.05,
            help="置信度≥此值的检测框才纳入密度统计, 过滤纯噪声",
        )
    with c2:
        density_min_samples = st.slider(
            "最少样本数 (帧)",
            min_value=60, max_value=600, value=DENSITY_MIN_SAMPLES, step=30,
            help="收集足够样本后才开始判定, 给MOG2预热时间",
        )

    # ── 应用按钮 ──
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("应用参数", type="primary", use_container_width=True):
            # 更新检测器实例参数 (当前会话立即生效)
            detector.confidence = new_conf
            detector.img_size = new_img_size
            detector.min_area = new_min_area
            detector.alert_blue_conf_high = blue_high
            detector.alert_yellow_conf_high = yellow_high
            detector.alert_orange_conf_high = orange_high

            # 密度监测参数 (热更新 DensityMonitor)
            if detector._density_monitor is not None and density_enabled:
                detector._density_monitor.window_sec = density_window
                detector._density_monitor.burst_zscore = density_zscore
                detector._density_monitor.conf_floor = density_conf_floor
                detector._density_monitor.min_samples = density_min_samples

            # 更新 RuntimeConfig 单例 (跨会话持久化, 新检测器启动时自动读取)
            from rockfall.config import RuntimeConfig as _RC
            _RC.set_batch({
                "DETECTION_CONFIDENCE": new_conf,
                "DETECTION_IMG_SIZE": new_img_size,
                "MOTION_MIN_AREA": new_min_area,
                "ALERT_BLUE_CONFIDENCE_HIGH": blue_high,
                "ALERT_YELLOW_CONFIDENCE_HIGH": yellow_high,
                "ALERT_ORANGE_CONFIDENCE_HIGH": orange_high,
                "SKIP_IDLE": new_skip_idle,
                "SKIP_ACTIVE": new_skip_active,
                "SKIP_CRITICAL": new_skip_critical,
                "MOTION_SCORE_LOW": new_motion_low[0],
                "MOTION_SCORE_HIGH": new_motion_low[1],
                "MOG2_LEARNING_RATE": new_mog2_lr,
                "DENSITY_ALERT_ENABLED": "true" if density_enabled else "false",
                "DENSITY_WINDOW_SEC": str(density_window),
                "DENSITY_BURST_ZSCORE": str(density_zscore),
                "DENSITY_CONF_FLOOR": str(density_conf_floor),
                "DENSITY_MIN_SAMPLES": str(density_min_samples),
            })

            # 更新会话状态
            st.session_state.detection_confidence = new_conf
            st.session_state.detection_img_size = new_img_size
            st.session_state.motion_min_area = new_min_area
            st.session_state.alert_blue_low = blue_low
            st.session_state.alert_blue_high = blue_high
            st.session_state.alert_yellow_high = yellow_high
            st.session_state.alert_orange_high = orange_high
            st.session_state.skip_idle = new_skip_idle
            st.session_state.skip_active = new_skip_active
            st.session_state.skip_critical = new_skip_critical
            st.session_state.motion_score_low = new_motion_low[0]
            st.session_state.motion_score_high = new_motion_low[1]
            st.session_state.mog2_learning_rate = new_mog2_lr
            st.session_state.density_alert_enabled = density_enabled
            st.session_state.density_window_sec = density_window
            st.session_state.density_burst_zscore = density_zscore
            st.session_state.density_conf_floor = density_conf_floor

            st.success("参数已应用 (含密度爆发监测)")

    with c2:
        if st.button("恢复默认", use_container_width=True):
            from rockfall.config import RuntimeConfig as _RC
            _RC.reset()  # 清除所有运行时覆盖
            for k, v in DEFAULT_PARAMS.items():
                st.session_state[k] = v
            # 恢复检测器参数
            detector.confidence = DEFAULT_PARAMS["detection_confidence"]
            detector.img_size = DEFAULT_PARAMS["detection_img_size"]
            detector.min_area = DEFAULT_PARAMS["motion_min_area"]
            detector.alert_blue_conf_high = DEFAULT_PARAMS["alert_blue_high"]
            detector.alert_yellow_conf_high = DEFAULT_PARAMS["alert_yellow_high"]
            detector.alert_orange_conf_high = DEFAULT_PARAMS["alert_orange_high"]
            st.rerun()

    # ── 当前配置状态 ──
    with st.expander("当前完整配置", expanded=False):
        from rockfall.config import RuntimeConfig as _RC
        _overrides = _RC.get_all_overrides()
        col_json, col_overrides = st.columns([1, 1])
        with col_json:
            st.caption("检测器当前参数")
            st.json({
                "detection": {
                    "confidence": detector.confidence,
                    "img_size": detector.img_size,
                    "min_area": detector.min_area,
                },
                "alert_thresholds": {
                    "blue_low": blue_low,
                    "blue_high": detector.alert_blue_conf_high,
                    "yellow_high": detector.alert_yellow_conf_high,
                    "orange_high": detector.alert_orange_conf_high,
                },
                "skip_strategy": {
                    "idle": st.session_state.get("skip_idle", SKIP_IDLE),
                    "active": st.session_state.get("skip_active", SKIP_ACTIVE),
                    "critical": st.session_state.get("skip_critical", SKIP_CRITICAL),
                },
                "device": config_get_device(),
            })
        with col_overrides:
            st.caption("RuntimeConfig 覆盖")
            if _overrides:
                st.json(_overrides)
            else:
                st.info("无覆盖 (全部使用默认值)")
