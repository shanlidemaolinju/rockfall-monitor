"""
预警分级决策树 — 四级预警等级判定
==================================
基于置信度、落石直径、运动状态、持续帧数的综合决策树，
对齐《公路自然灾害监测预警系统技术指南》第5.3节。

决策树结构::

    输入: 检测帧 + 跟踪轨迹
      │
      ├── max_conf > 0.90 ───────────────────→ 🔴 I 级
      │
      ├── max_conf 0.70-0.90 ──→ 直径 > 30cm  → 🔴 I 级 (升级)
      │                      ──→ 直径 20-30cm → 🟠 II 级
      │                      ──→ 直径 < 20cm ──→ 坠落 → 🟠 II 级
      │                                     ──→ 滚动 → 🟡 III 级
      │
      ├── max_conf 0.50-0.70 ──→ 直径 > 20cm  → 🟠 II 级 (升级)
      │                      ──→ 直径 10-20cm → 🟡 III 级
      │                      ──→ 直径 < 10cm ──→ >10帧 → 🟡 III 级
      │                                     ──→ <10帧 → 🔵 IV 级
      │
      ├── max_conf 0.30-0.50 ──→ 直径 > 10cm  → 🟡 III 级 (升级)
      │                      ──→ 直径 ≤ 10cm → 🔵 IV 级
      │
      └── max_conf < 0.30 ───────────────────→ 🟢 正常

使用方式::

    from rockfall.alert_classifier import classify_alert_level

    level = classify_alert_level(
        max_conf=0.85,
        rock_diameter_cm=25,
        motion_state="快速坠落",
        track_age=8,
    )
    # → "orange"
"""

from __future__ import annotations

# ── 等级常量 ──────────────────────────────────────────────
LEVEL_RED = "red"         # 🔴 I 级 · 特别严重
LEVEL_ORANGE = "orange"   # 🟠 II 级 · 严重
LEVEL_YELLOW = "yellow"   # 🟡 III 级 · 较重
LEVEL_BLUE = "blue"       # 🔵 IV 级 · 一般
LEVEL_GREEN = "green"     # 🟢 正常 (不预警)

# ── 决策树默认阈值 (可被 config.py / 环境变量覆盖) ──────────
_CONF_HIGH = 0.90          # Ⅰ 级直通阈值
_CONF_MEDIUM_HIGH = 0.70   # Ⅱ/Ⅲ 级分界
_CONF_MEDIUM_LOW = 0.50    # Ⅲ/Ⅳ 级分界
_CONF_LOW = 0.30           # 预警下限

_DIAMETER_XLARGE = 30      # cm, >30 → Ⅰ 级
_DIAMETER_LARGE = 20       # cm, 20-30 → Ⅱ 级
_DIAMETER_MEDIUM = 10      # cm, 10-20 → Ⅲ 级

_FRAME_LONG = 10           # 帧, >10 → 提升至 Ⅲ 级

# ── 运行时可覆盖阈值 (由 config.py 环境变量驱动) ────────────
# 延迟加载避免循环导入, 模块级函数访问时解析
_thresholds_loaded = False
CONF_HIGH = _CONF_HIGH
CONF_MEDIUM_HIGH = _CONF_MEDIUM_HIGH
CONF_MEDIUM_LOW = _CONF_MEDIUM_LOW
CONF_LOW = _CONF_LOW
DIAMETER_XLARGE = _DIAMETER_XLARGE
DIAMETER_LARGE = _DIAMETER_LARGE
DIAMETER_MEDIUM = _DIAMETER_MEDIUM
FRAME_LONG = _FRAME_LONG


def _load_thresholds_from_config():
    """从 rockfall.config 加载环境变量覆盖的阈值 (仅执行一次)。"""
    global _thresholds_loaded, \
        CONF_HIGH, CONF_MEDIUM_HIGH, CONF_MEDIUM_LOW, CONF_LOW, \
        DIAMETER_XLARGE, DIAMETER_LARGE, DIAMETER_MEDIUM, FRAME_LONG
    if _thresholds_loaded:
        return
    _thresholds_loaded = True
    try:
        from .config import (
            ALERT_BLUE_CONFIDENCE_LOW,
            ALERT_BLUE_CONFIDENCE_HIGH,
            ALERT_YELLOW_CONFIDENCE_HIGH,
            ALERT_ORANGE_CONFIDENCE_HIGH,
        )
        # 四级预警上下限映射到决策树阈值
        #   ALERT_BLUE_CONFIDENCE_LOW   → 预警下限 (进入 IV 级)
        #   ALERT_BLUE_CONFIDENCE_HIGH  → Ⅲ/Ⅳ 分界
        #   ALERT_YELLOW_CONFIDENCE_HIGH → Ⅱ/Ⅲ 分界
        #   ALERT_ORANGE_CONFIDENCE_HIGH → Ⅰ 级直通
        CONF_LOW = ALERT_BLUE_CONFIDENCE_LOW
        CONF_MEDIUM_LOW = ALERT_BLUE_CONFIDENCE_HIGH
        CONF_MEDIUM_HIGH = ALERT_YELLOW_CONFIDENCE_HIGH
        CONF_HIGH = ALERT_ORANGE_CONFIDENCE_HIGH
    except ImportError:
        pass  # 保持硬编码默认值

# ── 等级标签 (对齐交通部标准) ──────────────────────────────
LEVEL_LABELS: dict[str, str] = {
    LEVEL_RED:    "🔴 Ⅰ级·特别严重",
    LEVEL_ORANGE: "🟠 Ⅱ级·严重",
    LEVEL_YELLOW: "🟡 Ⅲ级·较重",
    LEVEL_BLUE:   "🔵 Ⅳ级·一般",
    LEVEL_GREEN:  "🟢 正常",
}

# 等级排序 (用于比较高低)
LEVEL_ORDER = [LEVEL_GREEN, LEVEL_BLUE, LEVEL_YELLOW, LEVEL_ORANGE, LEVEL_RED]


def classify_alert_level(
    max_conf: float,
    rock_diameter_cm: float = 0.0,
    motion_state: str = "",
    track_age: int = 0,
) -> str:
    """
    预警分级决策树 — 四级预警等级判定。

    参数:
        max_conf:         最高置信度 (0-1)，建议使用 smoothed_confidence
        rock_diameter_cm: 估算落石直径 (cm)，0 表示无法估算
        motion_state:     运动状态 ("快速坠落" / "横向滚动" / "缓慢滚动" / "静止" / "快速移动")
        track_age:        轨迹持续帧数 (age)

    返回:
        "red" / "orange" / "yellow" / "blue" / "green"

    示例:
        >>> classify_alert_level(0.95, 15, "横向滚动", 5)
        'red'

        >>> classify_alert_level(0.75, 12, "快速坠落", 3)
        'orange'

        >>> classify_alert_level(0.55, 8, "横向滚动", 12)
        'yellow'

        >>> classify_alert_level(0.40, 5, "静止", 2)
        'blue'

        >>> classify_alert_level(0.20, 0, "", 0)
        'green'
    """
    # 确保阈值已从环境变量加载 (首次调用时执行)
    _load_thresholds_from_config()

    # ── 第1层: 置信度 < 预警下限 → 正常 ──────────────────
    if max_conf < CONF_LOW:
        return LEVEL_GREEN

    # ── 第2层: 0.30 ≤ conf < 0.50 → IV 级 (可升级) ─────
    if max_conf < CONF_MEDIUM_LOW:
        if rock_diameter_cm > DIAMETER_MEDIUM:
            return LEVEL_YELLOW  # 升级至 III 级
        return LEVEL_BLUE

    # ── 第3层: 0.50 ≤ conf < 0.70 ─────────────────────
    if max_conf < CONF_MEDIUM_HIGH:
        if rock_diameter_cm > DIAMETER_LARGE:
            return LEVEL_ORANGE  # II 级 (升级)
        if rock_diameter_cm >= DIAMETER_MEDIUM:
            return LEVEL_YELLOW  # III 级
        # 直径 < 10cm: 按持续帧数判定
        if track_age > FRAME_LONG:
            return LEVEL_YELLOW  # III 级
        return LEVEL_BLUE       # IV 级

    # ── 第4层: 0.70 ≤ conf < 0.90 ─────────────────────
    if max_conf < CONF_HIGH:
        if rock_diameter_cm > DIAMETER_XLARGE:
            return LEVEL_RED     # I 级 (升级)
        if rock_diameter_cm >= DIAMETER_LARGE:
            return LEVEL_ORANGE  # II 级
        # 直径 < 20cm: 按运动状态判定
        if _is_falling(motion_state):
            return LEVEL_ORANGE  # II 级
        return LEVEL_YELLOW      # III 级 (滚动)

    # ── 第5层: conf ≥ 0.90 → I 级 ─────────────────────
    return LEVEL_RED


def classify_from_track(track: dict) -> str:
    """
    从单条跟踪记录直接判定预警等级 (便捷方法)。

    参数:
        track: RockTracker.update() 返回的 dict，
               含 confidence / smoothed_confidence / area / motion_state / age

    返回:
        "red" / "orange" / "yellow" / "blue" / "green"
    """
    conf = track.get("smoothed_confidence", track.get("confidence", 0))
    age = track.get("age", 0)
    motion = track.get("motion_state", "")
    area = track.get("area", 0)

    # 从 bbox 高度估算直径 (粗略)
    bbox = track.get("bbox", [0, 0, 0, 0])
    h_px = bbox[3] - bbox[1]
    # 假设 1080p 画面，10cm 对应 ~2% 高度比 (≈22px)
    diameter_cm = round(h_px / 22.0 * 10, 1) if h_px > 0 else 0.0

    return classify_alert_level(
        max_conf=conf,
        rock_diameter_cm=diameter_cm,
        motion_state=motion,
        track_age=age,
    )


def classify_from_tracks(
    tracks: list[dict],
    frame_h: int = 1080,
    height_ratio_ref: float = 0.02,
) -> str:
    """
    从跟踪结果列表聚合判定预警等级 (用于每帧综合判定)。

    取所有已确认轨迹中的最高置信度、最大直径和最高风险运动状态。

    参数:
        tracks:            RockTracker.update() 返回的 dict 列表
        frame_h:           画面高度 (像素)，用于直径估算
        height_ratio_ref:  基准高度比 (10cm 落石占画面比例)，默认 2%

    返回:
        "red" / "orange" / "yellow" / "blue" / "green"
    """
    confirmed = [t for t in tracks if t.get("confirmed")]
    if not confirmed:
        # 无已确认轨迹 → 用未确认轨迹中置信度最高的
        if not tracks:
            return LEVEL_GREEN
        best = max(tracks, key=lambda t: t.get("confidence", 0))
        conf = best.get("confidence", 0)
        if conf < CONF_LOW:
            return LEVEL_GREEN
        # 单帧未确认: 仅按置信度快速判定
        if conf >= CONF_HIGH:
            return LEVEL_RED
        if conf >= CONF_MEDIUM_HIGH:
            return LEVEL_ORANGE
        if conf >= CONF_MEDIUM_LOW:
            return LEVEL_YELLOW
        return LEVEL_BLUE

    # 聚合已确认轨迹
    max_conf = max(t.get("smoothed_confidence", t.get("confidence", 0)) for t in confirmed)
    max_age = max(t.get("age", 0) for t in confirmed)

    # 最大直径
    max_diameter_cm = 0.0
    for t in confirmed:
        bbox = t.get("bbox", [0, 0, 0, 0])
        h_px = bbox[3] - bbox[1]
        if frame_h > 0 and h_px > 0:
            height_ratio = h_px / frame_h
            diameter_cm = (height_ratio / height_ratio_ref) * 10
            if diameter_cm > max_diameter_cm:
                max_diameter_cm = round(diameter_cm, 1)

    # 运动状态: 有任一坠落即视为坠落
    motion_state = ""
    for t in confirmed:
        ms = t.get("motion_state", "")
        if _is_falling(ms):
            motion_state = ms
            break
    if not motion_state:
        # 取第一个非静止的运动状态
        for t in confirmed:
            ms = t.get("motion_state", "")
            if ms and ms != "静止":
                motion_state = ms
                break

    return classify_alert_level(
        max_conf=max_conf,
        rock_diameter_cm=max_diameter_cm,
        motion_state=motion_state,
        track_age=max_age,
    )


# ── 响应流程配置 ────────────────────────────────────────────


def get_response_workflow(alert_level: str) -> dict:
    """
    获取指定预警等级的响应流程配置。

    返回:
        {
            "level": str,           # 等级标识
            "label": str,           # 中文标签
            "trigger_conditions": [str],  # 触发条件列表
            "disposal_steps": [str],      # 处置流程步骤
            "push_channels": [str],       # 推送渠道
            "requires_sound": bool,       # 是否触发声光报警
        }
    """
    workflows = {
        LEVEL_RED: {
            "level": LEVEL_RED,
            "label": LEVEL_LABELS[LEVEL_RED],
            "trigger_conditions": [
                "置信度 > 0.90",
                "落石直径 > 30cm",
                "检测到坠落状态 (快速坠落)",
            ],
            "disposal_steps": [
                "立即通知公路管理部门封闭相关车道",
                "电话通知值班领导 (5分钟内响应)",
                "通知交警部门协助交通管制",
                "调取现场实时画面确认灾情规模",
                "启动公路地质灾害应急预案",
                "派遣巡查人员赴现场评估",
            ],
            "push_channels": ["pushplus", "smtp", "wecom", "dingtalk", "feishu"],
            "requires_sound": True,
        },
        LEVEL_ORANGE: {
            "level": LEVEL_ORANGE,
            "label": LEVEL_LABELS[LEVEL_ORANGE],
            "trigger_conditions": [
                "置信度 0.70-0.90 且 直径 20-30cm",
                "置信度 0.50-0.70 且 直径 > 20cm",
                "置信度 0.70-0.90 且 坠落状态",
            ],
            "disposal_steps": [
                "通知公路管理部门关注该路段",
                "建议限速通行 (≤40km/h)，设置预警标志",
                "安排人员在30分钟内到场巡查",
                "加密监测频率至 5fps",
                "准备应急物资和抢修设备",
            ],
            "push_channels": ["pushplus", "smtp"],
            "requires_sound": False,
        },
        LEVEL_YELLOW: {
            "level": LEVEL_YELLOW,
            "label": LEVEL_LABELS[LEVEL_YELLOW],
            "trigger_conditions": [
                "置信度 0.50-0.70 且 直径 10-20cm",
                "置信度 0.50-0.70 且 直径 < 10cm 但持续 > 10帧",
                "置信度 0.30-0.50 且 直径 > 10cm (升级)",
                "置信度 0.70-0.90 且 直径 < 20cm 滚动状态",
            ],
            "disposal_steps": [
                "系统自动记录预警事件",
                "纳入当日监测日报汇总",
                "关注后续帧是否有等级升级趋势",
                "建议2小时内安排远程视频巡检",
            ],
            "push_channels": [],
            "requires_sound": False,
        },
        LEVEL_BLUE: {
            "level": LEVEL_BLUE,
            "label": LEVEL_LABELS[LEVEL_BLUE],
            "trigger_conditions": [
                "置信度 0.30-0.50 且 直径 ≤ 10cm",
                "置信度 0.50-0.70 且 直径 < 10cm 且 ≤ 10帧",
            ],
            "disposal_steps": [
                "静默记录至本地数据库",
                "用于历史趋势分析和模型优化",
                "无需主动处置",
            ],
            "push_channels": [],
            "requires_sound": False,
        },
    }
    return workflows.get(alert_level, {
        "level": LEVEL_GREEN,
        "label": LEVEL_LABELS[LEVEL_GREEN],
        "trigger_conditions": ["置信度 < 0.30"],
        "disposal_steps": ["正常运行，无需处置"],
        "push_channels": [],
        "requires_sound": False,
    })


# ── 内部辅助 ────────────────────────────────────────────────


def _is_falling(motion_state: str) -> bool:
    """判断运动状态是否为坠落类型。"""
    return motion_state in ("快速坠落",)
