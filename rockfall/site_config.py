"""
多监测点位管理 — 商用标准功能，贴合广西+东盟大赛场景
=======================================================
支持:
  - 4 个预设演示点位 (广西本地 + 东盟跨境)
  - 点位自由切换，每个点位独立存储报警记录
  - SAM冷启动 + 传统CV热运行 + 自动重校准 (原有功能保留)

预设点位:
  1. 南宁那安快速路 1 号边坡     — 广西首府核心路段
  2. 崇左合那高速 2 号边坡       — 通往东盟陆路大通道
  3. 防城港兰海高速 3 号边坡     — 北部湾沿海关键通道
  4. 凭祥中越跨境公路 4 号边坡   — 中国-东盟自贸区门户
"""

import json
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from .config import DATA_DIR
from .road_segmentation import ROIParams

# ============================================================
# 监测点位数据模型
# ============================================================


@dataclass
class MonitoringSite:
    """单个监测点位的完整元数据"""

    site_id: str
    name: str                          # 点位名称 (用于界面显示)
    location: str                      # 地理位置 (用于报警推送)
    region: str                        # 所属区域
    camera_url: str = ""               # RTSP/摄像头地址
    description: str = ""              # 点位描述
    latitude: float = 0.0              # 纬度
    longitude: float = 0.0             # 经度
    highway: str = ""                  # 所属公路
    stake_mark: str = ""               # 桩号
    risk_level: str = "medium"         # 地质灾害风险等级: high/medium/low

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MonitoringSite":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# 4 个预设演示点位 — 广西本地 + 东盟跨境全覆盖
# ============================================================

PRESET_SITES: list[MonitoringSite] = [
    MonitoringSite(
        site_id="nanning_naan_s1",
        name="南宁那安快速路 1 号边坡",
        location="南宁那安快速路 1 号边坡",
        region="广西·南宁",
        camera_url="",
        description="广西首府南宁核心环城快速路段，石灰岩边坡，雨季落石风险高，车流量大 (日均 >8万辆)",
        latitude=22.817,
        longitude=108.366,
        highway="G75 兰海高速 (那安段)",
        stake_mark="K1952+300",
        risk_level="high",
    ),
    MonitoringSite(
        site_id="chongzuo_hena_s2",
        name="崇左合那高速 2 号边坡",
        location="崇左合那高速 2 号边坡",
        region="广西·崇左",
        camera_url="",
        description="中国通往东盟陆路大通道关键节点，喀斯特地貌边坡，邻近中越边境，战略意义重大",
        latitude=22.379,
        longitude=107.365,
        highway="G7211 南友高速 (合那段)",
        stake_mark="K138+800",
        risk_level="high",
    ),
    MonitoringSite(
        site_id="fangchenggang_lanhai_s3",
        name="防城港兰海高速 3 号边坡",
        location="防城港兰海高速 3 号边坡",
        region="广西·防城港",
        camera_url="",
        description="北部湾沿海关键通道，台风暴雨频发区域，花岗岩风化边坡，海运+陆运交汇枢纽",
        latitude=21.687,
        longitude=108.355,
        highway="G75 兰海高速 (防城港段)",
        stake_mark="K2078+150",
        risk_level="medium",
    ),
    MonitoringSite(
        site_id="pingxiang_crossborder_s4",
        name="凭祥中越跨境公路 4 号边坡",
        location="凭祥中越跨境公路 4 号边坡",
        region="广西·凭祥 (中越边境)",
        camera_url="",
        description="中国-东盟自贸区门户，中越跨境公路咽喉段，直接服务RCEP贸易通道，东盟大赛标杆场景",
        latitude=22.094,
        longitude=106.767,
        highway="G322 中越跨境公路",
        stake_mark="K1042+600",
        risk_level="high",
    ),
]

# ============================================================
# 配置持久化路径
# ============================================================

SITE_CONFIG_PATH = DATA_DIR / "site_config.json"
SITE_STATE_PATH = DATA_DIR / "site_state.json"       # 当前激活点位 + 全局状态
ROI_CONFIG_PATH = DATA_DIR / "site_config.json"      # 兼容旧路径 (ROI校准数据)

# ============================================================
# 点位切换与管理 — 线程安全
# ============================================================

_site_lock = threading.RLock()
_active_site: MonitoringSite | None = None


def list_sites() -> list[MonitoringSite]:
    """获取全部预设点位列表"""
    return list(PRESET_SITES)


def get_site_by_id(site_id: str) -> MonitoringSite | None:
    """按 ID 查找点位"""
    for site in PRESET_SITES:
        if site.site_id == site_id:
            return site
    return None


def get_active_site() -> MonitoringSite:
    """
    获取当前激活的监测点位。

    优先级:
      1. 运行时通过 set_active_site() 设置的
      2. site_state.json 中持久化的
      3. 预设第一个点位 (南宁那安快速路)
    """
    global _active_site

    with _site_lock:
        if _active_site is not None:
            return _active_site

        # 从持久化状态恢复
        saved = _load_site_state()
        saved_id = saved.get("active_site_id", "")
        if saved_id:
            site = get_site_by_id(saved_id)
            if site is not None:
                _active_site = site
                return _active_site

        # 默认: 第一个预设点位
        _active_site = PRESET_SITES[0]
        _save_site_state(_active_site.site_id)
        return _active_site


def set_active_site(site_id: str) -> MonitoringSite:
    """
    切换当前激活的监测点位。

    返回:
        新激活的点位对象

    Raises:
        ValueError: site_id 无效
    """
    global _active_site

    site = get_site_by_id(site_id)
    if site is None:
        raise ValueError(
            f"无效的点位ID: {site_id}。可用点位: {[s.site_id for s in PRESET_SITES]}"
        )

    with _site_lock:
        _active_site = site
        _save_site_state(site_id)

    return site


def get_active_location() -> str:
    """获取当前激活点位的完整地理位置字符串"""
    site = get_active_site()
    return site.location


def get_active_site_id() -> str:
    """获取当前激活点位的 site_id"""
    return get_active_site().site_id


def get_active_site_name() -> str:
    """获取当前激活点位的名称"""
    return get_active_site().name


def get_active_region() -> str:
    """获取当前激活点位的所属区域"""
    return get_active_site().region


# ============================================================
# 点位状态持久化
# ============================================================


def _save_site_state(site_id: str):
    """保存当前激活点位到磁盘"""
    state = {
        "active_site_id": site_id,
        "last_switch_time": time.time(),
        "last_switch_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    SITE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_site_state() -> dict:
    """从磁盘加载点位状态"""
    if not SITE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SITE_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_site_state() -> dict:
    """获取完整的点位状态信息 (供 API 返回)"""
    site = get_active_site()
    state = _load_site_state()
    return {
        "active_site": site.to_dict(),
        "available_sites": [s.to_dict() for s in PRESET_SITES],
        "last_switch_time": state.get("last_switch_time", 0),
        "last_switch_iso": state.get("last_switch_iso", ""),
    }


# ============================================================
# ROI 配置管理 (原有功能，适配多点位)
# ============================================================

# 兼容旧 CONFIG_PATH 别名
CONFIG_PATH = ROI_CONFIG_PATH


def save_site_config(
    camera_id: str,
    roi_params: ROIParams,
    polygon: np.ndarray,
    road_mask: np.ndarray = None,
):
    """
    保存 ROI 标定配置 (按 camera_id 索引，兼容多点位)。

    camera_id 建议使用 site_id 或 site_id + 后缀，确保不同点位的
    ROI 校准数据独立存储。
    """
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}

    entry = {
        "roi_params": asdict(roi_params),
        "polygon": polygon.tolist() if polygon is not None else None,
        "last_calibration": time.time(),
        "last_calibration_iso": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        ),
    }

    # 同时记录当前激活点位信息
    try:
        entry["site_id"] = get_active_site_id()
        entry["site_name"] = get_active_site_name()
    except Exception:
        pass

    config[camera_id] = entry

    if road_mask is not None:
        mask_path = DATA_DIR / "masks" / f"{camera_id[:12]}_site.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), road_mask)
        config[camera_id]["mask_path"] = str(mask_path)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_site_config(camera_id: str) -> tuple:
    """
    加载 ROI 标定配置。

    返回:
        (ROIParams | None, polygon ndarray | None, road_mask ndarray | None)
    """
    if not CONFIG_PATH.exists():
        return None, None, None

    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None, None

    if camera_id not in config:
        return None, None, None

    site = config[camera_id]
    params = ROIParams(**site["roi_params"]) if site.get("roi_params") else None
    polygon = np.array(site["polygon"], np.int32) if site.get("polygon") else None
    road_mask = None
    if site.get("mask_path") and Path(site["mask_path"]).exists():
        road_mask = cv2.imread(site["mask_path"], cv2.IMREAD_GRAYSCALE)

    return params, polygon, road_mask


def auto_optimize_cv_params(
    frame: np.ndarray, sam_road_mask: np.ndarray
) -> ROIParams:
    """用SAM精准掩码自动优化传统CV参数"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    road_pixels = hsv[sam_road_mask > 0]
    if len(road_pixels) < 100:
        return ROIParams()
    s_values = road_pixels[:, 1]
    v_values = road_pixels[:, 2]
    return ROIParams(
        sat_max=int(np.percentile(s_values, 95)),
        val_min=max(20, int(np.percentile(v_values, 5)) - 10),
        val_max=min(255, int(np.percentile(v_values, 95)) + 10),
        morph_close=9,
        morph_open=5,
        min_area_ratio=0.05,
    )


# ============================================================
# 点位报警记录隔离 — 查询辅助
# ============================================================


def get_site_filter_clause(backend: str = "sqlite") -> tuple[str, tuple]:
    """
    获取当前激活点位的 SQL 过滤条件，用于报警记录隔离查询。

    参数:
        backend: "sqlite" (?) 或 "mysql" (%s)

    返回:
        (WHERE子句, 参数元组)

    用法:
        clause, params = get_site_filter_clause()
        rows = store._sqlite_query(
            f"SELECT * FROM alerts WHERE alert_level='red' {clause}",
            ("red",) + params,
        )
    """
    location = get_active_location()
    placeholder = "?" if backend == "sqlite" else "%s"
    return (
        f"AND monitoring_location = {placeholder}",
        (location,),
    )


def get_available_site_locations() -> list[str]:
    """获取所有预设点位的 location 值列表 (用于报警记录筛选下拉)"""
    return [s.location for s in PRESET_SITES]
