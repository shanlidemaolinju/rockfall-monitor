"""
隐患点排查与风险评估 — 数据模型与持久化存储
===========================================
对应"第1步：隐患点排查与风险评估"工作流程。

支持:
  - 隐患点基础信息录入（地理位置、地形地貌、地质条件）
  - 历史灾害数据关联
  - 边坡稳定性多维度评估
  - 风险等级分级: 高风险 / 中风险 / 一般风险
  - 关联监测点位（从排查到部署的完整链路）
  - 生成隐患点清单及风险评估报告

与 MonitoringSite 的关系:
  - 一个隐患点可对应 0..1 个监测点位（排查后可能未部署）
  - 一个监测点位必须对应 1 个隐患点（部署前必须先排查）
"""

import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from .config import DATA_DIR
from .config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from .db_utils import is_mysql_available

# ============================================================
# 隐患点数据模型
# ============================================================


@dataclass
class HistoricalIncident:
    """历史灾害事件记录"""
    incident_date: str = ""              # 发生日期 ISO "YYYY-MM-DD"
    incident_type: str = ""              # 类型: rockfall/landslide/collapse
    severity: str = ""                   # 严重程度: major/moderate/minor
    description: str = ""                # 事件描述
    casualties: str = ""                 # 伤亡情况
    road_closure: str = ""               # 道路中断情况
    source: str = ""                     # 数据来源 (交通局/地质队/媒体报道)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HistoricalIncident":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class SlopeStability:
    """边坡稳定性评估维度"""
    slope_angle: float = 0.0             # 坡度 (度)
    slope_height: float = 0.0            # 坡高 (米)
    slope_type: str = ""                 # 坡型: 直线坡/凸坡/凹坡/复合坡
    rock_type: str = ""                  # 岩性: 石灰岩/花岗岩/砂岩/页岩/其他
    weathering_degree: str = ""          # 风化程度: 强/中/弱/未风化
    joint_development: str = ""          # 节理发育: 发育/较发育/不发育
    vegetation_coverage: str = ""        # 植被覆盖: 好/中/差/裸露
    drainage_condition: str = ""         # 排水条件: 好/中/差
    geological_score: float = 0.0        # 地质稳定性综合评分 (0-100, 越低越危险)
    survey_date: str = ""                # 勘察日期
    survey_team: str = ""                # 勘察单位/人员
    remarks: str = ""                    # 备注

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SlopeStability":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class HazardPoint:
    """隐患点完整数据模型"""

    hazard_id: str                       # 隐患点唯一ID (如 HZ-G75-001)
    name: str                            # 隐患点名称
    location: str = ""                   # 地理位置
    region: str = ""                     # 所属区域
    highway: str = ""                    # 所属公路
    stake_mark: str = ""                 # 桩号
    latitude: float = 0.0                # 纬度
    longitude: float = 0.0              # 经度

    # ── 风险等级 ──
    risk_level: str = "medium"           # high / medium / low (高风险/中风险/一般风险)
    risk_score: float = 0.0             # 综合风险评分 (0-100, 越高越危险)

    # ── 边坡稳定性评估 (一个隐患点可有多期评估记录) ──
    slope_assessments: list[dict] = field(default_factory=list)

    # ── 历史灾害记录 ──
    historical_incidents: list[dict] = field(default_factory=list)

    # ── 关联监测点位 ──
    linked_site_id: str = ""             # 关联的 MonitoringSite.site_id (空=未部署)

    # ── 状态与阶段 ──
    status: str = "identified"           # identified/assessed/monitored/remediated/cleared
                                         # 已识别/已评估/监测中/已治理/已消除

    # ── 现场信息 ──
    photo_paths: list[str] = field(default_factory=list)      # 现场照片路径
    report_path: str = ""                # 评估报告文件路径
    description: str = ""               # 综合描述

    # ── 责任信息 ──
    responsible_unit: str = ""           # 责任单位
    contact_person: str = ""             # 联系人
    contact_phone: str = ""              # 联系电话

    # ── 保护措施 ──
    protection_measures: str = ""        # 已有防护措施 (防护网/挡石墙/被动网/无)
    recommended_measures: str = ""       # 建议防治措施

    # ── 时间戳 ──
    created_at: str = ""                 # 创建时间 ISO
    updated_at: str = ""                 # 更新时间 ISO
    surveyed_at: str = ""                # 排查日期 ISO

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HazardPoint":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# ============================================================
# 风险等级定义
# ============================================================

RISK_LEVELS = {
    "high":   {"label": "🔴 高风险", "min_score": 70, "color": "#DC3545", "description": "坡度>50°、岩体破碎、历史塌方>2次/年，需立即监测"},
    "medium": {"label": "🟡 中风险", "min_score": 40, "color": "#FFC107", "description": "坡度30-50°、局部节理发育、偶有落石，建议近期监测"},
    "low":    {"label": "🟢 一般风险", "min_score": 0,  "color": "#28A745", "description": "坡度<30°、岩体较完整、无明显历史灾害，定期巡查"},
}

HAZARD_STATUSES = {
    "identified":  "🔍 已识别",
    "assessed":    "📋 已评估",
    "monitored":   "📡 监测中",
    "remediated":  "🛠️ 已治理",
    "cleared":     "✅ 已消除",
}

# ============================================================
# 持久化存储 (MySQL / SQLite 双后端)
# ============================================================

_MYSQL_AVAILABLE = is_mysql_available()

_HAZARD_TABLE_MYSQL = """\
CREATE TABLE IF NOT EXISTS hazard_points (
    hazard_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    location VARCHAR(256) DEFAULT '',
    region VARCHAR(128) DEFAULT '',
    highway VARCHAR(128) DEFAULT '',
    stake_mark VARCHAR(64) DEFAULT '',
    latitude DOUBLE DEFAULT 0,
    longitude DOUBLE DEFAULT 0,
    risk_level VARCHAR(16) DEFAULT 'medium',
    risk_score DOUBLE DEFAULT 0,
    slope_assessments JSON DEFAULT ('[]'),
    historical_incidents JSON DEFAULT ('[]'),
    linked_site_id VARCHAR(64) DEFAULT '',
    status VARCHAR(32) DEFAULT 'identified',
    photo_paths JSON DEFAULT ('[]'),
    report_path VARCHAR(512) DEFAULT '',
    description TEXT,
    responsible_unit VARCHAR(128) DEFAULT '',
    contact_person VARCHAR(64) DEFAULT '',
    contact_phone VARCHAR(32) DEFAULT '',
    protection_measures TEXT,
    recommended_measures TEXT,
    created_at VARCHAR(19) NOT NULL,
    updated_at VARCHAR(19) NOT NULL,
    surveyed_at VARCHAR(19) DEFAULT '',
    INDEX idx_risk_level (risk_level),
    INDEX idx_status (status),
    INDEX idx_linked_site (linked_site_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""

_HAZARD_TABLE_SQLITE = """\
CREATE TABLE IF NOT EXISTS hazard_points (
    hazard_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT DEFAULT '',
    region TEXT DEFAULT '',
    highway TEXT DEFAULT '',
    stake_mark TEXT DEFAULT '',
    latitude REAL DEFAULT 0,
    longitude REAL DEFAULT 0,
    risk_level TEXT DEFAULT 'medium',
    risk_score REAL DEFAULT 0,
    slope_assessments TEXT DEFAULT '[]',
    historical_incidents TEXT DEFAULT '[]',
    linked_site_id TEXT DEFAULT '',
    status TEXT DEFAULT 'identified',
    photo_paths TEXT DEFAULT '[]',
    report_path TEXT DEFAULT '',
    description TEXT,
    responsible_unit TEXT DEFAULT '',
    contact_person TEXT DEFAULT '',
    contact_phone TEXT DEFAULT '',
    protection_measures TEXT,
    recommended_measures TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    surveyed_at TEXT DEFAULT ''
)"""


class HazardStore:
    """隐患点持久化 — MySQL 优先, SQLite 降级。线程安全。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._backend = "sqlite"
        if MYSQL_HOST and _MYSQL_AVAILABLE:
            try:
                from .db_engine import get_mysql_engine
                engine = get_mysql_engine()
                if engine is not None:
                    conn = engine.raw_connection()
                    conn.close()
                    self._backend = "mysql"
            except Exception:
                pass
        self._init_table()

    # ---- 建表 ----

    def _init_table(self):
        if self._backend == "mysql":
            self._init_mysql_table()
        else:
            self._init_sqlite_table()

    def _init_mysql_table(self):
        conn = None
        try:
            conn = self._mysql_conn()
            with conn.cursor() as cur:
                cur.execute(_HAZARD_TABLE_MYSQL)
            conn.commit()
        except Exception as e:
            from .logger import log_event
            log_event("system", level="ERROR",
                      msg=f"Hazard MySQL 建表失败 ({e}), 降级为 SQLite")
            self._backend = "sqlite"
            self._init_sqlite_table()
        finally:
            if conn is not None:
                conn.close()

    def _init_sqlite_table(self):
        conn = self._sqlite_conn()
        conn.execute(_HAZARD_TABLE_SQLITE)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hazard_risk ON hazard_points(risk_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hazard_status ON hazard_points(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hazard_site ON hazard_points(linked_site_id)")
        conn.commit()
        conn.close()

    # ---- 连接 ----

    def _mysql_conn(self):
        from .db_engine import get_mysql_engine
        engine = get_mysql_engine()
        if engine is None:
            raise RuntimeError("MySQL 引擎未初始化")
        return engine.raw_connection()

    def _sqlite_conn(self):
        import sqlite3
        db_path = DATA_DIR / "hazard_points.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    # ---- CRUD ----

    def list_all(self, risk_level: str = "", status_filter: str = "") -> list[HazardPoint]:
        """列出所有隐患点，可按风险等级和状态筛选"""
        conditions = []
        params = []

        if self._backend == "mysql":
            if risk_level:
                conditions.append("risk_level = %s")
                params.append(risk_level)
            if status_filter:
                conditions.append("status = %s")
                params.append(status_filter)
            sql = "SELECT * FROM hazard_points"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY risk_score DESC, created_at DESC"
            rows = self._mysql_query(sql, tuple(params))
        else:
            if risk_level:
                conditions.append("risk_level = ?")
                params.append(risk_level)
            if status_filter:
                conditions.append("status = ?")
                params.append(status_filter)
            sql = "SELECT * FROM hazard_points"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY risk_score DESC, created_at DESC"
            rows = self._sqlite_query(sql, tuple(params))

        return [self._row_to_hazard(r) for r in rows]

    def get_by_id(self, hazard_id: str) -> HazardPoint | None:
        """按 ID 查找"""
        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT * FROM hazard_points WHERE hazard_id = %s", (hazard_id,))
        else:
            rows = self._sqlite_query(
                "SELECT * FROM hazard_points WHERE hazard_id = ?", (hazard_id,))
        return self._row_to_hazard(rows[0]) if rows else None

    def get_by_linked_site(self, site_id: str) -> HazardPoint | None:
        """按关联监测点位查找"""
        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT * FROM hazard_points WHERE linked_site_id = %s", (site_id,))
        else:
            rows = self._sqlite_query(
                "SELECT * FROM hazard_points WHERE linked_site_id = ?", (site_id,))
        return self._row_to_hazard(rows[0]) if rows else None

    def insert(self, hazard: HazardPoint) -> bool:
        """插入新隐患点"""
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        hazard.created_at = now
        hazard.updated_at = now

        if self._backend == "mysql":
            return self._mysql_insert(hazard, now)
        return self._sqlite_insert(hazard, now)

    def update(self, hazard: HazardPoint) -> bool:
        """更新已有隐患点"""
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        hazard.updated_at = now

        if self._backend == "mysql":
            return self._mysql_update(hazard, now)
        return self._sqlite_update(hazard, now)

    def delete(self, hazard_id: str) -> bool:
        """删除隐患点"""
        if self._backend == "mysql":
            conn = None
            try:
                conn = self._mysql_conn()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM hazard_points WHERE hazard_id = %s", (hazard_id,))
                conn.commit()
                return True
            except Exception:
                return False
            finally:
                if conn is not None:
                    conn.close()
        else:
            with self._lock:
                conn = self._sqlite_conn()
                conn.execute("DELETE FROM hazard_points WHERE hazard_id = ?", (hazard_id,))
                conn.commit()
                conn.close()
            return True

    def count(self, risk_level: str = "") -> int:
        """统计隐患点数量"""
        if self._backend == "mysql":
            if risk_level:
                rows = self._mysql_query(
                    "SELECT COUNT(*) as cnt FROM hazard_points WHERE risk_level = %s", (risk_level,))
            else:
                rows = self._mysql_query("SELECT COUNT(*) as cnt FROM hazard_points", ())
        else:
            if risk_level:
                rows = self._sqlite_query(
                    "SELECT COUNT(*) as cnt FROM hazard_points WHERE risk_level = ?", (risk_level,))
            else:
                rows = self._sqlite_query("SELECT COUNT(*) as cnt FROM hazard_points", ())
        return rows[0]["cnt"] if rows else 0

    def count_by_level(self) -> dict[str, int]:
        """按风险等级统计"""
        result = {"high": 0, "medium": 0, "low": 0}
        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT risk_level, COUNT(*) as cnt FROM hazard_points GROUP BY risk_level", ())
        else:
            rows = self._sqlite_query(
                "SELECT risk_level, COUNT(*) as cnt FROM hazard_points GROUP BY risk_level", ())
        for r in rows:
            if r["risk_level"] in result:
                result[r["risk_level"]] = r["cnt"]
        return result

    # ---- 种子数据 ----

    def seed_demo_data(self) -> int:
        """写入演示隐患点数据，返回新增条数"""
        from datetime import datetime, timedelta
        today = datetime.now()

        demos = [
            HazardPoint(
                hazard_id="HZ-G75-001",
                name="南宁那安快速路 1 号边坡隐患点",
                location="南宁那安快速路 1 号边坡",
                region="广西·南宁",
                highway="G75 兰海高速 (那安段)",
                stake_mark="K1952+300",
                latitude=22.817, longitude=108.366,
                risk_level="high", risk_score=82.0,
                slope_assessments=[SlopeStability(
                    slope_angle=55.0, slope_height=42.0,
                    slope_type="复合坡", rock_type="石灰岩",
                    weathering_degree="中", joint_development="发育",
                    vegetation_coverage="差", drainage_condition="中",
                    geological_score=28.0,
                    survey_date=today.strftime("%Y-%m-%d"),
                    survey_team="广西地质工程勘察院",
                    remarks="坡顶存在张拉裂缝，雨季渗水明显，孤石群发育",
                ).to_dict()],
                historical_incidents=[
                    HistoricalIncident(
                        incident_date=(today - timedelta(days=180)).strftime("%Y-%m-%d"),
                        incident_type="rockfall", severity="moderate",
                        description="降雨诱发落石，最大块径约0.8m，砸坏护栏",
                        casualties="无", road_closure="半幅通行2小时",
                        source="南宁市交通局养护记录",
                    ).to_dict(),
                    HistoricalIncident(
                        incident_date=(today - timedelta(days=420)).strftime("%Y-%m-%d"),
                        incident_type="rockfall", severity="minor",
                        description="零星碎石掉落，影响应急车道",
                        casualties="无", road_closure="未中断",
                        source="交警巡逻记录",
                    ).to_dict(),
                ],
                linked_site_id="nanning_naan_s1", status="monitored",
                description="石灰岩高陡边坡，节理裂隙发育，坡面孤石群明显，汛期落石频发。车流量大(日均>8万辆)，一旦发生大规模落石后果严重。",
                responsible_unit="南宁市公路管理局", contact_person="张工",
                contact_phone="138-XXXX-XXXX",
                protection_measures="SNS主动防护网(局部), 挡石墙(K1952+250~350)",
                recommended_measures="加密主动防护网覆盖范围，增设被动防护网+监测摄像头",
                surveyed_at=today.strftime("%Y-%m-%d"),
            ),
            HazardPoint(
                hazard_id="HZ-G7211-001",
                name="崇左合那高速 2 号边坡隐患点",
                location="崇左合那高速 2 号边坡",
                region="广西·崇左",
                highway="G7211 南友高速 (合那段)",
                stake_mark="K138+800",
                latitude=22.379, longitude=107.365,
                risk_level="high", risk_score=75.0,
                slope_assessments=[SlopeStability(
                    slope_angle=48.0, slope_height=38.0,
                    slope_type="凹坡", rock_type="石灰岩",
                    weathering_degree="强", joint_development="发育",
                    vegetation_coverage="中", drainage_condition="差",
                    geological_score=35.0,
                    survey_date=today.strftime("%Y-%m-%d"),
                    survey_team="崇左市地质环境监测站",
                    remarks="喀斯特溶蚀裂隙发育，坡脚有溶洞，雨季地下水活动加剧",
                ).to_dict()],
                historical_incidents=[
                    HistoricalIncident(
                        incident_date=(today - timedelta(days=300)).strftime("%Y-%m-%d"),
                        incident_type="collapse", severity="major",
                        description="持续暴雨后局部崩塌，方量约30m³，阻断硬路肩",
                        casualties="轻伤1人", road_closure="封闭应急车道3天",
                        source="崇左市交通局",
                    ).to_dict(),
                ],
                linked_site_id="chongzuo_hena_s2", status="monitored",
                description="喀斯特地貌区典型隐患边坡，溶蚀裂隙发育，雨季地下水活跃。通往东盟陆路大通道关键节点，战略意义重大。",
                responsible_unit="崇左市公路管理局", contact_person="李工",
                contact_phone="139-XXXX-XXXX",
                protection_measures="被动防护网(坡脚)",
                recommended_measures="坡面排水系统改造+主动防护网+自动化监测",
                surveyed_at=today.strftime("%Y-%m-%d"),
            ),
            HazardPoint(
                hazard_id="HZ-G75-002",
                name="防城港兰海高速 3 号边坡隐患点",
                location="防城港兰海高速 3 号边坡",
                region="广西·防城港",
                highway="G75 兰海高速 (防城港段)",
                stake_mark="K2078+150",
                latitude=21.687, longitude=108.355,
                risk_level="medium", risk_score=55.0,
                slope_assessments=[SlopeStability(
                    slope_angle=38.0, slope_height=28.0,
                    slope_type="直线坡", rock_type="花岗岩",
                    weathering_degree="中", joint_development="较发育",
                    vegetation_coverage="中", drainage_condition="中",
                    geological_score=55.0,
                    survey_date=today.strftime("%Y-%m-%d"),
                    survey_team="防城港市地质环境监测站",
                    remarks="花岗岩球状风化明显，坡面有孤石分布，台风暴雨期风险升高",
                ).to_dict()],
                historical_incidents=[
                    HistoricalIncident(
                        incident_date=(today - timedelta(days=500)).strftime("%Y-%m-%d"),
                        incident_type="rockfall", severity="minor",
                        description="台风过境后零星落石，未影响通行",
                        casualties="无", road_closure="未中断",
                        source="防城港市交通局养护记录",
                    ).to_dict(),
                ],
                linked_site_id="fangchenggang_lanhai_s3", status="monitored",
                description="北部湾沿海关键通道，台风暴雨频发区域。花岗岩风化边坡，球状风化形成孤石。海运+陆运交汇枢纽。",
                responsible_unit="防城港市公路管理局", contact_person="王工",
                contact_phone="137-XXXX-XXXX",
                protection_measures="局部挂网",
                recommended_measures="全面排查孤石位置并加固，增设监测点",
                surveyed_at=today.strftime("%Y-%m-%d"),
            ),
            HazardPoint(
                hazard_id="HZ-G322-001",
                name="凭祥中越跨境公路 4 号边坡隐患点",
                location="凭祥中越跨境公路 4 号边坡",
                region="广西·凭祥 (中越边境)",
                highway="G322 中越跨境公路",
                stake_mark="K1042+600",
                latitude=22.094, longitude=106.767,
                risk_level="high", risk_score=78.0,
                slope_assessments=[SlopeStability(
                    slope_angle=52.0, slope_height=35.0,
                    slope_type="凸坡", rock_type="砂岩夹页岩",
                    weathering_degree="强", joint_development="发育",
                    vegetation_coverage="差", drainage_condition="差",
                    geological_score=25.0,
                    survey_date=today.strftime("%Y-%m-%d"),
                    survey_team="广西地质工程勘察院",
                    remarks="软硬岩互层差异风化显著，页岩遇水软化，凸坡临空面大",
                ).to_dict()],
                historical_incidents=[
                    HistoricalIncident(
                        incident_date=(today - timedelta(days=150)).strftime("%Y-%m-%d"),
                        incident_type="landslide", severity="major",
                        description="连续降雨后边坡局部滑塌，方量约50m³，堆积至行车道",
                        casualties="无", road_closure="全幅封闭4小时",
                        source="凭祥市交通运输局",
                    ).to_dict(),
                ],
                linked_site_id="pingxiang_crossborder_s4", status="monitored",
                description="中国-东盟自贸区门户路段，RCEP贸易通道咽喉。软硬岩互层差异风化严重，凸坡临空面大，稳定性差。",
                responsible_unit="凭祥市公路管理局", contact_person="陈工",
                contact_phone="136-XXXX-XXXX",
                protection_measures="SNS主动防护网(部分)",
                recommended_measures="坡面锚杆加固+截排水沟+全覆盖主动防护网+实时监测",
                surveyed_at=today.strftime("%Y-%m-%d"),
            ),
            HazardPoint(
                hazard_id="HZ-G85-001",
                name="宜宾高速滑坡监测点隐患点",
                location="四川宜宾 G85渝昆高速",
                region="四川·宜宾",
                highway="G85 渝昆高速 (宜宾段)",
                stake_mark="",
                latitude=28.750, longitude=104.620,
                risk_level="high", risk_score=90.0,
                slope_assessments=[SlopeStability(
                    slope_angle=60.0, slope_height=55.0,
                    slope_type="复合坡", rock_type="砂岩泥岩互层",
                    weathering_degree="强", joint_development="发育",
                    vegetation_coverage="差", drainage_condition="差",
                    geological_score=15.0,
                    survey_date=today.strftime("%Y-%m-%d"),
                    survey_team="四川省地质工程勘察院",
                    remarks="2026.3.7发生大规模滑坡，前兆小落石→大崩塌间隔仅43秒。红层软岩区，暴雨后极易再次滑动。",
                ).to_dict()],
                historical_incidents=[
                    HistoricalIncident(
                        incident_date="2026-03-07",
                        incident_type="landslide", severity="major",
                        description="大规模山体滑坡，滑坡体方量约500m³。事发前43秒出现前兆小落石，监测系统捕捉到异常并触发预警。",
                        casualties="无（预警及时）", road_closure="双向封闭48小时",
                        source="四川省交通运输厅/央视新闻",
                    ).to_dict(),
                ],
                linked_site_id="yibin_s1", status="monitored",
                description="四川盆地南缘红层软岩区高风险边坡。2026.3.7发生大规模滑坡，滑坡体约500m³。前兆小落石→大崩塌间隔仅43秒，凸显实时监测的必要性。",
                responsible_unit="四川省公路管理局", contact_person="刘工",
                contact_phone="135-XXXX-XXXX",
                protection_measures="临时防护网+监测摄像头",
                recommended_measures="坡面锚索框架梁加固+截排水系统工程+24小时自动化监测预警",
                surveyed_at=today.strftime("%Y-%m-d"),
            ),
        ]

        count = 0
        for h in demos:
            if self.get_by_id(h.hazard_id) is None:
                if self.insert(h):
                    count += 1
        return count

    # ---- 内部 ----

    def _row_to_hazard(self, row: dict) -> HazardPoint:
        """DB 行 → HazardPoint"""
        def _parse_json(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return []
            return v or []

        def _parse_str_list(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return []
            return v or []

        return HazardPoint(
            hazard_id=row["hazard_id"],
            name=row.get("name", ""),
            location=row.get("location", ""),
            region=row.get("region", ""),
            highway=row.get("highway", ""),
            stake_mark=row.get("stake_mark", ""),
            latitude=float(row.get("latitude", 0) or 0),
            longitude=float(row.get("longitude", 0) or 0),
            risk_level=row.get("risk_level", "medium"),
            risk_score=float(row.get("risk_score", 0) or 0),
            slope_assessments=_parse_json(row.get("slope_assessments")),
            historical_incidents=_parse_json(row.get("historical_incidents")),
            linked_site_id=row.get("linked_site_id", ""),
            status=row.get("status", "identified"),
            photo_paths=_parse_str_list(row.get("photo_paths")),
            report_path=row.get("report_path", ""),
            description=row.get("description", ""),
            responsible_unit=row.get("responsible_unit", ""),
            contact_person=row.get("contact_person", ""),
            contact_phone=row.get("contact_phone", ""),
            protection_measures=row.get("protection_measures", ""),
            recommended_measures=row.get("recommended_measures", ""),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
            surveyed_at=row.get("surveyed_at", ""),
        )

    def _mysql_insert(self, hazard: HazardPoint, now: str) -> bool:
        conn = None
        try:
            conn = self._mysql_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO hazard_points
                       (hazard_id, name, location, region, highway, stake_mark,
                        latitude, longitude, risk_level, risk_score,
                        slope_assessments, historical_incidents, linked_site_id,
                        status, photo_paths, report_path, description,
                        responsible_unit, contact_person, contact_phone,
                        protection_measures, recommended_measures,
                        created_at, updated_at, surveyed_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (hazard.hazard_id, hazard.name, hazard.location, hazard.region,
                     hazard.highway, hazard.stake_mark,
                     hazard.latitude, hazard.longitude,
                     hazard.risk_level, hazard.risk_score,
                     json.dumps(hazard.slope_assessments, ensure_ascii=False),
                     json.dumps(hazard.historical_incidents, ensure_ascii=False),
                     hazard.linked_site_id, hazard.status,
                     json.dumps(hazard.photo_paths, ensure_ascii=False),
                     hazard.report_path, hazard.description,
                     hazard.responsible_unit, hazard.contact_person, hazard.contact_phone,
                     hazard.protection_measures, hazard.recommended_measures,
                     now, now, hazard.surveyed_at),
                )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if conn is not None:
                conn.close()

    def _sqlite_insert(self, hazard: HazardPoint, now: str) -> bool:
        with self._lock:
            try:
                conn = self._sqlite_conn()
                conn.execute(
                    """INSERT INTO hazard_points
                       (hazard_id, name, location, region, highway, stake_mark,
                        latitude, longitude, risk_level, risk_score,
                        slope_assessments, historical_incidents, linked_site_id,
                        status, photo_paths, report_path, description,
                        responsible_unit, contact_person, contact_phone,
                        protection_measures, recommended_measures,
                        created_at, updated_at, surveyed_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (hazard.hazard_id, hazard.name, hazard.location, hazard.region,
                     hazard.highway, hazard.stake_mark,
                     hazard.latitude, hazard.longitude,
                     hazard.risk_level, hazard.risk_score,
                     json.dumps(hazard.slope_assessments, ensure_ascii=False),
                     json.dumps(hazard.historical_incidents, ensure_ascii=False),
                     hazard.linked_site_id, hazard.status,
                     json.dumps(hazard.photo_paths, ensure_ascii=False),
                     hazard.report_path, hazard.description,
                     hazard.responsible_unit, hazard.contact_person, hazard.contact_phone,
                     hazard.protection_measures, hazard.recommended_measures,
                     now, now, hazard.surveyed_at),
                )
                conn.commit()
                conn.close()
                return True
            except Exception:
                return False

    def _mysql_update(self, hazard: HazardPoint, now: str) -> bool:
        conn = None
        try:
            conn = self._mysql_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE hazard_points SET
                       name=%s, location=%s, region=%s, highway=%s, stake_mark=%s,
                       latitude=%s, longitude=%s, risk_level=%s, risk_score=%s,
                       slope_assessments=%s, historical_incidents=%s, linked_site_id=%s,
                       status=%s, photo_paths=%s, report_path=%s, description=%s,
                       responsible_unit=%s, contact_person=%s, contact_phone=%s,
                       protection_measures=%s, recommended_measures=%s,
                       updated_at=%s, surveyed_at=%s
                       WHERE hazard_id=%s""",
                    (hazard.name, hazard.location, hazard.region,
                     hazard.highway, hazard.stake_mark,
                     hazard.latitude, hazard.longitude,
                     hazard.risk_level, hazard.risk_score,
                     json.dumps(hazard.slope_assessments, ensure_ascii=False),
                     json.dumps(hazard.historical_incidents, ensure_ascii=False),
                     hazard.linked_site_id, hazard.status,
                     json.dumps(hazard.photo_paths, ensure_ascii=False),
                     hazard.report_path, hazard.description,
                     hazard.responsible_unit, hazard.contact_person, hazard.contact_phone,
                     hazard.protection_measures, hazard.recommended_measures,
                     now, hazard.surveyed_at, hazard.hazard_id),
                )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if conn is not None:
                conn.close()

    def _sqlite_update(self, hazard: HazardPoint, now: str) -> bool:
        with self._lock:
            try:
                conn = self._sqlite_conn()
                conn.execute(
                    """UPDATE hazard_points SET
                       name=?, location=?, region=?, highway=?, stake_mark=?,
                       latitude=?, longitude=?, risk_level=?, risk_score=?,
                       slope_assessments=?, historical_incidents=?, linked_site_id=?,
                       status=?, photo_paths=?, report_path=?, description=?,
                       responsible_unit=?, contact_person=?, contact_phone=?,
                       protection_measures=?, recommended_measures=?,
                       updated_at=?, surveyed_at=?
                       WHERE hazard_id=?""",
                    (hazard.name, hazard.location, hazard.region,
                     hazard.highway, hazard.stake_mark,
                     hazard.latitude, hazard.longitude,
                     hazard.risk_level, hazard.risk_score,
                     json.dumps(hazard.slope_assessments, ensure_ascii=False),
                     json.dumps(hazard.historical_incidents, ensure_ascii=False),
                     hazard.linked_site_id, hazard.status,
                     json.dumps(hazard.photo_paths, ensure_ascii=False),
                     hazard.report_path, hazard.description,
                     hazard.responsible_unit, hazard.contact_person, hazard.contact_phone,
                     hazard.protection_measures, hazard.recommended_measures,
                     now, hazard.surveyed_at, hazard.hazard_id),
                )
                conn.commit()
                conn.close()
                return True
            except Exception:
                return False

    def _mysql_query(self, sql: str, params: tuple) -> list[dict]:
        conn = None
        cur = None
        try:
            conn = self._mysql_conn()
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()

    def _sqlite_query(self, sql: str, params: tuple) -> list[dict]:
        with self._lock:
            conn = self._sqlite_conn()
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            result = [dict(r) for r in rows]
            conn.close()
        return result


# 模块级单例
_hazard_store: HazardStore | None = None
_hazard_store_lock = threading.Lock()


def get_hazard_store() -> HazardStore:
    """获取 HazardStore 单例"""
    global _hazard_store
    if _hazard_store is None:
        with _hazard_store_lock:
            if _hazard_store is None:
                _hazard_store = HazardStore()
    return _hazard_store


# ============================================================
# 风险评分计算
# ============================================================

def calculate_risk_score(assessment: SlopeStability, incidents: list[HistoricalIncident]) -> float:
    """
    根据边坡稳定性和历史灾害记录计算综合风险评分。

    评分维度:
      1. 地质稳定性 (权重 50%) — 来自 SlopeStability.geological_score
      2. 历史灾害频率与严重性 (权重 30%)
      3. 道路重要性 (权重 20%) — 简化: 默认中等

    返回: 0-100 的综合风险评分，越高越危险
    """
    # 地质稳定性分 (geological_score 越低越危险, 反转)
    geo_danger = max(0, 100 - assessment.geological_score)

    # 历史灾害分
    incident_score = 0.0
    if incidents:
        severity_weights = {"major": 90, "moderate": 60, "minor": 30}
        for inc in incidents:
            w = severity_weights.get(inc.severity, 30)
            # 越近期权重越高 (简化: 全部均等)
            incident_score = max(incident_score, w)
    # 多次事件加分
    if len(incidents) >= 3:
        incident_score = min(100, incident_score + 10)
    elif len(incidents) >= 2:
        incident_score = min(100, incident_score + 5)

    # 综合评分: 地质50% + 历史30% + 重要性20%
    importance = 60.0  # 默认中等重要性（国道/高速）
    score = geo_danger * 0.50 + incident_score * 0.30 + importance * 0.20

    return round(min(100, max(0, score)), 1)


def determine_risk_level(score: float) -> str:
    """根据评分确定风险等级"""
    if score >= 70:
        return "high"
    elif score >= 40:
        return "medium"
    else:
        return "low"


# ============================================================
# 报告生成
# ============================================================

def generate_hazard_report(hazards: list[HazardPoint]) -> str:
    """
    生成隐患点清单及风险评估报告 (Markdown格式)。

    可直接在 Streamlit 中渲染, 也可导出。
    """
    from datetime import datetime

    now = datetime.now().strftime("%Y年%m月%d日")
    total = len(hazards)
    high_count = sum(1 for h in hazards if h.risk_level == "high")
    medium_count = sum(1 for h in hazards if h.risk_level == "medium")
    low_count = sum(1 for h in hazards if h.risk_level == "low")

    lines = [
        f"# 公路边坡隐患点排查与风险评估报告",
        f"",
        f"**编制日期**: {now}",
        f"**排查范围**: 共排查 {total} 处隐患点",
        f"",
        f"---",
        f"",
        f"## 一、总体概况",
        f"",
        f"| 风险等级 | 数量 | 占比 |",
        f"|---------|------|------|",
        f"| 🔴 高风险 | {high_count} | {high_count/total*100:.0f}% |" if total > 0 else "| 🔴 高风险 | 0 | 0% |",
        f"| 🟡 中风险 | {medium_count} | {medium_count/total*100:.0f}% |" if total > 0 else "| 🟡 中风险 | 0 | 0% |",
        f"| 🟢 一般风险 | {low_count} | {low_count/total*100:.0f}% |" if total > 0 else "| 🟢 一般风险 | 0 | 0% |",
        f"",
        f"**监测覆盖情况**:",
        f"- 已部署监测: {sum(1 for h in hazards if h.linked_site_id)} 处",
        f"- 待部署监测: {sum(1 for h in hazards if not h.linked_site_id)} 处",
        f"",
        f"---",
        f"",
        f"## 二、高风险隐患点详表",
        f"",
    ]

    high_hazards = [h for h in hazards if h.risk_level == "high"]
    if high_hazards:
        lines.append("| # | 隐患点名称 | 公路/桩号 | 风险评分 | 边坡概况 | 历史灾害 | 监测状态 |")
        lines.append("|---|-----------|----------|---------|---------|---------|---------|")
        for i, h in enumerate(high_hazards, 1):
            latest_assessment = h.slope_assessments[-1] if h.slope_assessments else {}
            slope_desc = f"坡度{latest_assessment.get('slope_angle','?')}° 坡高{latest_assessment.get('slope_height','?')}m" if latest_assessment else "待评估"
            incident_count = len(h.historical_incidents)
            incident_desc = f"{incident_count}次" if incident_count > 0 else "无记录"
            monitor_status = "✅ 已部署" if h.linked_site_id else "⚠️ 待部署"
            lines.append(
                f"| {i} | {h.name} | {h.highway} {h.stake_mark} | {h.risk_score} | {slope_desc} | {incident_desc} | {monitor_status} |"
            )
        lines.append("")
    else:
        lines.append("无高风险隐患点。")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 三、各隐患点详细信息",
        "",
    ])

    for h in hazards:
        level_label = RISK_LEVELS.get(h.risk_level, {}).get("label", h.risk_level)
        status_label = HAZARD_STATUSES.get(h.status, h.status)
        lines.extend([
            f"### {level_label} — {h.name}",
            f"",
            f"- **隐患点编号**: {h.hazard_id}",
            f"- **地理位置**: {h.region} / {h.location}",
            f"- **公路桩号**: {h.highway} {h.stake_mark}" if h.stake_mark else f"- **公路**: {h.highway}",
            f"- **坐标**: ({h.latitude:.4f}, {h.longitude:.4f})",
            f"- **综合风险评分**: {h.risk_score}/100",
            f"- **状态**: {status_label}",
            f"- **关联监测点位**: {h.linked_site_id if h.linked_site_id else '未部署'}",
            f"",
            f"**综合描述**: {h.description}" if h.description else "",
            f"",
        ])

        # 边坡评估
        if h.slope_assessments:
            lines.append("**边坡稳定性评估**:")
            lines.append("")
            for sa in h.slope_assessments:
                lines.extend([
                    f"| 评估项目 | 数据 |",
                    f"|---------|------|",
                    f"| 勘察日期 | {sa.get('survey_date', '')} |",
                    f"| 勘察单位 | {sa.get('survey_team', '')} |",
                    f"| 坡度 | {sa.get('slope_angle', '')}° |",
                    f"| 坡高 | {sa.get('slope_height', '')} m |",
                    f"| 坡型 | {sa.get('slope_type', '')} |",
                    f"| 岩性 | {sa.get('rock_type', '')} |",
                    f"| 风化程度 | {sa.get('weathering_degree', '')} |",
                    f"| 节理发育 | {sa.get('joint_development', '')} |",
                    f"| 植被覆盖 | {sa.get('vegetation_coverage', '')} |",
                    f"| 排水条件 | {sa.get('drainage_condition', '')} |",
                    f"| 地质稳定性评分 | {sa.get('geological_score', '')}/100 |",
                    f"| 备注 | {sa.get('remarks', '')} |",
                    f"",
                ])

        # 历史灾害
        if h.historical_incidents:
            lines.append("**历史灾害记录**:")
            lines.append("")
            lines.append("| 日期 | 类型 | 严重程度 | 描述 | 道路影响 |")
            lines.append("|------|------|---------|------|---------|")
            for inc in h.historical_incidents:
                type_label = {"rockfall": "落石", "landslide": "滑坡", "collapse": "崩塌"}.get(
                    inc.get("incident_type", ""), inc.get("incident_type", ""))
                severity_label = {"major": "严重", "moderate": "中等", "minor": "轻微"}.get(
                    inc.get("severity", ""), inc.get("severity", ""))
                lines.append(
                    f"| {inc.get('incident_date', '')} | {type_label} | {severity_label} | "
                    f"{inc.get('description', '')[:40]} | {inc.get('road_closure', '')} |"
                )
            lines.append("")

        # 防护与建议
        if h.protection_measures or h.recommended_measures:
            lines.append("**防治措施**:")
            lines.append("")
            if h.protection_measures:
                lines.append(f"- 已有防护: {h.protection_measures}")
            if h.recommended_measures:
                lines.append(f"- 建议措施: {h.recommended_measures}")
            lines.append("")

        # 责任信息
        if h.responsible_unit:
            lines.append(f"- **责任单位**: {h.responsible_unit} / {h.contact_person} ({h.contact_phone})")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.extend([
        "",
        "---",
        "",
        f"*本报告由 RockGuard 公路落石灾害监测预警系统自动生成, 基于地质勘察数据与历史灾害数据综合分析。*",
        f"*报告日期: {now}*",
    ])

    return "\n".join(lines)
