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
from .config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from .road_segmentation import ROIParams
from .db_utils import is_mysql_available

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
    # ---- v2.2 扩展字段 (DB 存储) ----
    roi_polygon: list | None = None    # ROI 多边形坐标 [[x,y], ...]
    alert_contacts: list | None = None # 报警接收人 [{"name":"","phone":"","email":""}, ...]
    is_active: bool = True             # 是否启用
    model_override: str = ""           # 点位专用模型路径 (空=使用全局默认)
    # ---- v2.3 点位级检测阈值 (空/0=使用全局默认) ----
    detection_confidence: float = 0.0  # YOLO检测置信度 (0=使用全局)
    alert_blue_low: float = 0.0        # 蓝色预警下限
    alert_blue_high: float = 0.0       # 蓝→黄分界
    alert_yellow_high: float = 0.0     # 黄→橙分界
    alert_orange_high: float = 0.0     # 橙→红分界
    # ---- 时间戳 ----
    created_at: str = ""               # 创建时间 ISO
    updated_at: str = ""               # 更新时间 ISO

    def to_dict(self) -> dict:
        d = asdict(self)
        # 确保 JSON 可序列化
        if self.roi_polygon is not None:
            d["roi_polygon"] = self.roi_polygon
        if self.alert_contacts is not None:
            d["alert_contacts"] = self.alert_contacts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MonitoringSite":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def get_thresholds(self) -> dict:
        """
        获取该点位的有效检测阈值。
        点位字段 > 0 → 使用点位值; 否则 → 使用全局默认 (config.py)。
        返回:
            {detection_confidence, alert_blue_low, alert_blue_high,
             alert_yellow_high, alert_orange_high}
        """
        from .config import (
            DETECTION_CONFIDENCE as _DEF_CONF,
            ALERT_BLUE_CONFIDENCE_LOW as _DEF_BLUE_LOW,
            ALERT_BLUE_CONFIDENCE_HIGH as _DEF_BLUE_HIGH,
            ALERT_YELLOW_CONFIDENCE_HIGH as _DEF_YELLOW_HIGH,
            ALERT_ORANGE_CONFIDENCE_HIGH as _DEF_ORANGE_HIGH,
        )
        return {
            "detection_confidence": (
                self.detection_confidence if self.detection_confidence > 0
                else _DEF_CONF
            ),
            "alert_blue_low": (
                self.alert_blue_low if self.alert_blue_low > 0
                else _DEF_BLUE_LOW
            ),
            "alert_blue_high": (
                self.alert_blue_high if self.alert_blue_high > 0
                else _DEF_BLUE_HIGH
            ),
            "alert_yellow_high": (
                self.alert_yellow_high if self.alert_yellow_high > 0
                else _DEF_YELLOW_HIGH
            ),
            "alert_orange_high": (
                self.alert_orange_high if self.alert_orange_high > 0
                else _DEF_ORANGE_HIGH
            ),
        }


# ============================================================
# 站点持久化存储 (MySQL / SQLite 双后端)
# ============================================================

_MYSQL_AVAILABLE = is_mysql_available()

_SITE_TABLE_MYSQL = """\
CREATE TABLE IF NOT EXISTS monitoring_sites (
    site_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    location VARCHAR(256) DEFAULT '',
    region VARCHAR(128) DEFAULT '',
    camera_url VARCHAR(512) DEFAULT '',
    description TEXT,
    latitude DOUBLE DEFAULT 0,
    longitude DOUBLE DEFAULT 0,
    highway VARCHAR(128) DEFAULT '',
    stake_mark VARCHAR(64) DEFAULT '',
    risk_level VARCHAR(16) DEFAULT 'medium',
    roi_polygon JSON DEFAULT ('[]'),
    alert_contacts JSON DEFAULT ('[]'),
    is_active TINYINT DEFAULT 1,
    model_override VARCHAR(256) DEFAULT '',
    detection_confidence DOUBLE DEFAULT 0,
    alert_blue_low DOUBLE DEFAULT 0,
    alert_blue_high DOUBLE DEFAULT 0,
    alert_yellow_high DOUBLE DEFAULT 0,
    alert_orange_high DOUBLE DEFAULT 0,
    created_at VARCHAR(19) NOT NULL,
    updated_at VARCHAR(19) NOT NULL,
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""

_SITE_TABLE_SQLITE = """\
CREATE TABLE IF NOT EXISTS monitoring_sites (
    site_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT DEFAULT '',
    region TEXT DEFAULT '',
    camera_url TEXT DEFAULT '',
    description TEXT,
    latitude REAL DEFAULT 0,
    longitude REAL DEFAULT 0,
    highway TEXT DEFAULT '',
    stake_mark TEXT DEFAULT '',
    risk_level TEXT DEFAULT 'medium',
    roi_polygon TEXT DEFAULT '[]',
    alert_contacts TEXT DEFAULT '[]',
    is_active INTEGER DEFAULT 1,
    model_override TEXT DEFAULT '',
    detection_confidence REAL DEFAULT 0,
    alert_blue_low REAL DEFAULT 0,
    alert_blue_high REAL DEFAULT 0,
    alert_yellow_high REAL DEFAULT 0,
    alert_orange_high REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)"""


class SiteStore:
    """监测点位持久化 — MySQL 优先, SQLite 降级。线程安全。"""

    def __init__(self):

        self._lock = threading.RLock()
        # 后端探测
        self._backend = "sqlite"
        if MYSQL_HOST and _MYSQL_AVAILABLE:
            try:
                from .db_engine import get_mysql_engine
                engine = get_mysql_engine()
                if engine is not None:
                    conn = engine.raw_connection()
                    conn.close()  # 归还到池 (仅探测)
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
                cur.execute(_SITE_TABLE_MYSQL)
            conn.commit()
        except Exception as e:
            from .logger import log_event
            log_event("system", level="ERROR",
                      msg=f"MySQL 建表失败 ({e}), 降级为 SQLite")
            self._backend = "sqlite"
            self._init_sqlite_table()
        finally:
            if conn is not None:
                conn.close()  # 归还到池

    def _init_sqlite_table(self):
        conn = self._sqlite_conn()
        conn.execute(_SITE_TABLE_SQLITE)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sites_active ON monitoring_sites(is_active)")
        conn.commit()
        conn.close()

    # ---- 连接 ----

    def _mysql_conn(self):
        """从连接池获取一个 MySQL 连接。调用者 MUST 调用 .close() 归还到池。"""
        from .db_engine import get_mysql_engine
        engine = get_mysql_engine()
        if engine is None:
            raise RuntimeError("MySQL 引擎未初始化")
        return engine.raw_connection()

    def _sqlite_conn(self):
        import sqlite3
        db_path = DATA_DIR / "sites.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    # ---- CRUD ----

    def list_all(self, active_only: bool = False) -> list[MonitoringSite]:
        """列出所有站点"""
        if self._backend == "mysql":
            sql = "SELECT * FROM monitoring_sites"
            if active_only:
                sql += " WHERE is_active = 1"
            sql += " ORDER BY site_id ASC"
            rows = self._mysql_query(sql, ())
        else:
            sql = "SELECT * FROM monitoring_sites"
            if active_only:
                sql += " WHERE is_active = 1"
            sql += " ORDER BY site_id ASC"
            rows = self._sqlite_query(sql, ())
        return [self._row_to_site(r) for r in rows]

    def get_by_id(self, site_id: str) -> MonitoringSite | None:
        """按 ID 查找"""
        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT * FROM monitoring_sites WHERE site_id = %s", (site_id,))
        else:
            rows = self._sqlite_query(
                "SELECT * FROM monitoring_sites WHERE site_id = ?", (site_id,))
        return self._row_to_site(rows[0]) if rows else None

    def insert(self, site: MonitoringSite) -> bool:
        """插入新站点。返回 True 成功, False 重复 ID。"""
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        site.created_at = now
        site.updated_at = now
        roi_json = json.dumps(site.roi_polygon or [], ensure_ascii=False)
        contacts_json = json.dumps(site.alert_contacts or [], ensure_ascii=False)

        if self._backend == "mysql":
            return self._mysql_insert(site, roi_json, contacts_json, now)
        return self._sqlite_insert(site, roi_json, contacts_json, now)

    def update(self, site: MonitoringSite) -> bool:
        """更新已有站点。返回 True 成功, False 不存在。"""
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        site.updated_at = now
        roi_json = json.dumps(site.roi_polygon or [], ensure_ascii=False)
        contacts_json = json.dumps(site.alert_contacts or [], ensure_ascii=False)

        if self._backend == "mysql":
            return self._mysql_update(site, roi_json, contacts_json, now)
        return self._sqlite_update(site, roi_json, contacts_json, now)

    def delete(self, site_id: str) -> bool:
        """删除站点。返回 True 成功。"""
        if self._backend == "mysql":
            conn = None
            try:
                conn = self._mysql_conn()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM monitoring_sites WHERE site_id = %s", (site_id,))
                conn.commit()
                return True
            except Exception:
                return False
            finally:
                if conn is not None:
                    conn.close()  # 归还到池
        else:
            with self._lock:
                conn = self._sqlite_conn()
                conn.execute("DELETE FROM monitoring_sites WHERE site_id = ?", (site_id,))
                conn.commit()
                conn.close()
            return True

    def count(self) -> int:
        """返回站点总数"""
        if self._backend == "mysql":
            rows = self._mysql_query("SELECT COUNT(*) as cnt FROM monitoring_sites", ())
        else:
            rows = self._sqlite_query("SELECT COUNT(*) as cnt FROM monitoring_sites", ())
        return rows[0]["cnt"] if rows else 0

    # ---- 种子数据迁移 ----

    def seed_from_presets(self, presets: list[MonitoringSite]) -> int:
        """将预设点位写入 DB（仅当 DB 为空时执行），返回写入条数。"""
        if self.count() > 0:
            return 0
        count = 0
        for site in presets:
            if self.insert(site):
                count += 1
        return count

    # ---- 内部 ----

    def _row_to_site(self, row: dict) -> MonitoringSite:
        """DB 行 → MonitoringSite"""
        def _parse_json(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return []
            return v or []

        return MonitoringSite(
            site_id=row["site_id"],
            name=row.get("name", ""),
            location=row.get("location", ""),
            region=row.get("region", ""),
            camera_url=row.get("camera_url", ""),
            description=row.get("description", ""),
            latitude=float(row.get("latitude", 0) or 0),
            longitude=float(row.get("longitude", 0) or 0),
            highway=row.get("highway", ""),
            stake_mark=row.get("stake_mark", ""),
            risk_level=row.get("risk_level", "medium"),
            roi_polygon=_parse_json(row.get("roi_polygon")),
            alert_contacts=_parse_json(row.get("alert_contacts")),
            is_active=bool(row.get("is_active", 1)),
            model_override=row.get("model_override", ""),
            detection_confidence=float(row.get("detection_confidence", 0) or 0),
            alert_blue_low=float(row.get("alert_blue_low", 0) or 0),
            alert_blue_high=float(row.get("alert_blue_high", 0) or 0),
            alert_yellow_high=float(row.get("alert_yellow_high", 0) or 0),
            alert_orange_high=float(row.get("alert_orange_high", 0) or 0),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )

    def _mysql_insert(self, site, roi_json, contacts_json, now):
        conn = None
        try:
            conn = self._mysql_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO monitoring_sites
                       (site_id, name, location, region, camera_url, description,
                        latitude, longitude, highway, stake_mark, risk_level,
                        roi_polygon, alert_contacts, is_active, model_override,
                        detection_confidence, alert_blue_low, alert_blue_high,
                        alert_yellow_high, alert_orange_high,
                        created_at, updated_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (site.site_id, site.name, site.location, site.region,
                     site.camera_url, site.description,
                     site.latitude, site.longitude, site.highway, site.stake_mark,
                     site.risk_level, roi_json, contacts_json,
                     1 if site.is_active else 0, site.model_override,
                     site.detection_confidence, site.alert_blue_low,
                     site.alert_blue_high, site.alert_yellow_high,
                     site.alert_orange_high, now, now),
                )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if conn is not None:
                conn.close()  # 归还到池

    def _sqlite_insert(self, site, roi_json, contacts_json, now):
        with self._lock:
            try:
                conn = self._sqlite_conn()
                conn.execute(
                    """INSERT INTO monitoring_sites
                       (site_id, name, location, region, camera_url, description,
                        latitude, longitude, highway, stake_mark, risk_level,
                        roi_polygon, alert_contacts, is_active, model_override,
                        detection_confidence, alert_blue_low, alert_blue_high,
                        alert_yellow_high, alert_orange_high,
                        created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (site.site_id, site.name, site.location, site.region,
                     site.camera_url, site.description,
                     site.latitude, site.longitude, site.highway, site.stake_mark,
                     site.risk_level, roi_json, contacts_json,
                     1 if site.is_active else 0, site.model_override,
                     site.detection_confidence, site.alert_blue_low,
                     site.alert_blue_high, site.alert_yellow_high,
                     site.alert_orange_high, now, now),
                )
                conn.commit()
                conn.close()
                return True
            except Exception:
                return False

    def _mysql_update(self, site, roi_json, contacts_json, now):
        conn = None
        try:
            conn = self._mysql_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE monitoring_sites SET
                       name=%s, location=%s, region=%s, camera_url=%s,
                       description=%s, latitude=%s, longitude=%s,
                       highway=%s, stake_mark=%s, risk_level=%s,
                       roi_polygon=%s, alert_contacts=%s, is_active=%s,
                       model_override=%s,
                       detection_confidence=%s, alert_blue_low=%s,
                       alert_blue_high=%s, alert_yellow_high=%s,
                       alert_orange_high=%s,
                       updated_at=%s
                       WHERE site_id=%s""",
                    (site.name, site.location, site.region, site.camera_url,
                     site.description, site.latitude, site.longitude,
                     site.highway, site.stake_mark, site.risk_level,
                     roi_json, contacts_json,
                     1 if site.is_active else 0, site.model_override,
                     site.detection_confidence, site.alert_blue_low,
                     site.alert_blue_high, site.alert_yellow_high,
                     site.alert_orange_high,
                     now, site.site_id),
                )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if conn is not None:
                conn.close()  # 归还到池

    def _sqlite_update(self, site, roi_json, contacts_json, now):
        with self._lock:
            try:
                conn = self._sqlite_conn()
                conn.execute(
                    """UPDATE monitoring_sites SET
                       name=?, location=?, region=?, camera_url=?,
                       description=?, latitude=?, longitude=?,
                       highway=?, stake_mark=?, risk_level=?,
                       roi_polygon=?, alert_contacts=?, is_active=?,
                       model_override=?,
                       detection_confidence=?, alert_blue_low=?,
                       alert_blue_high=?, alert_yellow_high=?,
                       alert_orange_high=?,
                       updated_at=?
                       WHERE site_id=?""",
                    (site.name, site.location, site.region, site.camera_url,
                     site.description, site.latitude, site.longitude,
                     site.highway, site.stake_mark, site.risk_level,
                     roi_json, contacts_json,
                     1 if site.is_active else 0, site.model_override,
                     site.detection_confidence, site.alert_blue_low,
                     site.alert_blue_high, site.alert_yellow_high,
                     site.alert_orange_high,
                     now, site.site_id),
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
            result = [dict(zip(cols, r)) for r in rows]
            return result
        except Exception:
            return []
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()  # 归还到池

    def _sqlite_query(self, sql: str, params: tuple) -> list[dict]:
        with self._lock:
            conn = self._sqlite_conn()
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            result = [dict(r) for r in rows]
            conn.close()
        return result


# 模块级单例
_site_store: SiteStore | None = None
_site_store_lock = threading.Lock()


def get_site_store() -> SiteStore:
    """获取 SiteStore 单例"""
    global _site_store
    if _site_store is None:
        with _site_store_lock:
            if _site_store is None:
                _site_store = SiteStore()
    return _site_store


# ============================================================
# 4 个预设演示点位 — 广西本地 + 东盟跨境全覆盖 (DB 种子数据)
# ============================================================

PRESET_SITES: list[MonitoringSite] = [
    MonitoringSite(
        site_id="qinzhou_s0",
        name="钦州公路边坡监测点",
        location="钦州公路边坡监测点",
        region="广西·钦州",
        camera_url="",
        description="钦州落石试验监测点",
        latitude=21.96,
        longitude=108.62,
        highway="G75 兰海高速 (钦州段)",
        stake_mark="",
        risk_level="high",
    ),
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
    MonitoringSite(
        site_id="yibin_s1",
        name="宜宾高速滑坡监测点",
        location="四川宜宾 G85渝昆高速",
        region="四川·宜宾",
        camera_url="",
        description="四川盆地南缘高风险边坡，2026.3.7发生大规模滑坡，前兆小落石→大崩塌间隔43秒",
        latitude=28.750,
        longitude=104.620,
        highway="G85 渝昆高速 (宜宾段)",
        stake_mark="",
        risk_level="high",
        detection_confidence=0.10,
        alert_blue_low=0.10,
        alert_blue_high=0.20,
        alert_yellow_high=0.40,
        alert_orange_high=0.70,
    ),
]

# ============================================================
# 配置持久化路径
# ============================================================

SITE_CONFIG_PATH = DATA_DIR / "site_config.json"
SITE_STATE_PATH = DATA_DIR / "site_state.json"       # 当前激活点位 + 全局状态
ROI_CONFIG_PATH = DATA_DIR / "roi_config.json"      # 兼容旧路径 (ROI校准数据), 与 config.py 保持一致

# ============================================================
# 点位切换与管理 — 线程安全
# ============================================================

_site_lock = threading.RLock()
_active_site: MonitoringSite | None = None


# ---- DB 优先读取的辅助 ----

def _get_all_sites() -> list[MonitoringSite]:
    """
    获取全部站点。优先级: DB > PRESET_SITES fallback。
    首次调用时若 DB 为空，自动将 PRESET_SITES 写入 DB。
    """
    try:
        store = get_site_store()
        sites = store.list_all()
        if not sites:
            # DB 为空 → 种子迁移 + 返回预设
            count = store.seed_from_presets(PRESET_SITES)
            if count > 0:
                from .logger import log_event
                log_event("system", level="INFO",
                          msg=f"监测点位种子数据已写入 DB ({count} 条)")
            return list(PRESET_SITES)
        # 只返回启用的站点 (DB 来源)
        active = [s for s in sites if s.is_active]
        if not active:
            # 所有站点都被停用 → 至少返回第一个（保证 get_active_site 不会崩溃）
            from .logger import log_event
            log_event("system", level="WARN",
                      msg="所有监测点位均为停用状态，将返回第一个可用点位")
            return sites[:1]
        return active
    except Exception as e:
        # DB 不可用时回退到硬编码预设 (记录警告)
        from .logger import log_event
        log_event("system", level="WARN",
                  msg=f"监测点位 DB 读取失败 ({e})，回退到 PRESET_SITES 硬编码列表")
        return list(PRESET_SITES)


def list_sites() -> list[MonitoringSite]:
    """获取全部启用的点位列表 (DB 优先, fallback 到 PRESET_SITES)"""
    return _get_all_sites()


def list_all_sites_admin() -> list[MonitoringSite]:
    """
    获取全部点位（含停用），供管理页面使用。
    与 list_sites() 不同：此函数返回包括 is_active=False 的点位。
    """
    try:
        store = get_site_store()
        sites = store.list_all()
        if not sites:
            store.seed_from_presets(PRESET_SITES)
            return list(PRESET_SITES)
        return sites
    except Exception:
        return list(PRESET_SITES)


def get_site_by_id(site_id: str) -> MonitoringSite | None:
    """按 ID 查找点位 (DB 优先)"""
    try:
        store = get_site_store()
        site = store.get_by_id(site_id)
        if site is not None:
            return site
    except Exception as e:
        from .logger import log_event
        log_event("system", level="WARN",
                  msg=f"DB 查询点位 {site_id} 失败 ({e})，回退 PRESET_SITES")
    # fallback
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
      3. DB 中第一个启用的点位
      4. 预设第一个点位
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

        # 默认: DB 第一个点位，否则 PRESET_SITES 第一个
        all_sites = _get_all_sites()
        _active_site = all_sites[0] if all_sites else PRESET_SITES[0]
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
        all_sites = _get_all_sites()
        raise ValueError(
            f"无效的点位ID: {site_id}。可用点位: {[s.site_id for s in all_sites]}"
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
    """获取所有已启用点位的 location 值列表 (用于报警记录筛选下拉)"""
    return [s.location for s in _get_all_sites()]
