"""
持久化层 — MySQL/SQLite 预警存储 + 失败重试
============================================
优先使用 MySQL (多节点共享), 未配置或连接失败时降级到 SQLite。

四级预警等级 (对齐《公路自然灾害监测预警系统技术指南》):
  Ⅰ 级 (特别严重，红色):   置信度 > 0.9 或 直径 > 30cm
  Ⅱ 级 (严重，橙色):       置信度 0.7-0.9 或 直径 20-30cm
  Ⅲ 级 (较重，黄色):       置信度 0.5-0.7 或 直径 10-20cm
  Ⅳ 级 (一般，蓝色):       置信度 0.3-0.5 或 直径 < 10cm

使用方式:
    from rockfall.alert_store import AlertStore
    store = AlertStore()
    store.save_alert(count, max_conf, track_ids, saved_to, ...)
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from .logger import log_event
from .config import (
    DATA_DIR, PUSHPLUS_TOKEN, PUSHPLUS_URL, PUSHPLUS_TOPIC,
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    ALERT_HASH_CHAIN_ENABLED, ALERT_HASH_GENESIS,
)
from .db_utils import is_mysql_available

_MYSQL_AVAILABLE = is_mysql_available()

_MYSQL_SCHEMA = """\
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    time VARCHAR(19) NOT NULL,
    alert_level VARCHAR(10) DEFAULT 'yellow',
    count INTEGER DEFAULT 0,
    max_confidence DOUBLE DEFAULT 0,
    track_ids JSON DEFAULT ('[]'),
    class_summary VARCHAR(255) DEFAULT '',
    saved_frame VARCHAR(500) DEFAULT '',
    clip_path VARCHAR(500) DEFAULT '',
    push_status VARCHAR(20) DEFAULT 'pending',
    push_msg VARCHAR(500) DEFAULT '',
    rock_diameter_cm DOUBLE DEFAULT 0,
    monitoring_location VARCHAR(100) DEFAULT '',
    review_status VARCHAR(20) DEFAULT '',
    reviewer_note VARCHAR(500) DEFAULT '',
    workflow_state VARCHAR(20) DEFAULT 'pending',
    workflow_history JSON DEFAULT ('[]'),
    operator VARCHAR(50) DEFAULT '',
    request_id VARCHAR(32) DEFAULT '',
    session_id VARCHAR(32) DEFAULT '',
    data_hash VARCHAR(64) DEFAULT '',
    prev_hash VARCHAR(64) DEFAULT '',
    created_at VARCHAR(19) NOT NULL,
    INDEX idx_workflow_state (workflow_state),
    INDEX idx_push_status (push_status),
    INDEX idx_time (time),
    INDEX idx_alert_level (alert_level),
    INDEX idx_request_id (request_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""

_SQLITE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    alert_level TEXT DEFAULT 'yellow',
    count INTEGER DEFAULT 0,
    max_confidence REAL DEFAULT 0,
    track_ids TEXT DEFAULT '[]',
    class_summary TEXT DEFAULT '',
    saved_frame TEXT DEFAULT '',
    clip_path TEXT DEFAULT '',
    push_status TEXT DEFAULT 'pending',
    push_msg TEXT DEFAULT '',
    rock_diameter_cm REAL DEFAULT 0,
    monitoring_location TEXT DEFAULT '',
    review_status TEXT DEFAULT '',
    reviewer_note TEXT DEFAULT '',
    workflow_state TEXT DEFAULT 'pending',
    workflow_history TEXT DEFAULT '[]',
    operator TEXT DEFAULT '',
    request_id TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    data_hash TEXT DEFAULT '',
    prev_hash TEXT DEFAULT '',
    created_at TEXT NOT NULL
)"""

# 增量迁移: 为旧表补充新增列 (兼容已部署系统)
_MIGRATIONS = {
    "sqlite": [
        "ALTER TABLE alerts ADD COLUMN rock_diameter_cm REAL DEFAULT 0",
        "ALTER TABLE alerts ADD COLUMN monitoring_location TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN review_status TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN reviewer_note TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN clip_path TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN workflow_state TEXT DEFAULT 'pending'",
        "ALTER TABLE alerts ADD COLUMN workflow_history TEXT DEFAULT '[]'",
        "ALTER TABLE alerts ADD COLUMN operator TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN request_id TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN session_id TEXT DEFAULT ''",
        # 哈希链防篡改 (v2.6+)
        "ALTER TABLE alerts ADD COLUMN data_hash TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN prev_hash TEXT DEFAULT ''",
    ],
    "mysql": [
        "ALTER TABLE alerts ADD COLUMN rock_diameter_cm DOUBLE DEFAULT 0",
        "ALTER TABLE alerts ADD COLUMN monitoring_location VARCHAR(100) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN review_status VARCHAR(20) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN reviewer_note VARCHAR(500) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN clip_path VARCHAR(500) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN workflow_state VARCHAR(20) DEFAULT 'pending'",
        "ALTER TABLE alerts ADD COLUMN workflow_history JSON DEFAULT ('[]')",
        "ALTER TABLE alerts ADD COLUMN operator VARCHAR(50) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN request_id VARCHAR(32) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN session_id VARCHAR(32) DEFAULT ''",
        # 哈希链防篡改 (v2.6+)
        "ALTER TABLE alerts ADD COLUMN data_hash VARCHAR(64) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN prev_hash VARCHAR(64) DEFAULT ''",
    ],
}


class AlertStore:
    """预警持久化 — MySQL 优先, SQLite 降级"""

    def __init__(self, db_path: str = ""):
        self._db_path = Path(db_path) if db_path else DATA_DIR / "alerts.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._retry_thread: threading.Thread | None = None
        self._stop_retry = threading.Event()

        self._backend = self._init_backend()
        self._init_db()

    # ---- 后端探测 ----

    def _init_backend(self) -> str:
        """探测可用的存储后端, 返回 'mysql' 或 'sqlite'"""
        if MYSQL_HOST and _MYSQL_AVAILABLE:
            try:
                from .db_engine import get_mysql_engine
                engine = get_mysql_engine()
                if engine is not None:
                    conn = engine.raw_connection()
                    conn.close()  # 归还到池 (仅探测)
                    return "mysql"
                else:
                    log_event("system", level="WARN", msg="MySQL 引擎创建失败, 降级为 SQLite")
            except Exception as e:
                log_event("system", level="WARN", msg=f"MySQL 连接失败 ({e}), 降级为 SQLite")
        return "sqlite"

    # ---- 建表 ----

    def _init_db(self):
        if self._backend == "mysql":
            self._init_mysql_table()
        else:
            self._init_sqlite_table()
        # 1. Alembic 管理主版本迁移 (建表/加列)
        try:
            from rockfall.migration import run_migrations
            run_migrations()
        except Exception as e:
            log_event("system", level="WARN",
                      msg=f"Alembic 迁移运行失败 ({e}), 回退手工迁移")
        # 2. 始终执行手工增量迁移 (补充 Alembic 未覆盖的 ALTER TABLE, 幂等安全)
        self._run_migrations()

    def _init_mysql_table(self):
        conn = None
        try:
            conn = self._mysql_conn()
            with conn.cursor() as cur:
                cur.execute(_MYSQL_SCHEMA)
            conn.commit()
        except Exception as e:
            log_event("system", level="ERROR", msg=f"MySQL 建表失败 ({e}), 降级为 SQLite")
            self._backend = "sqlite"
            self._init_sqlite_table()
        finally:
            if conn is not None:
                conn.close()  # 归还到池

    def _init_sqlite_table(self):
        with self._get_sqlite_conn() as conn:
            conn.execute(_SQLITE_SCHEMA)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_push_status ON alerts(push_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON alerts(time)")
            conn.commit()

    def _run_migrations(self):
        """增量迁移: 为新版本新增的列执行 ALTER TABLE (忽略已存在的列)"""
        backend = "mysql" if self._backend == "mysql" else "sqlite"
        migrations = _MIGRATIONS.get(backend, [])
        for sql in migrations:
            conn = None
            try:
                if self._backend == "mysql":
                    conn = self._mysql_conn()
                    with conn.cursor() as cur:
                        cur.execute(sql)
                    conn.commit()
                else:
                    with self._lock:
                        with self._get_sqlite_conn() as conn:
                            conn.execute(sql)
                            conn.commit()
            except Exception:
                # 列已存在或其他迁移错误, 静默跳过
                pass
            finally:
                if conn is not None and self._backend == "mysql":
                    conn.close()  # 归还到池

    # ---- 连接 ----

    def _mysql_conn(self):
        """从连接池获取一个 MySQL 连接。调用者 MUST 调用 .close() 归还到池。"""
        from .db_engine import get_mysql_engine
        engine = get_mysql_engine()
        if engine is None:
            raise RuntimeError("MySQL 引擎未初始化")
        return engine.raw_connection()

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ---- 写入 ----

    def save_alert(self, count: int, max_confidence: float,
                   track_ids: list[int] | None = None,
                   alert_level: str = "yellow",
                   class_summary: str = "",
                   saved_frame: str = "",
                   clip_path: str = "",
                   push_status: str = "pending",
                   rock_diameter_cm: float = 0,
                   monitoring_location: str = "") -> int:
        """保存预警记录, 返回 row ID。自动携带 trace 上下文。

        当 ALERT_HASH_CHAIN_ENABLED 为 True 时，自动计算 SHA256 哈希链。
        """
        # 提取当前 trace 上下文
        try:
            from .trace import get_request_id, get_session_id
            _rid = get_request_id()
            _sid = get_session_id()
        except Exception:
            _rid, _sid = "", ""

        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        # ---- 哈希链计算 ----
        data_hash = ""
        prev_hash = ""
        if ALERT_HASH_CHAIN_ENABLED:
            prev_hash = self._get_latest_hash_internal()
            if not prev_hash:
                prev_hash = ALERT_HASH_GENESIS
            # 构造字段 dict 用于哈希计算
            from .hash_chain import compute_record_hash
            fields = {
                "time": ts,
                "alert_level": alert_level,
                "count": count,
                "max_confidence": round(max_confidence, 4),
                "track_ids": track_ids or [],
                "class_summary": class_summary,
                "saved_frame": saved_frame,
                "clip_path": clip_path,
                "rock_diameter_cm": float(round(rock_diameter_cm, 1)),
                "monitoring_location": monitoring_location,
            }
            data_hash = compute_record_hash(fields, prev_hash)

        params = (
            ts, alert_level, count, round(max_confidence, 4),
            json.dumps(track_ids or [], separators=(",", ":"), ensure_ascii=False),
            class_summary, saved_frame,
            clip_path, push_status, round(rock_diameter_cm, 1), monitoring_location,
            _rid, _sid, ts, data_hash, prev_hash,
        )
        if self._backend == "mysql":
            return self._mysql_insert(params)
        return self._sqlite_insert(params)

    def _mysql_insert(self, params: tuple) -> int:
        conn = None
        cur = None
        try:
            conn = self._mysql_conn()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO alerts
                   (time, alert_level, count, max_confidence, track_ids,
                    class_summary, saved_frame, clip_path, push_status,
                    rock_diameter_cm, monitoring_location,
                    request_id, session_id, created_at,
                    data_hash, prev_hash)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                params,
            )
            conn.commit()
            rid = cur.lastrowid
            return rid
        except Exception:
            return -1
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()  # 归还到池

    def _sqlite_insert(self, params: tuple) -> int:
        with self._lock:
            with self._get_sqlite_conn() as conn:
                cur = conn.execute(
                    """INSERT INTO alerts
                       (time, alert_level, count, max_confidence, track_ids,
                        class_summary, saved_frame, clip_path, push_status,
                        rock_diameter_cm, monitoring_location,
                        request_id, session_id, created_at,
                        data_hash, prev_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    params,
                )
                conn.commit()
                return cur.lastrowid

    def mark_sent(self, alert_id: int, msg: str = ""):
        self._update_status(alert_id, "sent", msg)

    def mark_failed(self, alert_id: int, msg: str = ""):
        self._update_status(alert_id, "failed", msg)

    def _update_status(self, alert_id: int, status: str, msg: str):
        if self._backend == "mysql":
            conn = None
            try:
                conn = self._mysql_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE alerts SET push_status=%s, push_msg=%s WHERE id=%s",
                        (status, msg, alert_id),
                    )
                conn.commit()
            except Exception:
                pass
            finally:
                if conn is not None:
                    conn.close()  # 归还到池
        else:
            with self._lock:
                with self._get_sqlite_conn() as conn:
                    conn.execute(
                        "UPDATE alerts SET push_status=?, push_msg=? WHERE id=?",
                        (status, msg, alert_id),
                    )
                    conn.commit()

    # ---- 查询 ----

    def get_pending(self, limit: int = 20) -> list[dict]:
        if self._backend == "mysql":
            return self._mysql_query(
                "SELECT * FROM alerts WHERE push_status='pending' "
                "ORDER BY id ASC LIMIT %s", (limit,)
            )
        return self._sqlite_query(
            "SELECT * FROM alerts WHERE push_status='pending' "
            "ORDER BY id ASC LIMIT ?", (limit,)
        )

    def get_recent(self, limit: int = 50) -> list[dict]:
        if self._backend == "mysql":
            return self._mysql_query(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT %s", (limit,)
            )
        return self._sqlite_query(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        )

    def get_latest_alert(self, min_level: str = "yellow") -> dict | None:
        """获取最近一条达到指定等级及以上的预警 (用于 SSE 推送和 UI 弹窗)。

        min_level: 最低推送等级 — "blue" | "yellow" | "orange" | "red"
        等级优先级: blue < yellow < orange < red
        """
        levels = ["blue", "yellow", "orange", "red"]
        try:
            idx = levels.index(min_level)
        except ValueError:
            idx = 1  # 默认 yellow
        target_levels = levels[idx:]

        if self._backend == "mysql":
            placeholders = ",".join(["%s"] * len(target_levels))
            rows = self._mysql_query(
                f"SELECT * FROM alerts WHERE alert_level IN ({placeholders}) "
                "ORDER BY id DESC LIMIT 1",
                tuple(target_levels),
            )
        else:
            placeholders = ",".join(["?"] * len(target_levels))
            rows = self._sqlite_query(
                f"SELECT * FROM alerts WHERE alert_level IN ({placeholders}) "
                "ORDER BY id DESC LIMIT 1",
                tuple(target_levels),
            )
        return rows[0] if rows else None

    def count_today_by_level(self) -> dict[str, int]:
        """统计今日各等级预警数量, 返回 {"blue": N, "yellow": N, "orange": N, "red": N}"""
        today = datetime.now().strftime("%Y-%m-%d")
        result = {"blue": 0, "yellow": 0, "orange": 0, "red": 0}
        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT alert_level, COUNT(*) as cnt FROM alerts "
                "WHERE time LIKE %s GROUP BY alert_level",
                (today + "%",),
            )
        else:
            rows = self._sqlite_query(
                "SELECT alert_level, COUNT(*) as cnt FROM alerts "
                "WHERE time LIKE ? GROUP BY alert_level",
                (today + "%",),
            )
        for r in rows:
            level = r.get("alert_level", "")
            if level in result:
                result[level] = r.get("cnt", 0)
        return result

    def _mysql_query(self, sql: str, params: tuple) -> list[dict]:
        conn = None
        cur = None
        try:
            conn = self._mysql_conn()
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            result = [self._row_to_dict(r, cur) for r in rows]
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
            with self._get_sqlite_conn() as conn:
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_dict(r, cur) for r in rows]

    # ---- 重试 ----

    def start_retry_loop(self, interval: int = 30):
        if self._retry_thread is not None and self._retry_thread.is_alive():
            return
        self._stop_retry.clear()
        self._retry_thread = threading.Thread(
            target=self._retry_loop, args=(interval,), daemon=True,
            name="alert-retry",
        )
        self._retry_thread.start()

    def stop_retry_loop(self):
        self._stop_retry.set()

    # 四级预警标签 (对齐交通部标准)
    LEVEL_LABELS = {
        "red":    "🔴 Ⅰ级·特别严重",
        "orange": "🟠 Ⅱ级·严重",
        "yellow": "🟡 Ⅲ级·较重",
        "blue":   "🔵 Ⅳ级·一般",
    }

    def _retry_loop(self, interval: int):
        while not self._stop_retry.wait(interval):
            pending = self.get_pending(limit=5)
            for alert in pending:
                if self._stop_retry.is_set():
                    return
                if not PUSHPLUS_TOKEN or PUSHPLUS_TOKEN == "your_token_here":
                    continue
                try:
                    import requests
                    level = alert.get("alert_level", "yellow")
                    level_label = self.LEVEL_LABELS.get(level, "⚠️ 预警")
                    class_info = alert.get("class_summary", "落石") or "落石"
                    data = {
                        "token": PUSHPLUS_TOKEN,
                        "title": f"{level_label} {class_info}报警（补发）",
                        "content": f"补发预警: {alert['time']}, "
                                   f"数量={alert['count']}, "
                                   f"置信度={alert['max_confidence']}",
                        "topic": PUSHPLUS_TOPIC,
                        "template": "html",
                    }
                    res = requests.post(PUSHPLUS_URL, json=data, timeout=10).json()
                    if res.get("code") == 200:
                        self.mark_sent(alert["id"], "retry_ok")
                    else:
                        self.mark_failed(alert["id"], str(res.get("msg", "")))
                except Exception as e:
                    self.mark_failed(alert["id"], str(e))

    # ---- 归档查询 (日期范围 + 等级筛选) ----

    def query_alerts(
        self, start_date: str = "", end_date: str = "",
        alert_level: str = "", limit: int = 10000,
        offset: int = 0,
    ) -> list[dict]:
        """
        灵活查询预警记录, 支持日期范围和等级筛选 (用于归档和导出)。

        参数:
            start_date: 起始日期 "2026-06-01" (含)
            end_date:   结束日期 "2026-06-12" (含)
            alert_level: 预警等级筛选 "red"/"orange"/"yellow"/"blue"/""=全部
            limit:       返回上限
            offset:      分页偏移
        返回: dict 列表, 按时间降序
        """
        conditions = []
        params = []

        if start_date:
            conditions.append("time >= ?" if self._backend != "mysql" else "time >= %s")
            params.append(start_date + " 00:00:00")
        if end_date:
            conditions.append("time <= ?" if self._backend != "mysql" else "time <= %s")
            params.append(end_date + " 23:59:59")
        if alert_level:
            conditions.append("alert_level = ?" if self._backend != "mysql" else "alert_level = %s")
            params.append(alert_level)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        placeholder = "?" if self._backend != "mysql" else "%s"

        if self._backend == "mysql":
            return self._mysql_query(
                f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT %s OFFSET %s",
                tuple(params) + (limit, offset),
            )
        return self._sqlite_query(
            f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            tuple(params) + (limit, offset),
        )

    def count_alerts(
        self, start_date: str = "", end_date: str = "", alert_level: str = "",
    ) -> int:
        """统计符合条件的预警记录数"""
        conditions = []
        params = []

        if start_date:
            conditions.append("time >= ?" if self._backend != "mysql" else "time >= %s")
            params.append(start_date + " 00:00:00")
        if end_date:
            conditions.append("time <= ?" if self._backend != "mysql" else "time <= %s")
            params.append(end_date + " 23:59:59")
        if alert_level:
            conditions.append("alert_level = ?" if self._backend != "mysql" else "alert_level = %s")
            params.append(alert_level)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        if self._backend == "mysql":
            rows = self._mysql_query(
                f"SELECT COUNT(*) as cnt FROM alerts {where}", tuple(params),
            )
        else:
            rows = self._sqlite_query(
                f"SELECT COUNT(*) as cnt FROM alerts {where}", tuple(params),
            )
        return rows[0].get("cnt", 0) if rows else 0

    # ---- 审核标记 (误报/确认) ----

    def mark_review(self, alert_id: int, review_status: str, note: str = "") -> bool:
        """
        标记预警审核状态。

        review_status:
            'confirmed'   — 确认为真实落石
            'false_alarm' — 确认为误报
            ''            — 清除审核标记

        返回 True 表示更新成功。
        """
        if self._backend == "mysql":
            conn = None
            try:
                conn = self._mysql_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE alerts SET review_status=%s, reviewer_note=%s WHERE id=%s",
                        (review_status, note, alert_id),
                    )
                conn.commit()
                return True
            except Exception:
                return False
            finally:
                if conn is not None:
                    conn.close()  # 归还到池
        else:
            with self._lock:
                with self._get_sqlite_conn() as conn:
                    conn.execute(
                        "UPDATE alerts SET review_status=?, reviewer_note=? WHERE id=?",
                        (review_status, note, alert_id),
                    )
                    conn.commit()
            return True

    # ---- 统计数据 ----

    def get_daily_trends(self, days: int = 7) -> list[dict]:
        """
        最近 N 天的每日预警趋势 (按等级分组)。

        返回: [{"date": "2026-06-12", "red": 3, "orange": 5, "yellow": 8, "blue": 12, "total": 28}, ...]
        """
        from datetime import datetime as dt, timedelta
        result = []
        today = dt.now().date()
        for i in range(days - 1, -1, -1):
            d = today - timedelta(days=i)
            ds = d.strftime("%Y-%m-%d")
            entry = {"date": ds, "red": 0, "orange": 0, "yellow": 0, "blue": 0}
            for lv in ["red", "orange", "yellow", "blue"]:
                entry[lv] = self.count_alerts(start_date=ds, end_date=ds, alert_level=lv)
            entry["total"] = sum(entry[lv] for lv in ["red", "orange", "yellow", "blue"])
            result.append(entry)
        return result

    def get_false_alarm_stats(self, days: int = 30) -> dict:
        """
        统计最近 N 天的误报率。

        返回: {
            "total_reviewed": 100,        # 已审核总数
            "confirmed": 85,              # 确认真实
            "false_alarm": 15,            # 确认误报
            "false_alarm_rate": 0.15,     # 误报率
            "pending_review": 200,        # 待审核
        }
        """
        from datetime import datetime as dt, timedelta
        start = (dt.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT review_status, COUNT(*) as cnt FROM alerts "
                "WHERE time >= %s AND review_status != '' "
                "GROUP BY review_status",
                (start + " 00:00:00",),
            )
        else:
            rows = self._sqlite_query(
                "SELECT review_status, COUNT(*) as cnt FROM alerts "
                "WHERE time >= ? AND review_status != '' "
                "GROUP BY review_status",
                (start + " 00:00:00",),
            )

        confirmed = 0
        false_alarm = 0
        for r in rows:
            if r.get("review_status") == "confirmed":
                confirmed = r.get("cnt", 0)
            elif r.get("review_status") == "false_alarm":
                false_alarm = r.get("cnt", 0)

        total_reviewed = confirmed + false_alarm
        false_alarm_rate = round(false_alarm / total_reviewed, 4) if total_reviewed > 0 else 0

        # 待审核数
        if self._backend == "mysql":
            pending_rows = self._mysql_query(
                "SELECT COUNT(*) as cnt FROM alerts "
                "WHERE time >= %s AND (review_status = '' OR review_status IS NULL)",
                (start + " 00:00:00",),
            )
        else:
            pending_rows = self._sqlite_query(
                "SELECT COUNT(*) as cnt FROM alerts "
                "WHERE time >= ? AND (review_status = '' OR review_status IS NULL)",
                (start + " 00:00:00",),
            )
        pending = pending_rows[0].get("cnt", 0) if pending_rows else 0

        return {
            "total_reviewed": total_reviewed,
            "confirmed": confirmed,
            "false_alarm": false_alarm,
            "false_alarm_rate": false_alarm_rate,
            "pending_review": pending,
        }

    # ---- 哈希链验证 (Hash Chain Verification) ----

    def _get_latest_hash_internal(self) -> str:
        """内部方法: 获取最新一条记录的 data_hash (用于链式哈希计算)。"""
        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT data_hash FROM alerts WHERE data_hash != '' "
                "ORDER BY id DESC LIMIT 1", (),
            )
        else:
            rows = self._sqlite_query(
                "SELECT data_hash FROM alerts WHERE data_hash != '' "
                "ORDER BY id DESC LIMIT 1", (),
            )
        return rows[0].get("data_hash", "") if rows else ""

    def get_latest_hash(self) -> str | None:
        """获取最新记录的 data_hash，供外部健康检查使用。

        返回 None 表示没有已哈希的记录。
        """
        h = self._get_latest_hash_internal()
        return h if h else None

    def _get_record_by_id(self, alert_id: int) -> dict | None:
        """按 ID 获取单条记录。"""
        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT * FROM alerts WHERE id=%s", (alert_id,),
            )
        else:
            rows = self._sqlite_query(
                "SELECT * FROM alerts WHERE id=?", (alert_id,),
            )
        return rows[0] if rows else None

    def verify_alert(self, alert_id: int) -> dict:
        """验证单条预警记录的哈希完整性。

        返回:
            {"valid": bool, "stored_hash": str, "computed_hash": str,
             "prev_hash_match": bool, "msg": str}
        """
        record = self._get_record_by_id(alert_id)
        if record is None:
            return {"valid": False, "msg": f"记录 #{alert_id} 不存在"}

        from .hash_chain import verify_record

        # 获取前一条记录
        prev_record = None
        if alert_id > 1:
            prev_record = self._get_record_by_id(alert_id - 1)

        result = verify_record(record, prev_record, ALERT_HASH_GENESIS)

        if result.get("reason"):
            result["msg"] = result.pop("reason")
        else:
            result["msg"] = "验证通过" if result["valid"] else "验证失败"

        return result

    def verify_chain(self, start_id: int, end_id: int,
                     max_records: int | None = None) -> dict:
        """批量验证一个 ID 区间的哈希链完整性。

        参数:
            start_id:    起始 ID (含)
            end_id:      结束 ID (含)
            max_records: 单次最大验证条数 (默认 ALERT_HASH_VERIFY_BATCH_SIZE)

        返回:
            {"total_checked": int, "valid": int, "invalid": int,
             "skipped": int, "breaks": [...], "truncated": bool}
        """
        from .hash_chain import verify_chain_batch
        from .config import ALERT_HASH_VERIFY_BATCH_SIZE

        limit = max_records or ALERT_HASH_VERIFY_BATCH_SIZE
        truncated = False

        total_range = end_id - start_id + 1
        if total_range > limit:
            end_id = start_id + limit - 1
            truncated = True

        if self._backend == "mysql":
            rows = self._mysql_query(
                "SELECT * FROM alerts WHERE id >= %s AND id <= %s "
                "ORDER BY id ASC LIMIT %s",
                (start_id, end_id, limit),
            )
        else:
            rows = self._sqlite_query(
                "SELECT * FROM alerts WHERE id >= ? AND id <= ? "
                "ORDER BY id ASC LIMIT ?",
                (start_id, end_id, limit),
            )

        result = verify_chain_batch(rows, ALERT_HASH_GENESIS)
        result["truncated"] = truncated
        return result

    # ---- 预警工单流转 (闭环管理) ----

    WORKFLOW_STATES = {
        'pending':     '待审核',
        'confirmed':   '已确认·待派单',
        'false_alarm': '误报·已关闭',
        'dispatched':  '已派单·处置中',
        'arrived':     '现场已到场',
        'handled':     '已处置·待归档',
        'archived':    '已归档',
    }

    WORKFLOW_TRANSITIONS = {
        'pending':     ['confirmed', 'false_alarm'],
        'confirmed':   ['dispatched'],
        'dispatched':  ['arrived'],
        'arrived':     ['handled'],
        'handled':     ['archived'],
        'archived':    [],
        'false_alarm': [],
    }

    def transition_workflow(self, alert_id: int, new_state: str,
                            operator: str = '', note: str = '') -> dict:
        """
        执行工单状态流转。返回 {"ok": True/False, "msg": ...}。

        流转规则:
          pending -> confirmed/false_alarm
          confirmed -> dispatched
          dispatched -> arrived
          arrived -> handled
          handled -> archived
        """
        if new_state not in self.WORKFLOW_STATES:
            return {'ok': False, 'msg': f'未知状态: {new_state}'}

        # 读取当前状态
        current = self._get_workflow_state(alert_id)
        if current is None:
            return {'ok': False, 'msg': f'预警 #{alert_id} 不存在'}

        allowed = self.WORKFLOW_TRANSITIONS.get(current, [])
        if new_state not in allowed and current != '':
            return {'ok': False, 'msg': f'不允许从 {current} 转换到 {new_state}，允许: {allowed}'}

        # 更新状态
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        history_entry = {
            'from': current,
            'to': new_state,
            'time': now,
            'operator': operator,
            'note': note,
        }

        if self._backend == 'mysql':
            ok = self._mysql_update_workflow(alert_id, new_state, history_entry, operator)
        else:
            ok = self._sqlite_update_workflow(alert_id, new_state, history_entry, operator)

        if ok:
            label = self.WORKFLOW_STATES.get(new_state, new_state)
            return {'ok': True, 'msg': f'预警 #{alert_id} 状态已更新为: {label}',
                    'state': new_state, 'label': label}
        return {'ok': False, 'msg': '数据库更新失败'}

    def _get_workflow_state(self, alert_id: int) -> str | None:
        if self._backend == 'mysql':
            rows = self._mysql_query(
                'SELECT workflow_state FROM alerts WHERE id=%s', (alert_id,))
        else:
            rows = self._sqlite_query(
                'SELECT workflow_state FROM alerts WHERE id=?', (alert_id,))
        return rows[0].get('workflow_state', '') if rows else None

    def _mysql_update_workflow(self, alert_id, state, entry, operator):
        conn = None
        cur = None
        try:
            conn = self._mysql_conn()
            cur = conn.cursor()
            cur.execute(
                'SELECT workflow_history FROM alerts WHERE id=%s', (alert_id,))
            row = cur.fetchone()
            history = json.loads(row[0]) if row and row[0] else []
            history.append(entry)
            cur.execute(
                'UPDATE alerts SET workflow_state=%s, workflow_history=%s, operator=%s WHERE id=%s',
                (state, json.dumps(history, ensure_ascii=False), operator, alert_id))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()  # 归还到池

    def _sqlite_update_workflow(self, alert_id, state, entry, operator):
        with self._lock:
            with self._get_sqlite_conn() as conn:
                cur = conn.execute(
                    'SELECT workflow_history FROM alerts WHERE id=?', (alert_id,))
                row = cur.fetchone()
                history = json.loads(row[0]) if row and row[0] else []
                history.append(entry)
                conn.execute(
                    'UPDATE alerts SET workflow_state=?, workflow_history=?, operator=? WHERE id=?',
                    (state, json.dumps(history, ensure_ascii=False), operator, alert_id))
                conn.commit()
                return True

    def get_workflow_history(self, alert_id: int) -> list[dict]:
        """获取工单流转历史"""
        if self._backend == 'mysql':
            rows = self._mysql_query(
                'SELECT workflow_state, workflow_history, operator FROM alerts WHERE id=%s',
                (alert_id,))
        else:
            rows = self._sqlite_query(
                'SELECT workflow_state, workflow_history, operator FROM alerts WHERE id=?',
                (alert_id,))
        if not rows:
            return []
        r = rows[0]
        history = json.loads(r.get('workflow_history', '[]')) if isinstance(r.get('workflow_history'), str) else (r.get('workflow_history') or [])
        return history

    def count_by_workflow_state(self) -> dict:
        """按工单状态统计"""
        if self._backend == 'mysql':
            rows = self._mysql_query(
                'SELECT workflow_state, COUNT(*) as cnt FROM alerts GROUP BY workflow_state', ())
        else:
            rows = self._sqlite_query(
                'SELECT workflow_state, COUNT(*) as cnt FROM alerts GROUP BY workflow_state', ())
        return {r['workflow_state']: r['cnt'] for r in rows if r['workflow_state']}

    # ---- 数据归档与清理 (Data Retention) ----

    def archive_and_purge(
        self, retention_days: int | None = None,
        dry_run: bool = False,
        batch_size: int | None = None,
    ) -> dict:
        """
        事务性归档: 将超过保留期的预警记录导出 → 上传冷存储 → 从 DB 删除。

        流程:
          1. 查询超过 retention_days 的预警记录 (分批)
          2. 导出为 JSON Lines → data/archive/ 临时目录
          3. 上传到冷存储 (如已配置)
          4. 更新 .archive_progress.json
          5. 从 DB 删除已归档记录
          6. 清理临时文件

        跳过无数据的日期范围 (零记录不生成空文件)。

        返回:
            {archived_count, exported_files, uploaded_keys, errors, dry_run}
        """
        from .config import (
            ALERT_RETENTION_DAYS, ARCHIVE_BATCH_SIZE, DATA_DIR,
        )
        from .cold_storage import ColdStorageClient

        if retention_days is None:
            retention_days = ALERT_RETENTION_DAYS
        if batch_size is None:
            batch_size = ARCHIVE_BATCH_SIZE

        cutoff_date = (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            - timedelta(days=retention_days)
        ).strftime("%Y-%m-%d")

        # 查询过期记录数
        total = self.count_alerts(end_date=cutoff_date)
        if total == 0:
            return {
                "archived_count": 0,
                "exported_files": [],
                "uploaded_keys": [],
                "errors": [],
                "dry_run": dry_run,
                "msg": f"无超过 {retention_days} 天的预警记录 (截止 {cutoff_date})",
            }

        # 分批导出
        archive_dir = DATA_DIR / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        # 确定日期范围用于文件名
        oldest = self.query_alerts(
            end_date=cutoff_date, limit=1, offset=total - 1,
        )
        start_date = oldest[0]["time"][:10] if oldest else cutoff_date

        filename = f"alerts_{start_date}_to_{cutoff_date}.jsonl"
        export_path = archive_dir / filename

        exported_files = []
        uploaded_keys = []
        errors = []
        archived_ids = []

        # 分批查询 + 导出
        offset = 0
        with open(export_path, "w", encoding="utf-8") as f:
            while offset < total:
                batch = self.query_alerts(
                    end_date=cutoff_date,
                    limit=batch_size,
                    offset=offset,
                )
                if not batch:
                    break
                for record in batch:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    archived_ids.append(record["id"])
                offset += len(batch)

        exported_files.append(str(export_path))

        if dry_run:
            # Dry run: 不删除, 不实际上传
            export_path.unlink(missing_ok=True)
            return {
                "archived_count": len(archived_ids),
                "exported_files": exported_files,
                "uploaded_keys": [],
                "errors": [],
                "dry_run": True,
                "msg": f"Dry run: 将归档 {len(archived_ids)} 条记录 ({start_date} ~ {cutoff_date})",
            }

        # 上传到冷存储
        client = ColdStorageClient()
        if client.enabled:
            with open(export_path, "r", encoding="utf-8") as f:
                data_for_upload = [
                    json.loads(line) for line in f if line.strip()
                ]
            ok = client.upload_json(filename, data_for_upload)
            if ok:
                uploaded_keys.append(filename)
            else:
                errors.append(f"冷存储上传失败: {filename}")
                # 上传失败时保留本地文件和 DB 记录 (at-least-once)
                self._save_archive_progress(
                    pending_upload_keys=[filename],
                    last_archive_time=datetime.now().isoformat(),
                    status="upload_failed",
                )
                return {
                    "archived_count": 0,
                    "exported_files": exported_files,
                    "uploaded_keys": [],
                    "errors": errors,
                    "dry_run": False,
                    "msg": "冷存储上传失败, 记录未删除。稍后重试。",
                }

        # 上传成功 → 删除 DB 记录
        deleted = self.delete_archived(archived_ids)

        # 清理临时文件
        export_path.unlink(missing_ok=True)

        # 更新归档进度
        self._save_archive_progress(
            last_archive_time=datetime.now().isoformat(),
            last_archived_alert_id=max(archived_ids) if archived_ids else 0,
            pending_upload_keys=[],
            status="idle",
        )

        return {
            "archived_count": deleted,
            "exported_files": exported_files,
            "uploaded_keys": uploaded_keys,
            "errors": errors,
            "dry_run": False,
            "date_range": f"{start_date} ~ {cutoff_date}",
        }

    def delete_archived(self, alert_ids: list[int]) -> int:
        """批量删除已归档的预警记录。返回实际删除行数。"""
        if not alert_ids:
            return 0

        deleted = 0
        if self._backend == "mysql":
            conn = None
            try:
                conn = self._mysql_conn()
                with conn.cursor() as cur:
                    # 分批删除 (MySQL IN 子句限制)
                    for i in range(0, len(alert_ids), 500):
                        batch = alert_ids[i:i + 500]
                        placeholders = ",".join(["%s"] * len(batch))
                        cur.execute(
                            f"DELETE FROM alerts WHERE id IN ({placeholders})",
                            batch,
                        )
                        deleted += cur.rowcount
                conn.commit()
            except Exception as e:
                log_event("system", level="ERROR",
                          msg=f"归档删除失败 (MySQL): {e}")
            finally:
                if conn is not None:
                    conn.close()  # 归还到池
        else:
            try:
                with self._lock:
                    with self._get_sqlite_conn() as conn:
                        for i in range(0, len(alert_ids), 500):
                            batch = alert_ids[i:i + 500]
                            placeholders = ",".join(["?"] * len(batch))
                            cur = conn.execute(
                                f"DELETE FROM alerts WHERE id IN ({placeholders})",
                                batch,
                            )
                            deleted += cur.rowcount
                        conn.commit()
            except Exception as e:
                log_event("system", level="ERROR",
                          msg=f"归档删除失败 (SQLite): {e}")

        log_event("system", level="INFO",
                  msg=f"归档清理完成: 删除 {deleted} 条旧预警记录")
        return deleted

    @staticmethod
    def _save_archive_progress(
        last_archive_time: str = "",
        last_archived_alert_id: int = 0,
        pending_upload_keys: list[str] | None = None,
        status: str = "idle",
    ):
        """保存归档进度到 data/.archive_progress.json (断点续传)。"""
        from .config import DATA_DIR

        progress_path = DATA_DIR / ".archive_progress.json"
        progress = {
            "last_archive_time": last_archive_time,
            "last_archived_alert_id": last_archived_alert_id,
            "pending_upload_keys": pending_upload_keys or [],
            "status": status,
        }
        try:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---- 辅助 ----

    # 列顺序必须与 CREATE TABLE + 迁移后的物理列序一致 (SELECT * 返回此顺序)
    # 注意: 对于通过 ALTER TABLE 增量迁移的旧数据库, cursor.description 会给出正确的运行时顺序
    _COLS = ["id", "time", "alert_level", "count", "max_confidence",
             "track_ids", "class_summary", "saved_frame", "clip_path",
             "push_status", "push_msg", "rock_diameter_cm", "monitoring_location",
             "review_status", "reviewer_note",
             "workflow_state", "workflow_history", "operator",
             "request_id", "session_id", "data_hash", "prev_hash",
             "created_at"]

    @staticmethod
    def _row_to_dict(row: tuple, cursor=None) -> dict:
        """将查询行转为 dict。优先使用 cursor.description 获取列名（兼容非 SELECT * 查询）"""
        if cursor is not None and cursor.description is not None:
            cols = [d[0] for d in cursor.description]
            if len(cols) == len(row):
                return dict(zip(cols, row))
        return dict(zip(AlertStore._COLS, row))


# 模块级单例
_store: AlertStore | None = None
_store_lock = threading.Lock()


def get_alert_store() -> AlertStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = AlertStore()
                _store.start_retry_loop()
    return _store