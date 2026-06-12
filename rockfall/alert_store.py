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
from datetime import datetime
from pathlib import Path

from .config import (
    DATA_DIR, PUSHPLUS_TOKEN, PUSHPLUS_URL, PUSHPLUS_TOPIC,
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
)

_MYSQL_AVAILABLE = False
try:
    import pymysql
    _MYSQL_AVAILABLE = True
except ImportError:
    pass

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
    push_status VARCHAR(20) DEFAULT 'pending',
    push_msg VARCHAR(500) DEFAULT '',
    rock_diameter_cm DOUBLE DEFAULT 0,
    monitoring_location VARCHAR(100) DEFAULT '',
    created_at VARCHAR(19) NOT NULL,
    INDEX idx_push_status (push_status),
    INDEX idx_time (time),
    INDEX idx_alert_level (alert_level)
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
    push_status TEXT DEFAULT 'pending',
    push_msg TEXT DEFAULT '',
    rock_diameter_cm REAL DEFAULT 0,
    monitoring_location TEXT DEFAULT '',
    created_at TEXT NOT NULL
)"""

# 增量迁移: 为旧表补充新增列 (兼容已部署系统)
_MIGRATIONS = {
    "sqlite": [
        "ALTER TABLE alerts ADD COLUMN rock_diameter_cm REAL DEFAULT 0",
        "ALTER TABLE alerts ADD COLUMN monitoring_location TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN review_status TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN reviewer_note TEXT DEFAULT ''",
    ],
    "mysql": [
        "ALTER TABLE alerts ADD COLUMN rock_diameter_cm DOUBLE DEFAULT 0",
        "ALTER TABLE alerts ADD COLUMN monitoring_location VARCHAR(100) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN review_status VARCHAR(20) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN reviewer_note VARCHAR(500) DEFAULT ''",
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
                conn = pymysql.connect(
                    host=MYSQL_HOST, port=MYSQL_PORT,
                    user=MYSQL_USER, password=MYSQL_PASSWORD,
                    database=MYSQL_DATABASE,
                    charset='utf8mb4',
                    connect_timeout=3,
                )
                conn.close()
                return "mysql"
            except Exception as e:
                print(f"[AlertStore] MySQL 连接失败 ({e}), 降级为 SQLite")
        return "sqlite"

    # ---- 建表 ----

    def _init_db(self):
        if self._backend == "mysql":
            self._init_mysql_table()
        else:
            self._init_sqlite_table()
        self._run_migrations()

    def _init_mysql_table(self):
        try:
            conn = self._mysql_conn()
            with conn.cursor() as cur:
                cur.execute(_MYSQL_SCHEMA)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[AlertStore] MySQL 建表失败 ({e}), 降级为 SQLite")
            self._backend = "sqlite"
            self._init_sqlite_table()

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
            try:
                if self._backend == "mysql":
                    conn = self._mysql_conn()
                    with conn.cursor() as cur:
                        cur.execute(sql)
                    conn.commit()
                    conn.close()
                else:
                    with self._lock:
                        with self._get_sqlite_conn() as conn:
                            conn.execute(sql)
                            conn.commit()
            except Exception:
                # 列已存在或其他迁移错误, 静默跳过
                pass

    # ---- 连接 ----

    def _mysql_conn(self):
        return pymysql.connect(
            host=MYSQL_HOST, port=MYSQL_PORT,
            user=MYSQL_USER, password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4',
            autocommit=False,
        )

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
                   push_status: str = "pending",
                   rock_diameter_cm: float = 0,
                   monitoring_location: str = "") -> int:
        """保存预警记录, 返回 row ID"""
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        params = (
            ts, alert_level, count, round(max_confidence, 4),
            json.dumps(track_ids or []), class_summary, saved_frame,
            push_status, round(rock_diameter_cm, 1), monitoring_location, ts,
        )
        if self._backend == "mysql":
            return self._mysql_insert(params)
        return self._sqlite_insert(params)

    def _mysql_insert(self, params: tuple) -> int:
        try:
            conn = self._mysql_conn()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO alerts
                   (time, alert_level, count, max_confidence, track_ids,
                    class_summary, saved_frame, push_status,
                    rock_diameter_cm, monitoring_location, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                params,
            )
            conn.commit()
            rid = cur.lastrowid
            cur.close()
            conn.close()
            return rid
        except Exception:
            return -1

    def _sqlite_insert(self, params: tuple) -> int:
        with self._lock:
            with self._get_sqlite_conn() as conn:
                cur = conn.execute(
                    """INSERT INTO alerts
                       (time, alert_level, count, max_confidence, track_ids,
                        class_summary, saved_frame, push_status,
                        rock_diameter_cm, monitoring_location, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            try:
                conn = self._mysql_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE alerts SET push_status=%s, push_msg=%s WHERE id=%s",
                        (status, msg, alert_id),
                    )
                conn.commit()
                conn.close()
            except Exception:
                pass
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
        try:
            conn = self._mysql_conn()
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            return []

    def _sqlite_query(self, sql: str, params: tuple) -> list[dict]:
        with self._lock:
            with self._get_sqlite_conn() as conn:
                rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

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
            try:
                conn = self._mysql_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE alerts SET review_status=%s, reviewer_note=%s WHERE id=%s",
                        (review_status, note, alert_id),
                    )
                conn.commit()
                conn.close()
                return True
            except Exception:
                return False
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

    # ---- 辅助 ----

    # 列顺序必须与 CREATE TABLE / ALTER TABLE 物理列序一致 (SELECT * 返回此顺序)
    _COLS = ["id", "time", "alert_level", "count", "max_confidence",
             "track_ids", "class_summary", "saved_frame", "push_status",
             "push_msg", "created_at", "rock_diameter_cm", "monitoring_location",
             "review_status", "reviewer_note"]

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
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