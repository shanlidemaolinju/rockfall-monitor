"""
审计日志模块 — 完整操作审计跟踪
==============================
记录: 谁 + 何时 + 做了什么 + 变更前后值 + 结果 + IP + User-Agent

用法:
    from rockfall.audit import AuditLogger, audit_log
    audit = AuditLogger()
    audit.log("user_login", operator="admin", detail="登录成功", ip="192.168.1.1")
    audit.log("config_update", operator="zhangsan",
              detail="修改检测阈值", before={"confidence": 0.3}, after={"confidence": 0.5})

变更追踪:
    before/after 参数自动序列化为 JSON，便于审计回溯变更前后的具体值。
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DATA_DIR


class AuditLogger:
    """操作审计日志 — SQLite 持久化, 线程安全。

    扩展字段:
      - before/after: 变更前后的值 (JSON)，用于敏感操作的可回溯审计
      - user_agent:   客户端标识
      - request_id:   关联 trace.py 中的 X-Request-ID
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = ""):
        self._db_path = Path(db_path) if db_path else DATA_DIR / "audit.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""\
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    operator TEXT DEFAULT '',
                    detail TEXT DEFAULT '',
                    alert_id INTEGER DEFAULT 0,
                    ip TEXT DEFAULT '',
                    result TEXT DEFAULT '',
                    before_value TEXT DEFAULT '',
                    after_value TEXT DEFAULT '',
                    user_agent TEXT DEFAULT '',
                    request_id TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )""")
            conn.execute("""\
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)""")
            conn.execute("""\
                CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(created_at)""")
            conn.execute("""\
                CREATE INDEX IF NOT EXISTS idx_audit_operator ON audit_log(operator)""")
            conn.execute("""\
                CREATE INDEX IF NOT EXISTS idx_audit_alert ON audit_log(alert_id)""")

            # 迁移旧表：添加新列（如果不存在）
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN before_value TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN after_value TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN user_agent TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN request_id TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass

    def log(self, action: str, operator: str = "", detail: str = "",
            alert_id: int = 0, ip: str = "", result: str = "ok",
            before: Any = None, after: Any = None,
            user_agent: str = "", request_id: str = ""):
        """记录一条审计日志。

        参数:
          action:     操作类型 (如 config_update, site_switch, model_switch)
          operator:   操作人标识 (来自 API Key client_id 或 JWT claims)
          detail:     人类可读的操作描述
          alert_id:   关联的预警 ID（如有）
          ip:         操作来源 IP
          result:     操作结果 ("ok" / "error: <reason>")
          before:     变更前的值（任意可 JSON 序列化的对象）
          after:      变更后的值（任意可 JSON 序列化的对象）
          user_agent: 客户端 User-Agent
          request_id: 关联的 X-Request-ID (来自 trace.py)
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        before_json = json.dumps(before, ensure_ascii=False, default=str) if before is not None else ""
        after_json = json.dumps(after, ensure_ascii=False, default=str) if after is not None else ""

        # 尝试获取当前 request_id
        if not request_id:
            try:
                from .trace import get_request_id
                request_id = get_request_id() or ""
            except Exception:
                pass

        with self._lock:
            try:
                with sqlite3.connect(str(self._db_path)) as conn:
                    conn.execute(
                        "INSERT INTO audit_log "
                        "(action, operator, detail, alert_id, ip, result, "
                        " before_value, after_value, user_agent, request_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (action, operator, detail, alert_id, ip, result,
                         before_json, after_json, user_agent, request_id, now),
                    )
            except Exception:
                pass  # 审计失败不应阻塞主流程

    def query(self, action: str = "", operator: str = "", alert_id: int = 0,
              start: str = "", end: str = "", limit: int = 100, offset: int = 0) -> list[dict]:
        """查询审计日志（含 before/after 变更详情）"""
        conditions = []
        params = []
        if action:
            conditions.append("action=?")
            params.append(action)
        if operator:
            conditions.append("operator=?")
            params.append(operator)
        if alert_id:
            conditions.append("alert_id=?")
            params.append(alert_id)
        if start:
            conditions.append("created_at>=?")
            params.append(start)
        if end:
            conditions.append("created_at<=?")
            params.append(end + " 23:59:59")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            # 尝试将 before_value/after_value 反序列化为可读的 dict
            for field in ("before_value", "after_value"):
                raw = d.get(field, "")
                if raw and isinstance(raw, str):
                    try:
                        d[field] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        return results

    def count(self, action: str = "", operator: str = "", start: str = "",
              end: str = "") -> int:
        """统计审计日志条数"""
        conditions = []
        params = []
        if action:
            conditions.append("action=?")
            params.append(action)
        if operator:
            conditions.append("operator=?")
            params.append(operator)
        if start:
            conditions.append("created_at>=?")
            params.append(start)
        if end:
            conditions.append("created_at<=?")
            params.append(end + " 23:59:59")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM audit_log {where}", params
            ).fetchone()
        return row[0] if row else 0

    def get_actions_summary(self) -> list[dict]:
        """按操作类型汇总"""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT action, COUNT(*) as cnt FROM audit_log GROUP BY action ORDER BY cnt DESC"
            ).fetchall()
        return [dict(r) for r in rows]


# 模块级单例
_audit: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit


def audit_log(action: str, operator: str = "", detail: str = "",
              alert_id: int = 0, ip: str = "", result: str = "ok",
              before: Any = None, after: Any = None,
              user_agent: str = "", request_id: str = ""):
    """便捷函数: 写审计日志（含变更前后值）。

    示例:
        audit_log("config_update", operator="admin",
                  detail="修改检测置信度",
                  before={"detection_confidence": 0.3},
                  after={"detection_confidence": 0.5})
    """
    get_audit_logger().log(action, operator, detail, alert_id, ip, result,
                           before, after, user_agent, request_id)
