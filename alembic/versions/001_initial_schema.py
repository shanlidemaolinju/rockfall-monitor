"""初始数据库 Schema — 落石预警系统

Revision ID: 001
Revises: None
Create Date: 2026-06-13
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _detect_backend() -> str:
    """检测后端类型: 'mysql' 或 'sqlite'."""
    url = op.get_bind().engine.url.drivername
    if "mysql" in url:
        return "mysql"
    return "sqlite"


def upgrade() -> None:
    backend = _detect_backend()

    if backend == "mysql":
        _upgrade_mysql()
    else:
        _upgrade_sqlite()


def _upgrade_mysql() -> None:
    # 新建表
    op.execute("""
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
            workflow_state VARCHAR(20) DEFAULT 'pending',
            workflow_history JSON DEFAULT ('[]'),
            operator VARCHAR(50) DEFAULT '',
            request_id VARCHAR(32) DEFAULT '',
            session_id VARCHAR(32) DEFAULT '',
            created_at VARCHAR(19) NOT NULL,
            INDEX idx_workflow_state (workflow_state),
            INDEX idx_push_status (push_status),
            INDEX idx_time (time),
            INDEX idx_alert_level (alert_level),
            INDEX idx_request_id (request_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    # 增量迁移：为旧表补充新列
    for sql in [
        "ALTER TABLE alerts ADD COLUMN request_id VARCHAR(32) DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN session_id VARCHAR(32) DEFAULT ''",
    ]:
        try:
            op.execute(sql)
        except Exception:
            pass  # 列已存在


def _upgrade_sqlite() -> None:
    # 新建表
    op.execute("""
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
            workflow_state TEXT DEFAULT 'pending',
            workflow_history TEXT DEFAULT '[]',
            operator TEXT DEFAULT '',
            request_id TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    # 增量迁移：为旧表补充新列（SQLite 不支持 IF NOT EXISTS，使用 try/except）
    for sql in [
        "ALTER TABLE alerts ADD COLUMN request_id TEXT DEFAULT ''",
        "ALTER TABLE alerts ADD COLUMN session_id TEXT DEFAULT ''",
    ]:
        try:
            op.execute(sql)
        except Exception:
            pass  # 列已存在


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS alerts")
