"""点位级检测阈值 — 支持不同监测点位使用独立阈值

Revision ID: 003
Revises: 002
Create Date: 2026-06-19
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NEW_COLUMNS = [
    ("detection_confidence", "DOUBLE DEFAULT 0", "REAL DEFAULT 0"),
    ("alert_blue_low", "DOUBLE DEFAULT 0", "REAL DEFAULT 0"),
    ("alert_blue_high", "DOUBLE DEFAULT 0", "REAL DEFAULT 0"),
    ("alert_yellow_high", "DOUBLE DEFAULT 0", "REAL DEFAULT 0"),
    ("alert_orange_high", "DOUBLE DEFAULT 0", "REAL DEFAULT 0"),
]


def _detect_backend() -> str:
    url = op.get_bind().engine.url.drivername
    if "mysql" in url:
        return "mysql"
    return "sqlite"


def _column_exists(conn, table: str, column: str, backend: str) -> bool:
    """检查列是否已存在，避免重复 ALTER 报错。"""
    if backend == "mysql":
        result = conn.execute(
            sa.text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = :t AND COLUMN_NAME = :c"
            ),
            {"t": table, "c": column},
        )
        return result.scalar() > 0
    else:
        # SQLite: PRAGMA table_info
        result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
        return any(row[1] == column for row in result)


def upgrade() -> None:
    backend = _detect_backend()
    col_idx = 0 if backend == "mysql" else 2

    conn = op.get_bind()
    for name, mysql_def, sqlite_def in NEW_COLUMNS:
        if _column_exists(conn, "monitoring_sites", name, backend):
            continue
        col_def = mysql_def if backend == "mysql" else sqlite_def
        op.execute(f"ALTER TABLE monitoring_sites ADD COLUMN {name} {col_def}")


def downgrade() -> None:
    backend = _detect_backend()
    conn = op.get_bind()
    for name, mysql_def, sqlite_def in NEW_COLUMNS:
        if not _column_exists(conn, "monitoring_sites", name, backend):
            continue
        op.execute(f"ALTER TABLE monitoring_sites DROP COLUMN {name}")
