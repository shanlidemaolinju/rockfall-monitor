"""新增监测点位表 — 支持多站点数据库管理

Revision ID: 002
Revises: 001
Create Date: 2026-06-13
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _detect_backend() -> str:
    url = op.get_bind().engine.url.drivername
    if "mysql" in url:
        return "mysql"
    return "sqlite"


def upgrade() -> None:
    backend = _detect_backend()

    if backend == "mysql":
        op.execute("""
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
                created_at VARCHAR(19) NOT NULL,
                updated_at VARCHAR(19) NOT NULL,
                INDEX idx_sites_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    else:
        op.execute("""
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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_sites_active "
            "ON monitoring_sites(is_active)"
        )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS monitoring_sites")
