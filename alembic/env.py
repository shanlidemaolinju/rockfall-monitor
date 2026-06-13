"""
Alembic 迁移环境配置 — 支持 MySQL / SQLite 双后端
==================================================
使用 SQLAlchemy 仅作为连接层（非 ORM），迁移脚本执行原始 SQL。

数据库 URL 来源（优先级）:
  1. 环境变量 DATABASE_URL（如 mysql+pymysql://user:pass@host/db）
  2. MYSQL_* 环境变量自动拼接
  3. 默认 SQLite: sqlite:///data/alerts.db
"""

import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _build_db_url() -> str:
    """从环境变量构建数据库连接 URL。"""
    url = os.getenv("DATABASE_URL", "")
    if url:
        return url

    mysql_host = os.getenv("MYSQL_HOST", "")
    if mysql_host:
        mysql_port = os.getenv("MYSQL_PORT", "3306")
        mysql_user = os.getenv("MYSQL_USER", "")
        mysql_password = os.getenv("MYSQL_PASSWORD", "")
        mysql_database = os.getenv("MYSQL_DATABASE", "rock")
        return (
            f"mysql+pymysql://{mysql_user}:{mysql_password}"
            f"@{mysql_host}:{mysql_port}/{mysql_database}"
        )

    # 默认 SQLite
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return f"sqlite:///{os.path.join(data_dir, 'alerts.db')}"


def run_migrations_offline() -> None:
    url = _build_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _build_db_url()
    connectable = create_engine(url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
