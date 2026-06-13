"""
数据库迁移运行器 — Alembic in-process 封装
==========================================
在 AlertStore 初始化时自动调用，确保 Schema 最新。
由于服务以 --workers 1 运行，不存在并发迁移风险。

使用方式:
    from rockfall.migration import run_migrations
    run_migrations()  # 自动检测后端并升级到 head
"""

import os
from pathlib import Path
from alembic.config import Config
from alembic import command


def run_migrations(db_url: str = "") -> None:
    """运行所有待处理的数据库迁移到最新版本。

    参数:
        db_url: 可选；不传则从环境变量自动构建
                (MYSQL_* 环境变量 → MySQL，否则 SQLite)
    """
    alembic_dir = Path(__file__).parent.parent / "alembic"
    alembic_ini = Path(__file__).parent.parent / "alembic.ini"

    if not alembic_ini.exists():
        raise FileNotFoundError(f"alembic.ini 不存在: {alembic_ini}")

    cfg = Config(str(alembic_ini))
    # 将工作目录指向项目根目录，确保相对路径正确
    cfg.set_main_option("script_location", str(alembic_dir))

    if db_url:
        cfg.set_main_option("sqlalchemy.url", db_url)

    # 设置环境变量供 env.py 读取
    if db_url:
        os.environ["DATABASE_URL"] = db_url

    command.upgrade(cfg, "head")
