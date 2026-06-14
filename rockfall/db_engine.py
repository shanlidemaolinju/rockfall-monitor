"""
数据库连接池引擎 — SQLAlchemy 统一连接池
========================================
替代 alert_store / site_config 中分散的 pymysql.connect(),
提供连接复用、自动重连、池状态监控。

使用:
    from rockfall.db_engine import get_mysql_engine, get_pool_status

    # 获取连接 (调用者负责 .close() 归还到池)
    conn = get_mysql_engine().raw_connection()
    cur = conn.cursor()
    cur.execute("SELECT ...")
    conn.commit()
    conn.close()  # 归还到池, 不关闭物理连接

    # 或使用上下文管理器 (推荐)
    from rockfall.db_engine import mysql_connection
    with mysql_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT ...")
        conn.commit()

    # 查询池状态
    status = get_pool_status()
    # {"pool_size": 5, "checkedin": 3, "checkedout": 2, "overflow": 0}
"""

import logging
import threading
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# 模块级引擎单例 + 锁
_engine: Engine | None = None
_engine_lock = threading.Lock()

# 用于追踪当前后端配置 (检测配置变更时重建引擎)
_engine_config_hash: str = ""


def _build_mysql_url() -> str:
    """从环境变量构建 SQLAlchemy MySQL URL。"""
    from .config import (
        MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
    )
    # 对密码中的特殊字符进行 URL 编码
    password_encoded = quote_plus(MYSQL_PASSWORD) if MYSQL_PASSWORD else ""
    user_encoded = quote_plus(MYSQL_USER) if MYSQL_USER else ""
    return (
        f"mysql+pymysql://{user_encoded}:{password_encoded}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        f"?charset=utf8mb4"
    )


def _create_engine() -> Engine:
    """创建 SQLAlchemy engine (configured pooling)。"""
    from .config import (
        MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
        DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_PRE_PING, DB_POOL_RECYCLE,
        DB_CONNECT_TIMEOUT, DB_READ_TIMEOUT, DB_WRITE_TIMEOUT,
    )
    url = _build_mysql_url()
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_pre_ping=DB_POOL_PRE_PING,
        pool_recycle=DB_POOL_RECYCLE,
        # echo=False,  # 生产环境不打印 SQL
        connect_args={
            "connect_timeout": DB_CONNECT_TIMEOUT,
            "read_timeout": DB_READ_TIMEOUT,
            "write_timeout": DB_WRITE_TIMEOUT,
        },
    )


def _config_hash() -> str:
    """生成当前 MySQL 配置的哈希 (用于检测配置变更)。"""
    from .config import (
        MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE,
        DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_PRE_PING, DB_POOL_RECYCLE,
    )
    return (
        f"{MYSQL_HOST}:{MYSQL_PORT}:{MYSQL_USER}:{MYSQL_DATABASE}"
        f":{DB_POOL_SIZE}:{DB_MAX_OVERFLOW}:{DB_POOL_PRE_PING}:{DB_POOL_RECYCLE}"
    )


def get_mysql_engine() -> Engine | None:
    """
    获取 MySQL 引擎单例 (线程安全)。

    若 MySQL 未配置或 pymysql 不可用, 返回 None。
    配置变更时自动重建引擎。
    """
    global _engine, _engine_config_hash

    from .config import MYSQL_HOST
    from .db_utils import is_mysql_available

    if not MYSQL_HOST or not is_mysql_available():
        return None

    current_hash = _config_hash()
    if _engine is not None and _engine_config_hash == current_hash:
        return _engine

    with _engine_lock:
        # 双重检查
        current_hash = _config_hash()
        if _engine is not None and _engine_config_hash == current_hash:
            return _engine

        # 释放旧引擎 (如果有)
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass

        try:
            _engine = _create_engine()
            _engine_config_hash = current_hash
            logger.info(
                "MySQL 连接池已初始化 pool_size=%s max_overflow=%s",
                _engine.pool.size(), _engine.pool.overflow()
            )
        except Exception as e:
            logger.error("MySQL 引擎创建失败: %s", e)
            _engine = None

        return _engine


def get_pool_status() -> dict | None:
    """
    获取当前连接池状态, 用于 Prometheus 指标和健康检查。

    返回:
        None — MySQL 未启用
        {
            "pool_size": 5,
            "checkedin": 3,    # 空闲连接数
            "checkedout": 2,   # 使用中连接数
            "overflow": 0,     # 当前溢出连接数
            "max_total": 15,   # 配置的最大总连接数 = pool_size + max_overflow
        }
    """
    from .config import DB_POOL_SIZE, DB_MAX_OVERFLOW

    engine = get_mysql_engine()
    if engine is None:
        return None

    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checkedin": pool.checkedin(),
        "checkedout": pool.checkedout(),
        "overflow": pool.overflow(),
        "max_total": DB_POOL_SIZE + DB_MAX_OVERFLOW,
    }


class _ConnectionProxy:
    """
    MySQL 连接上下文管理器 — 自动归还连接到池。

    用法:
        with mysql_connection() as conn:
            cur = conn.cursor()
            cur.execute(...)
            conn.commit()
    """

    def __init__(self):
        self._conn = None

    def __enter__(self):
        engine = get_mysql_engine()
        if engine is None:
            raise RuntimeError("MySQL 未配置或不可用")
        self._conn = engine.raw_connection()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            # raw_connection().close() 自动回滚未提交事务并归还到池
            self._conn.close()
        return False  # 不吞异常


def mysql_connection():
    """返回一个上下文管理器, 用于安全获取/归还池连接。"""
    return _ConnectionProxy()


def dispose_engine():
    """释放引擎 (测试/关闭时使用)。"""
    global _engine, _engine_config_hash
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:
                pass
            _engine = None
            _engine_config_hash = ""
