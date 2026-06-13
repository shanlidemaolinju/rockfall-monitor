"""
数据库工具 — MySQL 可用性检测（消除跨模块重复）
================================================
alert_store 和 site_config 都需要检测 MySQL 是否可用,
之前各自独立实现了相同的 try/except 块。

使用:
    from rockfall.db_utils import is_mysql_available, get_pymysql
    if is_mysql_available():
        pymysql = get_pymysql()
        conn = pymysql.connect(...)
"""

_pymysql_checked: bool = False
_pymysql_available: bool = False
_pymysql_module = None


def is_mysql_available() -> bool:
    """检测 pymysql 是否已安装（模块级缓存，仅检测一次）。"""
    global _pymysql_checked, _pymysql_available, _pymysql_module
    if _pymysql_checked:
        return _pymysql_available

    try:
        import pymysql as _pm
        _pymysql_module = _pm
        _pymysql_available = True
    except ImportError:
        _pymysql_available = False
    _pymysql_checked = True
    return _pymysql_available


def get_pymysql():
    """返回 pymysql 模块引用（仅在 is_mysql_available() 返回 True 后调用）。"""
    return _pymysql_module
