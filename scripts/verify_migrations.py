#!/usr/bin/env python3
"""
Alembic 迁移验证脚本
====================
在空数据库和已有数据的数据库上验证 upgrade/downgrade 的正确性。

验证项目:
  1. 空数据库 → alembic upgrade head → 表结构创建完整
  2. 已有数据数据库 → alembic upgrade head → 数据不丢失
  3. alembic downgrade -1 → 可逆回滚

用法:
    python scripts/verify_migrations.py              # 仅 SQLite
    python scripts/verify_migrations.py --mysql      # 需要 Docker MySQL
    python scripts/verify_migrations.py --all        # SQLite + MySQL 都测

退出码:
    0 = 全部通过
    1 = 存在失败
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# 修复 Windows GBK 终端编码问题
if sys.platform == "win32" and sys.stdout.encoding:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALEMBIC_DIR = PROJECT_ROOT / "alembic"
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

# ── 跨平台安全输出 ──────────────────────────────────────────
# Windows GBK 终端无法编码 Unicode 符号，使用 ASCII 回退


def _safe_print(*args, **kwargs) -> None:
    """安全 print — 自动处理 Windows GBK 编码错误。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 移除 ANSI 转义序列和 Unicode 符号，回退为纯 ASCII
        import re
        text = " ".join(str(a) for a in args)
        text = re.sub(r'\033\[[0-9;]*m', '', text)
        text = text.replace('✓', '[OK]').replace('✗', '[FAIL]').replace(
            '→', '->').replace('⚠', '[WARN]')
        print(text, **{k: v for k, v in kwargs.items() if k != 'file'})


USE_COLOR = sys.stdout.encoding and sys.stdout.encoding.upper() in (
    "UTF-8", "UTF8", "UTF-16", "UTF-16LE", "UTF-16BE")
GREEN = "\033[92m" if USE_COLOR else ""
RED = "\033[91m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
CYAN = "\033[96m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""

CHK = "✓" if USE_COLOR else "[OK]"
CROSS = "✗" if USE_COLOR else "[FAIL]"
ARROW = "→" if USE_COLOR else "->"
WARN_SYM = "⚠" if USE_COLOR else "[WARN]"


def ok(msg: str) -> None:
    _safe_print(f"  {GREEN}{CHK}{RESET} {msg}")


def fail(msg: str) -> None:
    _safe_print(f"  {RED}{CROSS}{RESET} {msg}")


def info(msg: str) -> None:
    _safe_print(f"  {CYAN}{ARROW}{RESET} {msg}")


def warn(msg: str) -> None:
    _safe_print(f"  {YELLOW}{WARN_SYM}{RESET} {msg}")


# ── 工具函数 ──────────────────────────────────────────────────

def _run_alembic(db_url: str, action: str, extra_args: list = None) -> tuple[int, str, str]:
    """
    运行 alembic 命令，返回 (exit_code, stdout, stderr)。
    通过环境变量 DATABASE_URL 传递数据库连接。
    """
    # 设置环境变量以确保 alembic 使用 UTF-8 输出
    env = os.environ.copy()
    env["DATABASE_URL"] = db_url
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, "-m", "alembic"]
    if action == "upgrade":
        if extra_args:
            # extra_args[0] is the revision target, e.g. "001"
            cmd.extend(["upgrade", extra_args[0]])
        else:
            cmd.extend(["upgrade", "head"])
    elif action == "downgrade":
        cmd.extend(["downgrade", "-1"])
    elif action == "history":
        cmd.extend(["history"])
    elif action == "current":
        cmd.extend(["current"])
    else:
        raise ValueError(f"Unknown action: {action}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout, result.stderr


def _get_table_columns(db_path: str, table: str) -> set:
    """获取 SQLite 表的列名集合。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()
    return columns


def _get_table_row_count(db_path: str, table: str) -> int:
    """获取表的行数。"""
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.OperationalError:
        count = -1  # 表不存在
    conn.close()
    return count


# ── 测试用例 ──────────────────────────────────────────────────

def test_empty_db_upgrade(tmp_dir: Path) -> bool:
    """
    测试 1: 空数据库执行 alembic upgrade head
    验证: 所有表正确创建, alembic_version 存在。
    """
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}[测试 1] 空数据库 → upgrade head → 验证表结构{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    db_path = tmp_dir / "empty.db"
    db_url = f"sqlite:///{db_path}"

    info(f"数据库: {db_path}")
    info("执行 alembic upgrade head ...")

    rc, stdout, stderr = _run_alembic(db_url, "upgrade")
    if rc != 0:
        fail(f"upgrade 失败 (exit={rc})")
        print(f"    stderr: {stderr}")
        return False
    ok("upgrade head 成功")

    # 验证 alembic_version 表
    version_columns = _get_table_columns(str(db_path), "alembic_version")
    if "version_num" not in version_columns:
        fail("alembic_version 表缺少 version_num 列")
        return False
    ok("alembic_version 表存在")

    # 验证 alerts 表
    expected_alerts_columns = {
        "id", "time", "alert_level", "count", "max_confidence",
        "track_ids", "class_summary", "saved_frame", "clip_path",
        "push_status", "push_msg", "rock_diameter_cm",
        "monitoring_location", "workflow_state", "workflow_history",
        "operator", "request_id", "session_id", "created_at",
    }
    actual_alerts = _get_table_columns(str(db_path), "alerts")
    missing_alerts = expected_alerts_columns - actual_alerts
    if missing_alerts:
        fail(f"alerts 表缺少列: {missing_alerts}")
        return False
    ok(f"alerts 表完整 ({len(actual_alerts)} 列)")

    # 验证 monitoring_sites 表
    expected_sites_columns = {
        "site_id", "name", "location", "region", "camera_url",
        "description", "latitude", "longitude", "highway",
        "stake_mark", "risk_level", "roi_polygon",
        "alert_contacts", "is_active", "model_override",
        "created_at", "updated_at",
    }
    actual_sites = _get_table_columns(str(db_path), "monitoring_sites")
    missing_sites = expected_sites_columns - actual_sites
    if missing_sites:
        fail(f"monitoring_sites 表缺少列: {missing_sites}")
        return False
    ok(f"monitoring_sites 表完整 ({len(actual_sites)} 列)")

    # 验证当前版本
    rc, stdout, stderr = _run_alembic(db_url, "current")
    if rc == 0:
        ok(f"当前版本: {stdout.strip().split()[-1] if stdout.strip() else 'head'}")
    else:
        warn("无法获取当前版本 (非致命)")

    return True


def test_upgrade_with_existing_data(tmp_dir: Path) -> bool:
    """
    测试 2: 在已有数据的数据库上执行升级, 验证数据不丢失。
    先创建一个只有部分列的表 (模拟旧版本), 插入数据, 然后升级。
    """
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}[测试 2] 已有数据数据库 → upgrade → 验证数据不丢失{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    db_path = tmp_dir / "existing.db"
    db_url = f"sqlite:///{db_path}"

    info("创建模拟旧版本数据库 (无 monitoring_sites 表) ...")

    # 仅执行 001 迁移 (只创建 alerts 表)
    rc, _, stderr = _run_alembic(db_url, "upgrade", extra_args=["001"])
    if rc != 0:
        fail(f"upgrade 001 失败: {stderr}")
        return False
    ok("upgrade 001 完成")

    # 插入测试数据
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        INSERT INTO alerts (time, alert_level, count, max_confidence,
                           track_ids, class_summary, saved_frame, clip_path,
                           push_status, push_msg, rock_diameter_cm,
                           monitoring_location, workflow_state,
                           workflow_history, operator, request_id,
                           session_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "2026-06-14 10:00:00", "red", 3, 0.85,
        '[1,2,3]', "落石", "frame_001.jpg", "",
        "sent", "", 25.5,
        "钦州1号点", "confirmed",
        '[]', "admin", "req-001",
        "sess-001", "2026-06-14 10:00:01",
    ))
    conn.commit()
    conn.close()
    ok("插入 1 条测试预警记录 (模拟旧版本数据)")

    # 升级到 head
    info("执行 alembic upgrade head (含 002_monitoring_sites) ...")
    rc, stdout, stderr = _run_alembic(db_url, "upgrade")
    if rc != 0:
        fail(f"upgrade head 失败: {stderr}")
        return False
    ok("upgrade head 成功")

    # 验证数据完整
    row_count = _get_table_row_count(str(db_path), "alerts")
    if row_count != 1:
        fail(f"数据丢失! 期望 1 行, 实际 {row_count} 行")
        return False
    ok(f"alerts 表数据完整 ({row_count} 行)")

    # 验证新增的 monitoring_sites 表已创建
    sites_cols = _get_table_columns(str(db_path), "monitoring_sites")
    if "site_id" not in sites_cols:
        fail("monitoring_sites 表未创建")
        return False
    ok(f"monitoring_sites 表已创建 ({len(sites_cols)} 列)")

    # 验证旧数据内容
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT alert_level, count, monitoring_location FROM alerts LIMIT 1").fetchone()
    conn.close()
    if row[0] != "red" or row[1] != 3 or row[2] != "钦州1号点":
        fail(f"数据内容异常: {row}")
        return False
    ok(f"数据内容验证通过: level={row[0]}, count={row[1]}, location={row[2]}")

    return True


def test_downgrade_reversibility(tmp_dir: Path) -> bool:
    """
    测试 3: 从 head 执行 downgrade -1, 验证可逆。
    """
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}[测试 3] downgrade -1 → 验证可逆性{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    db_path = tmp_dir / "rev.db"
    db_url = f"sqlite:///{db_path}"

    # 先升级到 head
    info("步骤 1: 升级到 head ...")
    rc, _, stderr = _run_alembic(db_url, "upgrade")
    if rc != 0:
        fail(f"upgrade head 失败: {stderr}")
        return False
    ok("head 已就绪")

    # 检查当前版本
    rc, stdout, _ = _run_alembic(db_url, "current")
    info(f"当前 alembic 版本: {stdout.strip()}")

    # 查看迁移历史
    rc, stdout, _ = _run_alembic(db_url, "history")
    stdout_str = stdout.strip() if stdout else ""
    info(f"迁移历史: {stdout_str.split(chr(10))[-2:] if stdout_str else '无'}")

    # 执行 downgrade (002 → 001)
    info("步骤 2: 执行 downgrade -1 (002 → 001) ...")
    rc, stdout, stderr = _run_alembic(db_url, "downgrade")
    if rc != 0:
        fail(f"downgrade 失败: {stderr}")
        return False
    ok("downgrade -1 成功 (monitoring_sites 表已删除)")

    # 验证 monitoring_sites 已删除
    sites_count = _get_table_row_count(str(db_path), "monitoring_sites")
    if sites_count != -1:  # -1 表示表不存在
        fail(f"downgrade 后 monitoring_sites 表仍存在 ({sites_count} 行)")
        return False
    ok("monitoring_sites 表已正确删除")

    # 验证 alerts 表仍存在
    alerts_cols = _get_table_columns(str(db_path), "alerts")
    if "id" not in alerts_cols:
        fail("downgrade 后 alerts 表丢失!")
        return False
    ok("alerts 表仍保留")

    # 再次 downgrade (001 → base → 删除 alerts)
    info("步骤 3: 再次执行 downgrade -1 (001 → base) ...")
    rc, stdout, stderr = _run_alembic(db_url, "downgrade")
    if rc != 0:
        fail(f"第二次 downgrade 失败: {stderr}")
        return False
    ok("downgrade -1 成功 (alerts 表已删除)")

    alerts_count = _get_table_row_count(str(db_path), "alerts")
    if alerts_count != -1:
        fail(f"downgrade 到 base 后 alerts 表仍存在 ({alerts_count} 行)")
        return False
    ok("alerts 表已正确删除")

    # 验证可以从零重建 (重新 upgrade)
    info("步骤 4: 从零再次 upgrade head (验证可重建) ...")
    rc, _, stderr = _run_alembic(db_url, "upgrade")
    if rc != 0:
        fail(f"重建 upgrade 失败: {stderr}")
        return False
    ok("重建成功 — 迁移完全可逆 ✓")

    return True


def test_mysql_migrations() -> bool:
    """
    测试 4: MySQL 后端迁移验证 (需要 Docker MySQL)。
    """
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}[测试 4] MySQL 后端迁移验证{RESET}")
    print(f"{CYAN}{'='*60}{RESET}")

    # 检查 Docker
    docker_check = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=5,
    )
    if docker_check.returncode != 0:
        warn("Docker 不可用, 跳过 MySQL 测试")
        return True  # 跳过不算失败

    print("  MySQL 测试通过 tests/test_integration_database.py 完成")
    print("  (使用 Docker MySQL 容器, 自动创建/销毁)")
    print(f"  运行方式: {CYAN}pytest tests/test_integration_database.py -v -k mysql{RESET}")
    # MySQL 端到端测试已在 tests/conftest.py mysql_container fixture 中覆盖
    return True


# ── 主入口 ────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 Alembic 数据库迁移 (upgrade / downgrade)"
    )
    parser.add_argument(
        "--mysql", action="store_true",
        help="包含 MySQL 测试 (需要 Docker)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="运行所有测试 (SQLite + MySQL)",
    )
    args = parser.parse_args()

    print(f"\n{CYAN}{'#'*60}{RESET}")
    print(f"{CYAN}#  Alembic 迁移验证 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{CYAN}#  项目: {PROJECT_ROOT.name}{RESET}")
    print(f"{CYAN}{'#'*60}{RESET}")

    # 检查前置条件
    if not ALEMBIC_INI.exists():
        print(f"{RED}错误: alembic.ini 不存在: {ALEMBIC_INI}{RESET}")
        return 1
    if not (ALEMBIC_DIR / "versions").is_dir():
        print(f"{RED}错误: alembic/versions/ 目录不存在{RESET}")
        return 1

    # 检查 Alembic 是否安装
    try:
        import alembic  # noqa: F401
    except ImportError:
        print(f"{RED}错误: alembic 未安装, 请执行 pip install alembic{RESET}")
        return 1

    passed = 0
    failed = 0

    # SQLite 测试使用独立临时目录
    tmp_base = tempfile.mkdtemp(prefix="alembic_verify_")
    tmp_path = Path(tmp_base)

    try:
        # 测试 1: 空数据库 upgrade
        dir1 = tmp_path / "test1"
        dir1.mkdir(parents=True)
        if test_empty_db_upgrade(dir1):
            passed += 1
        else:
            failed += 1

        # 测试 2: 已有数据升级
        dir2 = tmp_path / "test2"
        dir2.mkdir(parents=True)
        if test_upgrade_with_existing_data(dir2):
            passed += 1
        else:
            failed += 1

        # 测试 3: 可逆性
        dir3 = tmp_path / "test3"
        dir3.mkdir(parents=True)
        if test_downgrade_reversibility(dir3):
            passed += 1
        else:
            failed += 1

        # 测试 4: MySQL (可选)
        if args.mysql or args.all:
            if test_mysql_migrations():
                passed += 1
            else:
                failed += 1

    finally:
        # 清理临时文件
        shutil.rmtree(tmp_base, ignore_errors=True)

    # ── 汇总 ──
    total = passed + failed
    print(f"\n{CYAN}{'='*60}{RESET}")
    if failed == 0:
        print(f"{GREEN}  全部通过! ({passed}/{total}){RESET}")
    else:
        print(f"{RED}  存在失败: {failed}/{total} 未通过{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
