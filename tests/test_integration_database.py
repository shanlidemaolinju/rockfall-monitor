"""
集成测试 — 数据库读写与查询
===========================
验证 AlertStore SQLite 后端的完整 CRUD + 查询 + 统计链路.

运行: python -m pytest tests/test_integration_database.py -v
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from rockfall import alert_store as _alert_store_mod
from rockfall.alert_store import AlertStore

# 全局强制 SQLite: 用户的 .env 可能配置了 MySQL, 测试需要独立 SQLite
_alert_store_mod.MYSQL_HOST = ""

# 阻止 Alembic 以 MySQL 模式运行，手动执行增量迁移
import rockfall.migration as _mig_mod
_orig_run_migrations = _mig_mod.run_migrations
def _raise_skip(*args, **kwargs):
    raise RuntimeError("skipped — use _run_migrations instead")
_mig_mod.run_migrations = _raise_skip

# 测试专用数据库路径
_TEST_DB = Path(__file__).resolve().parent / "_test_alerts.db"


def _cleanup_db():
    """删除测试数据库及其 WAL 文件"""
    for ext in ("", "-wal", "-shm"):
        p = _TEST_DB.parent / f"{_TEST_DB.name}{ext}"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


@pytest.fixture
def store():
    """每个测试函数独立的 SQLite AlertStore (含增量列)"""
    _cleanup_db()
    s = AlertStore(db_path=str(_TEST_DB))
    s._stop_retry.set()
    # 手动运行增量迁移 (Alembic 已被模块级 monkeypatch 跳过)
    s._run_migrations()
    yield s
    _cleanup_db()


# ================================================================
# 批量写入 + 查询
# ================================================================

class TestBulkWriteAndQuery:

    @staticmethod
    def _seed(s, n=20):
        levels = ["blue", "yellow", "orange", "red"]
        for i in range(n):
            s.save_alert(count=(i % 5) + 1, max_confidence=0.3 + i * 0.03,
                         alert_level=levels[i % 4], track_ids=[i, i + 1],
                         push_status="pending")

    def test_bulk_write_and_get_recent(self, store):
        self._seed(store, n=30)
        recent = store.get_recent(limit=10)
        assert len(recent) == 10
        ids = [r["id"] for r in recent]
        assert ids == sorted(ids, reverse=True)

    def test_pagination(self, store):
        self._seed(store, n=25)
        page1 = store.get_recent(limit=10)
        assert len(page1) == 10

    def test_count_today_by_level(self, store):
        # 直接验证写入 + 读取
        for lvl, conf in [("red", 0.8), ("yellow", 0.5), ("blue", 0.4)]:
            store.save_alert(count=1, max_confidence=conf, alert_level=lvl)

        recent = store.get_recent(limit=10)
        assert len(recent) >= 3, f"应有至少 3 条记录, 实际: {len(recent)}"

        counts = store.count_today_by_level()
        assert isinstance(counts, dict)
        assert "red" in counts
        assert "yellow" in counts
        assert "blue" in counts

    def test_get_latest_alert(self, store):
        store.save_alert(count=5, max_confidence=0.95, alert_level="red")
        store.save_alert(count=2, max_confidence=0.55, alert_level="yellow")
        latest = store.get_latest_alert(min_level="yellow")
        assert latest is not None
        assert latest["alert_level"] in ("yellow", "orange", "red")


# ================================================================
# 日期范围 + 等级筛选
# ================================================================

class TestQueryFilters:

    @staticmethod
    def _seed_dated(s):
        levels = ["red", "orange", "yellow", "blue"]
        for i, days_ago in enumerate([0, 0, 1, 3, 5, 7, 10]):
            s.save_alert(count=i + 1, max_confidence=0.5 + i * 0.05,
                         alert_level=levels[i % 4], push_status="sent")
        s.save_alert(count=10, max_confidence=0.99,
                     alert_level="red", push_status="sent")

    def test_query_by_date_range(self, store):
        self._seed_dated(store)
        today = datetime.now().strftime("%Y-%m-%d")
        today_rows = store.query_alerts(start_date=today, end_date=today)
        assert len(today_rows) > 0

    def test_query_by_level(self, store):
        self._seed_dated(store)
        level_rows = store.query_alerts(alert_level="red")
        assert all(r["alert_level"] == "red" for r in level_rows)

    def test_query_combined_filters(self, store):
        self._seed_dated(store)
        today = datetime.now().strftime("%Y-%m-%d")
        rows = store.query_alerts(start_date=today, end_date=today, alert_level="red")
        for r in rows:
            assert r["alert_level"] == "red"

    def test_count_alerts(self, store):
        self._seed_dated(store)
        total = store.count_alerts()
        assert total > 0
        red_count = store.count_alerts(alert_level="red")
        assert red_count > 0
        assert red_count <= total

    def test_query_with_offset(self, store):
        self._seed_dated(store)
        all_rows = store.query_alerts(limit=100)
        if len(all_rows) >= 3:
            page = store.query_alerts(limit=2, offset=1)
            assert len(page) <= 2


# ================================================================
# 推送状态流转
# ================================================================

class TestStatusLifecycle:

    def test_status_flow(self, store):
        aid = store.save_alert(count=3, max_confidence=0.88,
                               alert_level="orange", push_status="pending")
        assert aid > 0

        pending = store.get_pending(limit=100)
        assert any(r["id"] == aid for r in pending)

        store.mark_sent(aid, "push_ok")
        recent = store.get_recent(limit=50)
        record = next((r for r in recent if r["id"] == aid), None)
        assert record is not None
        assert record["push_status"] == "sent"
        assert record["push_msg"] == "push_ok"

    def test_failed_status(self, store):
        aid = store.save_alert(count=1, max_confidence=0.5,
                               alert_level="blue", push_status="pending")
        store.mark_failed(aid, "network_error")
        recent = store.get_recent(limit=50)
        record = next((r for r in recent if r["id"] == aid), None)
        assert record is not None
        assert record["push_status"] == "failed"
        assert record["push_msg"] == "network_error"

    def test_pending_after_mark_sent_returns_others(self, store):
        aid = store.save_alert(count=1, max_confidence=0.6, push_status="pending")
        store.mark_sent(aid, "ok")
        pending = store.get_pending(limit=100)
        assert not any(r["id"] == aid for r in pending)


# ================================================================
# 审核标记 + 统计
# ================================================================

class TestReviewAndStats:

    def test_mark_review_confirmed(self, store):
        aid = store.save_alert(count=2, max_confidence=0.7, alert_level="yellow")
        ok = store.mark_review(aid, "confirmed", "边坡确有落石")
        assert ok is True

    def test_mark_review_false_alarm(self, store):
        aid = store.save_alert(count=1, max_confidence=0.35, alert_level="blue")
        ok = store.mark_review(aid, "false_alarm", "树叶飘落")
        assert ok is True

    def test_daily_trends(self, store):
        trends = store.get_daily_trends(days=3)
        assert len(trends) == 3
        for entry in trends:
            assert entry["total"] == (entry["red"] + entry["orange"] +
                                      entry["yellow"] + entry["blue"])

    def test_false_alarm_stats(self, store):
        stats = store.get_false_alarm_stats(days=30)
        assert "false_alarm_rate" in stats
        assert 0 <= stats["false_alarm_rate"] <= 1


# ================================================================
# Excel 导出
# ================================================================

class TestExcelExport:

    def test_export_empty(self, store):
        from rockfall.utils import export_alerts_to_excel
        data = export_alerts_to_excel([])
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_export_with_data(self, store):
        from rockfall.utils import export_alerts_to_excel
        store.save_alert(count=3, max_confidence=0.95, alert_level="red",
                         push_status="sent", rock_diameter_cm=35.0)
        rows = store.get_recent(limit=5)
        data = export_alerts_to_excel(rows, sheet_title="测试导出")
        assert isinstance(data, bytes)
        assert len(data) > 100

    def test_export_handles_missing_fields(self, store):
        from rockfall.utils import export_alerts_to_excel
        store.save_alert(count=0, max_confidence=0.0)
        rows = store.get_recent(limit=1)
        data = export_alerts_to_excel(rows)
        assert len(data) > 0


# ================================================================
# 边界条件
# ================================================================

class TestEdgeCases:

    def test_save_with_extreme_values(self, store):
        aid = store.save_alert(count=9999, max_confidence=1.0,
                               alert_level="red", push_status="pending",
                               rock_diameter_cm=999.9,
                               track_ids=list(range(100)))
        assert aid > 0

    def test_query_empty_result(self, store):
        rows = store.query_alerts(start_date="2000-01-01", end_date="2000-01-02")
        assert rows == []

    def test_count_empty_result(self, store):
        cnt = store.count_alerts(start_date="2000-01-01", end_date="2000-01-02")
        assert cnt == 0

    def test_get_latest_alert_empty(self, store):
        # 空 store → get_latest_alert 返回 None (强制 SQLite)
        import rockfall.alert_store as _m
        _orig = _m.MYSQL_HOST
        _m.MYSQL_HOST = ""
        try:
            empty = AlertStore(db_path=str(Path(__file__).parent / "_empty_test.db"))
            empty._stop_retry.set()
            assert empty.get_latest_alert() is None
        finally:
            _m.MYSQL_HOST = _orig
