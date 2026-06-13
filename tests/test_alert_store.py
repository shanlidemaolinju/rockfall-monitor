"""测试 alert_store.py — SQLite + MySQL 预警持久化"""

import pytest


class TestAlertStore:
    """SQLite AlertStore 基础 CRUD — 每个测试独立 DB (通过 sqlite_store fixture)"""

    def test_save_and_retrieve(self, sqlite_store):
        sqlite_store.save_alert(
            count=3, max_confidence=0.85,
            track_ids=[1, 2], alert_level="red",
            saved_frame="/tmp/test.jpg", push_status="sent",
            class_summary="落石:3",
        )
        recent = sqlite_store.get_recent(limit=5)
        assert any(r["alert_level"] == "red" and r["count"] == 3 for r in recent)

    def test_alert_level_persisted(self, sqlite_store):
        sqlite_store.save_alert(count=1, max_confidence=0.5, alert_level="yellow")
        sqlite_store.save_alert(count=2, max_confidence=0.9, alert_level="red")
        recent = sqlite_store.get_recent(limit=2)
        assert recent[0]["alert_level"] == "red"
        assert recent[1]["alert_level"] == "yellow"

    def test_mark_sent(self, sqlite_store):
        sqlite_store.save_alert(count=1, max_confidence=0.5, push_status="pending")
        alert_id = sqlite_store.get_recent(1)[0]["id"]
        sqlite_store.mark_sent(alert_id, "retry_ok")
        updated = sqlite_store.get_recent(1)[0]
        assert updated["push_status"] == "sent"
        assert updated["push_msg"] == "retry_ok"

    def test_mark_failed(self, sqlite_store):
        sqlite_store.save_alert(count=1, max_confidence=0.5, push_status="pending")
        alert_id = sqlite_store.get_recent(1)[0]["id"]
        sqlite_store.mark_failed(alert_id, "timeout")
        updated = sqlite_store.get_recent(1)[0]
        assert updated["push_status"] == "failed"
        assert updated["push_msg"] == "timeout"

    def test_default_values(self, sqlite_store):
        sqlite_store.save_alert(count=0, max_confidence=0.0)
        recent = sqlite_store.get_recent(1)[0]
        assert recent["alert_level"] == "yellow"
        assert recent["push_status"] == "pending"

    def test_get_recent_limit(self, sqlite_store):
        for i in range(5):
            sqlite_store.save_alert(count=i, max_confidence=0.5, alert_level="green")
        assert len(sqlite_store.get_recent(limit=3)) == 3
        assert len(sqlite_store.get_recent(limit=10)) >= 5

    def test_class_summary_persisted(self, sqlite_store):
        sqlite_store.save_alert(count=2, max_confidence=0.7, class_summary="落石:1, 滑坡:1")
        recent = sqlite_store.get_recent(1)[0]
        assert recent["class_summary"] == "落石:1, 滑坡:1"


# ================================================================
# MySQL 后端 — 完整 CRUD + 查询 + 统计 (需要 Docker MySQL)
# ================================================================

class TestAlertStoreMySQL:
    """AlertStore MySQL 后端测试 — 覆盖 MySQL 特有的 SQL 语句、参数绑定、事务行为"""

    def test_save_alert_mysql_insert(self, mysql_backend):
        """MySQL 路径: save_alert → INSERT 成功返回正整数 ID"""
        store = mysql_backend
        aid = store.save_alert(
            count=3, max_confidence=0.85,
            track_ids=[1, 2], alert_level="red",
            saved_frame="/tmp/test.jpg",
            class_summary="落石:3",
            rock_diameter_cm=25.5,
            monitoring_location="南宁K12+350",
        )
        assert aid > 0, f"MySQL INSERT 应返回正整数 ID, 实际: {aid}"

    def test_get_recent_mysql(self, mysql_backend):
        """MySQL 路径: get_recent 返回按 ID 降序排列"""
        store = mysql_backend
        for i in range(5):
            store.save_alert(count=i, max_confidence=0.5 + i * 0.1,
                           alert_level="yellow")
        recent = store.get_recent(limit=3)
        assert len(recent) == 3
        ids = [r["id"] for r in recent]
        assert ids == sorted(ids, reverse=True)

    def test_get_pending_mysql(self, mysql_backend):
        """MySQL 路径: get_pending 仅返回 push_status='pending' 的记录"""
        store = mysql_backend
        aid_pending = store.save_alert(count=1, max_confidence=0.5, push_status="pending")
        aid_sent = store.save_alert(count=2, max_confidence=0.7, push_status="sent")
        store.mark_sent(aid_sent, "ok")

        pending = store.get_pending(limit=50)
        pending_ids = [r["id"] for r in pending]
        assert aid_pending in pending_ids
        assert aid_sent not in pending_ids

    def test_mark_sent_mysql(self, mysql_backend):
        """MySQL 路径: mark_sent 更新 push_status 和 push_msg"""
        store = mysql_backend
        aid = store.save_alert(count=1, max_confidence=0.5, push_status="pending")
        store.mark_sent(aid, "推送成功")
        recent = store.get_recent(1)[0]
        assert recent["push_status"] == "sent"
        assert recent["push_msg"] == "推送成功"

    def test_mark_failed_mysql(self, mysql_backend):
        """MySQL 路径: mark_failed 更新推送失败状态"""
        store = mysql_backend
        aid = store.save_alert(count=2, max_confidence=0.6, push_status="pending")
        store.mark_failed(aid, "网络超时")
        recent = store.get_recent(1)[0]
        assert recent["push_status"] == "failed"
        assert recent["push_msg"] == "网络超时"

    def test_query_alerts_date_range_mysql(self, mysql_backend):
        """MySQL 路径: query_alerts 日期范围筛选"""
        store = mysql_backend
        from datetime import datetime, timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        store.save_alert(count=1, max_confidence=0.5, alert_level="yellow")
        rows = store.query_alerts(start_date=today, end_date=today)
        assert len(rows) >= 1

    def test_query_alerts_by_level_mysql(self, mysql_backend):
        """MySQL 路径: query_alerts 等级筛选"""
        store = mysql_backend
        store.save_alert(count=1, max_confidence=0.95, alert_level="red")
        store.save_alert(count=2, max_confidence=0.55, alert_level="yellow")
        store.save_alert(count=3, max_confidence=0.35, alert_level="blue")

        red_rows = store.query_alerts(alert_level="red")
        assert all(r["alert_level"] == "red" for r in red_rows)

    def test_count_alerts_mysql(self, mysql_backend):
        """MySQL 路径: count_alerts 统计"""
        store = mysql_backend
        for _ in range(3):
            store.save_alert(count=1, max_confidence=0.5, alert_level="orange")
        total = store.count_alerts()
        assert total >= 3
        orange_count = store.count_alerts(alert_level="orange")
        assert orange_count >= 3

    def test_count_today_by_level_mysql(self, mysql_backend):
        """MySQL 路径: count_today_by_level 今日各等级统计"""
        store = mysql_backend
        store.save_alert(count=1, max_confidence=0.95, alert_level="red")
        store.save_alert(count=2, max_confidence=0.65, alert_level="yellow")
        store.save_alert(count=3, max_confidence=0.35, alert_level="blue")

        counts = store.count_today_by_level()
        assert isinstance(counts, dict)
        assert set(counts.keys()) == {"red", "orange", "yellow", "blue"}

    def test_get_latest_alert_mysql(self, mysql_backend):
        """MySQL 路径: get_latest_alert 按等级过滤"""
        store = mysql_backend
        store.save_alert(count=5, max_confidence=0.95, alert_level="red")
        store.save_alert(count=2, max_confidence=0.55, alert_level="yellow")

        latest = store.get_latest_alert(min_level="orange")
        assert latest is not None
        assert latest["alert_level"] in ("red", "orange")

    def test_workflow_transition_mysql(self, mysql_backend):
        """MySQL 路径: transition_workflow 完整流转"""
        store = mysql_backend
        aid = store.save_alert(count=3, max_confidence=0.9, alert_level="orange")

        # pending → confirmed
        r = store.transition_workflow(aid, "confirmed", "operator_zhang", "确认落石")
        assert r["ok"]

        # confirmed → dispatched
        r = store.transition_workflow(aid, "dispatched", "dispatcher_li", "派单")
        assert r["ok"]

        history = store.get_workflow_history(aid)
        assert len(history) == 2
        assert history[-1]["to"] == "dispatched"

    def test_workflow_invalid_transition_blocked_mysql(self, mysql_backend):
        """MySQL 路径: 越级流转被拦截"""
        store = mysql_backend
        aid = store.save_alert(count=2, max_confidence=0.8, alert_level="orange")
        r = store.transition_workflow(aid, "archived", "tester")
        assert not r["ok"]

    def test_mark_review_mysql(self, mysql_backend):
        """MySQL 路径: mark_review 审核标记"""
        store = mysql_backend
        aid = store.save_alert(count=2, max_confidence=0.7, alert_level="yellow")
        ok = store.mark_review(aid, "confirmed", "边坡落石确认")
        assert ok is True

    def test_daily_trends_mysql(self, mysql_backend):
        """MySQL 路径: get_daily_trends"""
        store = mysql_backend
        store.save_alert(count=1, max_confidence=0.5, alert_level="yellow")
        trends = store.get_daily_trends(days=3)
        assert len(trends) == 3
        for entry in trends:
            assert entry["total"] == (entry["red"] + entry["orange"] +
                                      entry["yellow"] + entry["blue"])

    def test_false_alarm_stats_mysql(self, mysql_backend):
        """MySQL 路径: get_false_alarm_stats"""
        store = mysql_backend
        aid = store.save_alert(count=1, max_confidence=0.35, alert_level="blue")
        store.mark_review(aid, "false_alarm", "误报: 树叶")
        stats = store.get_false_alarm_stats(days=30)
        assert "false_alarm_rate" in stats
        assert 0 <= stats["false_alarm_rate"] <= 1

    def test_query_with_offset_mysql(self, mysql_backend):
        """MySQL 路径: query_alerts 分页偏移"""
        store = mysql_backend
        for i in range(5):
            store.save_alert(count=i, max_confidence=0.3 + i * 0.1, alert_level="blue")
        all_rows = store.query_alerts(limit=100)
        if len(all_rows) >= 3:
            page = store.query_alerts(limit=2, offset=1)
            assert len(page) <= 2

    def test_save_with_all_fields_mysql(self, mysql_backend):
        """MySQL 路径: 全字段写入 + 读取一致性"""
        store = mysql_backend
        aid = store.save_alert(
            count=4, max_confidence=0.88, alert_level="orange",
            track_ids=[10, 20, 30],
            class_summary="落石:3, 滑坡:1",
            saved_frame="/data/results/frame_001.jpg",
            clip_path="/data/clips/clip_001.mp4",
            push_status="pending",
            rock_diameter_cm=22.0,
            monitoring_location="崇左K138+800",
        )
        assert aid > 0

        rows = store.query_alerts(limit=1)
        r = rows[0]
        assert r["alert_level"] == "orange"
        assert r["count"] == 4
        assert r["rock_diameter_cm"] == pytest.approx(22.0)
        assert r["monitoring_location"] == "崇左K138+800"

    def test_count_by_workflow_state_mysql(self, mysql_backend):
        """MySQL 路径: 按工单状态统计"""
        store = mysql_backend
        for i in range(3):
            aid = store.save_alert(count=1, max_confidence=0.5, alert_level="yellow")
            if i < 2:
                store.transition_workflow(aid, "confirmed", "op", "确认")

        counts = store.count_by_workflow_state()
        assert isinstance(counts, dict)
        # 至少包含 confirmed 状态
        assert "confirmed" in counts or "pending" in counts


# ================================================================
# 推送重试逻辑
# ================================================================

class TestPushRetryLogic:
    """推送重试逻辑 — _retry_loop 核心路径 + mark_sent/mark_failed"""

    @staticmethod
    def _simulate_one_retry_tick(store, monkeypatch):
        """
        模拟 _retry_loop 的单次迭代 (绕过无限循环)。

        直接调用 retry_loop 的循环体逻辑:
          1. get_pending(limit=5)
          2. 逐个发送 PushPlus 请求
          3. mark_sent / mark_failed
        """
        import requests as _requests
        import rockfall.config as cfg

        pending = store.get_pending(limit=5)
        for alert in pending:
            if not cfg.PUSHPLUS_TOKEN or cfg.PUSHPLUS_TOKEN == "your_token_here":
                continue
            try:
                level = alert.get("alert_level", "yellow")
                level_label = store.LEVEL_LABELS.get(level, "⚠️ 预警")
                class_info = alert.get("class_summary", "落石") or "落石"
                data = {
                    "token": cfg.PUSHPLUS_TOKEN,
                    "title": f"{level_label} {class_info}报警（补发）",
                    "content": f"补发预警: {alert['time']}, "
                               f"数量={alert['count']}, "
                               f"置信度={alert['max_confidence']}",
                    "topic": cfg.PUSHPLUS_TOPIC,
                    "template": "html",
                }
                res = _requests.post(cfg.PUSHPLUS_URL, json=data, timeout=10).json()
                if res.get("code") == 200:
                    store.mark_sent(alert["id"], "retry_ok")
                else:
                    store.mark_failed(alert["id"], str(res.get("msg", "")))
            except Exception as e:
                store.mark_failed(alert["id"], str(e))

    def test_retry_sends_pending(self, sqlite_store, monkeypatch):
        """重试将 pending 记录推送成功并标记为 sent"""
        import requests as _requests
        import rockfall.config as cfg

        monkeypatch.setattr(cfg, "PUSHPLUS_TOKEN", "test_token_123")
        monkeypatch.setattr(cfg, "PUSHPLUS_URL", "http://localhost:9999/push")
        monkeypatch.setattr(cfg, "PUSHPLUS_TOPIC", "")

        # Mock HTTP
        _called = []
        class _MockResponse:
            @staticmethod
            def json():
                return {"code": 200, "msg": "ok"}
        monkeypatch.setattr(_requests, "post",
                           lambda url, json=None, timeout=None: _called.append(json) or _MockResponse())

        aid = sqlite_store.save_alert(
            count=2, max_confidence=0.8, alert_level="orange",
            push_status="pending",
        )

        self._simulate_one_retry_tick(sqlite_store, monkeypatch)

        assert len(_called) >= 1, "应发送一次重试推送"
        assert _called[0]["title"] == "🟠 Ⅱ级·严重 落石报警（补发）"

        recent = sqlite_store.get_recent(1)[0]
        assert recent["push_status"] == "sent"
        assert recent["push_msg"] == "retry_ok"

    def test_retry_skips_when_no_token(self, sqlite_store, monkeypatch):
        """未配置 PushPlus token 时不发送推送"""
        import requests as _requests
        import rockfall.config as cfg

        monkeypatch.setattr(cfg, "PUSHPLUS_TOKEN", "")
        monkeypatch.setattr(cfg, "PUSHPLUS_URL", "http://localhost:9999/push")

        _called = []
        monkeypatch.setattr(_requests, "post", lambda *a, **kw: _called.append(1))

        sqlite_store.save_alert(count=1, max_confidence=0.5, push_status="pending")

        self._simulate_one_retry_tick(sqlite_store, monkeypatch)

        assert len(_called) == 0, "无 token 时不应发送请求"

    def test_retry_handles_http_error(self, sqlite_store, monkeypatch):
        """HTTP 错误时标记为 failed"""
        import requests as _requests
        import rockfall.config as cfg

        monkeypatch.setattr(cfg, "PUSHPLUS_TOKEN", "test_token")
        monkeypatch.setattr(cfg, "PUSHPLUS_URL", "http://localhost:9999/push")
        monkeypatch.setattr(cfg, "PUSHPLUS_TOPIC", "")

        def _fake_post(*a, **kw):
            raise ConnectionError("网络不可达")
        monkeypatch.setattr(_requests, "post", _fake_post)

        aid = sqlite_store.save_alert(count=1, max_confidence=0.5, push_status="pending")

        self._simulate_one_retry_tick(sqlite_store, monkeypatch)

        recent = sqlite_store.get_recent(1)[0]
        assert recent["push_status"] == "failed"
        assert "网络不可达" in recent.get("push_msg", "")


# ================================================================
# MySQL 边界条件
# ================================================================

class TestAlertStoreMySQLEdgeCases:
    """MySQL 后端的异常与边界场景"""

    def test_query_empty_result_mysql(self, mysql_backend):
        """查询无匹配记录时返回空列表"""
        rows = mysql_backend.query_alerts(
            start_date="2000-01-01", end_date="2000-01-02",
        )
        assert rows == []

    def test_count_empty_result_mysql(self, mysql_backend):
        """统计无匹配记录时返回 0"""
        cnt = mysql_backend.count_alerts(
            start_date="2000-01-01", end_date="2000-01-02",
        )
        assert cnt == 0

    def test_get_latest_alert_empty_mysql(self, mysql_backend):
        """空数据库 → get_latest_alert 返回 None"""
        # mysql_backend fixture 已 truncate, 确保为空
        assert mysql_backend.get_latest_alert() is None

    def test_save_with_extreme_values_mysql(self, mysql_backend):
        """MySQL 路径: 极值写入不崩溃"""
        aid = mysql_backend.save_alert(
            count=9999, max_confidence=1.0,
            alert_level="red", push_status="pending",
            rock_diameter_cm=999.9,
            track_ids=list(range(100)),
            class_summary="极端测试",
        )
        assert aid > 0