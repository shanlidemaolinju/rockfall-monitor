"""测试 alert_store.py — SQLite 预警持久化"""

import tempfile
import time
from pathlib import Path


class TestAlertStore:
    @classmethod
    def setup_class(cls):
        from rockfall.alert_store import AlertStore
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls.tmpdir.name) / "test_alerts.db")
        cls.store = AlertStore(db_path=cls.db_path)
        cls.store._stop_retry.set()  # 阻止重试线程启动

    @classmethod
    def teardown_class(cls):
        cls.tmpdir.cleanup()

    def test_save_and_retrieve(self):
        self.store.save_alert(
            count=3, max_confidence=0.85,
            track_ids=[1, 2], alert_level="red",
            saved_frame="/tmp/test.jpg", push_status="sent",
            class_summary="落石:3",
        )
        recent = self.store.get_recent(limit=5)
        assert any(r["alert_level"] == "red" and r["count"] == 3 for r in recent)

    def test_alert_level_persisted(self):
        self.store.save_alert(count=1, max_confidence=0.5, alert_level="yellow")
        self.store.save_alert(count=2, max_confidence=0.9, alert_level="red")
        recent = self.store.get_recent(limit=2)
        assert recent[0]["alert_level"] == "red"
        assert recent[1]["alert_level"] == "yellow"

    def test_mark_sent(self):
        self.store.save_alert(count=1, max_confidence=0.5, push_status="pending")
        alert_id = self.store.get_recent(1)[0]["id"]
        self.store.mark_sent(alert_id, "retry_ok")
        updated = self.store.get_recent(1)[0]
        assert updated["push_status"] == "sent"
        assert updated["push_msg"] == "retry_ok"

    def test_mark_failed(self):
        self.store.save_alert(count=1, max_confidence=0.5, push_status="pending")
        alert_id = self.store.get_recent(1)[0]["id"]
        self.store.mark_failed(alert_id, "timeout")
        updated = self.store.get_recent(1)[0]
        assert updated["push_status"] == "failed"
        assert updated["push_msg"] == "timeout"

    def test_default_values(self):
        self.store.save_alert(count=0, max_confidence=0.0)
        recent = self.store.get_recent(1)[0]
        assert recent["alert_level"] == "yellow"
        assert recent["push_status"] == "pending"

    def test_get_recent_limit(self):
        for i in range(5):
            self.store.save_alert(count=i, max_confidence=0.5, alert_level="green")
        assert len(self.store.get_recent(limit=3)) == 3
        assert len(self.store.get_recent(limit=10)) >= 5

    def test_class_summary_persisted(self):
        self.store.save_alert(count=2, max_confidence=0.7, class_summary="落石:1, 滑坡:1")
        recent = self.store.get_recent(1)[0]
        assert recent["class_summary"] == "落石:1, 滑坡:1"