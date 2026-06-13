"""
端到端测试: 预警触发 → 工单流转 → 审计记录 → 归档
====================================================
验证完整业务闭环。
"""

import pytest
from rockfall.alert_store import AlertStore
from rockfall.audit import AuditLogger, audit_log
from rockfall.health import SystemHealth
from rockfall.storage import StorageManager


@pytest.fixture
def store():
    s = AlertStore(db_path="data/_test_e2e.db")
    yield s
    import os
    try:
        os.unlink("data/_test_e2e.db")
    except Exception:
        pass


@pytest.fixture
def audit():
    a = AuditLogger(db_path="data/_test_audit_e2e.db")
    yield a
    import os
    try:
        os.unlink("data/_test_audit_e2e.db")
    except Exception:
        pass


class TestAlertWorkflowE2E:
    """完整预警工单流转闭环测试"""

    def test_full_workflow_lifecycle(self, store, audit):
        # 1. 系统触发预警
        alert_id = store.save_alert(
            count=3, max_confidence=0.95, alert_level="red",
            rock_diameter_cm=45, monitoring_location="钦州K12+350",
        )
        assert alert_id > 0
        audit.log("alert_triggered", "system", f"Alert#{alert_id} triggered", alert_id)

        # 2. 值班员审核确认真实落石 (pending -> confirmed)
        r = store.transition_workflow(alert_id, "confirmed", "operator_zhang", "确认真实落石")
        assert r["ok"]
        audit.log("workflow_transition", "operator_zhang",
                  f"Alert#{alert_id}: pending->confirmed", alert_id)

        # 3. 调度员派单 (confirmed -> dispatched)
        r = store.transition_workflow(alert_id, "dispatched", "dispatcher_li", "派单至巡查1组")
        assert r["ok"]
        audit.log("workflow_transition", "dispatcher_li",
                  f"Alert#{alert_id}: confirmed->dispatched", alert_id)

        # 4. 现场人员到场 (dispatched -> arrived)
        r = store.transition_workflow(alert_id, "arrived", "crew_wang", "已到场确认落石")
        assert r["ok"]

        # 5. 处置完毕 (arrived -> handled)
        r = store.transition_workflow(alert_id, "handled", "crew_wang", "落石已清除")
        assert r["ok"]

        # 6. 归档 (handled -> archived)
        r = store.transition_workflow(alert_id, "archived", "admin", "案件归档")
        assert r["ok"]
        audit.log("workflow_transition", "admin",
                  f"Alert#{alert_id}: handled->archived (closed)", alert_id)

        # 验证流转历史
        history = store.get_workflow_history(alert_id)
        assert len(history) == 5, f"Expected 5 transitions, got {len(history)}"
        assert history[-1]["to"] == "archived"

        # 验证无法回退
        r = store.transition_workflow(alert_id, "pending", "tester")
        assert not r["ok"], "Should not allow backward transition"

        # 验证审计日志
        audit_rows = audit.query(alert_id=alert_id)
        assert len(audit_rows) >= 2

    def test_false_alarm_workflow(self, store):
        """误报流程: pending -> false_alarm (直接关闭)"""
        alert_id = store.save_alert(count=1, max_confidence=0.35, alert_level="blue")
        r = store.transition_workflow(alert_id, "false_alarm", "operator_liu", "确认是风吹树叶")
        assert r["ok"]
        # 误报后不可再转
        r = store.transition_workflow(alert_id, "confirmed", "tester")
        assert not r["ok"], "Should not allow transition from false_alarm"

    def test_invalid_transition_blocked(self, store):
        """越级流转应被拦截"""
        alert_id = store.save_alert(count=2, max_confidence=0.8, alert_level="orange")
        # pending 不能直接到 archived
        r = store.transition_workflow(alert_id, "archived", "tester")
        assert not r["ok"], "Should block pending->archived"


class TestHealthCheck:
    """系统健康检查测试"""

    def test_health_returns_all_checks(self):
        health = SystemHealth()
        status = health.check_all()
        assert "healthy" in status
        assert "checks" in status
        assert "warnings" in status
        assert "uptime_hours" in status

    def test_health_check_disk(self):
        health = SystemHealth()
        disk = health._check_disk()
        assert "total_gb" in disk
        assert "free_gb" in disk
        assert "percent" in disk

    def test_health_check_model(self):
        health = SystemHealth()
        model = health._check_model()
        assert model["exists"], "Model file should exist"


class TestStorageManagement:
    """存储管理测试"""

    def test_get_storage_stats(self):
        sm = StorageManager()
        stats = sm.get_storage_stats()
        assert "total_mb" in stats
        assert "results" in stats

    def test_cleanup_dry_run(self):
        sm = StorageManager()
        result = sm.cleanup_old_files(retention_days=365, dry_run=True)
        assert result["dry_run"]
        assert "deleted_count" in result

    def test_quota_check_not_exceeded(self):
        sm = StorageManager()
        result = sm.enforce_quota(max_total_mb=100000)  # 100GB quota
        assert not result["enforced"], "Should not enforce with high quota"
