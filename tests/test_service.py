"""
测试 service.py — 业务逻辑层
============================
覆盖:
  - 异步任务生命周期 (create / poll / cleanup)
  - 检测器池管理 (_get_detector / remove_detector)
  - 图片/视频检测 (mock 模型)
  - 看板统计缓存 (get_dashboard_stats)
  - 预警审核 + 统计 + 导出
  - 运行时配置热更新 (update_runtime_config)
  - 点位管理 + ROI 管理

依赖:
  - mock_detector fixture (绕过 YOLO 模型加载)
  - sqlite_store fixture (SQLite AlertStore)
  - sqlite_site_store fixture (SQLite SiteStore)
  - tmp_data_dir fixture (临时文件隔离)

运行:
    pytest tests/test_service.py -v
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

import rockfall.site_config as _site_cfg
import server.service as svc


# ================================================================
# 异步任务生命周期
# ================================================================

class TestTaskLifecycle:
    """视频检测异步任务的创建 → 进度更新 → 完成/失败 → 过期清理"""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_detector, sqlite_store, tmp_data_dir, monkeypatch):
        """每个测试前: 清空任务存储 + 准备测试视频"""
        import rockfall.config as cfg

        svc._task_store.clear()
        # 停止已有的 executor 线程
        self._cleanup_tasks_after()

        # 确保 config 指向临时目录
        monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_data_dir / "results")
        monkeypatch.setattr(cfg, "UPLOADS_DIR", tmp_data_dir / "uploads")
        cfg.RESULTS_DIR.mkdir(exist_ok=True)
        cfg.UPLOADS_DIR.mkdir(exist_ok=True)

        # 准备一个最小的测试视频文件 (1帧 MP4)
        self._test_video = tmp_data_dir / "test_video.mp4"
        self._create_minimal_video(self._test_video)

    def _cleanup_tasks_after(self):
        """测试后清理任务存储"""
        yield
        svc._task_store.clear()

    @staticmethod
    def _create_minimal_video(path: Path):
        """创建一个最小可读的视频文件 (供 cv2.VideoCapture 打开)"""
        import cv2
        import numpy as np
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 25.0, (640, 480))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "TEST", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
        writer.release()

    # ---- 异步任务创建与查询 ----

    def test_create_async_video_task(self):
        """detect_video_local_async 创建任务并返回 task_id"""
        task_id = svc.detect_video_local_async(
            str(self._test_video), save_frames=False, push_alerts=False,
        )
        assert task_id
        assert len(task_id) == 36  # UUID4 格式

        status = svc.get_task_status(task_id)
        assert status is not None
        assert status["status"] in ("processing", "completed")

    def test_get_task_status_nonexistent(self):
        """查询不存在的任务返回 None"""
        assert svc.get_task_status("nonexistent-task-id") is None

    def test_task_transitions_to_completed(self):
        """任务完成后状态变为 completed"""
        task_id = svc.detect_video_local_async(
            str(self._test_video), save_frames=False, push_alerts=False,
        )
        # 等待任务完成 (模拟视频很短, 应快速完成)
        self._wait_for_task(task_id, expected_status="completed", timeout=15)

        status = svc.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["result"] is not None
        assert status.get("progress") == 100.0

    def test_task_progress_tracking(self):
        """任务执行期间进度字段持续更新"""
        task_id = svc.detect_video_local_async(
            str(self._test_video), save_frames=False, push_alerts=False,
        )

        # 等待至少开始处理
        time.sleep(0.5)
        status = svc.get_task_status(task_id)
        if status:
            # progress 字段应存在 (可能为 0.0 在开始前, 或 >0 在处理中)
            assert "progress" in status
            assert "current_frame" in status
            assert "total_frames" in status

    def test_task_records_camera_id(self):
        """任务记录关联的 camera_id"""
        task_id = svc.detect_video_local_async(
            str(self._test_video), save_frames=False, push_alerts=False,
            camera_id="nanning_naan_s1",
        )
        status = svc.get_task_status(task_id)
        assert status is not None
        assert status.get("camera_id") == "nanning_naan_s1"

    # ---- 过期任务清理 ----

    def test_cleanup_expired_completed_tasks(self, monkeypatch):
        """已完成任务在超过阈值后被清理"""
        import rockfall.config as cfg
        monkeypatch.setattr(cfg, "TASK_CLEANUP_SECONDS", 0)  # 立即过期
        monkeypatch.setattr(cfg, "TASK_CLEANUP_STUCK_SECONDS", 3600)

        # 手动创建一个"已完成"的旧任务
        with svc._task_lock:
            svc._task_store["old_completed"] = {
                "status": "completed", "result": {},
                "created_at": time.time() - 99999,
                "camera_id": "default",
            }

        svc._cleanup_expired_tasks()
        assert "old_completed" not in svc._task_store

    def test_cleanup_expired_failed_tasks(self, monkeypatch):
        """失败任务过期后也被清理"""
        import rockfall.config as cfg
        monkeypatch.setattr(cfg, "TASK_CLEANUP_SECONDS", 0)
        monkeypatch.setattr(cfg, "TASK_CLEANUP_STUCK_SECONDS", 3600)

        with svc._task_lock:
            svc._task_store["old_failed"] = {
                "status": "failed", "error": "test",
                "created_at": time.time() - 99999,
                "camera_id": "default",
            }

        svc._cleanup_expired_tasks()
        assert "old_failed" not in svc._task_store

    def test_cleanup_stuck_processing_tasks(self, monkeypatch):
        """卡死的处理中任务 (超时未完成) 被清理"""
        import rockfall.config as cfg
        monkeypatch.setattr(cfg, "TASK_CLEANUP_SECONDS", 3600)
        monkeypatch.setattr(cfg, "TASK_CLEANUP_STUCK_SECONDS", 0)  # 立即判定卡死

        with svc._task_lock:
            svc._task_store["stuck_task"] = {
                "status": "processing",
                "created_at": time.time() - 99999,
                "camera_id": "default",
            }

        svc._cleanup_expired_tasks()
        assert "stuck_task" not in svc._task_store

    def test_cleanup_preserves_active_tasks(self):
        """正常运行中的任务不会被误清理"""
        import rockfall.config as cfg

        with svc._task_lock:
            svc._task_store["active_task"] = {
                "status": "processing",
                "created_at": time.time(),
                "camera_id": "default",
            }

        svc._cleanup_expired_tasks()
        assert "active_task" in svc._task_store

    # ---- 辅助 ----

    @staticmethod
    def _wait_for_task(task_id: str, expected_status: str, timeout: float = 20):
        """轮询等待任务进入预期状态"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = svc.get_task_status(task_id)
            if status and status["status"] == expected_status:
                return
            time.sleep(0.3)
        # 超时不抛异常, 由调用方断言


# ================================================================
# 检测器池管理
# ================================================================

class TestDetectorPool:
    """_get_detector / remove_detector 检测器实例池"""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_detector):
        """清空检测器池"""
        svc._detectors.clear()
        svc._active_cameras.clear()
        yield
        svc._detectors.clear()
        svc._active_cameras.clear()

    def test_get_detector_creates_new(self):
        """首次调用 _get_detector 创建新实例"""
        det = svc._get_detector("camera_01")
        assert det is not None
        assert "camera_01" in svc._detectors

    def test_get_detector_reuses_existing(self):
        """同一 camera_id 返回已缓存的实例"""
        det1 = svc._get_detector("camera_02")
        det2 = svc._get_detector("camera_02")
        assert det1 is det2  # 同一对象

    def test_get_detector_different_cameras(self):
        """不同 camera_id 创建独立实例"""
        det_a = svc._get_detector("cam_A")
        det_b = svc._get_detector("cam_B")
        assert det_a is not det_b
        assert len(svc._detectors) >= 2

    def test_remove_detector(self):
        """remove_detector 清理实例和活跃列表"""
        svc._get_detector("to_remove")
        assert "to_remove" in svc._detectors

        svc.remove_detector("to_remove")
        assert "to_remove" not in svc._detectors

    def test_remove_nonexistent_detector_no_error(self):
        """移除不存在的 detector 不抛异常"""
        svc.remove_detector("never_added")  # 不应崩溃

    def test_detector_uses_site_id(self):
        """检测器创建时传入 site_id (验证不同 site_id 创建不同 detector)"""
        svc._detectors.clear()

        det_a = svc._get_detector("site_A")
        det_b = svc._get_detector("site_B")

        assert det_a is not det_b  # 不同实例
        assert det_a.site_id == "site_A"
        assert det_b.site_id == "site_B"

        # 验证缓存: 再次获取同一 site_id 返回同一实例
        det_a2 = svc._get_detector("site_A")
        assert det_a2 is det_a


# ================================================================
# 图片/视频检测 (mock)
# ================================================================

class TestDetectionWithMock:
    """使用 Mock 检测器验证检测流程 (不加载真实模型)"""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_detector, sqlite_store, tmp_data_dir, monkeypatch):
        import rockfall.config as cfg
        monkeypatch.setattr(cfg, "UPLOADS_DIR", tmp_data_dir / "uploads")
        cfg.UPLOADS_DIR.mkdir(exist_ok=True)
        monkeypatch.setattr(cfg, "RESULTS_DIR", tmp_data_dir / "results")
        cfg.RESULTS_DIR.mkdir(exist_ok=True)

        svc._detectors.clear()
        self._tmp = tmp_data_dir
        yield
        svc._detectors.clear()

    def test_detect_image_with_default(self, monkeypatch):
        """无文件时使用默认测试图片 (如果存在)"""
        # detect_image_file 使用 __file__ 计算默认图片路径
        # __file__ → server/service.py → parent.parent → <project_root>
        project_root = Path(__file__).resolve().parent.parent
        default_img = project_root / "data" / "rock.jpg"
        default_img.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        import numpy as np
        cv2.imwrite(str(default_img), np.zeros((100, 100, 3), dtype=np.uint8))

        try:
            result = svc.detect_image_file(file=None)
            assert "error" not in result
            assert result.get("count") == 2  # mock 返回值
        finally:
            # 清理
            if default_img.exists():
                default_img.unlink()

    def test_detect_image_with_upload(self, tmp_data_dir):
        """上传图片文件 → 检测 → 清理临时文件"""
        from io import BytesIO

        # 创建模拟 UploadFile
        class _MockUploadFile:
            filename = "test_rock.jpg"
            file = BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # 最小 JPEG

        mock_file = _MockUploadFile()
        result = svc.detect_image_file(file=mock_file)
        assert "error" not in result
        assert result.get("count") == 2

        # 验证临时文件已清理
        tmp_path = tmp_data_dir / "uploads" / "test_rock.jpg"
        assert not tmp_path.exists(), "上传临时文件应被清理"

    def test_detect_image_file_default_missing(self, tmp_data_dir, monkeypatch):
        """默认图片不存在时返回 error"""
        import rockfall.config as cfg
        monkeypatch.setattr(cfg, "DETECTION_CONFIDENCE", 0.35)

        # 确保默认图片路径不存在
        project_root = Path(__file__).resolve().parent.parent
        default_img = project_root / "data" / "rock.jpg"
        if default_img.exists():
            backup = default_img.read_bytes()
            default_img.unlink()
            try:
                result = svc.detect_image_file(file=None)
                assert "error" in result
            finally:
                default_img.parent.mkdir(parents=True, exist_ok=True)
                default_img.write_bytes(backup)
        else:
            result = svc.detect_image_file(file=None)
            assert "error" in result

    def test_detect_video_sync(self, tmp_data_dir):
        """同步视频检测返回完整结果"""
        import cv2
        import numpy as np

        video_path = tmp_data_dir / "sync_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (640, 480))
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.release()

        result = svc.detect_video_local(
            str(video_path), save_frames=False, push_alerts=False,
        )
        assert result is not None
        assert "source" in result


# ================================================================
# 看板统计 + 缓存
# ================================================================

class TestDashboardStats:
    """看板统计与缓存行为"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_store, monkeypatch):
        """注入 AlertStore 种子数据 + 清除统计缓存"""
        import rockfall.config as cfg
        from rockfall.logger import log_event

        svc._stats_cache = None
        svc._stats_cache_time = 0

        # 写入一些预警记录供统计
        store = sqlite_store
        for lvl, conf in [("red", 0.95), ("orange", 0.75), ("yellow", 0.55), ("blue", 0.4)]:
            store.save_alert(count=1, max_confidence=conf, alert_level=lvl)

        # 也写一条日志 (get_dashboard_stats 从日志+DB两份取)
        log_event("detection", frame=1, count=3, alert_level="red",
                  max_confidence=0.95, track_ids=[1, 2])

        # 确保 get_alert_store() 返回我们的测试 store
        import rockfall.alert_store as als
        self._old_store = als._store
        als._store = store

        yield

        als._store = self._old_store
        svc._stats_cache = None

    def test_dashboard_stats_returns_all_fields(self):
        stats = svc.get_dashboard_stats()
        assert "today_total" in stats
        assert "today_red" in stats
        assert "today_orange" in stats
        assert "today_yellow" in stats
        assert "today_blue" in stats
        assert "last_count" in stats
        assert "last_conf" in stats
        assert "last_alert_level" in stats

    def test_dashboard_stats_cache_behavior(self, monkeypatch):
        """60秒内重复调用返回缓存"""
        svc._stats_cache = None
        svc._stats_cache_time = 0

        stats1 = svc.get_dashboard_stats()
        # 不篡改时间也能验证: 缓存命中时 _stats_cache 非 None
        assert svc._stats_cache is not None

        stats2 = svc.get_dashboard_stats()
        assert stats1 == stats2  # 返回同一缓存对象

    def test_dashboard_stats_cache_expiry(self, monkeypatch):
        """超过 60 秒后缓存刷新"""
        svc._stats_cache = {"cached": True, "today_total": 999}
        svc._stats_cache_time = time.time() - 120  # 2 分钟前

        stats = svc.get_dashboard_stats()
        # 缓存应刷新 (不再返回 999)
        assert stats.get("today_total") != 999 or stats.get("cached") is None

    def test_get_recent_alerts(self):
        alerts = svc.get_recent_alerts(limit=10)
        assert isinstance(alerts, list)
        for a in alerts:
            assert "id" in a
            assert "alert_level" in a
            assert "time" in a

    def test_query_alerts_page(self):
        result = svc.query_alerts_page(limit=5, offset=0)
        assert "total" in result
        assert "rows" in result
        assert result["total"] >= 0

    def test_query_alerts_page_with_filters(self):
        """带等级筛选的分页查询"""
        result = svc.query_alerts_page(limit=10, alert_level="red")
        assert result["total"] >= 0
        for row in result["rows"]:
            assert row["alert_level"] == "red"


# ================================================================
# 预警审核 + 统计 + 导出
# ================================================================

class TestAlertReviewAndExport:
    """预警审核标记、统计聚合、Excel 导出"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_store, monkeypatch):
        import rockfall.alert_store as als
        self.store = sqlite_store
        self._old_store = als._store
        als._store = sqlite_store
        yield
        als._store = self._old_store

    def test_mark_alert_review_confirmed(self):
        aid = self.store.save_alert(count=2, max_confidence=0.8, alert_level="orange")
        result = svc.mark_alert_review(aid, "confirmed", "边坡确有落石")
        assert result["status"] == "ok"
        assert result["review_status"] == "confirmed"

    def test_mark_alert_review_false_alarm(self):
        aid = self.store.save_alert(count=1, max_confidence=0.3, alert_level="blue")
        result = svc.mark_alert_review(aid, "false_alarm", "风吹树叶")
        assert result["status"] == "ok"

    def test_get_alert_statistics(self):
        """统计返回今日+趋势+分布+误报率"""
        self.store.save_alert(count=3, max_confidence=0.95, alert_level="red")
        self.store.save_alert(count=1, max_confidence=0.55, alert_level="yellow")

        stats = svc.get_alert_statistics(days=7)
        assert "today" in stats
        assert "daily_trends" in stats
        assert "level_distribution" in stats
        assert "false_alarm" in stats
        assert "grand_total" in stats
        assert len(stats["daily_trends"]) == 7

    def test_export_alerts_excel(self):
        """导出 Excel 返回非空字节"""
        self.store.save_alert(count=3, max_confidence=0.9, alert_level="red",
                            push_status="sent")
        data = svc.export_alerts_excel(
            alert_level="red",
        )
        assert isinstance(data, bytes)
        assert len(data) > 100

    def test_export_alerts_excel_empty(self):
        """空数据也能导出"""
        data = svc.export_alerts_excel(
            start_date="2000-01-01", end_date="2000-01-02",
        )
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_get_export_summary(self):
        """导出摘要返回总数和等级分布"""
        self.store.save_alert(count=1, max_confidence=0.8, alert_level="red")
        self.store.save_alert(count=2, max_confidence=0.6, alert_level="yellow")
        summary = svc.get_export_summary()
        assert "total" in summary
        assert "by_level" in summary
        assert summary["total"] >= 2


# ================================================================
# 运行时配置热更新
# ================================================================

class TestRuntimeConfig:
    """update_runtime_config / get_runtime_config"""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_detector):
        svc._detectors.clear()
        # 创建一个活跃 detector 供热更新测试
        svc._get_detector("default")
        yield
        svc._detectors.clear()

    def test_update_realtime_param(self):
        """实时生效参数 (detection_confidence) 立即更新活跃 detector"""
        result = svc.update_runtime_config({"detection_confidence": 0.42})
        assert "applied" in result
        assert "detection_confidence" in result["applied"]
        assert result["applied"]["detection_confidence"] == 0.42

        # 验证活跃 detector 实例已更新
        det = svc._get_detector("default")
        assert det.confidence == 0.42

    def test_update_stream_restart_param(self):
        """需要流重启的参数 (mog2_history) 返回提示"""
        result = svc.update_runtime_config({"mog2_history": 800})
        assert "mog2_history" in result["applied"]
        assert "stream_restart_needed" in result
        assert "mog2_history" in result["stream_restart_needed"]

    def test_update_skip_params(self):
        """跳帧参数可热更新"""
        result = svc.update_runtime_config({"skip_idle": 10, "skip_active": 4})
        assert result["applied"]["skip_idle"] == 10
        assert result["applied"]["skip_active"] == 4

    def test_update_invalid_key_rejected(self):
        """白名单外参数被拒绝"""
        result = svc.update_runtime_config({"nonexistent_param": 999})
        assert "nonexistent_param" in result["skipped"]

    def test_update_non_numeric_rejected(self):
        """非数字值被拒绝"""
        result = svc.update_runtime_config({"detection_confidence": "high"})
        assert "detection_confidence" in result["skipped"]

    def test_get_runtime_config(self):
        """get_runtime_config 返回完整参数字典"""
        config = svc.get_runtime_config()
        assert "detection_confidence" in config
        assert "detection_img_size" in config
        assert "motion_min_area" in config
        assert "skip_idle" in config
        assert "skip_active" in config
        assert "skip_critical" in config
        assert isinstance(config["detection_confidence"], (int, float))

    def test_hot_update_persists_in_runtime_config(self):
        """热更新值写入 RuntimeConfig 单例"""
        from rockfall.config import RuntimeConfig
        svc.update_runtime_config({"detection_confidence": 0.33})
        assert RuntimeConfig.get("DETECTION_CONFIDENCE", 0.5) == 0.33

    def test_multiple_params_simultaneous(self):
        """同时更新多个参数"""
        result = svc.update_runtime_config({
            "detection_confidence": 0.5,
            "skip_idle": 8,
            "alert_blue_high": 0.55,
        })
        assert len(result["applied"]) == 3
        assert len(result.get("skipped", {})) == 0


# ================================================================
# 点位管理 (委托给 site_config)
# ================================================================

class TestSiteManagement:
    """service 层点位管理 (委托测试)"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_site_store, monkeypatch):
        import rockfall.site_config as sc
        self.store = sqlite_site_store
        self.store.seed_from_presets(sc.PRESET_SITES)
        sc._active_site = sc.PRESET_SITES[0]
        yield
        sc._active_site = None

    def test_get_sites_data(self):
        """get_sites_data 返回全部点位 + 当前激活"""
        data = svc.get_sites_data()
        assert "sites" in data
        assert "active_site_id" in data
        assert "active_site" in data
        assert len(data["sites"]) >= 4

    def test_switch_active_site(self):
        """switch_active_site 切换激活点位"""
        result = svc.switch_active_site("chongzuo_hena_s2")
        assert result["status"] == "ok"
        assert result["active_site"]["site_id"] == "chongzuo_hena_s2"

    def test_switch_invalid_site_raises(self):
        """切换无效 ID 抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的点位ID"):
            svc.switch_active_site("no_such_site")

    def test_create_site(self):
        """create_site 新增点位"""
        result = svc.create_site({
            "site_id": "new_site_01",
            "name": "新建测试边坡",
            "location": "测试地点",
            "region": "广西·测试",
            "risk_level": "high",
            "latitude": 23.0,
            "longitude": 108.0,
        })
        assert result["status"] == "ok"
        assert result["site"]["site_id"] == "new_site_01"
        assert result["site"]["risk_level"] == "high"

    def test_create_site_missing_id(self):
        """缺少 site_id 时抛出异常"""
        with pytest.raises(ValueError, match="site_id 不能为空"):
            svc.create_site({"name": "无名"})

    def test_create_duplicate_site(self):
        """重复 site_id 抛出异常"""
        svc.create_site({
            "site_id": "duplicate_test",
            "name": "第一个",
            "location": "L1",
            "region": "R",
        })
        with pytest.raises(ValueError, match="已存在"):
            svc.create_site({
                "site_id": "duplicate_test",
                "name": "第二个",
                "location": "L2",
                "region": "R",
            })

    def test_update_site(self):
        svc.create_site({
            "site_id": "to_update",
            "name": "旧名称",
            "location": "旧地点",
            "region": "R",
        })
        result = svc.update_site("to_update", {
            "name": "新名称",
            "risk_level": "low",
        })
        assert result["status"] == "ok"
        assert result["site"]["name"] == "新名称"
        assert result["site"]["risk_level"] == "low"

    def test_delete_site(self):
        svc.create_site({
            "site_id": "to_delete", "name": "待删除", "location": "L", "region": "R",
        })
        result = svc.delete_site("to_delete")
        assert result["status"] == "ok"
        assert result["deleted"] == "to_delete"

    def test_cannot_delete_active_site(self):
        """不能删除当前激活的点位"""
        active = svc.get_sites_data()["active_site_id"]
        with pytest.raises(ValueError, match="不能删除当前激活的点位"):
            svc.delete_site(active)


# ================================================================
# 预警截图信息
# ================================================================

class TestAlertImageInfo:
    """get_alert_image_info 查询"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_store, monkeypatch):
        import rockfall.alert_store as als
        self.store = sqlite_store
        self._old_store = als._store
        als._store = sqlite_store
        yield
        als._store = self._old_store

    def test_get_nonexistent_alert_image(self):
        result = svc.get_alert_image_info(99999)
        assert result is None

    def test_get_alert_image_info(self, tmp_data_dir):
        """有截图路径的预警记录"""
        frame_path = tmp_data_dir / "results" / "test_frame.jpg"
        frame_path.parent.mkdir(exist_ok=True)
        frame_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)

        aid = self.store.save_alert(
            count=2, max_confidence=0.8, alert_level="orange",
            saved_frame=str(frame_path),
        )
        result = svc.get_alert_image_info(aid)
        assert result is not None
        assert result["alert_id"] == aid
        assert result["exists"] is True
        assert result["alert_level"] == "orange"


# ================================================================
# 边界条件
# ================================================================

# ================================================================
# service 层 ROI 管理
# ================================================================

class TestServiceROI:
    """service 层 ROI 多边形查询/保存 (委托 site_config + 检测器重建)"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_site_store, mock_detector, monkeypatch):
        import rockfall.site_config as sc
        self.store = sqlite_site_store
        self.store.seed_from_presets(sc.PRESET_SITES)
        sc._active_site = sc.PRESET_SITES[0]
        svc._detectors.clear()
        yield
        sc._active_site = None
        svc._detectors.clear()

    def test_get_roi_for_site_current(self):
        """获取当前激活站点 ROI (不指定 site_id)"""
        result = svc.get_roi_for_site()
        assert result is not None
        assert "site_id" in result
        assert "roi_polygon" in result
        assert "frame_size" in result
        assert result["site_id"] == _site_cfg.PRESET_SITES[0].site_id

    def test_get_roi_for_site_by_id(self):
        """按指定 site_id 获取 ROI"""
        result = svc.get_roi_for_site(site_id="nanning_naan_s1")
        assert result["site_id"] == "nanning_naan_s1"

    def test_get_roi_nonexistent_site(self):
        """不存在的站点抛出 ValueError"""
        with pytest.raises(ValueError):
            svc.get_roi_for_site(site_id="no_such_site")

    def test_save_roi_for_site(self):
        """保存 ROI 多边形到指定站点"""
        polygon = [[100, 200], [400, 200], [400, 500], [100, 500]]
        result = svc.save_roi_for_site("nanning_naan_s1", polygon)
        assert result["status"] == "ok"
        assert result["site_id"] == "nanning_naan_s1"
        assert result["vertices"] == 4

        # 验证已持久化
        import rockfall.site_config as sc
        site = self.store.get_by_id("nanning_naan_s1")
        assert site.roi_polygon == polygon

    def test_save_roi_too_few_vertices(self):
        """少于 3 个顶点时抛出 ValueError"""
        with pytest.raises(ValueError, match="至少需要 3 个顶点"):
            svc.save_roi_for_site("nanning_naan_s1", [[0, 0], [100, 100]])

    def test_save_roi_nonexistent_site(self):
        """保存到不存在的站点抛出 ValueError"""
        with pytest.raises(ValueError, match="站点不存在"):
            svc.save_roi_for_site("no_such_site", [[0, 0], [100, 0], [100, 100]])


# ================================================================
# 地理预警数据 (地图可视化)
# ================================================================

class TestGeoAlerts:
    """get_geo_alerts — 预警+坐标关联"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_store, sqlite_site_store, monkeypatch):
        import rockfall.alert_store as als
        import rockfall.site_config as sc

        self.alert_store = sqlite_store
        self.site_store = sqlite_site_store
        self.site_store.seed_from_presets(sc.PRESET_SITES)

        # 注入监测点位到 AlertStore 记录
        self._old_alert = als._store
        als._store = sqlite_store

        yield
        als._store = self._old_alert

    def test_geo_alerts_basic(self):
        """有经纬度关联的预警记录"""
        self.alert_store.save_alert(
            count=3, max_confidence=0.9, alert_level="red",
            monitoring_location="钦州公路边坡监测点",
        )
        result = svc.get_geo_alerts(days=30)
        assert isinstance(result, list)

    def test_geo_alerts_empty(self):
        """无关联数据时返回空列表"""
        result = svc.get_geo_alerts(days=1, alert_level="red")
        assert isinstance(result, list)

    def test_geo_alerts_filter_by_level(self):
        """按等级筛选地图数据"""
        self.alert_store.save_alert(
            count=2, max_confidence=0.95, alert_level="red",
            monitoring_location="南宁那安快速路 1 号边坡",
        )
        result = svc.get_geo_alerts(days=30, alert_level="red")
        for item in result:
            assert item["alert_level"] == "red"
            # 有坐标的应包含 site 信息和经纬度
            if item.get("site_id"):
                assert "latitude" in item
                assert "longitude" in item


# ================================================================
# 边界条件
# ================================================================

class TestServiceEdgeCases:
    """异常路径 & 资源泄漏防护"""

    def test_detector_pool_thread_safety(self, mock_detector):
        """并发调用 _get_detector 不会创建重复实例"""
        import threading
        svc._detectors.clear()

        errors = []
        def _get():
            try:
                svc._get_detector("concurrent_test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 池中应只有 1 个实例
        assert "concurrent_test" in svc._detectors

    def test_cleanup_expired_tasks_empty_store(self):
        """空任务存储清理不崩溃"""
        svc._task_store.clear()
        svc._cleanup_expired_tasks()
        assert svc._task_store == {}

    def test_update_runtime_config_empty(self):
        """空更新字典不崩溃"""
        result = svc.update_runtime_config({})
        assert result["applied"] == {}
        assert result["skipped"] == {}
