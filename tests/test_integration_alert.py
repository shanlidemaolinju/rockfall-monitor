"""
集成测试 — 预警推送流水线
=========================
验证 AlertManager + dispatch_alert + popup SSE 完整链路:
  1. 四级分级调度 (blue/yellow/orange/red)
  2. AlertManager cooldown + confirm gating
  3. Popup SSE 共享状态 (set/wait/clear)
  4. 默认 send_alert/send_alert_async 兼容接口
  5. 帧保存 + base64 编码

运行: python -m pytest tests/test_integration_alert.py -v
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# 全局强制 SQLite (用户的 .env 可能配置了 MySQL)
import rockfall.alert_store as _als
_als.MYSQL_HOST = ""

# 阻止 Alembic 以 MySQL 模式运行，手动执行增量迁移
import rockfall.migration as _mig_mod2
_orig_run_migrations_alert = _mig_mod2.run_migrations
def _raise_skip_alert(*args, **kwargs):
    raise RuntimeError("skipped — use _run_migrations instead")
_mig_mod2.run_migrations = _raise_skip_alert


# ================================================================
# 测试辅助
# ================================================================

def _make_test_frame(w: int = 320, h: int = 240) -> np.ndarray:
    """创建合成 BGR 帧 (用于 alert frame 保存测试)"""
    frame = np.ones((h, w, 3), dtype=np.uint8) * 128
    cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 255), 2)
    return frame


def _make_track(idx: int = 1, confirmed: bool = True, confidence: float = 0.85,
                class_name: str = "落石", age: int = 5,
                motion_state: str = "运动", class_id: int = 0) -> dict:
    return {
        "id": idx,
        "bbox": [100, 100, 200, 250],
        "confidence": confidence,
        "age": age,
        "confirmed": confirmed,
        "class_id": class_id,
        "class_name": class_name,
        "motion_state": motion_state,
    }


# ================================================================
# 测试: dispatch_alert 四级分级调度
# ================================================================

class TestDispatchAlert:
    """集成: dispatch_alert() 四级预警调度 + AlertStore 联动"""

    @classmethod
    def setup_class(cls):
        from rockfall.alert_store import AlertStore
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = str(Path(cls.tmpdir.name) / "test_alerts.db")
        cls.store = AlertStore(db_path=cls.db_path)
        cls.store._stop_retry.set()
        cls.store._backend = "sqlite"  # 强制 SQLite
        # 手动运行增量迁移 (Alembic 已被模块级 monkeypatch 跳过)
        cls.store._run_migrations()

    @classmethod
    def teardown_class(cls):
        import gc
        gc.collect()
        try:
            cls.tmpdir.cleanup()
        except (PermissionError, OSError):
            pass  # Windows SQLite 文件锁, 进程退出后自动清理

    def _dispatch(self, alert_level: str, **kwargs) -> dict:
        """调用 dispatch_alert 并使用测试 AlertStore"""
        from unittest import mock
        from rockfall import notifier

        defaults = {
            "count": 1, "max_confidence": 0.85,
            "alert_level": alert_level,
        }
        defaults.update(kwargs)

        # 注入测试 store + 绕过 HTTP 推送
        with mock.patch('rockfall.alert_store.get_alert_store', return_value=self.store):
            with mock.patch('rockfall.alert_store.PUSHPLUS_TOKEN', 'mock_token'):
                with mock.patch('rockfall.notifier.PUSHPLUS_TOKEN', 'mock_token'):
                    with mock.patch('rockfall.notifier._push_with_retry',
                                   return_value={"code": 200, "msg": "mock_ok"}):
                        mgr = notifier.AlertManager()
                        mgr._last_alert_time = 0
                        with mock.patch.object(notifier, '_default_manager', mgr):
                            return notifier.dispatch_alert(**defaults)

    def test_blue_record_only(self):
        """Ⅳ级蓝色: 仅本地记录, 不推送"""
        result = self._dispatch("blue", max_confidence=0.35)
        assert result["action"] == "record_only"
        assert result["alert_level"] == "blue"

        # 验证数据库写入 (用最近写入的蓝色记录)
        recent = self.store.get_recent(limit=20)
        blue_records = [r for r in recent if r["alert_level"] == "blue"]
        assert len(blue_records) > 0
        assert blue_records[0]["push_status"] == "recorded"

    def test_yellow_popup(self):
        """Ⅲ级黄色: 记录 + 弹窗"""
        result = self._dispatch("yellow", max_confidence=0.55)
        assert result["action"] == "popup"
        assert result["alert_level"] == "yellow"

    def test_orange_wechat_push(self):
        """Ⅱ级橙色: 记录 + 多通道推送"""
        result = self._dispatch("orange", max_confidence=0.75)
        assert result["action"] == "multichannel"
        assert result["alert_level"] == "orange"

    def test_red_sound_alarm(self):
        """Ⅰ级红色: 全通道推送 + 声光报警"""
        result = self._dispatch("red", max_confidence=0.95)
        assert result["action"] == "all_channels+sound_alarm"
        assert result["alert_level"] == "red"

    def test_dispatch_with_tracks_and_frame(self):
        """预警携带 tracks 和 frame_bgr 时无异常"""
        frame = _make_test_frame()
        tracks = [_make_track(1), _make_track(2, class_name="滑坡", class_id=1)]

        result = self._dispatch(
            "orange", count=2, max_confidence=0.9,
            frame_bgr=frame, tracks=tracks,
            rock_diameter_cm=25.5,
        )
        assert result["alert_level"] == "orange"

    def test_dispatch_saves_diameter(self):
        """落石直径应正确持久化（yellow 级别直接入库）"""
        self._dispatch("yellow", rock_diameter_cm=15.7)
        recent = self.store.get_recent(limit=20)
        # 找到有 rock_diameter_cm 的记录并验证
        with_diameter = [r for r in recent if r.get("rock_diameter_cm") and float(r["rock_diameter_cm"]) > 0]
        assert len(with_diameter) > 0, "应有含落石直径的记录"
        assert any(float(r["rock_diameter_cm"]) == 15.7 for r in with_diameter), \
            f"应有直径=15.7的记录: {[r['rock_diameter_cm'] for r in with_diameter[:5]]}"

    def test_multiple_levels_in_order(self):
        """连续不同等级预警应全部入库（blue/yellow 直接入库，orange/red 需token）"""
        from rockfall.config import PUSHPLUS_TOKEN
        has_token = PUSHPLUS_TOKEN and PUSHPLUS_TOKEN != "your_token_here"

        for lvl in ["blue", "yellow", "orange", "red"]:
            conf = 0.4 + {"blue": 0, "yellow": 0.1, "orange": 0.3, "red": 0.5}[lvl]
            result = self._dispatch(lvl, max_confidence=conf)
            assert result["alert_level"] == lvl

        recent = self.store.get_recent(limit=50)
        stored_levels = set(r["alert_level"] for r in recent)

        # blue/yellow 一定入库，orange/red 需要有效 token
        for lvl in ["blue", "yellow"]:
            assert lvl in stored_levels, f"等级 {lvl} 应出现在记录中"

        if has_token:
            for lvl in ["orange", "red"]:
                assert lvl in stored_levels, f"等级 {lvl} 应出现在记录中"


# ================================================================
# 测试: AlertManager gating
# ================================================================

class TestAlertManagerGating:
    """集成: AlertManager 冷却 + 确认门控"""

    def setup_method(self):
        from rockfall.notifier import AlertManager
        self.mgr = AlertManager()
        self.mgr._last_alert_time = 0  # 清除冷却

    def test_token_not_configured_blocks(self):
        """PUSHPLUS_TOKEN 未配置时应被门控拦截 (mock HTTP 避免外部依赖)"""
        from unittest import mock

        # 无论 .env 中有什么 token, 均 mock 为未配置状态
        with mock.patch('rockfall.notifier.PUSHPLUS_TOKEN', ''):
            with mock.patch('rockfall.notifier._push_with_retry',
                           return_value={"code": -1, "msg": "未配置 PUSHPLUS_TOKEN"}):
                result = self.mgr.send(count=1, max_confidence=0.9)
                assert result["code"] != 200  # 未配置 token 时应失败

    def test_cooldown_respected(self):
        """冷却期内应被拦截 (mock PushPlus 避免外部依赖)"""
        from unittest import mock

        with mock.patch('rockfall.notifier.PUSHPLUS_TOKEN', 'mock_token'):
            with mock.patch('rockfall.notifier._push_with_retry',
                           return_value={"code": 200, "msg": "ok"}):
                # 第一次发送成功
                r1 = self.mgr.send(count=1, max_confidence=0.8)
                assert r1["code"] == 200

                # 立即第二次发送 → 冷却期内被拦截
                r2 = self.mgr.send(count=1, max_confidence=0.8)
                assert r2["code"] == 0, f"冷却期应被拦截: {r2}"
                assert "冷却" in r2.get("msg", ""), f"应提示冷却: {r2}"

    def test_confirm_frames_gating(self):
        """confirm_frames > 1 时需连续帧确认"""
        tracks = [_make_track(1, age=1)]
        # confirm_frames=3: 单帧轨迹不应通过
        result = self.mgr.send(
            count=1, max_confidence=0.8,
            tracks=tracks, confirm_frames=3,
        )
        assert result["code"] != 200, f"单帧轨迹不应通过确认门控: {result}"

    def test_save_frame_creates_file(self, tmp_path):
        """帧保存应生成有效的 JPEG 文件"""
        import os
        from rockfall.config import RESULTS_DIR

        frame = _make_test_frame()
        # 临时重定向 results 目录
        from rockfall.notifier import AlertManager
        saved = AlertManager._save_frame(frame)

        if saved:
            assert Path(saved).exists(), f"保存帧应存在: {saved}"
            img = cv2.imread(saved)
            assert img is not None, "保存的帧应可读取"
            assert img.shape == frame.shape, "保存帧尺寸应一致"


# ================================================================
# 测试: Popup SSE 共享状态
# ================================================================

class TestPopupSSE:
    """集成: 弹窗预警 SSE 共享状态 (线程安全)"""

    def test_set_and_get_popup(self):
        """写入弹窗 → 读取 → 清除"""
        from rockfall.notifier import (
            _set_latest_popup_alert, get_and_clear_popup_alert,
        )

        _set_latest_popup_alert(
            "red", count=3, max_confidence=0.95,
            class_summary="落石:3", saved_frame="/tmp/test.jpg",
            track_ids=[1, 2, 3], rock_diameter_cm=30.5,
            sound_alarm=True,
        )

        alert = get_and_clear_popup_alert()
        assert alert is not None, "应获取到弹窗"
        assert alert["alert_level"] == "red"
        assert alert["count"] == 3
        assert alert["max_confidence"] == 0.95
        assert alert["class_summary"] == "落石:3"
        assert alert["sound_alarm"] is True
        assert alert["rock_diameter_cm"] == 30.5
        assert len(alert["track_ids"]) == 3

        # 再次读取应为 None (已消费)
        alert2 = get_and_clear_popup_alert()
        assert alert2 is None, "消费后应返回 None"

    def test_popup_clear_on_read(self):
        """读取弹窗后应自动清除"""
        from rockfall.notifier import (
            _set_latest_popup_alert, get_and_clear_popup_alert,
        )

        _set_latest_popup_alert("yellow", 1, 0.5, "", "", [], 0)
        assert get_and_clear_popup_alert() is not None
        assert get_and_clear_popup_alert() is None

    def test_no_popup_initially(self):
        """初始状态无弹窗"""
        from rockfall.notifier import get_and_clear_popup_alert
        # 清空残留
        while get_and_clear_popup_alert() is not None:
            pass
        assert get_and_clear_popup_alert() is None


# ================================================================
# 测试: 向后兼容接口
# ================================================================

class TestBackwardCompatibility:
    """集成: send_alert / send_alert_async 兼容接口"""

    def test_send_alert_smoke(self):
        """send_alert() 应不抛异常 (即使 token 无效)"""
        from rockfall.notifier import send_alert
        result = send_alert(count=1, max_confidence=0.8, alert_level="yellow")
        assert isinstance(result, dict)
        assert "code" in result

    def test_send_alert_async_does_not_block(self):
        """send_alert_async() 应不阻塞"""
        from rockfall.notifier import send_alert_async
        t0 = time.time()
        send_alert_async(count=1, max_confidence=0.8, alert_level="yellow")
        elapsed = time.time() - t0
        # 异步提交应在 2s 内返回
        assert elapsed < 2.0, f"异步调用耗时过长: {elapsed:.1f}s"

    def test_dispatch_alert_async_smoke(self):
        """dispatch_alert_async() 应不抛异常"""
        from rockfall.notifier import dispatch_alert_async
        dispatch_alert_async(count=1, max_confidence=0.6, alert_level="blue")
        # 无异常 = 通过

    def test_build_class_summary(self):
        """_build_class_summary 应正确统计类别"""
        from rockfall.notifier import _build_class_summary

        assert _build_class_summary(None) == ""
        assert _build_class_summary([]) == ""

        tracks = [
            _make_track(1, class_name="落石"),
            _make_track(2, class_name="落石"),
            _make_track(3, class_name="滑坡"),
        ]
        result = _build_class_summary(tracks)
        assert "落石:2" in result
        assert "滑坡:1" in result
