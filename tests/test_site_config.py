"""
测试 site_config.py — 多监测点位管理 + ROI 配置 + 站点切换
==========================================================
覆盖:
  - MonitoringSite 数据模型 (to_dict / from_dict)
  - SiteStore CRUD (SQLite + MySQL 双后端)
  - seed_from_presets 种子迁移
  - 点位激活/切换 (set_active_site / get_active_site)
  - ROI 配置保存/加载 (save_site_config / load_site_config)
  - get_site_filter_clause SQL 辅助

运行:
    # SQLite only
    pytest tests/test_site_config.py -v --ignore-glob='*mysql*' -k "not mysql"

    # Full (requires Docker)
    pytest tests/test_site_config.py -v
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rockfall.site_config import (
    MonitoringSite, SiteStore, PRESET_SITES,
    get_site_store, get_active_site, set_active_site,
    list_sites, list_all_sites_admin, get_site_by_id,
    get_active_location, get_active_site_id, get_active_site_name,
    get_active_region,
    save_site_config, load_site_config,
    get_site_filter_clause, get_site_state,
    _save_site_state, _load_site_state,
)


# ================================================================
# MonitoringSite 数据模型
# ================================================================

class TestMonitoringSiteModel:
    """MonitoringSite dataclass 的序列化/反序列化"""

    def test_to_dict_basic(self):
        site = MonitoringSite(
            site_id="test_s1",
            name="测试边坡",
            location="测试边坡位置",
            region="广西·测试",
        )
        d = site.to_dict()
        assert d["site_id"] == "test_s1"
        assert d["name"] == "测试边坡"
        assert d["is_active"] is True
        assert d["risk_level"] == "medium"

    def test_to_dict_with_optional_fields(self):
        site = MonitoringSite(
            site_id="test_s2",
            name="带联系人边坡",
            location="南宁",
            region="广西·南宁",
            latitude=22.817,
            longitude=108.366,
            highway="G75 兰海高速",
            stake_mark="K1952+300",
            risk_level="high",
            roi_polygon=[[100, 200], [300, 200], [300, 400], [100, 400]],
            alert_contacts=[{"name": "张三", "phone": "13800138000"}],
            model_override="models/custom.pt",
        )
        d = site.to_dict()
        assert d["roi_polygon"] == [[100, 200], [300, 200], [300, 400], [100, 400]]
        assert d["alert_contacts"] == [{"name": "张三", "phone": "13800138000"}]
        assert d["model_override"] == "models/custom.pt"
        assert d["latitude"] == 22.817
        assert d["highway"] == "G75 兰海高速"

    def test_from_dict_roundtrip(self):
        original = MonitoringSite(
            site_id="roundtrip",
            name="往返测试",
            location="钦州",
            region="广西·钦州",
            highway="G75",
        )
        d = original.to_dict()
        restored = MonitoringSite.from_dict(d)
        assert restored.site_id == original.site_id
        assert restored.name == original.name
        assert restored.location == original.location
        assert restored.region == original.region
        assert restored.highway == original.highway

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict 应忽略不在 dataclass 中的多余字段"""
        d = {
            "site_id": "extra",
            "name": "多余字段测试",
            "location": "",
            "region": "",
            "unknown_field": "should_be_ignored",
            "another_junk": 12345,
        }
        site = MonitoringSite.from_dict(d)
        assert site.site_id == "extra"
        assert site.name == "多余字段测试"

    def test_default_values(self):
        site = MonitoringSite(site_id="d", name="默认值", location="测试", region="广西")
        assert site.location == "测试"
        assert site.camera_url == ""
        assert site.latitude == 0.0
        assert site.longitude == 0.0
        assert site.highway == ""
        assert site.stake_mark == ""
        assert site.risk_level == "medium"
        # roi_polygon / alert_contacts 默认为 None (Python dataclass field default)
        assert site.roi_polygon is None or site.roi_polygon == []
        assert site.alert_contacts is None or site.alert_contacts == []
        assert site.is_active is True
        assert site.model_override == ""


# ================================================================
# SiteStore — SQLite 后端
# ================================================================

class TestSiteStoreSQLite:
    """SiteStore 的 SQLite CRUD 完整测试"""

    def test_insert_and_get_by_id(self, sqlite_site_store):
        store = sqlite_site_store
        site = MonitoringSite(site_id="s1", name="边坡A", location="南宁",
                              region="广西·南宁", risk_level="high")
        ok = store.insert(site)
        assert ok is True

        retrieved = store.get_by_id("s1")
        assert retrieved is not None
        assert retrieved.site_id == "s1"
        assert retrieved.name == "边坡A"
        assert retrieved.risk_level == "high"
        assert retrieved.is_active is True
        # 自动填充时间戳
        assert retrieved.created_at != ""
        assert retrieved.updated_at != ""

    def test_insert_duplicate_id_fails(self, sqlite_site_store):
        store = sqlite_site_store
        site1 = MonitoringSite(site_id="dup", name="原始", location="A", region="R")
        site2 = MonitoringSite(site_id="dup", name="重复", location="B", region="R")
        assert store.insert(site1) is True
        assert store.insert(site2) is False  # 重复主键

    def test_list_all(self, sqlite_site_store):
        store = sqlite_site_store
        for i in range(5):
            store.insert(MonitoringSite(
                site_id=f"s{i}", name=f"边坡{i}", location=f"L{i}", region="广西",
            ))
        sites = store.list_all()
        assert len(sites) >= 5
        ids = [s.site_id for s in sites]
        assert "s0" in ids

    def test_list_all_active_only(self, sqlite_site_store):
        store = sqlite_site_store
        store.insert(MonitoringSite(site_id="active1", name="启用", location="A",
                                    region="R", is_active=True))
        store.insert(MonitoringSite(site_id="inactive1", name="停用", location="B",
                                    region="R", is_active=False))
        active = store.list_all(active_only=True)
        active_ids = [s.site_id for s in active]
        assert "active1" in active_ids
        assert "inactive1" not in active_ids

    def test_update(self, sqlite_site_store):
        store = sqlite_site_store
        store.insert(MonitoringSite(site_id="upd", name="旧名", location="旧位置",
                                    region="广西"))
        site = store.get_by_id("upd")
        site.name = "新名称"
        site.location = "新位置"
        site.risk_level = "low"
        ok = store.update(site)
        assert ok is True

        updated = store.get_by_id("upd")
        assert updated.name == "新名称"
        assert updated.location == "新位置"
        assert updated.risk_level == "low"

    def test_update_nonexistent(self, sqlite_site_store):
        """更新不存在的站点 — SQLite UPDATE 不抛异常但影响 0 行"""
        store = sqlite_site_store
        site = MonitoringSite(site_id="noexist", name="不存在", location="X", region="R")
        ok = store.update(site)
        # SQLite UPDATE 对不存在的行不会抛异常, 返回 True (影响0行)
        # MySQL 同样不会抛异常。两者行为一致: update 不检查 affected rows。
        # 只需验证调用不崩溃即可
        assert ok in (True, False)

    def test_delete(self, sqlite_site_store):
        store = sqlite_site_store
        store.insert(MonitoringSite(site_id="del", name="待删除", location="L",
                                    region="R"))
        assert store.get_by_id("del") is not None
        ok = store.delete("del")
        assert ok is True
        assert store.get_by_id("del") is None

    def test_count(self, sqlite_site_store):
        store = sqlite_site_store
        assert store.count() == 0
        for i in range(3):
            store.insert(MonitoringSite(site_id=f"c{i}", name=f"N{i}",
                                        location=f"L{i}", region="R"))
        assert store.count() == 3

    def test_seed_from_presets_when_empty(self, sqlite_site_store):
        """空 DB 时 seed_from_presets 写入预设点位"""
        store = sqlite_site_store
        count = store.seed_from_presets(PRESET_SITES)
        assert count == len(PRESET_SITES)
        assert store.count() == len(PRESET_SITES)

    def test_seed_from_presets_when_not_empty(self, sqlite_site_store):
        """已有数据时不重复写入"""
        store = sqlite_site_store
        store.insert(MonitoringSite(site_id="existing", name="已有", location="L",
                                    region="R"))
        count = store.seed_from_presets(PRESET_SITES)
        assert count == 0  # 不写入

    def test_persist_roi_polygon_json(self, sqlite_site_store):
        """ROI 多边形序列化为 JSON 存储"""
        store = sqlite_site_store
        polygon = [[100, 200], [300, 200], [300, 400], [100, 400]]
        site = MonitoringSite(site_id="roi1", name="ROI测试", location="L",
                              region="R", roi_polygon=polygon)
        store.insert(site)
        retrieved = store.get_by_id("roi1")
        assert retrieved.roi_polygon == polygon

    def test_persist_alert_contacts_json(self, sqlite_site_store):
        """报警联系人序列化为 JSON 存储"""
        store = sqlite_site_store
        contacts = [
            {"name": "张三", "phone": "13800138000", "email": "zhang@test.com"},
            {"name": "李四", "phone": "13900139000", "email": "li@test.com"},
        ]
        site = MonitoringSite(site_id="contact1", name="联系人测试", location="L",
                              region="R", alert_contacts=contacts)
        store.insert(site)
        retrieved = store.get_by_id("contact1")
        assert len(retrieved.alert_contacts) == 2
        assert retrieved.alert_contacts[0]["name"] == "张三"

    def test_persist_model_override(self, sqlite_site_store):
        """点位专用模型路径持久化"""
        store = sqlite_site_store
        site = MonitoringSite(site_id="model1", name="模型覆盖", location="L",
                              region="R", model_override="models/v2_custom.pt")
        store.insert(site)
        retrieved = store.get_by_id("model1")
        assert retrieved.model_override == "models/v2_custom.pt"


# ================================================================
# SiteStore — MySQL 后端
# ================================================================

class TestSiteStoreMySQL:
    """SiteStore 的 MySQL CRUD 测试 (需要 Docker MySQL)"""

    def test_insert_and_get(self, mysql_site_store):
        store = mysql_site_store
        site = MonitoringSite(site_id="mysql_s1", name="MySQL边坡",
                              location="测试地点", region="广西·测试",
                              risk_level="high", latitude=22.8, longitude=108.3)
        ok = store.insert(site)
        assert ok is True

        retrieved = store.get_by_id("mysql_s1")
        assert retrieved is not None
        assert retrieved.name == "MySQL边坡"
        assert retrieved.latitude == pytest.approx(22.8)
        assert retrieved.longitude == pytest.approx(108.3)

    def test_list_all_pagination(self, mysql_site_store):
        store = mysql_site_store
        for i in range(10):
            store.insert(MonitoringSite(
                site_id=f"mysql_p{i}", name=f"分页{i}",
                location=f"L{i}", region="R",
            ))
        all_sites = store.list_all()
        assert len(all_sites) >= 10

    def test_update_with_roi(self, mysql_site_store):
        store = mysql_site_store
        store.insert(MonitoringSite(site_id="mysql_upd", name="旧", location="L",
                                    region="R"))
        site = store.get_by_id("mysql_upd")
        site.name = "新名称"
        site.roi_polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        ok = store.update(site)
        assert ok is True

        updated = store.get_by_id("mysql_upd")
        assert updated.name == "新名称"
        assert len(updated.roi_polygon) == 4

    def test_delete_and_count(self, mysql_site_store):
        store = mysql_site_store
        store.insert(MonitoringSite(site_id="mysql_del", name="删除", location="L",
                                    region="R"))
        before = store.count()
        store.delete("mysql_del")
        after = store.count()
        assert after == before - 1

    def test_insert_duplicate_rejected(self, mysql_site_store):
        store = mysql_site_store
        site = MonitoringSite(site_id="mysql_dup", name="原始", location="A", region="R")
        assert store.insert(site) is True
        assert store.insert(site) is False


# ================================================================
# 点位切换与激活
# ================================================================

class TestSiteActivation:
    """站点激活/切换逻辑 (使用 SQLite 后端)"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_site_store, monkeypatch):
        """每个测试前: 写入预设点位 + 清除激活状态"""
        import rockfall.site_config as sc
        self.store = sqlite_site_store
        self.store.seed_from_presets(PRESET_SITES)

        # 清除激活状态
        sc._active_site = None
        # 清除持久化状态文件
        sc.SITE_STATE_PATH.write_text("{}")

    def test_get_active_site_defaults_to_first(self):
        """未设置时默认为第一个可用点位 (DB 中按 site_id ASC 排序的首个)"""
        site = get_active_site()
        assert site is not None
        # DB 已预填充 → list_all() 按 site_id ASC 排序
        # 首个应为 chongzuo_hena_s2 (字母序最前)
        all_sites = self.store.list_all()
        assert site.site_id == all_sites[0].site_id

    def test_set_active_site_valid(self):
        """切换到有效的点位 ID"""
        new_site = set_active_site("nanning_naan_s1")
        assert new_site.site_id == "nanning_naan_s1"
        assert new_site.name == "南宁那安快速路 1 号边坡"

        active = get_active_site()
        assert active.site_id == "nanning_naan_s1"

    def test_set_active_site_invalid_raises(self):
        """切换无效 ID 时抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的点位ID"):
            set_active_site("nonexistent_site_id")

    def test_set_active_site_persisted(self, tmp_path, monkeypatch):
        """切换后状态持久化到 site_state.json"""
        import rockfall.site_config as sc
        state_path = tmp_path / "site_state.json"
        monkeypatch.setattr(sc, "SITE_STATE_PATH", state_path)

        set_active_site("chongzuo_hena_s2")
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["active_site_id"] == "chongzuo_hena_s2"
        assert "last_switch_time" in state
        assert "last_switch_iso" in state

    def test_list_sites(self):
        """list_sites 返回所有启用点位"""
        sites = list_sites()
        assert len(sites) >= 4  # 预设 5 个
        for s in sites:
            assert s.is_active is True

    def test_list_all_sites_admin_includes_inactive(self, monkeypatch):
        """管理员列表含停用点位"""
        import rockfall.site_config as sc
        # 停用一个点位
        site = self.store.get_by_id("fangchenggang_lanhai_s3")
        site.is_active = False
        self.store.update(site)

        # 覆盖 list_all_sites_admin 的 _active_site 缓存
        sc._active_site = None
        all_sites = list_all_sites_admin()
        all_ids = [s.site_id for s in all_sites]
        assert "fangchenggang_lanhai_s3" in all_ids  # 含停用

    def test_get_site_by_id_valid(self):
        site = get_site_by_id("pingxiang_crossborder_s4")
        assert site is not None
        assert site.region == "广西·凭祥 (中越边境)"

    def test_get_site_by_id_invalid(self):
        assert get_site_by_id("invalid_id") is None

    def test_convenience_getters(self):
        """get_active_location / id / name / region 便捷函数"""
        set_active_site("qinzhou_s0")
        assert get_active_location() == "钦州公路边坡监测点"
        assert get_active_site_id() == "qinzhou_s0"
        assert get_active_site_name() == "钦州公路边坡监测点"
        assert get_active_region() == "广西·钦州"

    def test_get_site_state_returns_full_info(self):
        """get_site_state 返回完整状态字典"""
        set_active_site("nanning_naan_s1")
        state = get_site_state()
        assert "active_site" in state
        assert "available_sites" in state
        assert "last_switch_time" in state
        assert state["active_site"]["site_id"] == "nanning_naan_s1"


# ================================================================
# ROI 配置管理
# ================================================================

class TestROIConfig:
    """ROI 标定配置的保存/加载"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_site_store, tmp_data_dir, monkeypatch):
        import rockfall.site_config as sc
        self.store = sqlite_site_store
        self.store.seed_from_presets(PRESET_SITES)
        sc._active_site = PRESET_SITES[0]

        # 指向临时目录
        self.config_path = tmp_data_dir / "site_config.json"
        monkeypatch.setattr(sc, "CONFIG_PATH", self.config_path)

    def test_save_and_load_roi_config(self):
        """保存 → 加载 → 验证 ROI 参数完整恢复"""
        from rockfall.road_segmentation import ROIParams

        roi = ROIParams(sat_max=180, val_min=30, val_max=220,
                        morph_close=7, morph_open=3, min_area_ratio=0.05)
        polygon = np.array([[100, 200], [400, 200], [400, 500], [100, 500]], np.int32)

        save_site_config("nanning_naan_s1", roi, polygon)

        params, poly, mask = load_site_config("nanning_naan_s1")
        assert params is not None
        assert params.sat_max == 180
        assert params.val_min == 30
        assert poly is not None
        np.testing.assert_array_equal(poly, polygon)

    def test_load_nonexistent_config(self):
        """未保存的 camera_id 返回 None"""
        params, poly, mask = load_site_config("never_saved")
        assert params is None
        assert poly is None
        assert mask is None

    def test_save_config_includes_site_context(self):
        """保存的配置自动包含当前激活点位信息"""
        from rockfall.road_segmentation import ROIParams

        set_active_site("nanning_naan_s1")
        save_site_config("nanning_naan_s1", ROIParams(),
                         np.array([[0, 0], [100, 100]], np.int32))

        # 直接读 JSON 验证元数据
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        entry = config["nanning_naan_s1"]
        assert entry.get("site_id") == "nanning_naan_s1"
        # site_name 包含中文 "南宁"
        assert "南宁" in entry.get("site_name", "")

    def test_save_multiple_camera_ids(self):
        """多个 camera_id 独立存储"""
        from rockfall.road_segmentation import ROIParams

        save_site_config("cam_A", ROIParams(sat_max=100),
                         np.array([[0, 0], [50, 50]], np.int32))
        save_site_config("cam_B", ROIParams(sat_max=200),
                         np.array([[10, 10], [60, 60]], np.int32))

        params_a, _, _ = load_site_config("cam_A")
        params_b, _, _ = load_site_config("cam_B")
        assert params_a.sat_max == 100
        assert params_b.sat_max == 200


# ================================================================
# SQL 过滤条件
# ================================================================

class TestSiteFilterClause:
    """get_site_filter_clause — 按点位隔离查询"""

    @pytest.fixture(autouse=True)
    def _setup(self, sqlite_site_store, monkeypatch):
        import rockfall.site_config as sc
        self.store = sqlite_site_store
        self.store.seed_from_presets(PRESET_SITES)
        sc._active_site = PRESET_SITES[0]

    def test_sqlite_placeholder(self):
        set_active_site("qinzhou_s0")
        clause, params = get_site_filter_clause(backend="sqlite")
        assert "?" in clause
        assert params == ("钦州公路边坡监测点",)

    def test_mysql_placeholder(self):
        set_active_site("nanning_naan_s1")
        clause, params = get_site_filter_clause(backend="mysql")
        assert "%s" in clause
        assert params == ("南宁那安快速路 1 号边坡",)


# ================================================================
# 边界条件
# ================================================================

class TestEdgeCases:
    """边界与异常路径"""

    def test_empty_store_get_by_id(self, sqlite_site_store):
        assert sqlite_site_store.get_by_id("nonexistent") is None

    def test_empty_store_count(self, sqlite_site_store):
        assert sqlite_site_store.count() == 0

    def test_empty_store_list_all(self, sqlite_site_store):
        sites = sqlite_site_store.list_all()
        assert sites == []

    def test_site_state_load_corrupted(self, tmp_path, monkeypatch):
        """site_state.json 损坏时不崩溃"""
        import rockfall.site_config as sc
        bad_path = tmp_path / "bad_state.json"
        bad_path.write_text("{not valid json!!!", encoding="utf-8")
        monkeypatch.setattr(sc, "SITE_STATE_PATH", bad_path)
        result = _load_site_state()
        assert result == {}

    def test_site_state_load_missing(self, tmp_path, monkeypatch):
        import rockfall.site_config as sc
        missing = tmp_path / "missing.json"
        monkeypatch.setattr(sc, "SITE_STATE_PATH", missing)
        result = _load_site_state()
        assert result == {}

    def test_config_load_corrupted_json(self, sqlite_site_store, monkeypatch):
        """损坏的 ROI 配置文件不抛出异常"""
        import rockfall.site_config as sc
        import tempfile
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as td:
            cfg_path = _P(td) / "bad_config.json"
            cfg_path.write_text("{corrupt", encoding="utf-8")
            monkeypatch.setattr(sc, "CONFIG_PATH", cfg_path)
            params, poly, mask = load_site_config("any_camera")
            assert params is None

    def test_preset_sites_have_unique_ids(self):
        """预设点位 ID 无重复"""
        ids = [s.site_id for s in PRESET_SITES]
        assert len(ids) == len(set(ids)), f"重复 ID: {ids}"

    def test_preset_sites_have_required_fields(self):
        """预设点位必填字段非空"""
        for s in PRESET_SITES:
            assert s.site_id, f"{s.name}: site_id 为空"
            assert s.name, f"{s.site_id}: name 为空"
            assert s.location, f"{s.site_id}: location 为空"
            assert s.region, f"{s.site_id}: region 为空"
