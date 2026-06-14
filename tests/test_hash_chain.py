"""测试 hash_chain.py — SHA256 哈希链计算与验证"""

import pytest


# ---- 单元测试: hash_chain 核心函数 ----

class TestHashChainCore:
    """hash_chain 模块纯函数测试 (无 DB 依赖)"""

    def test_compute_record_hash_deterministic(self):
        """相同输入 → 相同 hash"""
        from rockfall.hash_chain import compute_record_hash

        fields = {
            "time": "2026-06-14 12:00:00",
            "alert_level": "yellow",
            "count": 3,
            "max_confidence": 0.8500,
            "track_ids": [1, 2],
            "class_summary": "落石:3",
            "saved_frame": "/tmp/test.jpg",
            "clip_path": "",
            "rock_diameter_cm": 15.0,
            "monitoring_location": "测试点",
        }
        prev = "0" * 64

        h1 = compute_record_hash(fields, prev)
        h2 = compute_record_hash(fields, prev)

        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_compute_record_hash_differs_on_field_change(self):
        """字段变化 → hash 不同"""
        from rockfall.hash_chain import compute_record_hash

        fields = {
            "time": "2026-06-14 12:00:00",
            "alert_level": "yellow",
            "count": 3,
            "max_confidence": 0.85,
            "track_ids": [1],
            "class_summary": "落石:3",
            "saved_frame": "",
            "clip_path": "",
            "rock_diameter_cm": 10.0,
            "monitoring_location": "",
        }
        prev = "0" * 64

        h1 = compute_record_hash(fields, prev)

        # 修改 count
        fields["count"] = 5
        h2 = compute_record_hash(fields, prev)

        assert h1 != h2

    def test_compute_record_hash_differs_on_prev_hash(self):
        """prev_hash 变化 → hash 不同 (形成链条)"""
        from rockfall.hash_chain import compute_record_hash

        fields = {
            "time": "2026-06-14 12:00:00",
            "alert_level": "red",
            "count": 1,
            "max_confidence": 0.95,
            "track_ids": [],
            "class_summary": "",
            "saved_frame": "",
            "clip_path": "",
            "rock_diameter_cm": 30.0,
            "monitoring_location": "",
        }

        h1 = compute_record_hash(fields, "a" * 64)
        h2 = compute_record_hash(fields, "b" * 64)

        assert h1 != h2

    def test_track_ids_deterministic_serialization(self):
        """track_ids 紧凑 JSON 序列化保证确定性"""
        from rockfall.hash_chain import compute_record_hash

        fields = {
            "time": "2026-06-14 12:00:00",
            "alert_level": "blue",
            "count": 2,
            "max_confidence": 0.45,
            "track_ids": [3, 1, 2],
            "class_summary": "",
            "saved_frame": "",
            "clip_path": "",
            "rock_diameter_cm": 5.0,
            "monitoring_location": "",
        }
        prev = "0" * 64

        h1 = compute_record_hash(fields, prev)
        # 再次计算应相同 (track_ids 顺序固定)
        h2 = compute_record_hash(fields, prev)

        assert h1 == h2

    def test_verify_record_with_genesis(self):
        """首条记录使用创世哈希验证"""
        from rockfall.hash_chain import compute_record_hash, verify_record

        genesis = "0" * 64
        fields = {
            "time": "2026-06-14 12:00:00",
            "alert_level": "yellow",
            "count": 1,
            "max_confidence": 0.7,
            "track_ids": [],
            "class_summary": "",
            "saved_frame": "",
            "clip_path": "",
            "rock_diameter_cm": 12.0,
            "monitoring_location": "",
        }
        h = compute_record_hash(fields, genesis)

        record = {
            "id": 1,
            "data_hash": h,
            "prev_hash": genesis,
            **fields,
        }

        result = verify_record(record, None, genesis)
        assert result["valid"]
        assert result["prev_hash_match"]
        assert result["hash_match"]

    def test_verify_record_detects_tampering(self):
        """检测字段篡改"""
        from rockfall.hash_chain import compute_record_hash, verify_record

        genesis = "0" * 64
        fields = {
            "time": "2026-06-14 12:00:00",
            "alert_level": "yellow",
            "count": 3,
            "max_confidence": 0.7,
            "track_ids": [],
            "class_summary": "",
            "saved_frame": "",
            "clip_path": "",
            "rock_diameter_cm": 15.0,
            "monitoring_location": "",
        }
        h = compute_record_hash(fields, genesis)

        record = {
            "id": 1,
            "data_hash": h,
            "prev_hash": genesis,
            **fields,
        }

        # 篡改 count
        tampered = dict(record)
        tampered["count"] = 999

        result = verify_record(tampered, None, genesis)
        assert not result["valid"]
        assert not result["hash_match"]

    def test_verify_record_empty_hash_skipped(self):
        """无 hash 的记录应报告跳过"""
        from rockfall.hash_chain import verify_record

        genesis = "0" * 64
        record = {
            "id": 1,
            "data_hash": "",
            "prev_hash": "",
            "time": "2026-06-14 12:00:00",
        }
        result = verify_record(record, None, genesis)
        assert not result["valid"]
        assert "未包含 data_hash" in result.get("reason", "")

    def test_build_chain(self):
        """测试批量构建哈希链"""
        from rockfall.hash_chain import build_chain, compute_record_hash

        genesis = "0" * 64
        records = [
            {
                "time": "2026-06-14 12:00:00",
                "alert_level": "yellow",
                "count": 1,
                "max_confidence": 0.5,
                "track_ids": [],
                "class_summary": "",
                "saved_frame": "",
                "clip_path": "",
                "rock_diameter_cm": 10.0,
                "monitoring_location": "",
            },
            {
                "time": "2026-06-14 12:01:00",
                "alert_level": "red",
                "count": 2,
                "max_confidence": 0.9,
                "track_ids": [1],
                "class_summary": "落石:2",
                "saved_frame": "/tmp/frame.jpg",
                "clip_path": "",
                "rock_diameter_cm": 25.0,
                "monitoring_location": "",
            },
        ]

        hashes = build_chain(records, genesis)
        assert len(hashes) == 2
        assert len(hashes[0]) == 64
        assert hashes[0] != hashes[1]

        # 验证链式连接
        assert compute_record_hash(records[0], genesis) == hashes[0]
        assert compute_record_hash(records[1], hashes[0]) == hashes[1]

    def test_verify_chain_batch(self):
        """测试批量验证"""
        from rockfall.hash_chain import (
            build_chain, verify_chain_batch,
        )

        genesis = "0" * 64
        records = [
            {"id": 1, "time": "2026-06-14 12:00:00", "alert_level": "yellow",
             "count": 1, "max_confidence": 0.5, "track_ids": [],
             "class_summary": "", "saved_frame": "", "clip_path": "",
             "rock_diameter_cm": 10.0, "monitoring_location": ""},
            {"id": 2, "time": "2026-06-14 12:01:00", "alert_level": "red",
             "count": 2, "max_confidence": 0.9, "track_ids": [1],
             "class_summary": "", "saved_frame": "", "clip_path": "",
             "rock_diameter_cm": 25.0, "monitoring_location": ""},
            {"id": 3, "time": "2026-06-14 12:02:00", "alert_level": "orange",
             "count": 1, "max_confidence": 0.8, "track_ids": [],
             "class_summary": "", "saved_frame": "", "clip_path": "",
             "rock_diameter_cm": 18.0, "monitoring_location": ""},
        ]

        hashes = build_chain(records, genesis)
        for i, r in enumerate(records):
            r["data_hash"] = hashes[i]
            r["prev_hash"] = genesis if i == 0 else hashes[i - 1]

        result = verify_chain_batch(records, genesis)
        assert result["total"] == 3
        assert result["valid"] == 3
        assert result["invalid"] == 0
        assert result["skipped"] == 0
        assert len(result["breaks"]) == 0

    def test_verify_chain_batch_detects_break(self):
        """批量验证检测到篡改"""
        from rockfall.hash_chain import build_chain, verify_chain_batch

        genesis = "0" * 64
        records = [
            {"id": 1, "time": "2026-06-14 12:00:00", "alert_level": "yellow",
             "count": 1, "max_confidence": 0.5, "track_ids": [],
             "class_summary": "", "saved_frame": "", "clip_path": "",
             "rock_diameter_cm": 10.0, "monitoring_location": ""},
            {"id": 2, "time": "2026-06-14 12:01:00", "alert_level": "red",
             "count": 2, "max_confidence": 0.9, "track_ids": [],
             "class_summary": "", "saved_frame": "", "clip_path": "",
             "rock_diameter_cm": 25.0, "monitoring_location": ""},
        ]

        hashes = build_chain(records, genesis)
        for i, r in enumerate(records):
            r["data_hash"] = hashes[i]
            r["prev_hash"] = genesis if i == 0 else hashes[i - 1]

        # 篡改中间记录
        records[0]["count"] = 999

        result = verify_chain_batch(records, genesis)
        assert result["invalid"] >= 1
        assert len(result["breaks"]) >= 1
        # 第一条应报告断裂
        assert result["breaks"][0]["id"] == 1


# ---- 集成测试: AlertStore + 哈希链 ----

class TestAlertStoreHashChain:
    """AlertStore 哈希链集成测试 (SQLite 后端)"""

    def test_hash_chain_enabled_saves_hash(self, sqlite_store, monkeypatch):
        """开启哈希链后 save_alert 自动生成 data_hash 和 prev_hash"""
        import rockfall.alert_store as als
        monkeypatch.setattr(als, "ALERT_HASH_CHAIN_ENABLED", True)
        monkeypatch.setattr(als, "ALERT_HASH_GENESIS", "0" * 64)

        aid = sqlite_store.save_alert(
            count=3, max_confidence=0.85, alert_level="yellow",
        )
        assert aid > 0

        recent = sqlite_store.get_recent(1)[0]
        assert recent["data_hash"] != ""
        assert len(recent["data_hash"]) == 64
        # 首条记录 prev_hash 应为 genesis
        assert recent["prev_hash"] == "0" * 64

    def test_hash_chain_links_consecutive_records(self, sqlite_store, monkeypatch):
        """连续记录的哈希形成链条"""
        import rockfall.alert_store as als
        monkeypatch.setattr(als, "ALERT_HASH_CHAIN_ENABLED", True)
        monkeypatch.setattr(als, "ALERT_HASH_GENESIS", "0" * 64)

        aid1 = sqlite_store.save_alert(count=1, max_confidence=0.5, alert_level="blue")
        aid2 = sqlite_store.save_alert(count=2, max_confidence=0.9, alert_level="red")

        r1 = sqlite_store._get_record_by_id(aid1)
        r2 = sqlite_store._get_record_by_id(aid2)

        # 第二条的 prev_hash 应等于第一条的 data_hash
        assert r2["prev_hash"] == r1["data_hash"]
        assert r1["prev_hash"] == "0" * 64  # 首条 = genesis

    def test_verify_alert_passes(self, sqlite_store, monkeypatch):
        """验证通过的记录"""
        import rockfall.alert_store as als
        monkeypatch.setattr(als, "ALERT_HASH_CHAIN_ENABLED", True)
        monkeypatch.setattr(als, "ALERT_HASH_GENESIS", "0" * 64)

        aid = sqlite_store.save_alert(count=1, max_confidence=0.7, alert_level="yellow")

        result = sqlite_store.verify_alert(aid)
        assert result["valid"]

    def test_verify_alert_chain(self, sqlite_store, monkeypatch):
        """连续记录的链式验证"""
        import rockfall.alert_store as als
        monkeypatch.setattr(als, "ALERT_HASH_CHAIN_ENABLED", True)
        monkeypatch.setattr(als, "ALERT_HASH_GENESIS", "0" * 64)

        a1 = sqlite_store.save_alert(count=1, max_confidence=0.5, alert_level="blue")
        a2 = sqlite_store.save_alert(count=2, max_confidence=0.8, alert_level="orange")
        a3 = sqlite_store.save_alert(count=1, max_confidence=0.95, alert_level="red")

        chain_result = sqlite_store.verify_chain(a1, a3)
        assert chain_result["total"] >= 3
        assert chain_result["invalid"] == 0

    def test_hash_chain_disabled_no_overhead(self, sqlite_store, monkeypatch):
        """关闭哈希链时不产生额外开销 (hash 为空)"""
        import rockfall.alert_store as als
        monkeypatch.setattr(als, "ALERT_HASH_CHAIN_ENABLED", False)

        aid = sqlite_store.save_alert(count=1, max_confidence=0.5)
        recent = sqlite_store.get_recent(1)[0]
        assert recent["data_hash"] == ""
        assert recent["prev_hash"] == ""

    def test_verify_nonexistent_alert(self, sqlite_store):
        """验证不存在的记录"""
        result = sqlite_store.verify_alert(99999)
        assert not result["valid"]
        assert "不存在" in result.get("msg", "")

    def test_get_latest_hash(self, sqlite_store, monkeypatch):
        """获取最新 hash"""
        import rockfall.alert_store as als
        monkeypatch.setattr(als, "ALERT_HASH_CHAIN_ENABLED", True)
        monkeypatch.setattr(als, "ALERT_HASH_GENESIS", "0" * 64)

        assert sqlite_store.get_latest_hash() is None

        sqlite_store.save_alert(count=1, max_confidence=0.5)
        h = sqlite_store.get_latest_hash()
        assert h is not None
        assert len(h) == 64
