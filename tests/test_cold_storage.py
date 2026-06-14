"""测试 cold_storage.py — 冷存储客户端 (S3/OSS)"""

import json
import sys
from unittest.mock import MagicMock

import pytest


class TestColdStorageClient:
    """ColdStorageClient 单元测试 (mock boto3)"""

    @pytest.fixture
    def mock_boto3_module(self):
        """在 sys.modules 中注入 mock boto3"""
        mock_b3 = MagicMock()
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock_b3.Session.return_value = mock_session
        # 注入 botocore.config (boto3 的依赖)
        mock_botocore = MagicMock()
        mock_botocore_config = MagicMock()

        old_boto3 = sys.modules.get("boto3")
        old_botocore = sys.modules.get("botocore")
        old_botocore_config = sys.modules.get("botocore.config")
        sys.modules["boto3"] = mock_b3
        sys.modules["botocore"] = mock_botocore
        sys.modules["botocore.config"] = mock_botocore_config

        yield mock_client

        # 恢复
        if old_boto3:
            sys.modules["boto3"] = old_boto3
        else:
            sys.modules.pop("boto3", None)
        if old_botocore:
            sys.modules["botocore"] = old_botocore
        else:
            sys.modules.pop("botocore", None)
        if old_botocore_config:
            sys.modules["botocore.config"] = old_botocore_config
        else:
            sys.modules.pop("botocore.config", None)

    @pytest.fixture
    def mock_s3_env(self, monkeypatch, mock_boto3_module):
        """设置 S3 环境配置"""
        import rockfall.cold_storage as cs

        monkeypatch.setattr(cs, "COLD_STORAGE_TYPE", "s3")
        monkeypatch.setattr(cs, "COLD_STORAGE_ACCESS_KEY", "test-key")
        monkeypatch.setattr(cs, "COLD_STORAGE_SECRET_KEY", "test-secret")
        monkeypatch.setattr(cs, "COLD_STORAGE_BUCKET", "test-bucket")
        monkeypatch.setattr(cs, "COLD_STORAGE_ENDPOINT", "http://minio:9000")
        monkeypatch.setattr(cs, "COLD_STORAGE_REGION", "us-east-1")
        monkeypatch.setattr(cs, "COLD_STORAGE_PREFIX", "test-prefix/")
        monkeypatch.setattr(cs, "_boto3_available", True)

    def test_disabled_when_type_empty(self, monkeypatch):
        """COLD_STORAGE_TYPE 为空时客户端禁用"""
        import rockfall.cold_storage as cs
        monkeypatch.setattr(cs, "COLD_STORAGE_TYPE", "")
        monkeypatch.setattr(cs, "COLD_STORAGE_ACCESS_KEY", "")
        monkeypatch.setattr(cs, "COLD_STORAGE_SECRET_KEY", "")

        client = cs.ColdStorageClient()
        assert not client.enabled

    def test_disabled_when_no_credentials(self, monkeypatch):
        """缺少凭证时客户端禁用"""
        import rockfall.cold_storage as cs
        monkeypatch.setattr(cs, "COLD_STORAGE_TYPE", "s3")
        monkeypatch.setattr(cs, "COLD_STORAGE_ACCESS_KEY", "")
        monkeypatch.setattr(cs, "COLD_STORAGE_SECRET_KEY", "")

        client = cs.ColdStorageClient()
        assert not client.enabled

    def test_s3_client_initializes(self, mock_s3_env):
        """S3 客户端正确初始化"""
        import rockfall.cold_storage as cs
        client = cs.ColdStorageClient()
        assert client.enabled

    def test_upload_json_success(self, mock_s3_env):
        """JSON 上传成功"""
        import rockfall.cold_storage as cs
        client = cs.ColdStorageClient()
        assert client.enabled

        data = [{"id": 1, "time": "2026-01-01", "alert_level": "yellow"}]
        ok = client.upload_json("test.jsonl", data)
        assert ok
        client._client.put_object.assert_called_once()

    def test_upload_json_disabled(self, monkeypatch):
        """客户端禁用时上传返回 False"""
        import rockfall.cold_storage as cs
        monkeypatch.setattr(cs, "COLD_STORAGE_TYPE", "")
        monkeypatch.setattr(cs, "COLD_STORAGE_ACCESS_KEY", "")

        client = cs.ColdStorageClient()
        assert not client.enabled
        assert not client.upload_json("test.jsonl", [])

    def test_upload_json_failure_graceful(self, mock_s3_env):
        """上传失败时不抛异常"""
        import rockfall.cold_storage as cs
        client = cs.ColdStorageClient()
        assert client.enabled

        client._client.put_object.side_effect = Exception("Network error")

        data = [{"id": 1}]
        ok = client.upload_json("test.jsonl", data)
        assert not ok

    def test_upload_file_success(self, mock_s3_env, tmp_path):
        """文件上传成功"""
        import rockfall.cold_storage as cs
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"id":1}\n{"id":2}\n')

        client = cs.ColdStorageClient()
        assert client.enabled
        ok = client.upload_file("remote/test.jsonl", str(test_file))
        assert ok

    def test_list_archives_empty(self, mock_s3_env):
        """空归档列表"""
        import rockfall.cold_storage as cs
        client = cs.ColdStorageClient()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": []}]
        client._client.get_paginator.return_value = mock_paginator

        archives = client.list_archives()
        assert archives == []
