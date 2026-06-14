"""
冷存储归档模块 — S3 / Alibaba OSS 客户端
========================================
将过期预警记录自动归档到对象存储（S3 兼容 / Alibaba OSS），
满足 ≥3 年合规保留要求，同时释放本地数据库空间。

支持的冷存储类型:
  - s3:   AWS S3 或兼容协议 (MinIO, Ceph RGW 等)
  - oss:  Alibaba Cloud OSS
  - (空):  禁用冷存储，仅本地导出

用法:
    from rockfall.cold_storage import ColdStorageClient

    client = ColdStorageClient()
    if client.enabled:
        client.upload_json("alerts_2023-01_to_2023-06.jsonl", data)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    COLD_STORAGE_TYPE,
    COLD_STORAGE_ENDPOINT,
    COLD_STORAGE_BUCKET,
    COLD_STORAGE_ACCESS_KEY,
    COLD_STORAGE_SECRET_KEY,
    COLD_STORAGE_REGION,
    COLD_STORAGE_PREFIX,
)

logger = logging.getLogger(__name__)

# 可选依赖
_boto3_available = False
_oss2_available = False

try:
    import boto3  # noqa: F401

    _boto3_available = True
except ImportError:
    pass

try:
    import oss2  # noqa: F401

    _oss2_available = True
except ImportError:
    pass


class ColdStorageClient:
    """冷存储客户端 — 将归档数据上传到 S3 或 Alibaba OSS。

    自动根据 COLD_STORAGE_TYPE 选择后端。
    所有方法失败时静默跳过（日志 WARN），不影响归档主流程的本地部分。
    """

    def __init__(self):
        self._client = None
        self._type = COLD_STORAGE_TYPE.lower() if COLD_STORAGE_TYPE else ""
        self._enabled = False

        if not self._type:
            return

        if not COLD_STORAGE_ACCESS_KEY or not COLD_STORAGE_SECRET_KEY:
            logger.warning("冷存储已配置类型 %s 但缺少 AccessKey/SecretKey，已禁用", self._type)
            return

        if self._type == "s3":
            self._init_s3()
        elif self._type == "oss":
            self._init_oss()
        else:
            logger.warning("不支持的冷存储类型: %s (支持 s3 / oss)", self._type)

    # ----------------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------------

    def _init_s3(self):
        """初始化 S3 兼容客户端 (boto3)。"""
        if not _boto3_available:
            logger.warning(
                "冷存储类型为 s3 但 boto3 未安装。"
                "请执行: pip install boto3"
            )
            return

        try:
            import boto3 as b3
            from botocore.config import Config as BotoConfig

            session = b3.Session(
                aws_access_key_id=COLD_STORAGE_ACCESS_KEY,
                aws_secret_access_key=COLD_STORAGE_SECRET_KEY,
            )
            self._client = session.client(
                "s3",
                endpoint_url=COLD_STORAGE_ENDPOINT or None,
                region_name=COLD_STORAGE_REGION or "us-east-1",
                config=BotoConfig(
                    connect_timeout=10,
                    read_timeout=30,
                    retries={"max_attempts": 2},
                ),
            )
            self._enabled = True
            logger.info("冷存储 S3 客户端已就绪: bucket=%s, endpoint=%s",
                        COLD_STORAGE_BUCKET, COLD_STORAGE_ENDPOINT or "AWS default")
        except Exception as e:
            logger.warning("S3 客户端初始化失败: %s", e)

    def _init_oss(self):
        """初始化 Alibaba OSS 客户端 (oss2)。"""
        if not _oss2_available:
            logger.warning(
                "冷存储类型为 oss 但 oss2 未安装。"
                "请执行: pip install oss2"
            )
            return

        try:
            import oss2 as o2

            auth = o2.Auth(COLD_STORAGE_ACCESS_KEY, COLD_STORAGE_SECRET_KEY)
            self._client = o2.Bucket(
                auth,
                COLD_STORAGE_ENDPOINT or "https://oss-cn-hangzhou.aliyuncs.com",
                COLD_STORAGE_BUCKET,
                connect_timeout=10,
            )
            self._enabled = True
            logger.info("冷存储 OSS 客户端已就绪: bucket=%s", COLD_STORAGE_BUCKET)
        except Exception as e:
            logger.warning("OSS 客户端初始化失败: %s", e)

    # ----------------------------------------------------------------
    # 公开属性
    # ----------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """冷存储是否可用。"""
        return self._enabled and self._client is not None

    # ----------------------------------------------------------------
    # 上传
    # ----------------------------------------------------------------

    def upload_json(self, key: str, data: list[dict]) -> bool:
        """上传 JSON Lines 格式数据到冷存储。

        参数:
            key:  对象存储的 key (路径)
            data: 记录列表，每条记录序列化为一行 JSON

        返回: True=成功, False=失败
        """
        if not self.enabled:
            return False

        full_key = f"{COLD_STORAGE_PREFIX}{key}" if COLD_STORAGE_PREFIX else key

        # 构建 JSON Lines 内容
        lines = "\n".join(
            json.dumps(r, ensure_ascii=False, default=str) for r in data
        )
        content = lines.encode("utf-8")

        try:
            if self._type == "s3":
                self._client.put_object(
                    Bucket=COLD_STORAGE_BUCKET,
                    Key=full_key,
                    Body=content,
                    ContentType="application/x-ndjson",
                )
            elif self._type == "oss":
                self._client.put_object(full_key, content)

            logger.info("冷存储上传成功: %s (%d 条记录, %.1f KB)",
                        full_key, len(data), len(content) / 1024)
            return True
        except Exception as e:
            logger.warning("冷存储上传失败 [%s]: %s", full_key, e)
            return False

    def upload_file(self, key: str, local_path: str | Path) -> bool:
        """上传本地文件到冷存储。

        参数:
            key:        对象存储的 key
            local_path: 本地文件路径

        返回: True=成功
        """
        if not self.enabled:
            return False

        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning("冷存储上传: 本地文件不存在 %s", local_path)
            return False

        full_key = f"{COLD_STORAGE_PREFIX}{key}" if COLD_STORAGE_PREFIX else key

        try:
            if self._type == "s3":
                self._client.upload_file(
                    str(local_path),
                    COLD_STORAGE_BUCKET,
                    full_key,
                )
            elif self._type == "oss":
                self._client.put_object_from_file(full_key, str(local_path))

            logger.info("冷存储文件上传成功: %s → %s", local_path, full_key)
            return True
        except Exception as e:
            logger.warning("冷存储文件上传失败 [%s]: %s", full_key, e)
            return False

    # ----------------------------------------------------------------
    # 列表与下载
    # ----------------------------------------------------------------

    def list_archives(
        self, prefix: str = "", start: str = "", end: str = ""
    ) -> list[dict]:
        """列出冷存储中的归档文件。

        参数:
            prefix: 额外前缀过滤
            start:  起始日期 (YYYY-MM-DD, 用于 key 过滤)
            end:    结束日期

        返回: [{"key": str, "size": int, "last_modified": str}, ...]
        """
        if not self.enabled:
            return []

        search_prefix = f"{COLD_STORAGE_PREFIX}{prefix}" if COLD_STORAGE_PREFIX else prefix

        try:
            result = []
            if self._type == "s3":
                paginator = self._client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=COLD_STORAGE_BUCKET,
                    Prefix=search_prefix,
                )
                for page in pages:
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        if not self._match_date_range(key, start, end):
                            continue
                        result.append({
                            "key": key,
                            "size": obj["Size"],
                            "last_modified": str(obj["LastModified"]),
                        })
            elif self._type == "oss":
                for obj in self._client.list_objects(prefix=search_prefix).object_list:
                    key = obj.key
                    if not self._match_date_range(key, start, end):
                        continue
                    result.append({
                        "key": key,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                    })

            return result
        except Exception as e:
            logger.warning("冷存储列表获取失败: %s", e)
            return []

    def download_archive(self, key: str, local_path: str | Path) -> bool:
        """从冷存储下载归档文件到本地。

        返回: True=成功
        """
        if not self.enabled:
            return False

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self._type == "s3":
                self._client.download_file(
                    COLD_STORAGE_BUCKET, key, str(local_path),
                )
            elif self._type == "oss":
                self._client.get_object_to_file(key, str(local_path))

            logger.info("冷存储下载成功: %s → %s", key, local_path)
            return True
        except Exception as e:
            logger.warning("冷存储下载失败 [%s]: %s", key, e)
            return False

    # ----------------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------------

    @staticmethod
    def _match_date_range(key: str, start: str, end: str) -> bool:
        """检查 key 是否匹配日期范围 (基于文件名中的日期)。"""
        if not start and not end:
            return True
        # key 格式: alerts-archive/alerts_2023-05-01_to_2023-06-30.jsonl
        if start and start > "":
            # 提取 key 中的日期段并比较
            if start not in key:
                return False
        return True
