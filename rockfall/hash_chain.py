"""
哈希链防篡改模块 — SHA256 链式完整性校验
=========================================
为每条预警记录计算 SHA256 摘要并链接成哈希链，
提供防篡改证据和完整性校验 API。

核心设计:
  每条的 data_hash = SHA256(time|alert_level|count|...|prev_hash)
  prev_hash = 上一条的 data_hash (首条使用创世哈希)

  任何记录被修改 → 其 data_hash 不匹配 → 后续所有记录的 prev_hash 断裂

字段拼接规范 (固定顺序, | 分隔, 确定性):
  time|alert_level|count|max_confidence|track_ids|class_summary|
  saved_frame|clip_path|rock_diameter_cm|monitoring_location|prev_hash

用法:
    from rockfall.hash_chain import compute_record_hash, verify_record, build_chain

    # 计算单条 hash
    h = compute_record_hash(fields_dict, prev_hash)

    # 验证单条
    result = verify_record(record_dict, prev_record_dict, genesis_hash)

    # 批量构建链
    hashes = build_chain(records_list, genesis_hash)
"""

import hashlib
import json
from typing import Any


# ---- 哈希计算用字段列表 (固定顺序, 不可变更) ----
_HASH_FIELDS = [
    "time",
    "alert_level",
    "count",
    "max_confidence",
    "track_ids",
    "class_summary",
    "saved_frame",
    "clip_path",
    "rock_diameter_cm",
    "monitoring_location",
]


def _serialize_field(value: Any) -> str:
    """将字段值序列化为确定性字符串。

    - track_ids: 紧凑 JSON (无空格) 以保证确定性
    - None / 空值: 空字符串
    - 数字: 保留合理精度
    """
    if value is None:
        return ""
    if isinstance(value, list):
        # 紧凑 JSON: separators=(',', ':') 无空格
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, str) and value.startswith("["):
        # 从 DB 读取的 JSON 字符串 → 解析后重新紧凑序列化
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        # 防御: 将整数转为浮点格式以避免与 DB 读取的 float 类型不匹配
        # (例如 save 时 round(0, 1) → int 0, DB 读取后 → float 0.0)
        return f"{float(value):.4f}" if value == 0 else str(value)
    return str(value)


def _extract_fields(record: dict) -> list[str]:
    """从记录 dict 中按固定顺序提取字段值并序列化。"""
    values = []
    for field in _HASH_FIELDS:
        values.append(_serialize_field(record.get(field, "")))
    return values


def compute_record_hash(fields: dict, prev_hash: str) -> str:
    """计算单条记录的 SHA256 哈希。

    参数:
        fields:    包含 _HASH_FIELDS 中所有字段的 dict
        prev_hash: 上一条记录的 data_hash (首条记录使用创世哈希)

    返回:
        64 字符小写 hex digest
    """
    parts = _extract_fields(fields)
    parts.append(prev_hash or "")
    content = "|".join(parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_record(
    record: dict,
    prev_record: dict | None,
    genesis_hash: str,
) -> dict:
    """验证单条记录的哈希完整性。

    参数:
        record:      待验证的预警记录 dict
        prev_record: 前一条记录 dict (首条为 None)
        genesis_hash: 创世哈希值

    返回:
        {
            "valid": bool,             # 总体有效
            "stored_hash": str,        # DB 中存储的 data_hash
            "computed_hash": str,      # 重新计算的 data_hash
            "prev_hash_match": bool,   # prev_hash 是否指向前一条
        }
    """
    stored_hash = record.get("data_hash", "")
    stored_prev = record.get("prev_hash", "")

    # 空 hash = 未启用哈希链的记录, 跳过验证
    if not stored_hash:
        return {
            "valid": False,
            "stored_hash": "",
            "computed_hash": "",
            "prev_hash_match": False,
            "reason": "记录未包含 data_hash (功能未启用或旧记录)",
        }

    # 确定 prev_hash 应指向的值
    if prev_record is None:
        expected_prev = genesis_hash
    else:
        expected_prev = prev_record.get("data_hash", genesis_hash)

    # 重新计算
    computed = compute_record_hash(record, expected_prev)

    # 验证
    hash_match = computed == stored_hash
    prev_match = stored_prev == expected_prev

    result = {
        "valid": hash_match and prev_match,
        "stored_hash": stored_hash,
        "computed_hash": computed,
        "prev_hash_match": prev_match,
        "hash_match": hash_match,
    }
    if not hash_match:
        result["reason"] = "data_hash 不匹配: 记录可能被篡改"
    elif not prev_match:
        result["reason"] = "prev_hash 不匹配: 哈希链在此处断裂"

    return result


def build_chain(records: list[dict], genesis_hash: str) -> list[str]:
    """为一组记录生成哈希链 (用于回填或批量处理)。

    返回: 与 records 等长的 data_hash 列表。
    """
    hashes = []
    prev = genesis_hash
    for r in records:
        h = compute_record_hash(r, prev)
        hashes.append(h)
        prev = h
    return hashes


def verify_chain_batch(
    records: list[dict],
    genesis_hash: str,
) -> dict:
    """批量验证已按 id 排序的记录列表的哈希链完整性。

    参数:
        records:      按 id 升序排列的预警记录列表
        genesis_hash: 创世哈希值

    返回:
        {
            "total": int,
            "valid": int,
            "invalid": int,
            "skipped": int,  # 无 hash 的旧记录
            "breaks": [{"id": int, "reason": str}, ...],
        }
    """
    breaks = []
    valid = 0
    invalid = 0
    skipped = 0

    for i, record in enumerate(records):
        prev_record = records[i - 1] if i > 0 else None
        result = verify_record(record, prev_record, genesis_hash)

        if result.get("reason") == "记录未包含 data_hash (功能未启用或旧记录)":
            skipped += 1
        elif result["valid"]:
            valid += 1
        else:
            invalid += 1
            breaks.append({
                "id": record.get("id", i),
                "reason": result.get("reason", "验证失败"),
                "stored_hash": result["stored_hash"],
                "computed_hash": result["computed_hash"],
            })

    return {
        "total": len(records),
        "valid": valid,
        "invalid": invalid,
        "skipped": skipped,
        "breaks": breaks,
    }
