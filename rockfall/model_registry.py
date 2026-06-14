"""
模型版本管理服务 — 下载、A/B 灰度、自动回滚
==========================================
从 S3/OSS 自动拉取新模型版本，支持 A/B 测试流量分割，
基于误报率和推理延迟指标自动回滚。

依赖:
  - ColdStorageClient (S3/OSS)
  - config.set_active_model() (原子符号链接切换)
  - health.py 自愈模式参考

用法:
    from rockfall.model_registry import ModelRegistry
    registry = ModelRegistry()
    versions = registry.check_remote_versions()
    registry.download_model("rock_best_v3.pt")
    model_path = registry.get_model_for_request(camera_id)
    registry.record_inference_metrics("rock_best_v3.pt", latency_ms=45.2)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    MODELS_DIR,
    MODEL_REGISTRY_ENABLED,
    MODEL_REGISTRY_S3_PREFIX,
    MODEL_REGISTRY_AB_SPLIT,
    MODEL_AUTO_ROLLBACK_ENABLED,
    MODEL_ROLLBACK_FP_RATE_INCREASE,
    MODEL_ROLLBACK_LATENCY_INCREASE,
    MODEL_ROLLBACK_MIN_SAMPLE,
)
from .cold_storage import ColdStorageClient
from .logger import log_event

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class ModelVersion:
    """模型版本元数据"""
    name: str                              # rock_best_v3.pt
    path: Path = field(default_factory=Path)
    sha256: str = ""                       # SHA256 校验和
    deployed_at: str = ""                  # ISO 格式时间戳
    is_stable: bool = False                # 是否为稳定版本
    is_active: bool = False                # 是否为当前激活版本
    # 运行指标 (滑动窗口)
    latency_samples: list[float] = field(default_factory=list)    # 推理耗时 (ms)
    false_alarm_count: int = 0             # 人工标记误报数
    total_inferences: int = 0              # 累计推理次数
    rollback_count: int = 0                # 被回滚次数

    @property
    def fp_rate(self) -> float:
        """误报率: 误报数 / 累计推理次数"""
        if self.total_inferences < 1:
            return 0.0
        return self.false_alarm_count / self.total_inferences

    @property
    def latency_p50_ms(self) -> float:
        """推理延迟 P50 (毫秒)"""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        mid = len(sorted_samples) // 2
        return sorted_samples[mid]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256[:16] + "..." if self.sha256 else "",
            "deployed_at": self.deployed_at,
            "is_stable": self.is_stable,
            "is_active": self.is_active,
            "fp_rate": round(self.fp_rate, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "total_inferences": self.total_inferences,
            "false_alarm_count": self.false_alarm_count,
            "rollback_count": self.rollback_count,
        }


# ══════════════════════════════════════════════════════════════
# 模型注册表
# ══════════════════════════════════════════════════════════════

class ModelRegistry:
    """模型版本管理 — 下载、A/B、回滚"""

    # 推理耗时滑动窗口大小
    LATENCY_WINDOW_SIZE = 500

    def __init__(self):
        self._manifest_path = MODELS_DIR / "manifest.json"
        self._cold_storage = ColdStorageClient()
        self._versions: dict[str, ModelVersion] = {}
        self._rollback_history: list[dict] = []
        self._lock = __import__('threading').Lock()  # 保护 _versions 并发访问
        self._load_manifest()

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return MODEL_REGISTRY_ENABLED and self._cold_storage.enabled

    @property
    def ab_split_pct(self) -> float:
        return MODEL_REGISTRY_AB_SPLIT

    @property
    def auto_rollback_enabled(self) -> bool:
        return MODEL_AUTO_ROLLBACK_ENABLED

    @property
    def active_version(self) -> ModelVersion | None:
        with self._lock:
            for v in self._versions.values():
                if v.is_active:
                    return v
        return None

    @property
    def stable_version(self) -> ModelVersion | None:
        with self._lock:
            for v in self._versions.values():
                if v.is_stable:
                    return v
        return None

    # ── 远程版本 ──────────────────────────────────────────────

    def check_remote_versions(self) -> list[dict]:
        """
        从 S3/OSS 列出可用模型版本。

        返回: [{"name": "rock_best_v3.pt", "size_mb": 45.2, "sha256": "abc...", ...}]
        """
        if not self.enabled:
            return []

        try:
            archives = self._cold_storage.list_archives(
                prefix=MODEL_REGISTRY_S3_PREFIX,
            )
        except Exception as e:
            logger.warning("远程模型列表获取失败: %s", e)
            return []

        # 过滤 .pt 文件和 .sha256 文件
        pt_files = {}
        sha256_files = {}
        for a in archives:
            key = a.get("key", "")
            name = Path(key).name
            if name.endswith(".pt"):
                pt_files[name] = a
            elif name.endswith(".sha256"):
                # 对应的 .pt 文件名
                base_name = name.replace(".sha256", "")
                sha256_files[base_name] = key

        result = []
        for name, info in pt_files.items():
            entry = {
                "name": name,
                "size_mb": round(info.get("size", 0) / (1024 ** 2), 1),
                "last_modified": info.get("last_modified", ""),
                "sha256_available": name in sha256_files,
                "is_local": name in self._versions,
            }
            result.append(entry)

        return sorted(result, key=lambda v: v["name"], reverse=True)

    def download_model(self, version_name: str) -> Path:
        """
        下载模型并 SHA256 校验，失败则清理并抛异常。

        流程:
          1. 下载到 MODELS_DIR / .downloading_{name} 临时文件
          2. 尝试下载同路径 .sha256 文件 → 计算本地 SHA256 → 对比
          3. 校验通过 → 原子 rename 到 MODELS_DIR / name
          4. 更新 manifest.json

        返回: 下载后的本地文件路径
        """
        if not self.enabled:
            raise RuntimeError("模型注册表未启用, 无法下载远程模型")

        remote_key = f"{MODEL_REGISTRY_S3_PREFIX}{version_name}"
        tmp_path = MODELS_DIR / f".downloading_{version_name}"
        final_path = MODELS_DIR / version_name

        # 1. 下载
        logger.info("开始下载模型: %s → %s", remote_key, tmp_path)
        success = self._cold_storage.download_archive(remote_key, tmp_path)
        if not success:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"模型下载失败: {remote_key}")

        # 2. SHA256 校验
        expected_sha256 = self._fetch_sha256(version_name)
        if expected_sha256:
            actual_sha256 = self._compute_sha256(tmp_path)
            if actual_sha256 != expected_sha256:
                tmp_path.unlink()
                raise RuntimeError(
                    f"SHA256 校验失败: {version_name}\n"
                    f"  期望: {expected_sha256[:16]}...\n"
                    f"  实际: {actual_sha256[:16]}..."
                )
            logger.info("SHA256 校验通过: %s", version_name)
        else:
            logger.warning("未找到 SHA256 校验文件, 跳过校验: %s", version_name)

        # 3. 原子 rename
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)
        logger.info("模型下载完成: %s", final_path)

        # 4. 更新 manifest
        version = ModelVersion(
            name=version_name,
            path=final_path,
            sha256=expected_sha256 or self._compute_sha256(final_path),
            deployed_at=datetime.now().isoformat(),
        )
        with self._lock:
            self._versions[version_name] = version
            self._save_manifest()

        log_event("model_registry", level="INFO",
                  msg=f"新模型已下载: {version_name} "
                      f"({round(final_path.stat().st_size / (1024**2), 1)}MB)")

        return final_path

    # ── 本地版本管理 ──────────────────────────────────────────

    def activate_model(self, version_name: str) -> None:
        """
        激活指定模型版本 (原子符号链接切换)。

        复用 config.set_active_model() 实现零停机切换。
        """
        from .config import set_active_model

        if version_name not in self._versions:
            raise ValueError(f"模型版本不存在: {version_name} (可用: {list(self._versions.keys())})")

        version = self._versions[version_name]
        if not version.path.exists():
            raise FileNotFoundError(f"模型文件不存在: {version.path}")

        # 原子切换
        set_active_model(version.path)

        # 更新状态 (持锁保护)
        with self._lock:
            for v in self._versions.values():
                v.is_active = False
            version.is_active = True
            version.is_stable = True
            self._save_manifest()

        log_event("model_registry", level="INFO",
                  msg=f"模型已激活: {version_name}")

    def get_model_for_request(self, request_id: str = "") -> Path | None:
        """
        A/B 分流: 基于 request_id 的哈希值决定使用稳定版还是候选版。

        同一 request_id (通常为 camera_id) 始终路由到同一模型,
        保证单路摄像头内的检测一致性。

        返回: 模型文件路径; 若注册表未启用则返回 None (调用方应回退到默认逻辑)
        """
        if not self.enabled:
            return None

        # 检查回滚
        if self.auto_rollback_enabled:
            rollback_target = self.check_rollback()
            if rollback_target is not None:
                # 回滚已由 check_rollback 执行
                pass

        stable = self.stable_version
        if stable is None or not stable.path.exists():
            return None

        # 无 A/B 分流 → 直接用稳定版
        if MODEL_REGISTRY_AB_SPLIT <= 0:
            return stable.path

        # A/B 分流: hash(request_id) % 100 < ab_split → 候选版
        # 候选版: 最近下载的非稳定版本
        candidate = self._get_candidate_version()
        if candidate is None or not candidate.path.exists():
            return stable.path

        # 确定性哈希
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest()[:8], 16)
        if hash_val % 100 < MODEL_REGISTRY_AB_SPLIT:
            return candidate.path
        return stable.path

    # ── 指标收集 ──────────────────────────────────────────────

    def record_inference_metrics(
        self, version_name: str, latency_ms: float = 0.0,
        is_false_alarm: bool = False,
    ):
        """
        记录推理指标 (供回滚判断)。

        参数:
            version_name:  模型版本名
            latency_ms:    推理耗时 (毫秒)
            is_false_alarm: 人工标记是否为误报
        """
        if version_name not in self._versions:
            return

        with self._lock:
            v = self._versions[version_name]
            v.total_inferences += 1

            if latency_ms > 0:
                v.latency_samples.append(latency_ms)
                # 滑动窗口: 只保留最近 N 条
                if len(v.latency_samples) > self.LATENCY_WINDOW_SIZE:
                    v.latency_samples = v.latency_samples[-self.LATENCY_WINDOW_SIZE:]

            if is_false_alarm:
                v.false_alarm_count += 1

    # ── 自动回滚 ──────────────────────────────────────────────

    def check_rollback(self) -> ModelVersion | None:
        """
        检查是否需要回滚 → 执行回滚并返回目标稳定版本; 不需要则返回 None。

        回滚条件 (同时满足):
          1. 审核样本 ≥ MODEL_ROLLBACK_MIN_SAMPLE
          2. 当前版本 FP 率 > 稳定版本 FP 率 × MODEL_ROLLBACK_FP_RATE_INCREASE
          3. 或 当前版本延迟 P50 > 稳定版本延迟 P50 × MODEL_ROLLBACK_LATENCY_INCREASE

        无审核数据时自动返回 None (不自动回滚)。
        """
        if not self.auto_rollback_enabled:
            return None

        current = self.active_version
        stable = self.stable_version

        if current is None or stable is None:
            return None
        if current.name == stable.name:
            return None  # 当前就是稳定版, 无需回滚

        # 样本不足 → 不自动回滚
        if current.total_inferences < MODEL_ROLLBACK_MIN_SAMPLE:
            return None
        if stable.total_inferences < MODEL_ROLLBACK_MIN_SAMPLE:
            return None

        should_rollback = False
        reason = ""

        # FP 率检查
        if stable.fp_rate > 0 and current.fp_rate > stable.fp_rate * MODEL_ROLLBACK_FP_RATE_INCREASE:
            should_rollback = True
            reason = (f"FP_rate 异常: current={current.fp_rate:.3f} "
                      f"vs baseline={stable.fp_rate:.3f} "
                      f"(阈值 ×{MODEL_ROLLBACK_FP_RATE_INCREASE})")

        # 延迟检查
        if stable.latency_p50_ms > 0 and current.latency_p50_ms > stable.latency_p50_ms * MODEL_ROLLBACK_LATENCY_INCREASE:
            should_rollback = True
            latency_reason = (f"latency 异常: current={current.latency_p50_ms:.0f}ms "
                             f"vs baseline={stable.latency_p50_ms:.0f}ms "
                             f"(阈值 ×{MODEL_ROLLBACK_LATENCY_INCREASE})")
            reason = reason + "; " + latency_reason if reason else latency_reason

        if not should_rollback:
            return None

        # 执行回滚
        logger.warning("触发自动回滚: %s → %s (%s)", current.name, stable.name, reason)
        log_event("model_rollback", level="WARN",
                  msg=f"模型自动回滚: {current.name} → {stable.name}",
                  reason=reason,
                  current_fp=round(current.fp_rate, 4),
                  baseline_fp=round(stable.fp_rate, 4),
                  current_latency_p50=round(current.latency_p50_ms, 1),
                  baseline_latency_p50=round(stable.latency_p50_ms, 1))

        try:
            self.activate_model(stable.name)
        except Exception as e:
            logger.error("回滚激活失败: %s", e)
            log_event("model_rollback", level="ERROR",
                      msg=f"模型回滚失败: {e}")
            return None

        # 记录回滚历史
        current.rollback_count += 1
        self._rollback_history.append({
            "time": datetime.now().isoformat(),
            "from": current.name,
            "to": stable.name,
            "reason": reason,
        })
        # 保留最近 50 条
        if len(self._rollback_history) > 50:
            self._rollback_history = self._rollback_history[-50:]

        self._save_manifest()

        # 通过 push_channels 发送运维告警 (异步, 等级 orange)
        try:
            from .notifier import _push_via_registry
            title = f"模型自动回滚: {current.name} → {stable.name}"
            content = (
                f"原因: {reason}\n"
                f"当前 FP 率: {current.fp_rate:.3f} (基线: {stable.fp_rate:.3f})\n"
                f"当前延迟 P50: {current.latency_p50_ms:.0f}ms (基线: {stable.latency_p50_ms:.0f}ms)\n"
                f"已自动回滚到稳定版本 {stable.name}，请运维人员确认。"
            )
            # 在线程池中异步发送, 不阻塞回滚流程
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollback-notify") as ex:
                ex.submit(_push_via_registry, title, content, "orange")
        except Exception:
            pass

        return stable

    # ── 查询 ──────────────────────────────────────────────────

    def get_status(self) -> dict:
        """获取注册表完整状态 (供管理 API)。"""
        with self._lock:
            versions_data = [v.to_dict() for v in self._versions.values()]
        return {
            "enabled": self.enabled,
            "ab_split_pct": self.ab_split_pct,
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "remote_enabled": self._cold_storage.enabled,
            "versions": versions_data,
            "active_version": self.active_version.name if self.active_version else None,
            "stable_version": self.stable_version.name if self.stable_version else None,
            "rollback_history": self._rollback_history[-10:],
            "manifest_path": str(self._manifest_path),
        }

    def get_rollback_history(self, limit: int = 20) -> list[dict]:
        """获取回滚历史"""
        return list(reversed(self._rollback_history[-limit:]))

    # ── 内部: 持久化 ──────────────────────────────────────────

    def _load_manifest(self):
        """从 MODELS_DIR/manifest.json 加载本地版本元数据"""
        if not self._manifest_path.exists():
            # 尝试从现有模型文件同步
            self._sync_from_local_files()
            return

        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            for v_data in data.get("versions", []):
                v = ModelVersion(
                    name=v_data["name"],
                    path=Path(v_data["path"]),
                    sha256=v_data.get("sha256", ""),
                    deployed_at=v_data.get("deployed_at", ""),
                    is_stable=v_data.get("is_stable", False),
                    is_active=v_data.get("is_active", False),
                    false_alarm_count=v_data.get("false_alarm_count", 0),
                    total_inferences=v_data.get("total_inferences", 0),
                    rollback_count=v_data.get("rollback_count", 0),
                )
                self._versions[v.name] = v
            self._rollback_history = data.get("rollback_history", [])
            logger.info("模型清单已加载: %d 个版本", len(self._versions))
        except Exception as e:
            logger.warning("模型清单加载失败: %s, 从本地文件同步", e)
            self._sync_from_local_files()

    def _sync_from_local_files(self):
        """从 MODELS_DIR/ 中现有的 .pt 文件恢复版本列表"""
        from .config import list_model_versions
        existing = list_model_versions()
        for m in existing:
            name = m["name"]
            self._versions[name] = ModelVersion(
                name=name,
                path=Path(m["path"]),
                deployed_at=m.get("modified", ""),
                is_stable=m.get("is_active", False),
                is_active=m.get("is_active", False),
            )
        if self._versions:
            self._save_manifest()
            logger.info("从本地文件同步了 %d 个模型版本", len(self._versions))

    def _save_manifest(self):
        """保存 manifest.json"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "versions": [
                {
                    "name": v.name,
                    "path": str(v.path),
                    "sha256": v.sha256,
                    "deployed_at": v.deployed_at,
                    "is_stable": v.is_stable,
                    "is_active": v.is_active,
                    "false_alarm_count": v.false_alarm_count,
                    "total_inferences": v.total_inferences,
                    "rollback_count": v.rollback_count,
                }
                for v in self._versions.values()
            ],
            "rollback_history": self._rollback_history,
        }
        self._manifest_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 内部: 辅助 ────────────────────────────────────────────

    def _get_candidate_version(self) -> ModelVersion | None:
        """获取 A/B 测试的候选版本 (最近下载的非稳定版本)"""
        with self._lock:
            candidates = [
                v for v in self._versions.values()
                if not v.is_stable and v.path.exists()
            ]
        if not candidates:
            return None
        # 按部署时间降序, 取最新的
        candidates.sort(key=lambda v: v.deployed_at, reverse=True)
        return candidates[0]

    def _fetch_sha256(self, version_name: str) -> str:
        """尝试从 S3/OSS 下载对应 .sha256 文件并返回内容"""
        sha256_key = f"{MODEL_REGISTRY_S3_PREFIX}{version_name}.sha256"
        tmp_sha = MODELS_DIR / f".tmp_{version_name}.sha256"
        try:
            success = self._cold_storage.download_archive(sha256_key, tmp_sha)
            if success and tmp_sha.exists():
                content = tmp_sha.read_text(encoding="utf-8").strip()
                tmp_sha.unlink()
                # 格式: "abc123def456  rock_best_v3.pt" 或 纯 hex
                if " " in content:
                    return content.split()[0].lower()
                return content.lower()
        except Exception:
            pass
        finally:
            if tmp_sha.exists():
                try:
                    tmp_sha.unlink()
                except Exception:
                    pass
        return ""

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """计算文件 SHA256 哈希 (分块读取, 支持大文件)"""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()


# ══════════════════════════════════════════════════════════════
# 模块级单例
# ══════════════════════════════════════════════════════════════

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """获取或创建 ModelRegistry 单例。首次调用时初始化。"""
    global _registry
    if _registry is not None:
        return _registry
    _registry = ModelRegistry()
    return _registry
