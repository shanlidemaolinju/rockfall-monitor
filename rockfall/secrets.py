"""
密钥管理模块 — 加密配置 + 外部 Secrets 支持
===========================================
特性:
  - AES-256-GCM 加密配置文件 (cryptography 库)
  - 支持 ENC: 前缀的环境变量自动解密
  - 支持文件型 Secrets (/run/secrets/, K8s Secret 挂载)
  - 日志中自动屏蔽敏感值 (密码/Token/Key)
  - HashiCorp Vault 集成接口 (预留)

用法:
    from rockfall.secrets import SecretsManager, get_secrets_manager

    sm = get_secrets_manager()
    # 生成加密密钥
    sm.generate_key()
    # 加密值
    encrypted = sm.encrypt("my_password")
    # 解密环境变量中的 ENC: 前缀值
    db_password = sm.resolve_env("MYSQL_PASSWORD", "default")

环境变量:
  SECRETS_KEY: 加密主密钥 (64 hex chars = 32 bytes for AES-256)
    不设置则尝试从 SECRETS_KEY_FILE 读取
  SECRETS_KEY_FILE: 加密主密钥文件路径 (Docker/K8s 推荐方式)
    e.g. /run/secrets/rockfall_encryption_key
"""

import os
import re
import sys
from pathlib import Path
from typing import Any


class SecretsManager:
    """密钥管理器单例。

    职责:
      1. 加解密敏感配置值 (AES-256-GCM)
      2. 解析 ENC:<base64> 前缀的环境变量
      3. 读取文件型 Secrets (K8s/Docker)
      4. 敏感值脱敏 (日志安全)
    """

    _instance = None

    def __init__(self):
        self._key: bytes | None = None
        self._key_loaded = False

    def _load_key(self):
        """加载加密主密钥。优先级:
        1. 环境变量 SECRETS_KEY (64 hex chars)
        2. 文件 SECRETS_KEY_FILE (raw bytes 或 hex)
        """
        if self._key_loaded:
            return

        key_hex = os.getenv("SECRETS_KEY", "")
        if key_hex and len(key_hex) >= 64:
            try:
                self._key = bytes.fromhex(key_hex[:64])
                self._key_loaded = True
                return
            except ValueError:
                pass

        key_file = os.getenv("SECRETS_KEY_FILE", "")
        if key_file and Path(key_file).exists():
            try:
                raw = Path(key_file).read_bytes()
                if len(raw) >= 32:
                    # 尝试 hex 解码
                    try:
                        self._key = bytes.fromhex(raw.decode().strip()[:64])
                    except (ValueError, UnicodeDecodeError):
                        self._key = raw[:32]
                    self._key_loaded = True
                    return
            except Exception:
                pass

        self._key_loaded = True  # 无可用密钥，加密功能不可用

    @property
    def is_available(self) -> bool:
        self._load_key()
        return self._key is not None

    # ============================================================
    # AES-256-GCM 加解密
    # ============================================================

    def encrypt(self, plaintext: str) -> str:
        """加密明文，返回 "ENC:<base64>" 格式的密文。"""
        self._load_key()
        if not self._key:
            raise RuntimeError("加密密钥未配置 (设置 SECRETS_KEY 或 SECRETS_KEY_FILE)")

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import os as _os

        aesgcm = AESGCM(self._key)
        nonce = _os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)

        # 格式: nonce(12) + ciphertext → base64
        import base64
        combined = nonce + ciphertext
        return "ENC:" + base64.b64encode(combined).decode()

    def decrypt(self, encrypted: str) -> str:
        """解密密文。支持 "ENC:<base64>" 格式或原始 base64。"""
        self._load_key()
        if not self._key:
            raise RuntimeError("加密密钥未配置 (设置 SECRETS_KEY 或 SECRETS_KEY_FILE)")

        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import base64

        # 去除 ENC: 前缀
        data = encrypted
        if data.startswith("ENC:"):
            data = data[4:]

        try:
            combined = base64.b64decode(data)
        except Exception:
            raise ValueError("无法解码密文 (不是有效的 base64)")

        if len(combined) < 13:  # nonce(12) + min ciphertext(1)
            raise ValueError("密文长度无效")

        nonce = combined[:12]
        ciphertext = combined[12:]

        aesgcm = AESGCM(self._key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext.decode("utf-8")
        except Exception:
            raise ValueError("解密失败 — 密钥不匹配或数据损坏")

    # ============================================================
    # 环境变量解析
    # ============================================================

    def resolve_env(self, key: str, default: str = "") -> str:
        """读取环境变量，自动解密 ENC: 前缀的值。

        也支持 _FILE 后缀：如果存在 MYSQL_PASSWORD_FILE 环境变量，
        则从该文件读取密码（K8s Secret / Docker Secret 模式）。
        """
        # 优先检查 _FILE 后缀 (K8s/Docker Secrets)
        file_env = os.getenv(f"{key}_FILE", "")
        if file_env:
            file_path = Path(file_env)
            if file_path.exists():
                return file_path.read_text(encoding="utf-8").strip()

        raw = os.getenv(key, default)
        if raw and raw.startswith("ENC:"):
            try:
                return self.decrypt(raw)
            except Exception:
                # 解密失败时返回原始值（向后兼容）
                return raw
        return raw

    # ============================================================
    # 批量敏感字段脱敏
    # ============================================================

    # 敏感环境变量名模式（用于日志脱敏）
    SENSITIVE_KEY_PATTERNS = [
        re.compile(r".*PASSWORD.*", re.IGNORECASE),
        re.compile(r".*SECRET.*", re.IGNORECASE),
        re.compile(r".*TOKEN.*", re.IGNORECASE),
        re.compile(r".*API_KEY.*", re.IGNORECASE),
        re.compile(r"CAMERA_URL", re.IGNORECASE),  # RTSP 含密码
        re.compile(r".*ENCRYPTION_KEY.*", re.IGNORECASE),
    ]

    @classmethod
    def is_sensitive_key(cls, key: str) -> bool:
        """判断环境变量名是否包含敏感信息"""
        return any(p.match(key) for p in cls.SENSITIVE_KEY_PATTERNS)

    @classmethod
    def mask_sensitive_value(cls, value: str) -> str:
        """脱敏处理单个值。

        规则:
          - 长度 ≤ 4: 完全不显示
          - 长度 > 4: 显示首2尾2字符，中间用 *** 替换
        """
        if not value:
            return "(empty)"
        if len(value) <= 4:
            return "***"
        return value[:2] + "***" + value[-2:]

    @classmethod
    def sanitize_config_for_log(cls, config_dict: dict) -> dict:
        """对配置字典中的敏感值脱敏，安全地用于日志输出。"""
        return {
            k: cls.mask_sensitive_value(str(v)) if cls.is_sensitive_key(k) else v
            for k, v in config_dict.items()
        }


# 模块级单例
_secrets: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    global _secrets
    if _secrets is None:
        _secrets = SecretsManager()
    return _secrets


# ============================================================
# 便捷函数：在 config.py 中使用
# ============================================================

def resolve_secret(key: str, default: str = "") -> str:
    """便捷函数：读取环境变量（自动处理 ENC: 和 _FILE 后缀）。"""
    return get_secrets_manager().resolve_env(key, default)


def mask_for_log(value: str) -> str:
    """便捷函数：脱敏单个值。"""
    return SecretsManager.mask_sensitive_value(value)
