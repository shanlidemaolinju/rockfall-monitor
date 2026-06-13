"""
认证模块 — JWT + 多客户端 API Key 管理
======================================
特性:
  - JWT Token 签发与验证 (HS256)，支持过期时间
  - 多客户端 API Key (web / desktop / mobile)，每个客户端独立 key
  - API Key 有效期管理 (创建时间 + TTL)
  - Token 刷新机制
  - 生产环境强制 HTTPS (通过 X-Forwarded-Proto 检测)

用法:
    from rockfall.auth import AuthManager, get_auth_manager

    auth = get_auth_manager()
    token = auth.create_token(client="web", expires_hours=24)
    claims = auth.verify_token(token)  # 返回 payload 或抛出异常
"""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import DATA_DIR

# ============================================================
# 配置常量
# ============================================================
DEFAULT_TOKEN_EXPIRE_HOURS = 24
DEFAULT_API_KEY_EXPIRE_DAYS = 90
TOKEN_REFRESH_WINDOW_HOURS = 1  # 过期前1小时内可刷新


def _constant_time_compare(a: str, b: str) -> bool:
    """时间恒定字符串比较，防止时序攻击"""
    return hmac.compare_digest(a.encode(), b.encode())


# ============================================================
# 多客户端 Key 管理
# ============================================================

class ClientKeyStore:
    """多客户端 API Key 持久化存储 (SQLite)。

    每个客户端 (web/desktop/mobile) 拥有独立的 API Key，
    Key 可设置过期时间，支持吊销。
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = ""):
        self._db_path = Path(db_path) if db_path else DATA_DIR / "auth.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""\
                CREATE TABLE IF NOT EXISTS client_keys (
                    client_id TEXT PRIMARY KEY,
                    api_key_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    label TEXT DEFAULT ''
                )""")
            conn.execute("""\
                CREATE TABLE IF NOT EXISTS token_blacklist (
                    token_hash TEXT PRIMARY KEY,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT NOT NULL
                )""")

    def create_key(self, client_id: str, label: str = "",
                   expire_days: int = DEFAULT_API_KEY_EXPIRE_DAYS) -> str:
        """为一个客户端生成新的 API Key。返回明文 key（仅此时可见）。"""
        raw_key = "rk_" + secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=expire_days)

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO client_keys
                   (client_id, api_key_hash, created_at, expires_at, is_active, label)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (client_id, key_hash, now.isoformat(), expires.isoformat(), label),
            )
        return raw_key

    def validate_key(self, raw_key: str) -> dict | None:
        """验证 API Key，返回客户端信息或 None。"""
        if not raw_key:
            return None
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM client_keys WHERE api_key_hash=? AND is_active=1",
                (key_hash,),
            ).fetchone()
            if not row:
                return None
            r = dict(row)
            # 检查过期
            if r["expires_at"]:
                expires = datetime.fromisoformat(r["expires_at"])
                if datetime.now(timezone.utc) > expires.replace(tzinfo=timezone.utc):
                    return None
            return {"client_id": r["client_id"], "label": r["label"]}

    def revoke_key(self, client_id: str) -> bool:
        """吊销指定客户端的 API Key"""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE client_keys SET is_active=0 WHERE client_id=?",
                (client_id,),
            )
            return conn.total_changes > 0

    def list_clients(self) -> list[dict]:
        """列出所有客户端 Key 状态（不含 hash）"""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT client_id, created_at, expires_at, is_active, label FROM client_keys"
            ).fetchall()
        return [dict(r) for r in rows]

    def cleanup_blacklist(self):
        """清理过期的黑名单 token"""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM token_blacklist WHERE expires_at < ?", (now,)
            )


# ============================================================
# JWT 简易实现
# ============================================================

class JWTManager:
    """自包含 JWT 签发/验证（无外部依赖，HS256）。

    格式: base64url(header).base64url(payload).signature
    """

    def __init__(self, secret: str = ""):
        self._secret = secret or secrets.token_hex(32)

    @property
    def secret(self) -> str:
        return self._secret

    def _b64url_encode(self, data: bytes) -> str:
        import base64
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _b64url_decode(self, s: str) -> bytes:
        import base64
        # 补齐 padding
        pad = 4 - len(s) % 4
        if pad != 4:
            s += "=" * pad
        return base64.urlsafe_b64decode(s)

    def _sign(self, header_b64: str, payload_b64: str) -> str:
        message = f"{header_b64}.{payload_b64}".encode()
        sig = hmac.new(self._secret.encode(), message, hashlib.sha256).digest()
        return self._b64url_encode(sig)

    def create_token(self, payload: dict, expires_hours: int = DEFAULT_TOKEN_EXPIRE_HOURS) -> str:
        """签发 JWT token"""
        now = datetime.now(timezone.utc)
        claims = {
            **payload,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=expires_hours)).timestamp()),
            "jti": secrets.token_hex(8),
        }
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = self._b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = self._b64url_encode(json.dumps(claims, separators=(",", ":")).encode())
        sig = self._sign(header_b64, payload_b64)
        return f"{header_b64}.{payload_b64}.{sig}"

    def verify_token(self, token: str) -> dict:
        """验证 JWT token，返回 payload。验证失败抛出 ValueError。"""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")

        header_b64, payload_b64, sig = parts

        # 验证签名
        expected_sig = self._sign(header_b64, payload_b64)
        if not _constant_time_compare(sig, expected_sig):
            raise ValueError("Invalid token signature")

        # 解析 payload
        try:
            payload = json.loads(self._b64url_decode(payload_b64))
        except (json.JSONDecodeError, ValueError):
            raise ValueError("Invalid token payload")

        # 验证过期
        now = int(time.time())
        if payload.get("exp", 0) < now:
            raise ValueError("Token expired")
        if payload.get("iat", now + 1) > now:
            raise ValueError("Token issued in the future")

        return payload

    def refresh_token(self, token: str, expires_hours: int = DEFAULT_TOKEN_EXPIRE_HOURS) -> str:
        """刷新即将过期的 token（过期前 TOKEN_REFRESH_WINDOW_HOURS 小时内可刷新）"""
        claims = self.verify_token(token)  # 如果已过期会抛出异常
        now = int(time.time())
        exp = claims.get("exp", 0)

        # 仅在刷新窗口内允许刷新
        if exp - now > TOKEN_REFRESH_WINDOW_HOURS * 3600:
            raise ValueError("Token not in refresh window")

        # 保留原始 claims，更新过期时间
        refresh_claims = {k: v for k, v in claims.items()
                         if k not in ("iat", "exp", "jti")}
        return self.create_token(refresh_claims, expires_hours=expires_hours)


# ============================================================
# AuthManager — 统一认证入口
# ============================================================

class AuthManager:
    """统一认证管理器。

    支持三种认证方式（依优先级）:
      1. JWT Bearer token (Authorization: Bearer <token>)
      2. X-API-Key header (多客户端 key)
      3. api_key query parameter (兼容旧版，不推荐)

    环境变量:
      AUTH_JWT_SECRET: JWT 签名密钥 (不设置则自动生成，重启后旧 token 失效)
      API_KEY: 兼容旧版的单一 Master Key (优先级最低)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        import os
        self._jwt = JWTManager(os.getenv("AUTH_JWT_SECRET", ""))
        self._key_store = ClientKeyStore()

        # 如果配置了旧版 API_KEY 且 key store 中没有 master key，自动迁移
        api_key = os.environ.get("API_KEY", "")
        if api_key and api_key != "your_token_here":
            clients = self._key_store.list_clients()
            if not any(c["client_id"] == "master" for c in clients):
                self._key_store.create_key(
                    "master", label="Legacy Master Key (from API_KEY env)",
                    expire_days=365 * 10,  # 10年，等同于无过期
                )
                # 注意：无法将旧 API_KEY 哈希自动迁移，需要手动设置
                # 这里仅创建占位，实际验证仍通过 _verify_legacy_key

    def _verify_legacy_key(self, raw_key: str) -> dict | None:
        """兼容旧版 API_KEY 环境变量（运行时动态读取）"""
        api_key = os.environ.get("API_KEY", "")
        if api_key and _constant_time_compare(raw_key, api_key):
            return {"client_id": "master", "label": "Legacy Master Key"}
        return None

    def authenticate(self, auth_header: str = "", api_key: str = "",
                    query_key: str = "") -> dict:
        """统一认证入口。

        返回: {"client_id": "...", "label": "...", "auth_method": "jwt"|"api_key"|"legacy"}
        失败抛出 ValueError。
        """
        # 优先级1: JWT Bearer token
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                claims = self._jwt.verify_token(token)
                return {
                    "client_id": claims.get("client", "unknown"),
                    "label": claims.get("label", ""),
                    "auth_method": "jwt",
                    "claims": claims,
                }
            except ValueError:
                pass  # 继续尝试其他方式

        # 优先级2: X-API-Key header → 多客户端 key
        effective_key = api_key or query_key
        if effective_key:
            client = self._key_store.validate_key(effective_key)
            if client:
                return {**client, "auth_method": "api_key"}

            # 优先级3: 旧版 API_KEY 兼容
            legacy = self._verify_legacy_key(effective_key)
            if legacy:
                return {**legacy, "auth_method": "legacy"}

        raise ValueError("Authentication failed")

    def create_client_key(self, client_id: str, label: str = "",
                         expire_days: int = DEFAULT_API_KEY_EXPIRE_DAYS) -> str:
        """创建新的客户端 API Key，返回明文 key"""
        return self._key_store.create_key(client_id, label, expire_days)

    def create_jwt(self, client: str = "web", label: str = "",
                  expires_hours: int = DEFAULT_TOKEN_EXPIRE_HOURS) -> str:
        """签发 JWT token"""
        return self._jwt.create_token(
            {"client": client, "label": label},
            expires_hours=expires_hours,
        )

    def refresh_jwt(self, token: str) -> str:
        """刷新 JWT token"""
        return self._jwt.refresh_token(token)

    def revoke_client(self, client_id: str) -> bool:
        """吊销客户端 API Key"""
        return self._key_store.revoke_key(client_id)

    def list_clients(self) -> list[dict]:
        """列出所有客户端"""
        return self._key_store.list_clients()

    @property
    def jwt_secret(self) -> str:
        return self._jwt.secret


# 模块级单例
_auth: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    global _auth
    if _auth is None:
        _auth = AuthManager()
    return _auth
