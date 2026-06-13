"""
通用 Webhook 推送通道
======================
支持任意 JSON Webhook，可适配各种第三方平台。

环境变量:
  WEBHOOK_URL          — Webhook 地址 (必填)
  WEBHOOK_SECRET       — 签名密钥 (可选，用于 HMAC-SHA256 签名)
  WEBHOOK_CUSTOM_HEADERS — 自定义请求头 JSON (可选)
     例如: '{"X-Custom": "value"}'

若配置了 WEBHOOK_URL，此通道可用于对接未内置支持的系统。
"""

import hashlib
import hmac
import json
import os
import time

import requests

from .base import PushChannel, PushResult


class WebhookChannel(PushChannel):
    name = "webhook"
    display_name = "通用 Webhook"

    def __init__(self):
        self._url = os.getenv("WEBHOOK_URL", "")
        self._secret = os.getenv("WEBHOOK_SECRET", "")
        self._custom_headers = self._parse_headers()

    @staticmethod
    def _parse_headers() -> dict:
        raw = os.getenv("WEBHOOK_CUSTOM_HEADERS", "")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def validate_config(self) -> bool:
        return bool(self._url)

    def _sign_body(self, body: str) -> str:
        """HMAC-SHA256 签名"""
        if not self._secret:
            return ""
        return hmac.new(
            self._secret.encode(), body.encode(), hashlib.sha256
        ).hexdigest()

    def send(self, title: str, content: str,
             alert_level: str = "yellow") -> PushResult:
        if not self.validate_config():
            return PushResult(False, self.name,
                              "WEBHOOK_URL 未配置", code=-1)

        payload = {
            "title": title,
            "content": content,
            "alert_level": alert_level,
            "timestamp": int(time.time()),
            "source": "RockGuard",
        }

        body = json.dumps(payload, ensure_ascii=False)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RockGuard/2.2",
            **self._custom_headers,
        }
        if self._secret:
            headers["X-Signature"] = self._sign_body(body)

        try:
            r = requests.post(self._url, data=body.encode("utf-8"),
                              headers=headers, timeout=10)
            ok = 200 <= r.status_code < 300
            return PushResult(ok, self.name,
                              f"HTTP {r.status_code}" if ok else r.text[:120],
                              code=r.status_code)
        except Exception as e:
            return PushResult(False, self.name, str(e), code=-1)
