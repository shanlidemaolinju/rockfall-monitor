"""
飞书群机器人通道
=================
通过飞书自定义机器人 Webhook 发送消息卡片。

环境变量:
  FEISHU_WEBHOOK_URL   — 飞书群机器人 Webhook 地址
  FEISHU_SECRET        — 签名校验密钥 (可选)
"""

import base64
import hashlib
import hmac
import os
import re
import time

import requests

from .base import PushChannel, PushResult


class FeishuChannel(PushChannel):
    name = "feishu"
    display_name = "飞书群机器人"

    def __init__(self):
        self._webhook_url = os.getenv("FEISHU_WEBHOOK_URL", "")
        self._secret = os.getenv("FEISHU_SECRET", "")

    def validate_config(self) -> bool:
        return bool(self._webhook_url)

    def _sign(self, timestamp: int) -> str:
        """飞书签名: timestamp + '\n' + secret → HMAC-SHA256 → base64"""
        if not self._secret:
            return ""
        string_to_sign = f"{timestamp}\n{self._secret}"
        return base64.b64encode(
            hmac.new(
                string_to_sign.encode(), digestmod=hashlib.sha256
            ).digest()
        ).decode()

    def send(self, title: str, content: str,
             alert_level: str = "yellow") -> PushResult:
        if not self.validate_config():
            return PushResult(False, self.name,
                              "飞书 Webhook 未配置 (FEISHU_WEBHOOK_URL)", code=-1)

        # HTML → 纯文本
        plain = re.sub(r"<[^>]+>", "", content)
        plain = re.sub(r"\n\s*\n", "\n", plain).strip()[:3000]

        level_colors = {
            "red": "red", "orange": "orange",
            "yellow": "yellow", "blue": "blue"
        }

        timestamp = int(time.time())
        payload = {
            "timestamp": str(timestamp),
            "sign": self._sign(timestamp),
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": level_colors.get(alert_level, "blue"),
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": plain,
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {"tag": "plain_text",
                             "content": "系统自动发送 · RockGuard"}
                        ]
                    }
                ]
            }
        }

        try:
            r = requests.post(self._webhook_url, json=payload, timeout=10)
            resp = r.json()
            if r.status_code == 200 and resp.get("code") == 0:
                return PushResult(True, self.name, "飞书推送成功", code=200)
            return PushResult(False, self.name,
                              f"飞书返回: {resp.get('msg', r.text[:100])}",
                              code=resp.get("code", r.status_code))
        except Exception as e:
            return PushResult(False, self.name, str(e), code=-1)
