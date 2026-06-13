"""
钉钉群机器人通道
=================
通过钉钉自定义机器人 Webhook 发送 Markdown 预警消息。

环境变量:
  DINGTALK_WEBHOOK_URL  — 钉钉群机器人 Webhook 地址
  DINGTALK_SECRET       — 加签密钥 (可选，安全建议配置)

签名算法: HMAC-SHA256, 参考钉钉开放平台文档
"""

import base64
import hashlib
import hmac
import os
import re
import time
import urllib.parse

import requests

from .base import PushChannel, PushResult


class DingTalkChannel(PushChannel):
    name = "dingtalk"
    display_name = "钉钉群机器人"

    def __init__(self):
        self._webhook_url = os.getenv("DINGTALK_WEBHOOK_URL", "")
        self._secret = os.getenv("DINGTALK_SECRET", "")

    def validate_config(self) -> bool:
        return bool(self._webhook_url)

    def _sign_url(self) -> str:
        """对 URL 加签 (钉钉安全设置)"""
        if not self._secret:
            return self._webhook_url

        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self._secret}"
        sign = base64.b64encode(
            hmac.new(
                self._secret.encode(), string_to_sign.encode(), hashlib.sha256
            ).digest()
        ).decode()
        return f"{self._webhook_url}&timestamp={timestamp}&sign={urllib.parse.quote(sign)}"

    def send(self, title: str, content: str,
             alert_level: str = "yellow") -> PushResult:
        if not self.validate_config():
            return PushResult(False, self.name,
                              "钉钉 Webhook 未配置 (DINGTALK_WEBHOOK_URL)", code=-1)

        # HTML → 纯文本 (钉钉 Markdown 不支持 HTML)
        plain = re.sub(r"<[^>]+>", "", content)
        plain = re.sub(r"\n\s*\n", "\n", plain).strip()[:4000]

        level_emoji = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "blue": "🔵"}
        emoji = level_emoji.get(alert_level, "⚠️")

        payload = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": f"## {emoji} {title}\n\n{plain}\n\n> 系统自动发送 · RockGuard"
            }
        }

        try:
            url = self._sign_url()
            r = requests.post(url, json=payload, timeout=10)
            resp = r.json()
            if r.status_code == 200 and resp.get("errcode") == 0:
                return PushResult(True, self.name, "钉钉推送成功", code=200)
            return PushResult(False, self.name,
                              f"钉钉返回: {resp.get('errmsg', r.text[:100])}",
                              code=resp.get("errcode", r.status_code))
        except Exception as e:
            return PushResult(False, self.name, str(e), code=-1)
