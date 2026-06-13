"""
企业微信机器人通道
===================
通过企业微信群机器人 Webhook 发送 Markdown 预警消息。

环境变量:
  WECOM_WEBHOOK_URL — 群机器人 Webhook 地址
"""

import os
import re

import requests

from .base import PushChannel, PushResult


class WeComChannel(PushChannel):
    name = "wecom"
    display_name = "企业微信群机器人"

    def __init__(self):
        self._webhook_url = os.getenv("WECOM_WEBHOOK_URL", "")

    def validate_config(self) -> bool:
        return bool(self._webhook_url)

    def send(self, title: str, content: str,
             alert_level: str = "yellow") -> PushResult:
        if not self.validate_config():
            return PushResult(False, self.name,
                              "企业微信 Webhook 未配置 (WECOM_WEBHOOK_URL)", code=-1)

        # HTML → Markdown 纯文本 (企业微信仅支持有限 Markdown)
        plain = re.sub(r"<[^>]+>", "", content)
        plain = re.sub(r"\n\s*\n", "\n", plain).strip()[:2000]

        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": f"## {title}\n\n{plain}\n\n> 系统自动发送 · RockGuard"
            }
        }
        try:
            r = requests.post(self._webhook_url, json=payload, timeout=10)
            if r.status_code == 200 and r.json().get("errcode") == 0:
                return PushResult(True, self.name, "企业微信推送成功", code=200)
            return PushResult(False, self.name,
                              f"企业微信返回: {r.text[:100]}", code=r.status_code)
        except Exception as e:
            return PushResult(False, self.name, str(e), code=-1)
