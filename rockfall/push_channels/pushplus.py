"""
PushPlus 微信推送通道
======================
通过 PushPlus API 发送微信模板消息。

环境变量:
  PUSHPLUS_TOKEN  — PushPlus Token (必填)
  PUSHPLUS_TOPIC  — 群组编码 (可选)
  PUSHPLUS_URL    — API 地址 (默认 http://www.pushplus.plus/send)
"""

import os
import time

import requests

from .base import PushChannel, PushResult


class PushPlusChannel(PushChannel):
    name = "pushplus"
    display_name = "PushPlus 微信推送"

    def __init__(self):
        self._token = self._resolve_token()
        self._topic = os.getenv("PUSHPLUS_TOPIC", "")
        self._url = os.getenv("PUSHPLUS_URL", "http://www.pushplus.plus/send")

    @staticmethod
    def _resolve_token() -> str:
        token = os.getenv("PUSHPLUS_TOKEN", "")
        token_file = os.getenv("PUSHPLUS_TOKEN_FILE", "")
        if token_file and __import__('pathlib').Path(token_file).exists():
            return __import__('pathlib').Path(token_file).read_text(encoding="utf-8").strip()
        if token.startswith("ENC:"):
            try:
                from rockfall.secrets import resolve_secret
                return resolve_secret("PUSHPLUS_TOKEN", "")
            except Exception:
                return token
        return token

    def validate_config(self) -> bool:
        return bool(self._token and self._token != "your_token_here")

    def is_enabled_for_level(self, alert_level: str) -> bool:
        return alert_level in ("red", "orange")

    def send(self, title: str, content: str,
             alert_level: str = "yellow") -> PushResult:
        if not self.validate_config():
            return PushResult(False, self.name, "PUSHPLUS_TOKEN 未配置", code=-1)

        data = {
            "token": self._token,
            "title": title,
            "content": content,
            "topic": self._topic,
            "template": "html",
        }
        for attempt in range(3):
            try:
                res = requests.post(self._url, json=data, timeout=10).json()
                code = res.get("code", -1)
                return PushResult(
                    success=(code == 200), channel=self.name,
                    message=res.get("msg", ""), code=code,
                    raw_response=res,
                )
            except Exception as e:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                else:
                    return PushResult(False, self.name, str(e), code=-1)
        return PushResult(False, self.name, "未知错误", code=-1)
