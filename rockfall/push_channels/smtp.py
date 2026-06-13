"""
SMTP 邮件推送通道
==================
通过 SMTP 发送预警邮件。

环境变量:
  SMTP_HOST        — SMTP 服务器地址
  SMTP_PORT        — 端口 (默认 587)
  SMTP_USER        — 发件人账号
  SMTP_PASSWORD    — 发件人密码
  ALERT_EMAIL_TO   — 收件人列表 (逗号分隔)
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .base import PushChannel, PushResult


class SMTPEmailChannel(PushChannel):
    name = "smtp"
    display_name = "邮件通知 (SMTP)"

    def __init__(self):
        self._host = os.getenv("SMTP_HOST", "")
        self._port = int(os.getenv("SMTP_PORT", "587"))
        self._user = os.getenv("SMTP_USER", "")
        self._password = os.getenv("SMTP_PASSWORD", "")
        self._to = self._resolve_recipients()

    @staticmethod
    def _resolve_recipients() -> list[str]:
        raw = os.getenv("ALERT_EMAIL_TO", "")
        return [e.strip() for e in raw.split(",") if e.strip()]

    def validate_config(self) -> bool:
        return bool(self._host and self._user and self._password and self._to)

    def send(self, title: str, content: str,
             alert_level: str = "yellow") -> PushResult:
        if not self.validate_config():
            return PushResult(False, self.name,
                              "SMTP 配置不完整 (需 SMTP_HOST/USER/PASSWORD/ALERT_EMAIL_TO)",
                              code=-1)

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = title
            msg["From"] = self._user
            msg["To"] = ", ".join(self._to)
            msg.attach(MIMEText(content, "html", "utf-8"))

            with smtplib.SMTP(self._host, self._port, timeout=10) as server:
                server.starttls()
                server.login(self._user, self._password)
                server.sendmail(self._user, self._to, msg.as_string())

            return PushResult(True, self.name,
                              f"邮件已发送至 {len(self._to)} 位收件人", code=200)
        except Exception as e:
            return PushResult(False, self.name, f"邮件发送失败: {e}", code=-1)
