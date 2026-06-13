"""
推送通道插件包 — 自动注册所有内置通道
======================================
模块首次导入时，所有内置通道自动注册到全局 ChannelRegistry。

扩展方式:
  1. 创建新文件 (如 mychannel.py)，继承 PushChannel
  2. 在此处 import 即可自动注册
  3. 或者运行时手动调用 registry.register(MyChannel())
"""

from .base import PushChannel, PushResult
from .registry import get_registry, ChannelRegistry

_registry = get_registry()

# ---- 内置通道自动注册 ----
# 实例化通道并通过 registry.register() 显式注册

from .pushplus import PushPlusChannel
_registry.register(PushPlusChannel())

from .smtp import SMTPEmailChannel
_registry.register(SMTPEmailChannel())

from .wecom import WeComChannel
_registry.register(WeComChannel())

from .webhook import WebhookChannel
_registry.register(WebhookChannel())

from .dingtalk import DingTalkChannel
_registry.register(DingTalkChannel())

from .feishu import FeishuChannel
_registry.register(FeishuChannel())
