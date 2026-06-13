"""
推送通道抽象基类 — 插件式架构核心
====================================
所有推送通道必须继承 PushChannel 并实现 send() 方法。

设计原则:
  - 每个通道自包含（读取自己的环境变量配置）
  - send() 不抛异常，始终返回 PushResult
  - validate_config() 让调度器提前判断通道是否可用
  - is_enabled_for_level() 支持按预警等级过滤
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PushResult:
    """推送结果 — 统一返回格式"""

    success: bool
    channel: str = ""           # 通道标识 (如 "pushplus", "dingtalk")
    message: str = ""           # 人类可读的结果描述
    code: int = 0               # HTTP 状态码或内部错误码
    raw_response: dict | None = field(default=None, repr=False)


class PushChannel(ABC):
    """
    推送通道抽象基类。

    子类必须实现:
      - name: 通道唯一标识 (如 "pushplus")
      - display_name: UI 显示名 (如 "PushPlus 微信推送")
      - send(): 发送推送的核心方法

    子类可选覆写:
      - validate_config(): 返回 True 表示配置就绪
      - is_enabled_for_level(): 按预警等级决定是否启用
    """

    name: str = "base"
    display_name: str = "基础通道"

    @abstractmethod
    def send(self, title: str, content: str,
             alert_level: str = "yellow") -> PushResult:
        """
        发送推送消息。

        参数:
            title:       推送标题
            content:     推送正文 (HTML)
            alert_level: 预警等级 (red/orange/yellow/blue)

        返回:
            PushResult — 始终返回，不抛异常
        """
        ...

    def validate_config(self) -> bool:
        """
        验证通道配置是否就绪（环境变量/配置文件）。

        返回 True 表示通道可用，False 表示未配置。
        调度器在发送前调用此方法，跳过不可用的通道。
        """
        return True

    def is_enabled_for_level(self, alert_level: str) -> bool:
        """
        该通道是否在指定预警等级下启用。

        默认: 红色和橙色启用，黄/蓝不启用。
        子类可覆写以自定义策略。
        """
        return alert_level in ("red", "orange")

    def __repr__(self) -> str:
        status = "ready" if self.validate_config() else "unconfigured"
        return f"<{self.name}: {status}>"
