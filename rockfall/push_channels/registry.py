"""
通道注册表 — 动态发现 + 按名查找
=================================
所有推送通道在模块加载时自动注册到全局单例 ChannelRegistry。

用法:
    from rockfall.push_channels.registry import get_registry
    registry = get_registry()
    channel = registry.get("dingtalk")
    if channel and channel.validate_config():
        result = channel.send("标题", "内容", "red")
"""

from __future__ import annotations
import threading
from typing import Iterator

from .base import PushChannel, PushResult


class ChannelRegistry:
    """推送通道注册表 — 线程安全"""

    def __init__(self):
        self._channels: dict[str, PushChannel] = {}
        self._lock = threading.Lock()

    def register(self, channel: PushChannel) -> None:
        """注册一个通道（同名通道后者覆盖前者）"""
        with self._lock:
            self._channels[channel.name] = channel

    def unregister(self, name: str) -> None:
        """移除一个通道"""
        with self._lock:
            self._channels.pop(name, None)

    def get(self, name: str) -> PushChannel | None:
        """按名称获取通道"""
        return self._channels.get(name)

    def list_all(self) -> list[PushChannel]:
        """列出所有已注册通道"""
        with self._lock:
            return list(self._channels.values())

    def list_ready(self, alert_level: str = "") -> list[PushChannel]:
        """
        列出所有配置就绪且对指定预警等级启用的通道。

        参数:
            alert_level: 预警等级筛选，空字符串 = 全部返回
        """
        channels = self.list_all()
        ready = []
        for ch in channels:
            if not ch.validate_config():
                continue
            if alert_level and not ch.is_enabled_for_level(alert_level):
                continue
            ready.append(ch)
        return ready

    def send_all(self, title: str, content: str,
                 alert_level: str = "yellow") -> dict[str, PushResult]:
        """
        向所有就绪且启用的通道并行发送推送。

        返回:
            {channel_name: PushResult, ...}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        channels = self.list_ready(alert_level)
        if not channels:
            return {}

        results: dict[str, PushResult] = {}
        with ThreadPoolExecutor(max_workers=min(len(channels), 6),
                                thread_name_prefix="push") as ex:
            futures = {
                ex.submit(ch.send, title, content, alert_level): ch.name
                for ch in channels
            }
            for future in as_completed(futures, timeout=30):
                name = futures[future]
                try:
                    results[name] = future.result(timeout=15)
                except Exception as e:
                    results[name] = PushResult(
                        success=False, channel=name,
                        message=f"发送异常: {e}", code=-1,
                    )
            # 超时未完成的任务标记为失败
            for future, name in futures.items():
                if name not in results:
                    results[name] = PushResult(
                        success=False, channel=name,
                        message="发送超时", code=-1,
                    )
        return results

    def __len__(self) -> int:
        return len(self._channels)

    def __iter__(self) -> Iterator[PushChannel]:
        return iter(self.list_all())

    def __contains__(self, name: str) -> bool:
        return name in self._channels


# ---- 模块级单例 ----

_registry: ChannelRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> ChannelRegistry:
    """获取全局通道注册表单例"""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ChannelRegistry()
    return _registry
