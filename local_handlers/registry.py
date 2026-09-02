"""
本地处理器注册表
===============
按注册顺序依次匹配，第一个 can_handle 返回 True 的执行；
都不匹配时返回兜底提示（能力清单随后续扩展自动并入）。
"""
from typing import Dict, Any, List, Optional
from .base import BaseLocalHandler


class LocalHandlerRegistry:
    """处理器注册表（顺序匹配）"""

    def __init__(self) -> None:
        self._handlers: List[BaseLocalHandler] = []

    def register(self, handler: BaseLocalHandler) -> BaseLocalHandler:
        """注册处理器（追加到匹配顺序末尾）。"""
        self._handlers.append(handler)
        return handler

    def match(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None, cwd: Optional[str] = None) -> Optional[BaseLocalHandler]:
        """按注册顺序返回第一个能处理的处理器，没有则返回 None。"""
        for handler in self._handlers:
            try:
                if handler.can_handle(text, attachments, cwd):
                    return handler
            except Exception:
                continue
        return None

    def supported_text(self) -> str:
        """兜底提示中的能力清单。"""
        names = [h.description for h in self._handlers if h.description]
        return "、".join(names) if names else "暂无"

    def handle(self, bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        """执行本地处理，返回回复文本；无匹配时返回兜底提示。"""
        handler = self.match(text, attachments, cwd=bs.get("cwd"))
        if handler is None:
            return f"我目前只支持：{self.supported_text()}等"
        return handler.handle(bs, text, attachments)


# 全局注册表实例
local_handler_registry = LocalHandlerRegistry()
