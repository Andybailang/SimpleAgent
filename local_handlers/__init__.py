"""
LocalLLM 本地处理器包
====================
按注册顺序匹配用户输入，第一个 can_handle 返回 True 的处理器执行；
都不匹配时由注册表返回兜底提示。当前支持：图片识别（OCR）、
PDF 转图片提取文字、文本文件内容透传、DeepSeek 对话（Local Deepseek，
兜底大脑），后续扩展按同目录结构新增即可。
"""
from typing import Dict, Any, List, Optional

from .registry import local_handler_registry
from . import pdf  # noqa: F401  （导入即注册，含 PDF 时优先）
from . import ocr  # noqa: F401  （导入即注册）
from . import text  # noqa: F401  （导入即注册，纯文本透传）
from . import chat  # noqa: F401  （导入即注册，DeepSeek 对话，最后兜底）


# 本地回复末尾统一附加的 markdown 标记（换行后独立起行，明示内容为本地生成）
_LOCAL_REPLY_FOOTER = "以上内容由本地处理生成（LocalLLM）"


def handle_local_turn(bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
    """执行本地处理，末尾附加本地生成标记后返回（server 层伪造流式事件）。"""
    result = local_handler_registry.handle(bs, text, attachments)
    return f"{result or ''}\n\n---\n\n> ⚡ {_LOCAL_REPLY_FOOTER}"
