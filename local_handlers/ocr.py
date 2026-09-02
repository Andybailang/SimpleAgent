"""
图片识别处理器
=============
用户附图时调用 PaddleOCR 提取文字并分段返回。
"""
import os
from typing import Dict, Any, List, Optional
from .base import BaseLocalHandler
from .registry import local_handler_registry
from ocr_engine import get_ocr, extract_rec_texts


class OCRHandler(BaseLocalHandler):
    """识别用户附带的图片文字"""

    name = "ocr"
    description = "图片识别"

    def can_handle(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None, cwd: Optional[str] = None) -> bool:
        return any(a.get("isImage") and a.get("path") for a in (attachments or []))

    def handle(self, bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        """识别全部图片附件，按图片名分段返回。"""
        cwd = bs.get("cwd") or os.getcwd()
        images = [a for a in (attachments or []) if a.get("isImage") and a.get("path")]
        parts: List[str] = []
        for att in images:
            path = att.get("path") or ""
            resolved = path if os.path.isabs(path) else os.path.join(cwd, path)
            resolved = os.path.realpath(resolved)
            name = att.get("name") or os.path.basename(resolved) or path
            if not os.path.isfile(resolved):
                parts.append(f"【{name}】错误：图片文件不存在（{resolved}）")
                continue
            try:
                ocr = get_ocr()
                result = ocr.predict(resolved)
                texts: List[str] = extract_rec_texts(result)
                if not texts:
                    parts.append(f"【{name}】未识别到文字")
                else:
                    parts.append(f"【{name}】识别结果：\n" + "\n".join(texts))
            except Exception as e:
                parts.append(f"【{name}】图片识别失败：{e}")
        return "\n\n".join(parts) if parts else "未发现可识别的图片附件"


# 导入即注册：处理器追加到匹配顺序末尾
local_handler_registry.register(OCRHandler())
