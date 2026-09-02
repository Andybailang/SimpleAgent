"""
本地处理器基类
=============
LocalLLM 模式下按注册顺序匹配，第一个 can_handle 返回 True 的处理器执行。
"""
import re
from typing import Dict, Any, List, Optional


_SKILL_NAME_RE = re.compile(r"【技能：([^】]+)】")
_USER_REQUEST_MARKER = "【用户请求】"


def extract_skill_name(text: str) -> str:
    """从技能注入文本中提取技能名（【技能：xxx】）；无技能包装时返回空字符串。"""
    if not text:
        return ""
    m = _SKILL_NAME_RE.search(text)
    return m.group(1).strip() if m else ""


def extract_user_request(text: str) -> str:
    """从技能注入文本中提取真实用户输入（【用户请求】之后）；无标记时返回原文。"""
    t = text or ""
    idx = t.rfind(_USER_REQUEST_MARKER)
    if idx >= 0:
        return t[idx + len(_USER_REQUEST_MARKER):].strip()
    return t.strip()


class BaseLocalHandler:
    """本地处理器基类"""

    #: 处理器标识（唯一）
    name: str = "base"
    #: 能力描述（兜底提示中展示，如“图片识别”）
    description: str = ""

    def can_handle(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None, cwd: Optional[str] = None) -> bool:
        """判断当前输入是否由本处理器处理。"""
        return False

    def handle(self, bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        """执行处理，返回回复文本。"""
        raise NotImplementedError
