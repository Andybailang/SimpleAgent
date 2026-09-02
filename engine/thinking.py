"""engine.thinking — DeepSeek 思考/推理与请求用户标识辅助。

这些属性/方法被 OpenAI Chat Completions 链路（openai_flow.py）与 Responses 链路
（response_flow.py）共用：
- _is_thinking / _deepseek_effort / _DEEPSEEK_EFFORT_MAP：把本地 thinking_level 映射为
  DeepSeek 的 reasoning effort，并判断是否开启思考；
- _session_user_id：把会话 id 清洗成 DeepSeek 可接受的 user_id / user 字段。
"""
from typing import Any, Dict, Optional


class ThinkingMixin:

    _DEEPSEEK_EFFORT_MAP = {
        "off": "none",      # none 表示关闭思考模式
        "low": "low",
        "medium": "high",   # medium 映射为 high
        "high": "high",
        "max": "max",
    }
    @property
    def _is_thinking(self) -> bool:
        """当前是否处于思考模式（thinking_level != off）。"""
        return self.thinking_level != "off"
    def _deepseek_effort(self) -> str:
        """把本地 thinking_level 映射为 DeepSeek 的 reasoning effort（low/high/max/none）。"""
        return self._DEEPSEEK_EFFORT_MAP.get(self.thinking_level, "high")
    def _session_user_id(self) -> str:
        """返回用于 user_id / user 的会话标识（我们的会话 uuid），越界字符剔除。

        DeepSeek 限制字符集 [a-zA-Z0-9-_]、最大长度 512。取自 traffic_context 的
        session_id（如 desk_xxx / cli_xxx）。无则返回空串（不传该参数）。
        """
        sid = str((self.traffic_context or {}).get("session_id") or "").strip()
        if not sid:
            return ""
        cleaned = "".join(ch for ch in sid if ch.isalnum() or ch in ("-", "_"))
        return cleaned[:512]
