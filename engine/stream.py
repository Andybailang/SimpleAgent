"""engine.stream — 流式分发与事件规范化。
"""
from typing import Any, Dict, List, Optional

from .util import Message

class StreamMixin:
    async def chat_stream_async(self, user_message: str, use_tools: bool = True, attachments: Optional[List[Dict[str, Any]]] = None, continue_only: bool = False):
        """
        Async streaming chat using AsyncOpenAI.

        Follows the original tokenicode / Claude Code model: attachments are
        passed as file paths in the message text and the model reads them with
        the Read tool, so document size is not limited by inlining.

        Args:
            user_message: The user input message
            use_tools: Whether to attach tool definitions. All tools are
                exposed; paths are resolved against the agent's cwd.
            attachments: 附件只传路径文本：图片由模型调用
                extract_text_from_image 提取文字，其他文件用 Read 读取。

        Yields:
            Streaming text fragments
        """
        self._ask_user_paused = False  # 每轮重新开始，避免上次提问的暂停状态残留
        self._pending_question_event = None
        self._refresh_system_message()
        if self.api_format == "anthropic":
            method = self.chat_nonstream_anthropic_async if not self.stream_supported else self.chat_stream_anthropic_async
            async for evt in method(user_message, use_tools=use_tools, attachments=attachments, continue_only=continue_only):
                yield evt
            return

        # OpenAI Chat Completions（openai）/ DeepSeek Responses API（response）：
        # 统一定义，按 stream_supported 选择流式/非流式；仅 openai/response 需要事件归一化。
        if self.api_format == "response":
            method = self.chat_nonstream_response_async if not self.stream_supported else self.chat_stream_response_async
        else:
            method = self.chat_nonstream_openai_async if not self.stream_supported else self.chat_stream_openai_async
        async for evt in method(user_message, use_tools=use_tools, attachments=attachments, continue_only=continue_only):
            yield self._normalize_openai_bridge_event(evt)
    def _normalize_openai_bridge_event(self, evt: Any) -> Any:
        """把 OpenAI / Responses 路径产出的事件归一为前端消费的形状。

        text / error 事件以字符串下发（前端当作文本片段追加），其余保持 dict。
        """
        et = evt.get("type") if isinstance(evt, dict) else None
        if et in ("usage", "system", "thinking", "tool_use", "tool_result"):
            return evt
        if et in ("text", "error"):
            return evt.get("text", "")
        return evt
