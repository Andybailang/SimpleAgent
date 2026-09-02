"""engine.agent — SimpleAgent 核心类（编排 + 会话操作）。

通过组合多个功能 mixin，把不同 API 格式的收发、提示词、工具执行、
上下文压缩拆到各自文件；本文件只保留状态初始化、模式/权限切换与历史操作。
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from path_util import realpath_clean, strip_vermagic
from .config import AUTO_COMPACT_CONTEXT_MAX_TOKENS, AUTO_COMPACT_THRESHOLD, PERSIST_INTERVAL_MS
from .util import Message
from .thinking import ThinkingMixin
from .traffic import TrafficMixin
from .prompt import PromptMixin
from .tools_runtime import ToolsRuntimeMixin
from .context import ContextMixin
from .openai_flow import OpenAIFlowMixin
from .anthropic_flow import AnthropicFlowMixin
from .response_flow import ResponseFlowMixin
from .stream import StreamMixin

from tools import tool_registry, ToolContext
from tools.base import Tool, ToolPermission

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class SimpleAgent(
    StreamMixin,
    ThinkingMixin,
    OpenAIFlowMixin,
    AnthropicFlowMixin,
    ResponseFlowMixin,
    TrafficMixin,
    PromptMixin,
    ToolsRuntimeMixin,
    ContextMixin,
):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        cwd: Optional[str] = None,
        permission_mode: str = "default",
        api_format: str = "openai",
        thinking_level: str = "off",
        api_auth: str = "x-api-key",
        stream_supported: bool = True,
        billing: str = "paid",
        plain_chat: bool = False,
        mode: str = "work",
        role_prompt: Optional[str] = None
    ):
        """
        初始化 Agent

        Args:
            api_key: OpenAI API 密钥
            base_url: API 基础 URL
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            cwd: 工作目录（文件工具的相对路径根），默认当前进程目录
            permission_mode: 工具权限模式：default（仅任务目录）/ readonly（只读）/ full（完全访问）
            api_format: 接口格式：openai（DeepSeek Chat Completions）/ anthropic（Messages API）
                / response（DeepSeek Responses API，OpenAI 兼容 /responses）
            thinking_level: 思考深度：off/low/medium/high/max
            stream_supported: 是否支持流式；false 时走非流式请求（用于某些模型
                在 anthropic 兼容端点 + tools 时正文为空的场景）
            billing: 计费属性 free/paid。free 模型对 token 不敏感，允许思考与工具调用
                及其结果写入持久上下文（self.messages）；paid 维持现状：仅在 turn 内
                参与，不写回以节省 token。
            role_prompt: 自定义角色提示词（仅工作模式生效，前置到系统提示词最前）
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cwd = strip_vermagic(os.path.realpath(cwd)) if cwd else strip_vermagic(os.getcwd())
        if permission_mode in ("default", "readonly", "full"):
            self.permission_mode = permission_mode
        else:
            self.permission_mode = "default"

        if api_format in ("openai", "anthropic", "response"):
            self.api_format = api_format
        elif api_format in ("chat", "chat_completion"):
            # 兼容旧模型配置：chat / chat_completion 一律归一为 openai（Chat Completions）
            self.api_format = "openai"
        else:
            self.api_format = "openai"
        if thinking_level in ("off", "low", "medium", "high", "max"):
            self.thinking_level = thinking_level
        else:
            self.thinking_level = "off"
        if api_auth in ("x-api-key", "bearer"):
            self.api_auth = api_auth
        else:
            self.api_auth = "x-api-key"
        self.stream_supported = bool(stream_supported)
        self.billing = "free" if str(billing or "").lower() == "free" else "paid"
        self.plain_chat = bool(plain_chat)
        self.mode = "chat" if (plain_chat or mode == "chat") else "work"
        self.role_prompt = str(role_prompt or "").strip()
        # 是否把思考/工具调用/结果写回持久上下文：
        # - 免费模型（billing=free）：token 不敏感，一律写回；
        # - 编程模式（mode=work）：长流程需断点续跑，即使收费模型也写回，避免中断后丢过程；
        # - 聊天模式收费模型：维持现状（仅在 turn 内参与，不写回，省 token）。
        self._persist_tool_context = (self.billing == "free" or self.mode == "work")
        # 工具轮次落盘钩子与节流：server 会在生成轮里注入 _persist_hook，
        # 每轮工具调用完整写回 self.messages 后通过 _maybe_persist_tool_round 节流写盘，
        # 保证长任务（大量工具轮/长时间思考）在进程意外终止时，磁盘始终保留已完成轮次的上下文。
        self._persist_hook = None  # type: Optional[callable]
        self._persist_interval_ms = PERSIST_INTERVAL_MS
        self._last_persist_ms = 0

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 消息历史
        self.messages: List[Message] = []

        # 工具上下文（包含 Agent 引用、工作目录、权限与当前模式）
        self.tool_context = ToolContext(self, self.cwd, self.permission_mode)
        self.tool_context.mode = self.mode

        # 可用工具：全部来自工具注册表（按当前模式过滤见 _tool_definitions / _execute_tool）
        self.tools: List[Tool] = tool_registry.get_all_tools()

        # AskUserQuestion 交互状态：当模型调用该工具时置暂停，等待用户回答后以
        # tool_result 续跑（server_state 登记 pending_question 并驱动续跑）。
        self._pending_question: Optional[Dict[str, Any]] = None
        self._pending_question_event: Optional[Dict[str, Any]] = None
        self._ask_user_paused: bool = False

        # 记录初始的系统提示
        self._add_system_message()

        # LLM 流量存档上下文（由 server.py 注入会话级属性：session_id/stdin_id/cwd/role 等）
        self.traffic_context: Dict[str, Any] = {}

        # 自动压缩阈值（由 server.py 注入；供发送前本地估算检测用）
        self.context_max_tokens: int = AUTO_COMPACT_CONTEXT_MAX_TOKENS
        self.compact_threshold: float = AUTO_COMPACT_THRESHOLD
    def set_mode(self, mode: str) -> str:
        """切换模式：work（工作模式）/ chat（聊天模式）。"""
        if mode not in ("work", "chat"):
            return f"错误：未知模式 '{mode}'"
        self.mode = mode
        self.plain_chat = (mode == "chat")
        self.tool_context.mode = mode
        # 模式切换后同步写回标志（编程模式工作需断点续跑，收费模型也写回）
        self._persist_tool_context = (self.billing == "free" or self.mode == "work")
        self._refresh_system_message()
        return f"已切换至 {mode} 模式"
    def set_permission_mode(self, mode: str) -> str:
        """切换工具权限模式：default / readonly / full。"""
        if mode not in ("default", "readonly", "full"):
            return f"错误：未知权限模式 '{mode}'"
        self.permission_mode = mode
        self.tool_context.permission_mode = mode
        return f"已切换工具权限模式为 {mode}"
    def get_messages(self) -> List[Message]:
        """获取消息历史"""
        return self.messages
    def clear_history(self):
        """清空消息历史"""
        self.messages = []
        self._add_system_message()
    def get_history_text(self) -> str:
        """获取消息历史的文本表示"""
        result = ""
        for msg in self.messages:
            if msg.role == "system":
                continue
            result += f"\n{'='*50}\n"
            result += f"[{msg.role.upper()}] ({msg.timestamp})\n"
            if msg.tool_name:
                result += f"工具: {msg.tool_name}\n"
            result += f"{msg.content}\n"
            result += f"{'='*50}\n"
        return result
