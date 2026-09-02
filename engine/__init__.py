"""engine — 后端引擎包。

将原来单文件 engine.py 拆分为功能明确的多文件：
- config.py：纯常量与辅助函数
- util.py：Message 数据结构 + 内容/媒体辅助
- traffic.py：LLM 流量存档与上下文占用事件
- thinking.py：DeepSeek 思考/推理与请求用户标识辅助（OpenAI-Chat 与 Responses 共用）
- prompt.py：系统提示词构建
- tools_runtime.py：工具执行与权限
- context.py：上下文压缩与本地估算
- openai_flow.py / anthropic_flow.py / response_flow.py：三种 API 格式收发
- stream.py：流式分发与事件规范化
- agent.py：SimpleAgent 核心类（组合以上 mixin）
"""

from .agent import SimpleAgent
from .util import Message

# 保留与原 engine.py 一致的外部导出（server.py / cli.py 直接使用）
from tools.semantic_config import _semantic_setting  # noqa: F401
from tools import tool_registry, ToolContext  # noqa: F401
from tools.base import Tool, ToolPermission  # noqa: F401

__all__ = ["SimpleAgent", "Message", "_semantic_setting", "tool_registry", "ToolContext", "Tool", "ToolPermission"]
