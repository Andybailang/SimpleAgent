"""
数据适配层 - 在简化 Agent 数据格式和 Tokenicore 前端期望的数据格式之间转换
"""
from typing import List, Dict, Any, Optional
from datetime import datetime


class MessageAdapter:
    """消息适配器"""

    @staticmethod
    def to_tokenicode_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将我们的消息格式转换为 Tokenicore 期望的格式

        我们的格式：
        {
            "id": "msg_xxx",
            "role": "user" | "assistant" | "system",
            "type": "text" | "tool_use" | "tool_result",
            "content": "...",
            "tool_name": "...",
            "timestamp": 1234567890
        }

        Tokenicore 格式：
        {
            "id": "msg_xxx",
            "role": "user" | "assistant" | "system",
            "type": "text" | "tool_use" | "thinking" | "tool_result" | "permission" | "plan" | "plan_review" | "question" | "todo",
            "content": "...",
            "toolName": "...",
            "toolInput": {...},
            "toolResult": "...",
            "timestamp": 1234567890
        }
        """
        adapted = []

        for msg in messages:
            adapted_msg = {
                "id": msg.get("id", ""),
                "role": msg.get("role", "system"),
                "type": "text",  # 默认类型
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", int(datetime.now().timestamp() * 1000))
            }

            # 根据不同类型添加额外字段
            if msg.get("type") == "tool_result":
                adapted_msg["type"] = "tool_result"
                adapted_msg["toolResult"] = msg.get("content", "")
                adapted_msg["toolName"] = msg.get("tool_name", "")
            elif msg.get("type") == "tool_use":
                adapted_msg["type"] = "tool_use"
                adapted_msg["toolName"] = msg.get("tool_name", "")
                adapted_msg["toolInput"] = msg.get("tool_input", {})

            adapted.append(adapted_msg)

        return adapted

    @staticmethod
    def from_tokenicode_format(msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Tokenicore 消息格式转换回我们的简化格式

        Tokenicore 格式：
        {
            "id": "msg_xxx",
            "role": "user" | "assistant" | "system",
            "type": "text" | "tool_use" | "thinking" | "tool_result" | ...,
            "content": "...",
            "toolName": "...",
            "toolInput": {...},
            "toolResult": "...",
            "timestamp": 1234567890
        }

        我们的格式：
        {
            "id": "msg_xxx",
            "role": "user" | "assistant" | "system",
            "type": "text" | "tool_use" | "tool_result",
            "content": "...",
            "tool_name": "...",
            "timestamp": 1234567890
        }
        """
        adapted_msg = {
            "id": msg.get("id", ""),
            "role": msg.get("role", "system"),
            "type": msg.get("type", "text"),
            "content": msg.get("content", ""),
            "timestamp": msg.get("timestamp", int(datetime.now().timestamp() * 1000))
        }

        # 添加工具相关字段
        if msg.get("type") == "tool_result":
            adapted_msg["type"] = "tool_result"
            adapted_msg["content"] = msg.get("toolResult", "")
            adapted_msg["tool_name"] = msg.get("toolName", "")
        elif msg.get("type") == "tool_use":
            adapted_msg["type"] = "tool_use"
            adapted_msg["tool_name"] = msg.get("toolName", "")
            adapted_msg["tool_input"] = msg.get("toolInput", {})

        return adapted_msg


class SessionAdapter:
    """会话适配器"""

    @staticmethod
    def to_tokenicode_format(session_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """转换会话格式"""
        return {
            "id": session_id,
            "title": f"Session {session_id}",
            "messages": MessageAdapter.to_tokenicode_format(messages),
            "created_at": datetime.now().isoformat()
        }

    @staticmethod
    def from_tokenicode_format(session_data: Dict[str, Any]) -> Dict[str, Any]:
        """转换回会话格式"""
        return {
            "id": session_data.get("id", ""),
            "messages": [MessageAdapter.from_tokenicode_format(msg) for msg in session_data.get("messages", [])]
        }


# ==================== 工具调用适配 ====================

class ToolCallAdapter:
    """工具调用适配器"""

    @staticmethod
    def create_tool_call_result(tool_name: str, result: str) -> Dict[str, Any]:
        """创建工具调用结果"""
        return {
            "tool_name": tool_name,
            "tool_input": {},
            "tool_result": result,
            "type": "tool_result",
            "role": "assistant",
            "content": result,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }

    @staticmethod
    def create_tool_call_request(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """创建工具调用请求"""
        return {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "type": "tool_use",
            "role": "assistant",
            "content": f"Calling {tool_name} with {tool_input}",
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
