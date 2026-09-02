"""
TodoWrite 工具：生成或更新任务分解列表
"""
from typing import Dict, Any, List
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class TodoWriteTool(BaseTool):
    """TodoWrite 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="TodoWrite",
            description=(
                "生成或更新任务分解列表，展示进行中的待办事项。参数：todos（待办事项数组），"
                "每个元素包含 content（待办内容）、status（pending/in_progress/completed）、"
                "activeForm（进行中的简短描述，可选）。用途：让用户清晰看到任务分解的步骤。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "待办事项内容"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "待办事项状态"
                                },
                                "activeForm": {"type": "string", "description": "进行中的简短描述"}
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.READONLY,
        )

    @classmethod
    def execute(cls, context: ToolContext, todos: List[Dict[str, Any]]) -> str:
        """TodoWrite 是声明式工具：todo 列表由前端渲染，这里仅返回简短确认，避免模型拿到空结果后重复调用。"""
        n = len(todos) if isinstance(todos, list) else 0
        return f"已更新待办列表（{n} 项）"
