"""
create_directory 工具：创建目录
"""
import os
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class CreateDirectoryTool(BaseTool):
    """create_directory 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="create_directory",
            description="创建目录。参数：path（目录路径）",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要创建的目录路径"
                    }
                },
                "required": ["path"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.DEFAULT,
        )

    @classmethod
    def execute(cls, context: ToolContext, path: str) -> str:
        """创建目录"""
        try:
            path = context.resolve_path(path, "create_directory")
            os.makedirs(path, exist_ok=True)
            return f"成功创建目录 {path}"
        except Exception as e:
            return f"创建目录 {path} 时出错: {str(e)}"
