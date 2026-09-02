"""
rename_file 工具：重命名或移动文件/目录
"""
import os
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class RenameFileTool(BaseTool):
    """rename_file 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="rename_file",
            description="重命名或移动文件/目录。参数：src（源路径）、dest（目标路径）",
            parameters={
                "type": "object",
                "properties": {
                    "src": {
                        "type": "string",
                        "description": "源路径"
                    },
                    "dest": {
                        "type": "string",
                        "description": "目标路径"
                    }
                },
                "required": ["src", "dest"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.DEFAULT,
        )

    @classmethod
    def execute(cls, context: ToolContext, src: str, dest: str) -> str:
        """重命名或移动文件/目录"""
        try:
            src = context.resolve_path(src, "rename_file")
            dest = context.resolve_path(dest, "rename_file")
            if not os.path.exists(src):
                return f"错误：源路径 {src} 不存在"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.rename(src, dest)
            return f"成功将 {src} 移动/重命名为 {dest}"
        except Exception as e:
            return f"重命名/移动 {src} 时出错: {str(e)}"
