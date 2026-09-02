"""
delete_file 工具：删除文件
"""
import os
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class DeleteFileTool(BaseTool):
    """delete_file 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="delete_file",
            description="删除文件。参数：path（文件路径）",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要删除的文件路径"
                    }
                },
                "required": ["path"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.DEFAULT,
        )

    @classmethod
    def execute(cls, context: ToolContext, path: str) -> str:
        """删除文件"""
        try:
            path = context.resolve_path(path, "delete_file")
            if not os.path.exists(path):
                return f"错误：路径 {path} 不存在"
            if os.path.isdir(path):
                return "错误：该路径是目录，请勿直接删除目录；如需清理请先删除目录内文件"
            os.remove(path)
            return f"成功删除文件 {path}"
        except PermissionError:
            return f"错误：没有权限删除文件 {path}"
        except Exception as e:
            return f"删除文件 {path} 时出错: {str(e)}"
