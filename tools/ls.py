"""
LS 工具：列出目录内容
"""
import os
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class LsTool(BaseTool):
    """LS 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="LS",
            description="列出目录下的文件和子目录。参数：path（可选，目录路径，默认当前目录）",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "目录路径"
                    }
                },
                "required": []
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
        )

    @classmethod
    def execute(cls, context: ToolContext, path: str = ".") -> str:
        """列出目录内容（对齐 Claude Code LS）"""
        try:
            path = context.resolve_path(path, "LS")
            if not os.path.exists(path):
                return f"错误：路径 {path} 不存在"
            if not os.path.isdir(path):
                return f"错误：路径 {path} 不是目录"

            entries = os.listdir(path)
            result = f"目录 {path} 下的内容：\n\n"
            for entry in entries:
                entry_path = os.path.join(path, entry)
                if os.path.isdir(entry_path):
                    result += f"[DIR]  {entry}\n"
                else:
                    result += f"[FILE] {entry}\n"
            return result
        except PermissionError:
            return f"错误：没有权限访问目录 {path}"
        except Exception as e:
            return f"列出目录 {path} 时出错: {str(e)}"
