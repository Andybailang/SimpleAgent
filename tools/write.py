"""
Write 工具：创建或覆盖写入文件
"""
import os
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class WriteTool(BaseTool):
    """Write 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="Write",
            description="创建或覆盖写入文件。参数：file_path（文件路径）、content（文件内容）。写入纯文本请用 .txt/.md/.py 等文本扩展名，不要用 .docx/.jpg 等二进制扩展名。",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "content": {
                        "type": "string",
                        "description": "文件内容"
                    }
                },
                "required": ["file_path", "content"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.DEFAULT,
            aliases={"file_path": ["path"]},
        )

    @classmethod
    def execute(cls, context: ToolContext, file_path: str, content: str) -> str:
        """写入文件"""
        try:
            path = context.resolve_path(file_path, "Write")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"成功写入文件 {path}"
        except Exception as e:
            return f"写入文件 {file_path} 时出错: {str(e)}"
