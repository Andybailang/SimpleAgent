"""
Glob 工具：按 glob 模式查找文件/目录
"""
import os
import glob as glob_module
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class GlobTool(BaseTool):
    """Glob 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="Glob",
            description="按 glob 模式查找文件/目录。参数：pattern（glob 模式，如 src/**/*.py）",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "glob 模式"
                    }
                },
                "required": ["pattern"]
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
        )

    @classmethod
    def execute(cls, context: ToolContext, pattern: str) -> str:
        """按 glob 模式查找文件/目录（对齐 Claude Code Glob）"""
        try:
            if not pattern:
                return "错误：pattern 不能为空"
            pat = os.path.expanduser(pattern).replace("\\", "/")
            if not os.path.isabs(pat):
                pat = os.path.join(context.cwd, pat)
            hits: list = []
            for p in glob_module.glob(pat, recursive=True):
                try:
                    context.resolve_path(p, "Glob")
                except ValueError:
                    continue
                if p.startswith(context.cwd):
                    hits.append(os.path.relpath(p, context.cwd))
                else:
                    hits.append(p)
                if len(hits) >= 200:
                    break
            if hits:
                return "匹配到的文件/目录：\n" + "\n".join(hits)
            return f"没有匹配 '{pattern}' 的路径"
        except Exception as e:
            return f"Glob 出错: {str(e)}"
