"""
Grep 工具：在文件或目录中搜索匹配的内容
"""
import os
import re
import fnmatch
from typing import Optional
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class GrepTool(BaseTool):
    """Grep 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="Grep",
            description=(
                "在文件或目录中搜索匹配的内容。参数：pattern（正则）、path（可选，目录或文件，默认当前目录）、"
                "glob（可选，按文件名模式过滤）、ignore_case（可选，忽略大小写）"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "要搜索的正则表达式"
                    },
                    "path": {
                        "type": "string",
                        "description": "目录或文件路径"
                    },
                    "glob": {
                        "type": "string",
                        "description": "按文件名 glob 模式过滤（如 *.py）"
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "true 时忽略大小写"
                    }
                },
                "required": ["pattern"]
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
        )

    @classmethod
    def execute(cls, context: ToolContext, pattern: str, path: Optional[str] = ".",
                glob: Optional[str] = None, ignore_case: bool = False) -> str:
        """搜索文件内容（对齐 Claude Code Grep）：pattern 必填，path 目录/文件，glob 过滤文件名"""
        try:
            if not pattern:
                return "错误：pattern 不能为空"
            target = context.resolve_path(path or ".", "Grep")
            flags = re.IGNORECASE if ignore_case else 0
            try:
                rx = re.compile(pattern, flags)
            except re.error as e:
                return f"错误：无效的正则 pattern：{e}"
            files: list = []
            if os.path.isfile(target):
                files = [target]
            elif os.path.isdir(target):
                skip_dirs = {".git", "node_modules", "venv", ".venv", "dist", "build", "__pycache__"}
                for root, dirs, names in os.walk(target):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip_dirs]
                    for name in names:
                        if glob and not fnmatch.fnmatch(name, glob):
                            continue
                        files.append(os.path.join(root, name))
            else:
                return f"错误：路径 {target} 不存在"
            matches: list = []
            for fp in files:
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if rx.search(line):
                                matches.append(f"{fp}:{line_num}: {line.rstrip()}")
                                if len(matches) >= 100:
                                    return f"找到 {len(matches)} 个匹配（截取前100个）：\n" + "\n".join(matches)
                except Exception:
                    continue
            if matches:
                return f"找到 {len(matches)} 个匹配：\n" + "\n".join(matches[:100])
            return f"在 {target} 中没有找到匹配 '{pattern}' 的内容"
        except Exception as e:
            return f"搜索时出错: {str(e)}"
