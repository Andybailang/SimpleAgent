"""
Read 工具：读取文件内容
"""
import re
import zipfile
from typing import Optional
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext

READ_FILE_LIMIT = 20000


def extract_docx_text(path: str) -> str:
    """Extract plain text from a .docx file (zip + XML) without extra deps."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = xml.replace("</w:p>", "\n")
    text = re.sub(r"<[^>]+>", "", xml)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def decode_content(raw: bytes) -> str:
    """解码文件内容，尝试常见编码。"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class ReadTool(BaseTool):
    """Read 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="Read",
            description=(
                "读取文件内容。参数：file_path（文件路径）、offset（可选，起始行号 0-based）、"
                "limit（可选，读取行数，与 length 等价；不传则读到末尾）。读取后直接阅读回答，不要立即调用 SemanticSearch。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "起始行号（0-based），默认从第 1 行开始"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "要读取的行数；不传则读取到文件末尾（超 20000 字符截断）"
                    },
                    "length": {
                        "type": "integer",
                        "description": "与 limit 等价（兼容旧参数名 length）"
                    }
                },
                "required": ["file_path"]
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
            aliases={"file_path": ["path"]},
        )

    @classmethod
    def execute(cls, context: ToolContext, file_path: str,
                offset: Optional[int] = None,
                limit: Optional[int] = None,
                length: Optional[int] = None) -> str:
        """执行读取文件"""
        try:
            path = context.resolve_path(file_path, "Read")
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if ext == "docx":
                content = extract_docx_text(path)
            else:
                raw = open(path, "rb").read()
                content = decode_content(raw)
            # offset/limit 行分片（offset 0-based，与 Claude Code Read 一致）
            n = limit if limit is not None else length
            if offset is not None or n is not None:
                try:
                    start = max(0, int(offset or 0))
                except (TypeError, ValueError):
                    return f"错误：offset 必须是整数（当前：{offset}）"
                try:
                    count = int(n) if n is not None else None
                except (TypeError, ValueError):
                    return f"错误：limit/length 必须是整数（当前：{n}）"
                lines = content.split("\n")
                if start >= len(lines):
                    return f"错误：文件 {path} 共 {len(lines)} 行，起始行 {start + 1} 超出范围"
                end = len(lines) if count is None else min(len(lines), start + count)
                sliced = "\n".join(lines[start:end])
                if len(sliced) > READ_FILE_LIMIT:
                    sliced = sliced[:READ_FILE_LIMIT] + "\n...[内容过长，已截断]"
                return f"文件 {path} 第 {start + 1}~{end} 行（共 {len(lines)} 行）的内容：\n\n{sliced}"
            if len(content) > READ_FILE_LIMIT:
                content = content[:READ_FILE_LIMIT] + "\n...[内容过长，已截断]"
            return f"文件 {path} 的内容：\n\n{content}"
        except FileNotFoundError:
            return f"错误：文件 {file_path} 不存在"
        except PermissionError:
            return f"错误：没有权限读取文件 {file_path}"
        except Exception as e:
            return f"读取文件 {file_path} 时出错: {str(e)}"
