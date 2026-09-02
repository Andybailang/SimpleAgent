"""
文本文件处理器
==============
用户上传纯文本类文件（或消息中含文本文件路径）时，把文件内容透传给 UI 呈现，
不返回“不支持”提示，代码类文件会用对应语言的 markdown 代码块包裹。其中 .doc / .docx 用 aspose-words-foss 转成 markdown 返回。
"""
import os
import re
import tempfile
from typing import Dict, Any, List, Optional

from .base import BaseLocalHandler
from .registry import local_handler_registry

# 文本文件扩展名（pdf / 图片由各自处理器负责，不在此列）
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".mdx", ".doc", ".docx", ".py", ".pyi", ".ts", ".tsx",
    ".js", ".jsx", ".mjs", ".cjs", ".json", ".jsonl", ".toml", ".yaml",
    ".yml", ".ini", ".cfg", ".conf", ".env", ".csv", ".tsv", ".log",
    ".xml", ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue",
    ".svelte", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql", ".proto", ".rst", ".tex", ".java", ".kt",
    ".swift", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb",
    ".php", ".lua", ".r", ".pl", ".ex", ".exs", ".erl", ".ml", ".mli",
    ".zig", ".tf", ".hcl", ".dockerfile", ".makefile", ".lock",
}

# Word 文档扩展名（二进制，走 aspose-words-foss 转 markdown）
WORD_EXTENSIONS = {".doc", ".docx"}

# 无扩展名但按文本处理的常见文件名（小写比较）
TEXT_FILENAMES = {
    "makefile", "dockerfile", "license", "copying", "readme", "changelog",
    "gitignore", "gitattributes", "editorconfig", "prettierrc", "eslintrc",
    "npmrc", "yarnrc",
}

# 图片扩展名（避免把图片当文本透传，交给 OCR 处理器）
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".svg", ".tif", ".tiff",
}

# 单文件透传上限（字符），超出截断并提示
MAX_TEXT_CHARS = 100_000

_BACKTICK_RE = re.compile(r"`([^`\n]+)`")


# 代码/配置类文件扩展名 → markdown 围栏语言（不在此列则按扩展名兜底）
CODE_LANGS = {
    ".py": "python", ".pyi": "python", ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx", ".mjs": "javascript", ".cjs": "javascript",
    ".json": "json", ".jsonl": "jsonl", ".toml": "toml", ".yaml": "yaml",
    ".yml": "yaml", ".ini": "ini", ".cfg": "ini", ".conf": "ini",
    ".env": "ini", ".csv": "csv", ".tsv": "tsv", ".log": "text",
    ".xml": "xml", ".html": "html", ".htm": "html", ".css": "css",
    ".scss": "scss", ".sass": "sass", ".less": "less", ".vue": "vue",
    ".svelte": "svelte", ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
    ".fish": "fish", ".ps1": "powershell", ".bat": "batch", ".cmd": "batch",
    ".sql": "sql", ".graphql": "graphql", ".gql": "graphql", ".proto": "protobuf",
    ".tex": "latex", ".java": "java", ".kt": "kotlin", ".swift": "swift",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".lua": "lua",
    ".r": "r", ".pl": "perl", ".ex": "elixir", ".exs": "elixir",
    ".erl": "erlang", ".ml": "ocaml", ".mli": "ocaml", ".zig": "zig",
    ".tf": "terraform", ".hcl": "hcl",
}

# 无扩展名代码/配置文件名 → 围栏语言（readme/license 等纯文本类不包裹）
_FILENAME_LANGS = {
    "makefile": "makefile",
    "dockerfile": "dockerfile",
    "gitignore": "gitignore",
    "gitattributes": "gitattributes",
    "editorconfig": "ini",
    "prettierrc": "json",
    "eslintrc": "json",
    "npmrc": "ini",
    "yarnrc": "ini",
}

# 纯文档类扩展名（不包代码围栏，保持原样渲染）
PROSE_EXTENSIONS = {".txt", ".md", ".markdown", ".mdx", ".rst"}


def _code_fence_language(path: str) -> Optional[str]:
    """返回代码文件的 markdown 围栏语言；纯文档类文件返回 None（不包裹）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in WORD_EXTENSIONS or ext in PROSE_EXTENSIONS:
        return None
    if ext in CODE_LANGS:
        return CODE_LANGS[ext]
    base = os.path.basename(path).lower()
    if base in _FILENAME_LANGS:
        return _FILENAME_LANGS[base]
    if base in TEXT_FILENAMES:
        return None  # readme / license / changelog 等纯文本类，不包裹
    if ext:
        return ext.lstrip(".")
    return None


def _wrap_content(name: str, path: str, content: str) -> str:
    """代码/配置类文件内容用对应语言代码围栏包裹，纯文档类原样返回。"""
    lang = _code_fence_language(path)
    if lang:
        return f"【{name}】\n```{lang}\n{content}\n```"
    return f"【{name}】\n{content}"


def _is_text_file_name(name: str) -> bool:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    ext = os.path.splitext(base)[1].lower()
    return ext in TEXT_EXTENSIONS or base.lower() in TEXT_FILENAMES


def _is_text_attachment(att: Dict[str, Any]) -> bool:
    if att.get("isImage"):
        return False
    name = att.get("name") or att.get("path") or ""
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf" or ext in IMAGE_EXTENSIONS:
        return False
    return _is_text_file_name(name)


def _resolve(cwd: str, path: str) -> str:
    resolved = path if os.path.isabs(path) else os.path.join(cwd, path)
    return os.path.realpath(resolved)


def _extract_text_paths(text: str, cwd: str) -> List[str]:
    """从消息文本中提取反引号包裹的、真实存在的文本文件路径。"""
    found: List[str] = []
    for token in _BACKTICK_RE.findall(text):
        token = token.strip()
        if not token:
            continue
        path = _resolve(cwd, token)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf" or ext in IMAGE_EXTENSIONS:
            continue
        if not os.path.isfile(path):
            continue
        if _is_text_file_name(path):
            found.append(path)
    return found


def _read_text(path: str) -> str:
    """读取文本文件：优先 UTF-8/GBK，失败时替换非法字节，超长截断。"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        return f"读取失败：{e}"
    for enc in ("utf-8-sig", "gbk"):
        try:
            content = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        content = raw.decode("utf-8", errors="replace")
    if len(content) > MAX_TEXT_CHARS:
        content = content[:MAX_TEXT_CHARS] + "\n…（内容过长，已截断）"
    return content


def _convert_word_to_markdown(path: str) -> str:
    """用 aspose-words-foss 把 doc/docx 转成 markdown 文本返回，超长截断。"""
    md_path = None
    try:
        import aspose.words_foss as aw
        doc = aw.Document(path)
        fd, md_path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        doc.save(md_path, aw.SaveFormat.MARKDOWN)
        with open(md_path, "rb") as f:
            raw = f.read()
        content = raw.decode("utf-8", errors="replace")
        if len(content) > MAX_TEXT_CHARS:
            content = content[:MAX_TEXT_CHARS] + "\n…（内容过长，已截断）"
        return content
    except Exception as e:
        return f"转换失败：{e}"
    finally:
        if md_path:
            try:
                os.remove(md_path)
            except OSError:
                pass


def _read_or_convert(path: str) -> str:
    """doc/docx 走 Word 转 markdown，其余文本直接读取。"""
    if os.path.splitext(path)[1].lower() in WORD_EXTENSIONS:
        return _convert_word_to_markdown(path)
    return _read_text(path)


class TextHandler(BaseLocalHandler):
    """透传纯文本文件内容给 UI 呈现"""

    name = "text"
    description = "文本文件内容透传"

    def can_handle(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None, cwd: Optional[str] = None) -> bool:
        if any(_is_text_attachment(a) for a in (attachments or [])):
            return True
        return bool(cwd) and bool(_extract_text_paths(text, cwd))

    def handle(self, bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        cwd = bs.get("cwd") or os.getcwd()
        parts: List[str] = []
        seen: set = set()
        for att in attachments or []:
            if not _is_text_attachment(att):
                continue
            path = _resolve(cwd, att.get("path") or "")
            if path in seen:
                continue
            seen.add(path)
            name = att.get("name") or os.path.basename(path)
            if not os.path.isfile(path):
                parts.append(f"【{name}】错误：文件不存在（{path}）")
                continue
            parts.append(_wrap_content(name, path, _read_or_convert(path)))
        for path in _extract_text_paths(text, cwd):
            if path in seen:
                continue
            seen.add(path)
            name = os.path.basename(path)
            parts.append(_wrap_content(name, path, _read_or_convert(path)))
        return "\n\n".join(parts) if parts else "未发现可读取的文本文件"


# 导入即注册：排在 PDF / OCR 之后，避免抢占图片与 PDF
local_handler_registry.register(TextHandler())
