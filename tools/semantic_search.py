"""
SemanticSearch 工具：基于向量（嵌入模型）的语义搜索
"""
import os
import fnmatch
import httpx
from typing import List, Optional, Tuple
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext
from .semantic_config import (
    SEMANTIC_SEARCH_ENABLED,
    SEMANTIC_SEARCH_API_BASE,
    SEMANTIC_SEARCH_API_KEY,
    SEMANTIC_SEARCH_MODEL,
    SEMANTIC_SEARCH_TOP_K,
    SEMANTIC_SEARCH_CHUNK_SIZE,
    SEMANTIC_SEARCH_CHUNK_OVERLAP,
    SEMANTIC_SEARCH_MAX_FILES,
    SEMANTIC_SEARCH_MAX_CHUNKS,
    SEMANTIC_SEARCH_MAX_FILE_BYTES,
    SEMANTIC_SEARCH_TIMEOUT,
)


class SemanticSearchTool(BaseTool):
    """SemanticSearch 工具实现"""

    # 最近一次向量化失败时的错误信息（供 execute 返回给模型）
    _semantic_error = ""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="SemanticSearch",
            description=(
                "基于向量（嵌入模型）做语义搜索：用自然语言 query 在工作区文本中找语义最相关的片段。"
                "参数：query（必填，自然语言查询）；path（可选，目录或文件，默认当前目录）；"
                "glob（可选，按文件名过滤，如 *.md）；top_k（可选，返回条数，默认 5）。"
                "适合描述性查找（如“处理附件上传的逻辑”），比 Grep 正则更灵活。注意：只搜本地文件，不支持网页/网址内容。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言查询内容"},
                    "path": {"type": "string", "description": "要搜索的目录或文件路径（默认当前目录）"},
                    "glob": {"type": "string", "description": "按文件名 glob 过滤（如 *.md、*.py）"},
                    "top_k": {"type": "integer", "description": "返回的相似片段数量（默认 5）"},
                },
                "required": ["query"],
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
            requires_semantic_key=True,
        )

    @classmethod
    def execute(cls, context: ToolContext, query: str, path: Optional[str] = ".",
                glob: Optional[str] = None, top_k: Optional[int] = None) -> str:
        """基于嵌入模型的语义搜索：把 query 与工作区文本分块分别向量化，按余弦相似度返回 Top-K。"""
        query = (query or "").strip()
        if not query:
            return "错误：query 不能为空"
        if not SEMANTIC_SEARCH_ENABLED:
            return "错误：semantic_search 未启用（SEMANTIC_SEARCH_ENABLED=false）"
        if not SEMANTIC_SEARCH_API_KEY:
            return ("错误：semantic_search 未配置 API key（请在 src/agent/semantic.env 中设置 "
                    "SEMANTIC_SEARCH_API_KEY，或设置环境变量 SILICONFLOW_API_KEY）")
        try:
            k = max(1, min(int(top_k) if top_k is not None else SEMANTIC_SEARCH_TOP_K, 50))
        except (TypeError, ValueError):
            k = SEMANTIC_SEARCH_TOP_K
        try:
            target = context.resolve_path(path or ".", "SemanticSearch")
        except ValueError as e:
            return f"错误：{e}"
        if not os.path.exists(target):
            return f"错误：路径 {target} 不存在"

        files: List[str] = []
        if os.path.isfile(target):
            files = [target]
        elif os.path.isdir(target):
            skip_dirs = {"node_modules", "venv", ".venv", "dist", "build", "__pycache__"}
            for root, dirs, names in os.walk(target):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip_dirs]
                for name in names:
                    if glob and not fnmatch.fnmatch(name, glob):
                        continue
                    files.append(os.path.join(root, name))
                    if len(files) >= SEMANTIC_SEARCH_MAX_FILES:
                        break
                if len(files) >= SEMANTIC_SEARCH_MAX_FILES:
                    break
        else:
            return f"错误：路径 {target} 不存在"

        chunks: List[Tuple[str, str, int]] = []  # (text, file_path, line_no)
        skipped = 0
        for fp in files:
            try:
                with open(fp, "rb") as f:
                    head = f.read(8192)
                    if b"\x00" in head:
                        skipped += 1
                        continue
                    f.seek(0)
                    raw = f.read(SEMANTIC_SEARCH_MAX_FILE_BYTES + 1)
                if len(raw) > SEMANTIC_SEARCH_MAX_FILE_BYTES:
                    skipped += 1
                    continue
                text = ""
                for enc in ("utf-8", "gbk", "latin-1"):
                    try:
                        text = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
            except OSError:
                skipped += 1
                continue
            if not text.strip():
                continue
            for chunk, line_no in cls._chunk_text(text):
                chunks.append((chunk, fp, line_no))
                if len(chunks) >= SEMANTIC_SEARCH_MAX_CHUNKS:
                    break
            if len(chunks) >= SEMANTIC_SEARCH_MAX_CHUNKS:
                break

        if not chunks:
            return "错误：在指定范围内没有可检索的文本文件"
        if len(chunks) == 1:
            chunk, fp, line_no = chunks[0]
            return (f"语义搜索：仅找到 1 个片段，无需向量比对。\n"
                    f"文件 {fp}:{line_no}\n{chunk.strip()[:500]}")

        cls._semantic_error = ""
        try:
            query_vec = cls._embed_texts([query])
            if query_vec is None:
                return cls._semantic_error
            chunk_vecs = cls._embed_texts([c[0] for c in chunks])
            if chunk_vecs is None:
                return cls._semantic_error
        except Exception as e:
            return f"错误：语义搜索执行失败：{e}"

        scored = [
            (cls._cosine_similarity(query_vec[0], vec), chunk)
            for chunk, vec in zip(chunks, chunk_vecs)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        lines = [f"语义搜索结果（query: {query!r}，扫描 {len(files)} 个文件 / {len(chunks)} 个片段）："]
        if skipped:
            lines.append(f"（已跳过 {skipped} 个超大或不可读文件）")
        for rank, (score, (chunk, fp, line_no)) in enumerate(scored[:k], 1):
            snippet = " ".join(chunk.strip().split())[:200]
            lines.append(f"{rank}. 相似度 {score:.3f} | {fp}:{line_no}")
            lines.append(f"   {snippet}")
        return "\n".join(lines)

    @classmethod
    def _embed_texts(cls, texts: List[str]) -> Optional[List[List[float]]]:
        """批量调用嵌入 API，返回向量列表；失败时返回 None（错误信息写入 cls._semantic_error）。"""
        url = f"{SEMANTIC_SEARCH_API_BASE}/embeddings"
        headers = {
            "Authorization": f"Bearer {SEMANTIC_SEARCH_API_KEY}",
            "Content-Type": "application/json",
        }
        vectors: List[List[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            try:
                resp = httpx.post(
                    url,
                    json={"model": SEMANTIC_SEARCH_MODEL, "input": batch},
                    headers=headers,
                    timeout=SEMANTIC_SEARCH_TIMEOUT,
                )
            except Exception as e:
                cls._semantic_error = f"错误：调用嵌入 API 失败：{e}"
                return None
            if resp.status_code != 200:
                cls._semantic_error = f"错误：嵌入 API 返回 {resp.status_code}：{resp.text[:300]}"
                return None
            try:
                data = resp.json().get("data") or []
            except Exception:
                cls._semantic_error = "错误：嵌入 API 响应解析失败"
                return None
            if len(data) != len(batch):
                cls._semantic_error = f"错误：嵌入 API 返回数量异常（期望 {len(batch)}，实际 {len(data)}）"
                return None
            for item in data:
                emb = item.get("embedding")
                if not emb:
                    cls._semantic_error = "错误：嵌入 API 响应缺少 embedding 字段"
                    return None
                vectors.append(emb)
        return vectors

    @classmethod
    def _cosine_similarity(cls, a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @classmethod
    def _chunk_text(cls, text: str) -> List[Tuple[str, int]]:
        """把文本切成带重叠的字符块，返回 (chunk 文本, 起始行号 1-based)。"""
        size = SEMANTIC_SEARCH_CHUNK_SIZE
        overlap = min(SEMANTIC_SEARCH_CHUNK_OVERLAP, size // 2)
        chunks: List[Tuple[str, int]] = []
        start = 0
        line_no = 1
        while start < len(text):
            end = min(len(text), start + size)
            chunks.append((text[start:end], line_no))
            if end >= len(text):
                break
            line_no += text.count("\n", start, end - overlap)
            start = end - overlap
        return chunks
