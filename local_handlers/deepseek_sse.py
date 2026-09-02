"""
DeepSeek 网页版 SSE 原始流解析
=============================
把 CodexWebProxy 返回的网页版 SSE 原始流（event:/data: 行）解析为可见回复文本。

- 与代理侧旧版 `_parse_sse_body` 的正文提取规则保持一致；
- 兼容 DOM 兜底包装的最小 SSE（初始 envelope 带 RESPONSE 片段）；
- 未知帧 / 未知字段一律跳过、不报错，为将来代理返回更完整下游流保留兼容。
"""
import json
from typing import List


def parse_sse_content(sse_text: str) -> str:
    """从 DeepSeek 网页版 SSE 原始流中提取可见正文（按出现顺序拼接）。

    只拼接正文内容，不修改文本本身——换行等按原样保留，
    单换行的展示由渲染层（remark-breaks）负责。
    """
    parts: List[str] = []
    for line in (sse_text or "").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        # 初始 envelope：v.response.fragments[].content（RESPONSE 片段）
        v = obj.get("v")
        if isinstance(v, dict):
            response = v.get("response")
            if isinstance(response, dict):
                for frag in response.get("fragments") or []:
                    if (
                        isinstance(frag, dict)
                        and frag.get("type") == "RESPONSE"
                        and isinstance(frag.get("content"), str)
                    ):
                        parts.append(frag["content"])

        # 增量追加：p="response/fragments/-1/content", o="APPEND", v="..."
        if obj.get("o") == "APPEND" and obj.get("p") == "response/fragments/-1/content":
            val = obj.get("v")
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, dict) and isinstance(val.get("v"), str):
                parts.append(val["v"])
            continue

        # BATCH 操作：批量补丁里可能追加新的 RESPONSE 片段（正文起始内容）
        if obj.get("o") == "BATCH" and isinstance(obj.get("v"), list):
            for item in obj["v"]:
                if (
                    isinstance(item, dict)
                    and item.get("p") == "fragments"
                    and item.get("o") == "APPEND"
                ):
                    frags = item.get("v")
                    if isinstance(frags, list):
                        for frag in frags:
                            if (
                                isinstance(frag, dict)
                                and frag.get("type") == "RESPONSE"
                                and isinstance(frag.get("content"), str)
                            ):
                                parts.append(frag["content"])
                    elif (
                        isinstance(frags, dict)
                        and frags.get("type") == "RESPONSE"
                        and isinstance(frags.get("content"), str)
                    ):
                        parts.append(frags["content"])
            continue

        # 裸增量：{"v":"文本"}
        if "v" in obj and "p" not in obj and "o" not in obj:
            val = obj.get("v")
            if isinstance(val, str):
                parts.append(val)

    return "".join(parts)


def parse_sse_references(sse_text: str) -> list:
    """从 SSE 原始流中提取引用条目（搜索结果 / 引用元数据）。

    收集 SEARCH 片段 results 补丁与 fragment 内 references 中的条目，
    按引用编号（cite_index / index）去重并升序返回，每项：
    {"index": int, "url": str, "title": str|None, "snippet": str|None}
    无法与 [citation:N] 对应的无编号条目会被忽略。
    """
    refs = {}
    for line in (sse_text or "").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        items = []
        # 1) 初始 envelope / fragment 内的 results 与 references
        v = obj.get("v")
        if isinstance(v, dict):
            response = v.get("response")
            if isinstance(response, dict):
                for frag in response.get("fragments") or []:
                    if not isinstance(frag, dict):
                        continue
                    if isinstance(frag.get("results"), list):
                        items.extend(frag["results"])
                    if isinstance(frag.get("references"), list):
                        items.extend(frag["references"])
        # 2) results 补丁：p 以 /results 结尾，v 为条目数组
        if str(obj.get("p") or "").endswith("/results") and isinstance(obj.get("v"), list):
            items.extend(obj["v"])

        for it in items:
            if not isinstance(it, dict):
                continue
            url = it.get("url")
            if not isinstance(url, str) or not url:
                continue
            index = it.get("cite_index", it.get("index"))
            if not isinstance(index, int):
                continue
            refs[index] = {
                "index": index,
                "url": url,
                "title": it.get("title") if isinstance(it.get("title"), str) else None,
                "snippet": it.get("snippet") if isinstance(it.get("snippet"), str) else None,
            }

    return [refs[i] for i in sorted(refs)]
