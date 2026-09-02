"""
search_and_extract 工具：SearXNG 搜索 + 正文/图片提取（搜索与提取解耦）。

- search_searxng(query, num_results, time_range)  调 SearXNG JSON 接口，返回 [{title,url,snippet}]
- search_and_extract(query, num_results, extract_images, max_body_chars, time_range)
    先搜索拿 URL 列表，再复用 extract_from_urls 的提取逻辑组装 JSON

提取逻辑复用 extract_from_urls，不重复实现。
"""
import json
import os
from typing import Any, Dict, List

import httpx

from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext
from .extract_from_urls import _extract_from_urls

# SearXNG 配置（与 mcptool/searxng_server 保持一致）
SEARXNG_BASE_URL = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080").rstrip("/")
SEARCH_PATH = "/search"
REQUEST_TIMEOUT = 30.0
DEFAULT_NUM_RESULTS = 5
MAX_NUM_RESULTS = 10
# 合法时间范围（SearXNG 支持）
VALID_TIME_RANGES = {"day", "week", "month", "year"}


def _search_searxng(query: str, num_results: int, time_range: str = "") -> List[Dict[str, str]]:
    """调 SearXNG JSON 接口，返回 [{title, url, snippet}]。"""
    params = {
        "q": query,
        "format": "json",
        "language": "auto",
        "categories": "general",
        "safesearch": 0,
    }
    if time_range in VALID_TIME_RANGES:
        params["time_range"] = time_range
    resp = httpx.get(
        SEARXNG_BASE_URL + SEARCH_PATH,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    items: List[Dict[str, str]] = []
    for r in results[:num_results]:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        items.append({
            "title": _clean(r.get("title")) or "(无标题)",
            "url": url,
            "snippet": _clean(r.get("content")),
        })
    return items


def _clean(text: Any) -> str:
    """清理标题/摘要：去空白、截断。"""
    if not text:
        return ""
    return " ".join(str(text).split())[:300]


def _search_and_extract(query: str, num_results: int, extract_images: bool,
                        max_body_chars: int, time_range: str = "") -> str:
    """主函数：搜索 + 提取，合并标题/摘要。"""
    try:
        items = _search_searxng(query, num_results, time_range)
    except Exception as e:
        return json.dumps([{
            "url": "",
            "title": "",
            "snippet": "",
            "body_text": None,
            "images": [],
            "status": f"failed: 搜索失败：{e}",
        }], ensure_ascii=False)

    if not items:
        return "未找到与 “%s” 相关的搜索结果。" % query

    urls = [it["url"] for it in items]
    extracted = json.loads(_extract_from_urls(urls, extract_images, max_body_chars))
    # 按 URL 合并标题/摘要
    title_by_url = {it["url"]: it["title"] for it in items}
    snippet_by_url = {it["url"]: it["snippet"] for it in items}
    for item in extracted:
        u = item.get("url", "")
        item["title"] = title_by_url.get(u, "")
        item["snippet"] = snippet_by_url.get(u, "")
    return json.dumps(extracted, ensure_ascii=False)


class SearchAndExtractTool(BaseTool):
    """search_and_extract 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="search_and_extract",
            description=(
                "用 SearXNG 搜索关键词，并对搜索结果中的每个网页批量提取正文与图片链接。"
                "参数：query（必填，搜索关键词）；num_results（可选，默认 5，最大 10）；"
                "extract_images（可选，默认 true，是否提取图片）；"
                "max_body_chars（可选，默认 3000，每篇正文最大字符数）；"
                "time_range（可选，搜索时间范围：day/week/month/year，如“最近新闻”用 week）。"
                "返回 JSON 数组，每项含 title/snippet/url/body_text/images/status。"
                "适合“搜索新闻并提取每篇正文与配图”的场景。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "num_results": {"type": "integer", "description": "返回结果条数（默认 5，最大 10）"},
                    "extract_images": {"type": "boolean", "description": "是否提取图片链接（默认 true）"},
                    "max_body_chars": {"type": "integer", "description": "每篇正文最大字符数（默认 3000）"},
                    "time_range": {
                        "type": "string",
                        "enum": ["", "day", "week", "month", "year"],
                        "description": "搜索时间范围（day/week/month/year；留空不限）",
                    },
                },
                "required": ["query"],
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
        )

    @classmethod
    def execute(cls, context: ToolContext, query: str,
                num_results: int = DEFAULT_NUM_RESULTS,
                extract_images: bool = True,
                max_body_chars: int = 3000,
                time_range: str = "") -> str:
        query = (query or "").strip()
        if not query:
            return "错误：query 不能为空。"
        try:
            n = max(1, min(int(num_results or DEFAULT_NUM_RESULTS), MAX_NUM_RESULTS))
        except (TypeError, ValueError):
            n = DEFAULT_NUM_RESULTS
        try:
            body_limit = max(100, min(int(max_body_chars or 3000), 20000))
        except (TypeError, ValueError):
            body_limit = 3000
        tr = str(time_range or "").strip().lower()
        if tr not in VALID_TIME_RANGES:
            tr = ""
        return _search_and_extract(query, n, bool(extract_images), body_limit, tr)
