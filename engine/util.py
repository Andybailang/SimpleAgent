"""engine.util — 核心数据模型与内容/媒体辅助。

包含 Message 数据结构、工具结果转换/小抄帖、文件转 data URL/base64 与模型响应记录等。
"""
import base64
import mimetypes
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cache_monitor import CACHE_MONITOR

def _usage_field(usage: Any, path: str, default: int = 0) -> int:
    """从 usage 对象/dict 读取 token 字段（支持点号路径），兼容 OpenAI/Responses/Anthropic。

    设计上统一三种用法：
    - OpenAI Chat / Responses / Anthropic 的 usage 有时是 object（OpenAI SDK），有时是
      dict（串流 raw JSON），这里全部兼容；读不到或解析失败一律返回 default。
    """
    cur: Any = usage
    for part in path.split("."):
        if cur is None:
            return default
        try:
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = getattr(cur, part, None)
        except Exception:
            return default
    if cur is None:
        return default
    try:
        return int(cur or 0)
    except Exception:
        return default

def _tool_result_content(result: Any) -> Any:
    """把工具执行结果转成 Anthropic tool_result.content。

    媒体结构（{"__media__": [...]}）一律转文本占位——图片原始数据绝不进入模型上下文
    （需要看图时由后端工具内部处理，如 generate_image / extract_text_from_image）。
    其余原样返回（字符串或错误文本）。
    """
    if isinstance(result, dict) and result.get("__media__"):
        blocks: List[Dict[str, Any]] = []
        if result.get("text"):
            blocks.append({"type": "text", "text": str(result["text"])})
        blocks.append({
            "type": "text",
            "text": f"[图片 × {len(result['__media__'])}（原始数据已省略，禁止读图）]",
        })
        return blocks
    return result
# ---- Agent 主动旁白（<agent-note>）----
# 工具结果命中反爬/拦截特征时，在结果前注入给模型的提示，引导模型换可行方式重试。
# 文案为静态模板，不拼接工具输出（防 prompt injection）。
# 规则：(工具名精确或前缀, 特征正则, 提示文案)

AGENT_NOTE_RULES: List[Tuple[str, str, str]] = [
    (
        "mcp_fetch_",
        r"403|cloudflare|captcha|验证码|人机验证|access\s*denied|access_denied|waf|反爬|rate\s*limit",
        "fetch 抓取被目标站点反爬拦截（403/验证码/Cloudflare 等），返回内容不可用。"
        "请换方式重试：用 Bash 执行 curl 并自定义 User-Agent/Referer 等请求头抓取；"
        "或用 Bash 运行 python/httpx 脚本带 Cookie 或代理抓取；"
        "也可以尝试抓取页面缓存或搜索引擎快照。",
    ),
]
def _maybe_attach_agent_note(tool_name: str, result: Any) -> Any:
    """工具结果命中反爬等特征时，在结果前注入 <agent-note> 旁白（仅字符串结果）。"""
    if not isinstance(result, str):
        return result
    name = (tool_name or "").lower()
    lower_result = result.lower()
    for match_name, pattern, template in AGENT_NOTE_RULES:
        if not (name == match_name or name.startswith(match_name)):
            continue
        if re.search(pattern, lower_result, re.IGNORECASE):
            return f"<agent-note>{template}</agent-note>\n\n{result}"
    return result
def _record_model_response(model: str, text: str) -> None:
    """把模型本轮返回的全部内容（思考 + 文本 + 工具调用，按到达顺序）记入缓存监控（From 方向）。

    与请求记录（To 方向）分开存储；任何异常都不影响主流程。
    """
    try:
        if not text:
            return
        CACHE_MONITOR.record_response(model, text)
    except Exception:
        pass
@dataclass
class Message:
    """消息结构"""
    role: str  # 'user' 或 'assistant'
    content: Any  # 普通文本 str，或 OpenAI 多模态内容块 list
    timestamp: int
    tool_name: Optional[str] = None  # 工具调用时记录工具名
    tool_input: Optional[Dict[str, Any]] = None  # 工具调用输入
    tool_result: Optional[str] = None  # 工具调用结果

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可持久化的 dict。"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_result": self.tool_result,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Message":
        """从 dict 恢复 Message。"""
        return Message(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0),
            tool_name=data.get("tool_name"),
            tool_input=data.get("tool_input"),
            tool_result=data.get("tool_result"),
        )
def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"
def _file_to_data_url(path: str) -> Optional[str]:
    """Read a local file and return it as a base64 data URL (for image blocks)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            return None
        return f"data:{_guess_mime(path)};base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception:
        return None
def _file_to_base64(path: str) -> Optional[str]:
    """Read a local file and return raw base64 payload (for Anthropic image blocks)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            return None
        return base64.b64encode(raw).decode('ascii')
    except Exception:
        return None


# 只读工具名（_resolve_path 的软链接越界放行与只读权限判定用；注册表权限声明为主）
READ_ONLY_TOOLS = {"Read", "LS", "Grep", "Glob", "TodoWrite", "SemanticSearch"}
