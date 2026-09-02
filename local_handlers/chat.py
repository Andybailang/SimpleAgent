"""
DeepSeek 对话处理器（Local Deepseek）
====================================
LocalLLM 模式的兜底大脑：把用户消息转发给本地 Deepseek 代理
（OpenAI 兼容 POST /v1/chat/completions），并把收到的回复原样回传。
- 只转发最新一条用户消息（single_turn），上下文由网页版 DeepSeek 维护；
- 首次使用（程序启动后第一次）自动发送「工具使用教学」：教网页版用文本协议
  声明本地工具调用（generate_image / generate_video 等白名单工具），本地解析
  后执行并把结果以文本回传，形成与 engine 解耦的本地工具微循环；
- /teach-tool-usage 技能命中时强制重发教学（用户手动新开网页对话后使用）；
- raw_sse=True 时带自定义请求头要求代理返回网页版 SSE 原始流，本处理器自行解析正文；
- 代理基本地址不可达时本处理器不参与匹配（can_handle=False）。
"""
import json
import os
import re
import socket
import time
import urllib.parse
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx

from path_util import strip_vermagic
from .base import BaseLocalHandler, extract_skill_name, extract_user_request
from .registry import local_handler_registry
from .deepseek_sse import parse_sse_content, parse_sse_references
from .deepseek_config import (
    LOCAL_DEEPSEEK_CONFIG,
    LOCAL_DEEPSEEK_TIMEOUT,
    LOCAL_DEEPSEEK_PROBE_TIMEOUT,
    LOCAL_DEEPSEEK_RAW_SSE_HEADER,
    LOCAL_DEEPSEEK_ALLOWED_TOOLS,
    LOCAL_DEEPSEEK_PASSTHROUGH_SKILLS,
    LOCAL_DEEPSEEK_TEACH_ON_START,
    LOCAL_DEEPSEEK_MAX_TOOL_ROUNDS,
)
from tools.base import ToolContext
from tools import tool_registry


# 程序启动后是否已经发送过工具教学（进程级全局标记：首次使用 LocalLLM 时教一次）
_TOOL_TEACHING_DONE = False

# 文本协议：网页版用 <tool_call>...</tool_call> 声明工具调用（JSON）
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)

# 工具执行结果回传给网页版的标记前缀
_TOOL_RESULT_MARKER = "【工具结果】"

# 原始出入日志文件（仓库根 local_llm_trace.log，已被 .gitignore 忽略）。
# 记录每次发给 Local Deepseek 的原始文本与代理返回的原始响应，便于排查
# 路径转义 / URL 编码等显示链路问题。
def _trace_log(kind: str, content: str, max_chars: int = 4000) -> None:
    try:
        root = strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
        path = os.path.join(root, "local_llm_trace.log")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snippet = (content or "")[:max_chars]
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n===== [{stamp}] {kind} =====\n")
            f.write(snippet)
            f.write("\n===== END =====\n")
    except Exception:
        pass


def _build_tool_teaching_prompt() -> str:
    """构建工具使用教学文本：从注册表动态取白名单工具定义，教网页版文本协议。

    协议：
    - 需要调用工具时输出 <tool_call>{"tool": "...", "arguments": {...}}</tool_call>
    - 本地执行后回传 【工具结果】文本，再组织成给用户的最终回复
    - 展示生成的图片/视频用标准 Markdown 语法，路径中的特殊字符不要转义
    """
    lines = [
        "你是 BigCodex 的本地工具协调大脑。除了普通对话，你还可以通过调用本地工具完成任务。",
        "当前可用工具如下（只有这些工具可以被调用）：",
        "",
    ]
    for name in LOCAL_DEEPSEEK_ALLOWED_TOOLS:
        tool = tool_registry.get_tool(name)
        if tool is None:
            continue
        lines.append(f"## 工具 {tool.name}")
        lines.append(f"说明：{tool.description}")
        lines.append("参数 JSON Schema：")
        lines.append(json.dumps(tool.parameters, ensure_ascii=False, indent=2))
        lines.append("")
    lines += [
        "调用规则（严格遵守）：",
        "1. 只有确实需要生成/处理图片或视频时才调用工具，普通对话不要调用；",
        "2. 调用格式（必须单独一段，不要放在代码块里）：",
        '   <tool_call>{"tool": "工具名", "arguments": {"参数名": 值}}</tool_call>',
        "3. 一次回复可以输出多个 <tool_call>，本地会按顺序执行并把每个结果回传；",
        "4. 每次工具执行后，你会收到一条以【工具结果】开头的文本，",
        "   请基于结果组织成给用户的最终回复；",
        "5. 如果结果是本地文件路径，展示时必须用标准 Markdown 语法：",
        '   图片用 ![](绝对路径)，视频用 ![video](绝对路径)；',
        "   路径中的点号、下划线、括号等字符一律不要转义（不要写成 \\. 或 \\_ 形式）；",
        "6. 本地文件路径（如 D:\\...\\xxx.png）不是网址：不要把它转成 https:// 链接，",
        "   不要做 URL 编码（不要出现 %5C、%E5%85%B1 等百分号编码），原样输出路径即可；",
        "7. 工具执行失败时，如实转述错误原因，不要编造成功。",
    ]
    return "\n".join(lines)


def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从网页版回复中解析 <tool_call> 标记；容错：单条解析失败跳过。"""
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text or ""):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, dict) or not data.get("tool"):
            continue
        calls.append({
            "tool": str(data["tool"]),
            "arguments": data.get("arguments") or {},
        })
    return calls


def _build_tool_result_text(tool_name: str, result: Any) -> str:
    """把工具执行结果转成回传给网页版的文本。"""
    if isinstance(result, dict) and result.get("__media__"):
        return f"{_TOOL_RESULT_MARKER} 工具 {tool_name} 返回媒体内容（原始数据已省略，禁止读图）"
    return (
        f"{_TOOL_RESULT_MARKER} 工具 {tool_name} 返回：\n{str(result or '')}\n\n"
        "（展示规则：上面的本地文件路径不是网址，是保存在本机的真实文件。"
        "展示给用户时必须用标准 Markdown 语法原样写出：图片 ![](路径)、视频 ![video](路径)；"
        "禁止把它转成 https:// 链接，禁止 URL 编码，禁止在路径字符前加反斜杠转义。）"
    )


def _emit_tool_events(bs: Dict[str, Any], tool_name: str, tool_input: Dict[str, Any], tool_use_id: str, result: Any) -> None:
    """向会话事件队列发射 tool_use / tool_result 事件（与正常 LLM 路径形状一致，
    前端无需改动即可渲染工具卡片）。"""
    try:
        events = bs.get("events")
        if events is None:
            return
        block_index = int(bs.get("_local_tool_block_index") or 0) + 1
        bs["_local_tool_block_index"] = block_index
        tool_block = {
            "type": "tool_use",
            "id": tool_use_id,
            "name": tool_name,
            "input": tool_input,
        }
        events.append({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "index": block_index,
                "content_block": tool_block,
            },
        })
        events.append({
            "type": "assistant",
            "message": {"content": [tool_block]},
            "session_id": bs.get("session_id"),
        })
        events.append({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "content": str(result) if not isinstance(result, dict) else "[图片 × 已省略]",
            "session_id": bs.get("session_id"),
        })
    except Exception:
        pass


class _LocalToolContext(ToolContext):
    """LocalLLM 模式的轻量工具上下文：不依赖 agent 实例，直接基于会话 cwd。
    仅白名单生成类工具使用（generate_image / generate_video），权限按默认放行。"""

    def __init__(self, cwd: str, permission_mode: str = "default"):
        super().__init__(agent=None, cwd=cwd, permission_mode=permission_mode)

    def resolve_path(self, path: str, tool_name: str = "") -> str:
        raw = os.path.expanduser(path or "")
        if not os.path.isabs(raw):
            raw = os.path.join(self.cwd, raw)
        return os.path.realpath(raw)

    def check_permission(self, tool_name: str, tool_input: Optional[Dict[str, Any]] = None) -> Optional[str]:
        return None


def _deepseek_reachable() -> bool:
    """探测 Local Deepseek 代理基本地址是否可达（TCP 连接，短超时）。"""
    url = str(LOCAL_DEEPSEEK_CONFIG.get("api_url") or "")
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=LOCAL_DEEPSEEK_PROBE_TIMEOUT):
            return True
    except OSError:
        return False


class ChatHandler(BaseLocalHandler):
    """Local Deepseek 对话兜底处理器"""

    name = "chat"
    description = "DeepSeek 对话（Local Deepseek）"

    def can_handle(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None, cwd: Optional[str] = None) -> bool:
        if not text or not text.strip():
            return False
        # 代理基本地址不可达时视为无法处理
        return _deepseek_reachable()

    def _post(self, content: str) -> str:
        """向本地 Deepseek 代理发送一条文本消息，返回解析后的正文。"""
        url = str(LOCAL_DEEPSEEK_CONFIG.get("api_url") or "")
        headers = {
            "Authorization": f"Bearer {LOCAL_DEEPSEEK_CONFIG.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        raw_sse = bool(LOCAL_DEEPSEEK_CONFIG.get("raw_sse", False))
        if raw_sse:
            headers[LOCAL_DEEPSEEK_RAW_SSE_HEADER] = "1"
        payload = {
            "model": LOCAL_DEEPSEEK_CONFIG.get("model", "deepseek-chat"),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(LOCAL_DEEPSEEK_CONFIG.get("max_tokens") or 32768),
            "stream": bool(LOCAL_DEEPSEEK_CONFIG.get("stream", False)),
        }
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=LOCAL_DEEPSEEK_TIMEOUT)
        except Exception as e:
            _trace_log("REQ", content)
            return f"错误：调用 Local Deepseek 失败：{e}"
        if resp.status_code != 200:
            _trace_log("REQ", content)
            _trace_log("RESP", f"HTTP {resp.status_code}: {resp.text[:500]}")
            return f"错误：Local Deepseek 返回 {resp.status_code}：{resp.text[:300]}"
        try:
            data = resp.json()
        except Exception:
            _trace_log("REQ", content)
            _trace_log("RESP", f"非 JSON 响应: {resp.text[:500]}")
            return f"错误：Local Deepseek 响应解析失败：{resp.text[:300]}"
        try:
            reply = data["choices"][0]["message"]["content"]
        except Exception:
            _trace_log("REQ", content)
            _trace_log("RESP", f"缺少回复内容: {str(data)[:500]}")
            return f"错误：Local Deepseek 响应缺少回复内容：{str(data)[:300]}"
        # 先记录原始响应（SSE 或纯文本），再解析
        _trace_log("REQ", content)
        _trace_log("RESP_RAW", str(reply))
        # raw_sse 模式下代理返回网页版 SSE 原始流，解析成可见正文；
        # 旧代理不识别请求头时返回纯文本，直接兜底使用
        if raw_sse and any(ln.startswith("data:") for ln in str(reply).splitlines()):
            raw_sse_text = str(reply)
            parsed = parse_sse_content(raw_sse_text)
            if parsed:
                reply = parsed
            refs = parse_sse_references(raw_sse_text) or None
            _trace_log("RESP_PARSED", str(reply))
            return str(reply).strip() or "(空回复)", refs
        _trace_log("RESP_PARSED", str(reply))
        return str(reply).strip() or "(空回复)", None

    def _run_tool(self, cwd: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """本地执行白名单工具；未知工具/参数错误返回错误文本。"""
        tool = tool_registry.get_tool(tool_name)
        if tool is None:
            return f"错误：未知工具 '{tool_name}'，可用工具：{', '.join(LOCAL_DEEPSEEK_ALLOWED_TOOLS)}"
        context = _LocalToolContext(cwd=cwd, permission_mode="default")
        try:
            return tool_registry.execute_tool(context, tool_name, **arguments)
        except Exception as e:
            return f"错误：工具 {tool_name} 执行失败：{e}"

    def handle(self, bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        global _TOOL_TEACHING_DONE

        # /teach-tool-usage 技能：强制重发工具教学（用户手动新开网页对话后使用）
        if extract_skill_name(text) == "teach-tool-usage":
            teach_text = _build_tool_teaching_prompt()
            _TOOL_TEACHING_DONE = True
            reply, _refs = self._post(teach_text)
            if reply.startswith("错误"):
                return reply
            return "已向 Local Deepseek 重新发送工具使用教学。\n\n" + (
                reply if len(reply) < 300 else reply[:300] + "…"
            )

        # 技能透传：命中白名单（image-builder / video-builder）时，把技能描述全文
        # 连同用户请求一起发给网页版，让其按技能规范执行；否则只透传真实用户输入。
        skill_name = extract_skill_name(text)
        if skill_name and skill_name in LOCAL_DEEPSEEK_PASSTHROUGH_SKILLS:
            user_text = (text or "").strip()
        else:
            user_text = extract_user_request(text) or (text or "").strip()
        if not user_text:
            return "错误：没有可发送的用户消息"

        # 程序启动后第一次使用 LocalLLM：先静默发送一次工具教学（回复不展示给用户）
        if LOCAL_DEEPSEEK_TEACH_ON_START and not _TOOL_TEACHING_DONE:
            teach_text = _build_tool_teaching_prompt()
            teach_reply, _refs = self._post(teach_text)
            if not teach_reply.startswith("错误"):
                _TOOL_TEACHING_DONE = True

        cwd = bs.get("cwd") or os.getcwd()
        max_rounds = int(LOCAL_DEEPSEEK_MAX_TOOL_ROUNDS or 3)
        final_reply = ""
        tool_seq = int(bs.get("_local_tool_seq") or 0)
        for _round in range(max_rounds):
            reply, refs = self._post(user_text if _round == 0 else final_reply)
            if refs:
                bs["local_references"] = refs
            tool_calls = _parse_tool_calls(reply)
            if not tool_calls:
                return reply
            # 有工具调用：逐个执行并把结果文本回传网页版，等待其组织最终回复
            result_parts: List[str] = []
            for idx, call in enumerate(tool_calls):
                tool_name = str(call.get("tool") or "")
                arguments = call.get("arguments") or {}
                tool_seq += 1
                bs["_local_tool_seq"] = tool_seq
                tool_use_id = f"local_{int(time.time() * 1000)}_{tool_seq}"
                result = self._run_tool(cwd, tool_name, arguments)
                _emit_tool_events(bs, tool_name, arguments, tool_use_id, result)
                result_parts.append(_build_tool_result_text(tool_name, result))
            final_reply = "\n\n".join(result_parts)

        return "（达到本地工具调用轮数上限，请重试）"


# 导入即注册：放在所有处理器最后，作为 LocalLLM 的兜底大脑
local_handler_registry.register(ChatHandler())
