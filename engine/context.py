"""engine.context — 上下文压缩、请求监控与本地估算。
"""
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    AUTO_COMPACT_CONTEXT_MAX_TOKENS,
    AUTO_COMPACT_THRESHOLD,
    CJK_TOKENS_FACTOR,
    COMPACT_KEEP_LAST,
    COMPACT_MAX_SUMMARY_CHARS,
    COMPACT_RECENT_HARD_CAP_CHARS,
    COMPACT_RECENT_MAX_CHARS,
    LOCAL_COMPACT_NOTICE,
    PAID_COMPACT_RECENT_MAX_TOKENS,
    RETRY_MAX_ATTEMPTS,
    _CACHE_PARAM_KEYS,
    _estimate_output_tokens,
    _is_rate_limited,
    _retry_delays,
)
from .util import Message, _usage_field
from cache_monitor import CACHE_MONITOR
from traffic import TRAFFIC
import httpx

class ContextMixin:
    def _serialize_for_cache(self, obj: Any) -> Any:
        """递归清理请求结构：图片等二进制块替换为固定标记，保留文本与工具结构。"""
        if isinstance(obj, dict):
            otype = obj.get("type")
            if otype in ("image", "image_url") or "image_url" in obj:
                return "__IMAGE__"
            return {k: self._serialize_for_cache(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_for_cache(v) for v in obj]
        return obj
    def _monitor_request_prompt(self, payload: Any, tools: Optional[List[Dict[str, Any]]] = None) -> None:
        """把本轮请求序列化为纯文本并记入缓存监控；任何异常都不影响主流程。

        序列化顺序固定为 system → tools 定义 → messages → 其余未知字段，
        并剔除 model/max_tokens/stream/temperature/thinking 等纯请求参数。
        这类固定内容每条请求都不变，置于字符串最前可使共享前缀最长，
        与真实 prompt 缓存的前缀行为一致。"""
        try:
            cleaned = self._serialize_for_cache(payload)
            ordered: Dict[str, Any] = {}
            system = None
            messages = None
            input_items = None
            if isinstance(cleaned, dict):
                system = cleaned.get("system")
                messages = cleaned.get("messages")
                input_items = cleaned.get("input")
                if tools is None:
                    tools = cleaned.get("tools")
            elif isinstance(cleaned, list):
                messages = cleaned
            # OpenAI 形态：system 提示内嵌在 messages 首条，拆出放到最前
            if isinstance(messages, list):
                first = messages[0] if messages else None
                if system is None and isinstance(first, dict) and first.get("role") == "system":
                    system = first.get("content")
                    messages = messages[1:]
            # Responses API（/responses）形态：对话在 input 列表，首条是 system message
            elif isinstance(input_items, list):
                first = input_items[0] if input_items else None
                if system is None and isinstance(first, dict) \
                        and first.get("type") == "message" and first.get("role") == "system":
                    system = first.get("content")
                    input_items = input_items[1:]
                messages = input_items
            if system is not None:
                ordered["system"] = system
            if tools:
                ordered["tools"] = self._serialize_for_cache(tools)
            if messages is not None:
                ordered["messages"] = messages
            if isinstance(cleaned, dict):
                for key, value in cleaned.items():
                    if key in ("system", "messages", "tools", "input") or key in _CACHE_PARAM_KEYS:
                        continue
                    ordered[key] = value
            CACHE_MONITOR.record(self.model_name, json.dumps(
                ordered, ensure_ascii=False, default=str))
        except Exception:
            pass
    def compact_context(self) -> str:
        """压缩 LLM 上下文：保留 system 与最近 COMPACT_KEEP_LAST 条消息，
        更早的消息用 LLM 生成摘要替换（失败时退化为截断拼接）。
        保留窗口超体积预算时，从最旧（靠近摘要）一端丢弃，保证最新消息完整。
        付费模型（billing != free）不调用 LLM 摘要，改为从最近一条开始累计，
        最多 COMPACT_KEEP_LAST 条且累计预估 token 不超过 PAID_COMPACT_RECENT_MAX_TOKENS，
        某条会使总和超限即停止（该条不并入），极端情况一条都不保留。
        只重建 self.messages（发给模型的上下文），不影响会话历史。"""
        conv = [m for m in self.messages if m.role != "system"]
        if self.billing != "free":
            # 付费模型省 token：从最近一条开始，最多 COMPACT_KEEP_LAST 条，
            # 累计预估 token <= PAID_COMPACT_RECENT_MAX_TOKENS，超预算即停止（该条不并入）。
            recent = self._recent_window_for_prune(conv)
            old = conv[:len(conv) - len(recent)]
            if not old:
                # 无旧消息可裁剪：数量与尺寸都未触发
                degraded = self._enforce_compact_cap()
                if degraded:
                    return f"上下文（{len(conv)} 条消息）未超预算，已将 {degraded} 个超大图片附件降级为文本占位"
                return f"上下文（{len(conv)} 条消息）无需压缩"
        else:
            if len(conv) <= COMPACT_KEEP_LAST:
                # 条数虽少，但如果最新消息本身超大（如 500K 图片），仍需降级图片才能救回上下文
                degraded = self._enforce_compact_cap()
                if degraded:
                    return f"上下文较短（{len(conv)} 条消息），已将 {degraded} 个超大图片附件降级为文本占位"
                return f"上下文较短（{len(conv)} 条消息），无需压缩"
            old = conv[:-COMPACT_KEEP_LAST]
            recent = self._trim_recent(conv[-COMPACT_KEEP_LAST:])
        summary = self._summarize_old_messages(old)
        kept_system = [m for m in self.messages if m.role == "system"]
        summary_text = self._wrap_summary_text(summary)
        if recent and recent[0].role == "user":
            if isinstance(recent[0].content, str):
                # 摘要合并进最近首条 user 消息，避免 Anthropic 接口出现连续 user 消息
                merged = Message(
                    role="user",
                    content=(summary_text + "\n\n---\n" + recent[0].content) if summary_text else recent[0].content,
                    timestamp=recent[0].timestamp,
                )
                self.messages = kept_system + [merged] + recent[1:]
            else:
                # 多模态（如图片）首条：把摘要作为 text 块插到前面，避免连续 user 消息
                blocks = list(recent[0].content) if isinstance(recent[0].content, list) else []
                if summary_text:
                    blocks = [{"type": "text", "text": summary_text}] + blocks
                merged = Message(
                    role="user",
                    content=blocks,
                    timestamp=recent[0].timestamp,
                )
                self.messages = kept_system + [merged] + recent[1:]
        elif summary_text:
            summary_msg = Message(
                role="user",
                content=summary_text,
                timestamp=int(datetime.now().timestamp() * 1000),
            )
            self.messages = kept_system + [summary_msg] + recent
        else:
            # old 为空（无可裁剪内容）：直接保留最近消息
            self.messages = kept_system + list(recent)
        degraded = self._enforce_compact_cap()
        extra = f"，另将 {degraded} 个超大图片降级为文本占位" if degraded else ""
        action = "本地裁剪" if self.billing != "free" else "合并为摘要"
        return f"已压缩上下文：{len(old)} 条旧消息{action}，保留最近 {len(recent)} 条消息{extra}"
    def _trim_recent(self, recent: List[Message]) -> List[Message]:
        """保留窗口超体积预算时，从最旧（靠近摘要）一端开始丢弃，保证最新消息完整。"""
        kept = list(recent)
        while len(kept) > 1 and self._messages_chars(kept) > COMPACT_RECENT_MAX_CHARS:
            kept.pop(0)
        return kept
    def _messages_chars(self, msgs: List[Message]) -> int:
        """估算消息体积（字符数）：文本按长度，图片按 data URL / base64 文本长度。"""
        return sum(self._message_chars(m) for m in msgs)
    def _message_chars(self, m: Message) -> int:
        """估算单条消息体积（字符数）：文本按长度，图片按 data URL / base64 文本长度。"""
        c = m.content
        if isinstance(c, str):
            return len(c)
        if isinstance(c, list):
            total = 0
            for b in c:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text":
                    total += len(str(b.get("text") or ""))
                elif btype == "image_url":
                    iu = b.get("image_url") or {}
                    if isinstance(iu, dict):
                        total += len(str(iu.get("url") or ""))
                elif btype == "image":
                    source = b.get("source") or {}
                    if isinstance(source, dict) and source.get("data"):
                        total += len(str(source["data"]))
            return total
        return len(str(c or ""))
    def _recent_window_for_prune(self, conv: List[Message]) -> List[Message]:
        """付费模型本地裁剪的有界窗口：从最近一条开始，最多纳入 COMPACT_KEEP_LAST 条，
        累计预估 token（= 各消息字符数 × CJK_TOKENS_FACTOR）不超过 PAID_COMPACT_RECENT_MAX_TOKENS。

        某条若会使累计总和超过预算，则立即停止（该条不并入）；极端情况下最近一条
        本身已超预算，则窗口为空（一条都不保留）。返回按时间顺序排列、仅包含被并入的
        最近消息列表。
        """
        window: List[Message] = []
        total = 0.0
        for m in reversed(conv):
            if len(window) >= COMPACT_KEEP_LAST:
                break
            tokens = self._message_chars(m) * CJK_TOKENS_FACTOR
            if total + tokens > PAID_COMPACT_RECENT_MAX_TOKENS:
                break
            window.append(m)
            total += tokens
        window.reverse()
        return window
    def _estimate_context_tokens(self, messages: List[Dict[str, Any]],
                                  tools: Optional[List[Dict[str, Any]]] = None,
                                  system_text: str = "") -> int:
        """本地估算完整请求体的 token 数（不等服务端 usage，防缓存命中率误导）。

        口径与 _estimate_output_tokens 一致：中文按 1 字≈0.6 token、ASCII 约 4 字符/token；
        含 system、全部 messages（含工具结果文本）、tools 定义。
        图片等二进制块在 _to_openai_user_content / _to_anthropic_user_content
        已替换为文本占位，不会重复计入原始字节。
        """
        def _tok(t: str) -> int:
            if not t:
                return 0
            return _estimate_output_tokens(str(t))

        total = 0
        if system_text:
            total += _tok(system_text)
        for m in messages or []:
            c = m.get("content")
            if isinstance(c, str):
                total += _tok(c)
            elif isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    btype = b.get("type")
                    if btype == "text":
                        total += _tok(b.get("text") or "")
                    elif btype == "thinking":
                        total += _tok(b.get("thinking") or "")
                    elif btype in ("image", "image_url"):
                        # 图像块：按固定预算估算（避免 base64 字符串虚高）
                        total += 200
                    elif btype == "tool_use":
                        total += _tok((b.get("name") or "") + " " + json.dumps(b.get("input") or {}, ensure_ascii=False))
                    elif btype == "tool_result":
                        # 工具执行结果文本：Anthropic 格式下工具结果以 tool_result
                        # 块存在于 user 消息 content 列表。此前未统计导致估算远小于
                        # 实际请求体（工具结果常是上下文中的大头）。
                        rc = b.get("content")
                        if isinstance(rc, str):
                            total += _tok(rc)
                        elif isinstance(rc, list):
                            for rb in rc:
                                if isinstance(rb, dict) and rb.get("type") == "text":
                                    total += _tok(rb.get("text") or "")
                                elif isinstance(rb, str):
                                    total += _tok(rb)
            else:
                total += _tok(c or "")
        for td in tools or []:
            total += _tok(json.dumps(td, ensure_ascii=False))
        return max(1, total)
    @property
    def _compact_limit_tokens(self) -> int:
        """自动压缩触发阈值 = context_max_tokens × compact_threshold。"""
        try:
            return max(1, int(self.context_max_tokens * self.compact_threshold))
        except Exception:
            return max(1, int(AUTO_COMPACT_CONTEXT_MAX_TOKENS * AUTO_COMPACT_THRESHOLD))
    def _find_turn_start_index(self, messages: List[Dict[str, Any]]) -> int:
        """定位当前 turn 的起点：从末尾往前找第一个真实 user 消息（本轮用户输入）。

        工具轮中工具结果以 user 角色写回（内容为 tool 引用或多个【工具...】文本段），
        需要跳过这类工具反馈，继续往前找真正的本轮起点。
        找不到时返回 0（保护全部消息）。
        """
        def _is_tool_feedback(m: Dict[str, Any]) -> bool:
            c = m.get("content")
            if isinstance(c, list):
                return any(isinstance(b, dict) and b.get("role") == "tool" for b in c)
            if isinstance(c, str):
                return c.lstrip().startswith("【工具")
            return False

        idx = len(messages) - 1
        while idx >= 0:
            m = messages[idx]
            if m.get("role") != "user":
                idx -= 1
                continue
            if not _is_tool_feedback(m):
                return idx
            idx -= 1
        return 0
    def _maybe_auto_compact(self, messages: List[Dict[str, Any]],
                            tools: Optional[List[Dict[str, Any]]] = None,
                            system_text: str = "") -> Optional[Dict[str, Any]]:
        """发送前统一入口：本地估算完整请求体，超阈值则压缩并返回 system 事件。

        返回 None 表示未触发；否则返回 {"type": "system", "subtype": "auto_compact",
        "message": ..., "estimated_tokens": ..., "limit_tokens": ...}（调用方 yield 出去）。
        压缩只重建 self.messages，不影响会话历史与当前轮状态。
        """
        try:
            if self.billing == "free":
                return None  # 免费模型：token 不敏感，不需要压缩
            est = self._estimate_context_tokens(messages, tools=tools, system_text=system_text)
            limit = self._compact_limit_tokens
            if est <= limit:
                return None
            result = self.auto_compact_context(protect_turn=True)
            return {
                "type": "system",
                "subtype": "auto_compact",
                "message": result,
                "estimated_tokens": est,
                "limit_tokens": limit,
            }
        except Exception:
            return None
    def auto_compact_context(self, protect_turn: bool = False) -> str:
        """带保护段的自动压缩：protect_turn=True 时优先从当前 turn 的 user 起点
        起整段保留（含其后的全部工具轮次），更早的历史用 LLM 摘要替换。

        若保护段本身已超阈值（说明当前 turn 内部已膨胀到极限），彻底退化为
        只保留最近 COMPACT_KEEP_LAST 条（丢工具轮的旧部分），确保压缩后可继续。

        付费模型（billing != free）为省 token 一律直接本地裁剪：从最近一条开始累计，
        最多 COMPACT_KEEP_LAST 条且累计预估 token 不超过 PAID_COMPACT_RECENT_MAX_TOKENS，
        某条会使总和超限即停止（该条不并入），极端情况一条都不保留；不调用 LLM 生成摘要
        （避免一次近乎全量的 prompt 造成缓存完全未命中、token 浪费）。
        """
        conv = [m for m in self.messages if m.role != "system"]
        if not conv:
            return "上下文为空，无需压缩"
        if self.billing != "free":
            # 付费模型省 token：从最近一条开始，最多 COMPACT_KEEP_LAST 条，
            # 累计预估 token <= PAID_COMPACT_RECENT_MAX_TOKENS，超预算即停止（该条不并入）；
            # 不发起 LLM 摘要请求。
            recent = self._recent_window_for_prune(conv)
            old = conv[:len(conv) - len(recent)]
            return self._rebuild_with_summary(old, recent)
        if protect_turn:
            api_msgs = [{"role": m.role, "content": m.content} for m in conv]
            start = self._find_turn_start_index(api_msgs)
            protect = conv[start:]
            protect_est = self._estimate_context_tokens(
                [{"role": m.role, "content": m.content} for m in protect])
            if protect_est <= self._compact_limit_tokens:
                old = conv[:start]
                recent = protect
                return self._rebuild_with_summary(old, recent)
            # 保护段自身已超阈值 → 彻底退化为只保留最近 COMPACT_KEEP_LAST 条
            # （原 compact_context 在条数少时短路返回"无需压缩"，这里必须强制压缩）
            old = conv[:-COMPACT_KEEP_LAST]
            recent = conv[-COMPACT_KEEP_LAST:]
            return self._rebuild_with_summary(old, recent)
        # 无保护段：沿用 /compact 逻辑（条数少时返回无需压缩属正常）
        return self.compact_context()
    def _rebuild_with_summary(self, old: List[Message], recent: List[Message]) -> str:
        """把 old 摘要为一条 user 消息并入 recent 首条（或前置独立摘要消息），
        重建 self.messages。与 compact_context 的重建逻辑一致。

        付费模型省 token（_summarize_old_messages 返回本地裁剪说明，不调用 LLM）时，
        同样复用此路径重建，保证消息结构与前缀一致。
        """
        summary = self._summarize_old_messages(old)
        kept_system = [m for m in self.messages if m.role == "system"]
        summary_text = self._wrap_summary_text(summary)
        if recent and recent[0].role == "user":
            if isinstance(recent[0].content, str):
                merged = Message(
                    role="user",
                    content=(summary_text + "\n\n---\n" + recent[0].content) if summary_text else recent[0].content,
                    timestamp=recent[0].timestamp,
                )
                self.messages = kept_system + [merged] + recent[1:]
            else:
                blocks = list(recent[0].content) if isinstance(recent[0].content, list) else []
                if summary_text:
                    blocks = [{"type": "text", "text": summary_text}] + blocks
                merged = Message(
                    role="user",
                    content=blocks,
                    timestamp=recent[0].timestamp,
                )
                self.messages = kept_system + [merged] + recent[1:]
        elif summary_text:
            summary_msg = Message(
                role="user",
                content=summary_text,
                timestamp=int(datetime.now().timestamp() * 1000),
            )
            self.messages = kept_system + [summary_msg] + recent
        else:
            # old 为空（无可裁剪内容）：直接保留最近消息
            self.messages = kept_system + list(recent)
        degraded = self._enforce_compact_cap()
        extra = f"，另将 {degraded} 个超大图片降级为文本占位" if degraded else ""
        action = "本地裁剪" if self.billing != "free" else "合并为摘要"
        return f"已自动压缩（发送前）：{len(old)} 条旧消息{action}，保留最近 {len(recent)} 条消息{extra}"
    def _wrap_summary_text(self, summary: str) -> str:
        """把摘要/裁剪说明包装成发给模型的 user 消息文本。

        付费模型本地裁剪时直接使用裁剪说明，不再叠加“以下是此前对话的摘要”前缀
        （避免与“已裁剪”语义打架）；免费模型使用 LLM 摘要并附加引导语。
        """
        if not summary:
            return ""
        if self.billing != "free":
            return summary
        return f"[以下是此前对话的摘要，请基于它继续当前任务]\n{summary}"
    def _enforce_compact_cap(self) -> int:
        """压缩后兜底：若保留上下文仍超过硬上限（如最新一条是超大图片），
        把超大图片块降级为文本占位，确保压缩后的上下文可继续使用。
        返回被降级的图片块数量。"""
        if self._messages_chars(self.messages) <= COMPACT_RECENT_HARD_CAP_CHARS:
            return 0
        targets: List[Dict[str, Any]] = []
        for mi, msg in enumerate(self.messages):
            if not isinstance(msg.content, list):
                continue
            for bi, block in enumerate(msg.content):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                size = 0
                if btype == "image_url":
                    iu = block.get("image_url") or {}
                    if isinstance(iu, dict):
                        size = len(str(iu.get("url") or ""))
                elif btype == "image":
                    source = block.get("source") or {}
                    if isinstance(source, dict) and source.get("data"):
                        size = len(str(source["data"]))
                if size > 0:
                    targets.append({
                        "mi": mi,
                        "bi": bi,
                        "size": size,
                        "name": str(block.get("name") or ""),
                    })
        if not targets:
            return 0
        targets.sort(key=lambda t: t["size"], reverse=True)
        degraded = 0
        for t in targets:
            if self._messages_chars(self.messages) <= COMPACT_RECENT_HARD_CAP_CHARS:
                break
            size_kb = t["size"] * 3 / 4 / 1024
            name = t["name"]
            if name:
                placeholder = f"[图片附件 {name}（约{size_kb:.0f}KB）因体积过大，已从上下文中移除]"
            else:
                placeholder = f"[图片附件（约{size_kb:.0f}KB）因体积过大，已从上下文中移除]"
            self.messages[t["mi"]].content[t["bi"]] = {"type": "text", "text": placeholder}
            degraded += 1
        return degraded
    def _summarize_old_messages(self, old: List[Message]) -> str:
        """汇总旧消息。

        付费模型（billing != free）为省 token 直接本地裁剪，返回一个本地占位说明，
        不再调用 LLM 生成摘要（避免一次近乎全量的 prompt 造成缓存完全未命中、token 浪费）；
        免费模型才调用 LLM 生成摘要；LLM 失败时退化为截断拼接。
        """
        if not old:
            return ""
        if self.billing != "free":
            return LOCAL_COMPACT_NOTICE.format(n=len(old))
        text = self._messages_to_text(old)
        summary = self._llm_summarize(text)
        if summary:
            return summary[:COMPACT_MAX_SUMMARY_CHARS]
        if len(text) > COMPACT_MAX_SUMMARY_CHARS:
            return text[:COMPACT_MAX_SUMMARY_CHARS] + "\n...(旧消息过长，已截断)"
        return text
    def _messages_to_text(self, msgs: List[Message]) -> str:
        """把消息列表转成可摘要的纯文本。"""
        parts: List[str] = []
        for m in msgs:
            role = "用户" if m.role == "user" else ("助手" if m.role == "assistant" else str(m.role))
            content = m.content
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            s = str(content or "").strip()
            if not s:
                continue
            parts.append(f"{role}: {s}")
        return "\n\n".join(parts)
    def _llm_summarize(self, text: str) -> str:
        """调用当前模型生成对话摘要；任何失败返回空串（调用方退化为截断）。"""
        prompt = (
            "请用简洁的中文总结以下 AI 编程助手的对话内容。"
            "需要保留：已完成的任务、关键决策、涉及的文件路径、用户偏好、尚未完成的事项。"
            "直接输出摘要正文，不要输出任何解释或前后缀。\n\n"
            "<conversation>\n" + text + "\n</conversation>"
        )
        try:
            if self.api_format == "anthropic":
                return self._anthropic_summarize(prompt)
            self._monitor_request_prompt([{"role": "user", "content": prompt}])
            _payload_sum = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.3,
            }
            _req_start_ms = time.time() * 1000
            self._traffic_request(_payload_sum, message_type="summarize", stream=False)
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
            )
            _req_dur_ms = int(time.time() * 1000 - _req_start_ms)
            _sum_text = (resp.choices[0].message.content or "").strip()
            _uhit, _umiss, _uout, _ureason = self._openai_usage(resp.usage)
            self._traffic_response(
                text=_sum_text,
                input_tokens=_uhit + _umiss,
                cache_hit_tokens=_uhit,
                cache_miss_tokens=_umiss,
                output_tokens=_uout,
                reasoning_tokens=_ureason,
                message_type="summarize",
                duration_ms=_req_dur_ms,
            )
            return _sum_text
        except Exception:
            return ""
    def _anthropic_summarize(self, prompt: str) -> str:
        """Anthropic Messages API 非流式摘要请求。"""
        base = self.base_url.rstrip("/")
        url = base + "/messages" if base.endswith("/v1") else base + "/v1/messages"
        headers = self._anthropic_headers()
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        self._monitor_request_prompt(payload)
        delays = _retry_delays()
        for attempt in range(RETRY_MAX_ATTEMPTS + 1):
            _req_start_ms = time.time() * 1000
            self._traffic_request(payload, message_type="summarize", stream=False, retry_attempt=attempt)
            resp = httpx.post(url, headers=headers, json=payload, timeout=httpx.Timeout(120.0, connect=30.0))
            _dur_ms = int(time.time() * 1000 - _req_start_ms)
            if resp.status_code == 200:
                break
            self._traffic_response(
                text="", message_type="summarize", stream=False,
                status_code=resp.status_code, duration_ms=_dur_ms, error=resp.text[:300],
            )
            if not _is_rate_limited(resp.status_code, resp.text):
                return ""
            if attempt >= RETRY_MAX_ATTEMPTS:
                return ""
            time.sleep(delays[attempt])
        data = resp.json()
        _sum_text = "".join(
            b.get("text", "") for b in (data.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
        _usage = data.get("usage") or {}
        _uhit = _usage_field(_usage, "cache_read_input_tokens")
        _umiss = _usage_field(_usage, "cache_creation_input_tokens")
        _in = _usage.get("input_tokens") or 0
        _out = _usage.get("output_tokens") or 0
        if _uhit == 0 and _umiss == 0:
            _umiss = _in
        self._traffic_response(
            text=_sum_text,
            input_tokens=_in,
            cache_hit_tokens=_uhit,
            cache_miss_tokens=_umiss,
            output_tokens=_out,
            reasoning_tokens=_usage_field(_usage, "output_tokens_details.reasoning_tokens"),
            message_type="summarize",
            duration_ms=_dur_ms if resp.status_code == 200 else 0,
        )
        return _sum_text
        content = data.get("content") or []
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts).strip()
