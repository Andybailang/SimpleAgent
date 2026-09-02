"""engine.anthropic_flow — Anthropic Messages API（流式与非流式）工具循环。
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    MAX_TOOL_ROUNDS,
    RETRY_MAX_ATTEMPTS,
    THINKING_BUDGETS,
    _estimate_output_tokens,
    _is_rate_limited,
    _retry_delays,
    _retry_event,
)
from .util import Message, _record_model_response, _tool_result_content, _usage_field
import httpx

def _usage_token_fields(usage: Dict[str, Any]) -> tuple:
    """从 Anthropic/OpenAI 兼容 usage dict 提取 (input_total, hit, miss, output, reasoning)。

    兼容三种格式：
    - Anthropic：input_tokens / cache_read_input_tokens(命中) /
      cache_creation_input_tokens(未命中)；
    - OpenAI Chat / Zhipu-anthropic：prompt_tokens / prompt_tokens_details.cached_tokens(命中)，
      或 prompt_cache_hit_tokens / prompt_cache_miss_tokens；
    - Responses 风格：output_tokens / output_tokens_details.reasoning_tokens。
    usage 允许为空 dict；缓存字段均缺失时按全部未命中计（miss=input_total，hit=0）。
    """
    if not usage:
        return 0, 0, 0, 0, 0
    hit_pc = _usage_field(usage, "prompt_cache_hit_tokens")
    miss_pc = _usage_field(usage, "prompt_cache_miss_tokens")
    if hit_pc or miss_pc:
        hit = hit_pc
        miss = miss_pc
        input_total = hit + miss
    else:
        total = _usage_field(usage, "input_tokens") or _usage_field(usage, "prompt_tokens")
        hit = _usage_field(usage, "cache_read_input_tokens") or \
            _usage_field(usage, "prompt_tokens_details.cached_tokens")
        miss = max(0, total - hit)
        input_total = total
    output = _usage_field(usage, "output_tokens") or _usage_field(usage, "completion_tokens")
    reasoning = _usage_field(usage, "output_tokens_details.reasoning_tokens") or \
        _usage_field(usage, "completion_tokens_details.reasoning_tokens")
    return input_total, hit, miss, output, reasoning


class AnthropicFlowMixin:
    def _with_attachments_anthropic(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Build Anthropic-compatible content for a user turn with attachments.

        图片附件不再内联上传：仅把路径放进文本，模型通过
        extract_text_from_image 工具读取文字；其他文件同样只传路径。
        始终返回普通字符串。
        """
        if not attachments:
            return text
        file_paths: List[str] = []
        for att in attachments:
            path = att.get("path") or ""
            file_paths.append(path or att.get("name") or "")
        text_parts: List[str] = []
        if text:
            text_parts.append(text)
        if file_paths:
            text_parts.append("已附加文件：\n" + "\n".join(file_paths))
        return "\n\n".join(text_parts)
    def _to_anthropic_user_content(self, content: Any) -> Any:
        """Normalize stored user content (str / OpenAI blocks / Anthropic blocks) to Anthropic format."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")
        blocks: List[Dict[str, Any]] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype == "text":
                blocks.append({"type": "text", "text": b.get("text", "")})
            elif btype in ("image", "image_url"):
                # 历史中的图片：Agnes 等 Anthropic 兼容网关不传递图片内容，
                # 图片轮次走 OpenAI 回退，后续 anthropic 请求不再携带图片块。
                blocks.append({"type": "text", "text": "[图片附件已省略]"})
            elif btype == "tool_result":
                blocks.append(b)
        if len(blocks) == 1 and blocks[0].get("type") == "text":
            return blocks[0]["text"]
        return blocks
    def _build_anthropic_messages(self) -> List[Dict[str, Any]]:
        """Convert self.messages to Anthropic messages format (system handled separately)."""
        api_messages: List[Dict[str, Any]] = []
        for msg in self.messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                api_messages.append({"role": "user", "content": self._to_anthropic_user_content(msg.content)})
            elif msg.role == "assistant":
                content = msg.content
                if isinstance(content, str):
                    if not content:
                        continue
                    blocks = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    blocks = [b for b in content if isinstance(b, dict) and b.get("type") in ("text", "tool_use")]
                    if not blocks:
                        continue
                else:
                    continue
                api_messages.append({"role": "assistant", "content": blocks})
        return api_messages
    def _anthropic_headers(self) -> Dict[str, str]:
        """Anthropic Messages 请求头：默认 x-api-key（Anthropic 原生），
        部分聚合网关（如 SiliconFlow）要求 Authorization: Bearer。"""
        headers: Dict[str, str] = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self.api_auth == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        return headers
    async def _anthropic_stream_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        round_type: str = "user_turn",
    ):
        """Anthropic 流式响应；遇限流（HTTP 429 / 智谱 1302、1305）按指数退避重试。

        每次实际发出 HTTP 都记录一条 traffic request（重试也消耗真实流量）；
        非 200 时记录失败 response（status/error），200 成功由上层解析完一轮后记录。
        产出两类对象：
        - dict：system/api_retry 重试事件（前端活动条显示原因、次数与延时）；
        - httpx.Response：HTTP 200 的流式响应对象。
        重试耗尽或遇非限流错误时抛出 RuntimeError。
        """
        delays = _retry_delays()
        for attempt in range(RETRY_MAX_ATTEMPTS + 1):
            _req_start_ms = time.time() * 1000
            self._traffic_request(payload, message_type=round_type, stream=True, retry_attempt=attempt)
            stream = client.stream("POST", url, headers=headers, json=payload)
            async with stream as resp:
                if resp.status_code == 200:
                    yield resp
                    return
                body = (await resp.aread()).decode("utf-8", errors="replace")
                _dur_ms = int(time.time() * 1000 - _req_start_ms)
                self._traffic_response(
                    text="", status_code=resp.status_code, duration_ms=_dur_ms,
                    message_type=round_type, stream=True, error=body[:300],
                )
            if not _is_rate_limited(resp.status_code, body):
                raise RuntimeError(f"Anthropic API {resp.status_code}: {body[:500]}")
            if attempt >= RETRY_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Anthropic API {resp.status_code}（限流，已重试 {RETRY_MAX_ATTEMPTS} 次仍失败）: {body[:300]}"
                )
            yield _retry_event(
                attempt=attempt + 1,
                delay_ms=int(delays[attempt] * 1000),
                error_status=resp.status_code,
                error=body[:200],
            )
            await asyncio.sleep(delays[attempt])
    async def _anthropic_post_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        round_type: str = "user_turn",
    ):
        """Anthropic 非流式请求；限流重试逻辑与 _anthropic_stream_with_retry 一致。

        每次实际发出的 HTTP 尝试都记录一条 traffic request（重试也消耗真实流量）；
        非 200 时记录失败 response（status/error），200 成功由上层解析完一轮后记录。
        产出两类对象：
        - dict：system/api_retry 重试事件；
        - dict：HTTP 200 的完整 JSON 响应。
        """
        delays = _retry_delays()
        for attempt in range(RETRY_MAX_ATTEMPTS + 1):
            _req_start_ms = time.time() * 1000
            self._traffic_request(payload, message_type=round_type, stream=False, retry_attempt=attempt)
            resp = await client.post(url, headers=headers, json=payload)
            _dur_ms = int(time.time() * 1000 - _req_start_ms)
            if resp.status_code == 200:
                yield resp.json()
                return
            body = resp.text
            self._traffic_response(
                text="", message_type=round_type, stream=False,
                status_code=resp.status_code, duration_ms=_dur_ms, error=body[:300],
            )
            if not _is_rate_limited(resp.status_code, body):
                raise RuntimeError(f"Anthropic API {resp.status_code}: {body[:500]}")
            if attempt >= RETRY_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Anthropic API {resp.status_code}（限流，已重试 {RETRY_MAX_ATTEMPTS} 次仍失败）: {body[:300]}"
                )
            yield _retry_event(
                attempt=attempt + 1,
                delay_ms=int(delays[attempt] * 1000),
                error_status=resp.status_code,
                error=body[:200],
            )
            await asyncio.sleep(delays[attempt])
    async def chat_stream_anthropic_async(
        self,
        user_message: str,
        use_tools: bool = True,
        attachments: Optional[List[Dict[str, Any]]] = None,
        continue_only: bool = False,
    ):
        """Anthropic Messages API streaming chat with a native tool loop.

        Yields structured event dicts:
          {"type": "text", "text": ...}        assistant text fragments
          {"type": "thinking", "text": ...}    thinking fragments
          {"type": "tool_use", "id", "name", "input"}                        tool_use block
          {"type": "tool_result", "tool_use_id", "tool_name", "content"}     tool result
          {"type": "usage", "input_tokens", "output_tokens"}                 token usage
          {"type": "error", "text": ...}                                     error message
        """
        # 图片附件不再内联上传（见 _with_attachments_anthropic）：
        # 附件只传路径文本，模型通过 extract_text_from_image 工具读取图片文字，
        # 因此无需再回退 OpenAI 路径，统一走 Anthropic 原生工具循环。

        if not continue_only:
            user_content = self._with_attachments_anthropic(user_message, attachments)
            self.messages.append(Message(
                role="user",
                content=user_content,
                timestamp=int(datetime.now().timestamp() * 1000)
            ))

        api_messages = self._build_anthropic_messages()
        system_text = self._anthropic_system_text()
        tool_defs = self._tool_definitions_anthropic() if use_tools else None
        base = self.base_url.rstrip("/")
        url = base + "/messages" if base.endswith("/v1") else base + "/v1/messages"
        headers = self._anthropic_headers()
        thinking = None
        if self.thinking_level != "off" and self.thinking_level in THINKING_BUDGETS:
            thinking = {"type": "enabled", "budget_tokens": THINKING_BUDGETS[self.thinking_level]}

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                for _round in range(MAX_TOOL_ROUNDS):
                    payload: Dict[str, Any] = {
                        "model": self.model_name,
                        "max_tokens": self.max_tokens,
                        "messages": api_messages,
                        "stream": True,
                    }
                    if system_text:
                        payload["system"] = system_text
                    if tool_defs:
                        payload["tools"] = tool_defs
                    if thinking is not None:
                        payload["thinking"] = thinking
                    else:
                        payload["temperature"] = self.temperature

                    text_parts: List[str] = []
                    thinking_parts: List[str] = []
                    current_tools: Dict[int, Dict[str, Any]] = {}
                    resp_buf: List[str] = []
                    usage_acc: Dict[str, Any] = {}
                    _round_start_ms = time.time() * 1000

                    self._monitor_request_prompt(payload)
                    # 发送前读取流量库本会话最近 request（专供 UI 实时显示）
                    ctx = self._context_usage_event()
                    if ctx:
                        yield ctx
                    # 自动压缩：发送前本地估算上下文体积，超阈值则压缩并通知 UI
                    ac = self._maybe_auto_compact(payload.get("messages") or [], tools=payload.get("tools"), system_text=payload.get("system") or "")
                    if ac:
                        yield ac
                        self._refresh_system_message()
                        api_messages = self._build_anthropic_messages()
                        payload["messages"] = api_messages
                    _round_type = "tool_round" if api_messages and str(api_messages[-1].get("role")) == "tool" else "user_turn"
                    async for retry_item in self._anthropic_stream_with_retry(
                        client, url, headers, payload,
                        round_type=_round_type,
                    ):
                        if isinstance(retry_item, dict):
                            yield retry_item
                            continue
                        resp = retry_item
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[len("data:"):].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                evt = json.loads(data)
                            except Exception:
                                continue
                            evt_type = evt.get("type")
                            if evt_type == "message_start":
                                usage_acc.update((evt.get("message") or {}).get("usage") or {})
                            elif evt_type == "content_block_start":
                                block = evt.get("content_block") or {}
                                btype = block.get("type")
                                index = evt.get("index")
                                if btype == "tool_use":
                                    current_tools[index] = {
                                        "id": block.get("id"),
                                        "name": block.get("name"),
                                        "input_json": "",
                                    }
                            elif evt_type == "content_block_delta":
                                delta = evt.get("delta") or {}
                                dtype = delta.get("type")
                                index = evt.get("index")
                                if dtype == "text_delta":
                                    frag = delta.get("text", "")
                                    if frag:
                                        text_parts.append(frag)
                                        resp_buf.append(frag)
                                        yield {"type": "text", "text": frag}
                                elif dtype == "thinking_delta":
                                    frag = delta.get("thinking", "")
                                    if frag:
                                        thinking_parts.append(frag)
                                        resp_buf.append(frag)
                                        yield {"type": "thinking", "text": frag}
                                elif dtype == "input_json_delta" and index in current_tools:
                                    current_tools[index]["input_json"] += delta.get("partial_json", "")
                            elif evt_type == "message_delta":
                                usage_acc.update(evt.get("usage") or {})

                        break
                    # 解析本轮完整的 tool_use 块
                    tool_uses = []
                    for index in sorted(current_tools.keys()):
                        t = current_tools[index]
                        try:
                            input_data = json.loads(t["input_json"]) if t["input_json"].strip() else {}
                        except Exception:
                            input_data = {}
                        tool_uses.append({
                            "id": t["id"],
                            "name": t["name"],
                            "input": input_data,
                        })
                        try:
                            _inp = json.dumps(input_data, ensure_ascii=False)
                        except Exception:
                            _inp = "{}"
                        resp_buf.append(f"\n[tool_use] {t['name']}: {_inp}")

                    assistant_text = "".join(text_parts)
                    if not (self._persist_tool_context and tool_uses):
                        # 写回上下文且有工具调用时，assistant 消息（含思考与 tool_use 块）
                        # 由 _persist_tool_round 统一写回，避免重复。
                        self.messages.append(Message(
                            role="assistant",
                            content=assistant_text,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))

                    input_tokens, cache_hit_tokens, cache_miss_tokens, output_tokens, reasoning_tokens = _usage_token_fields(usage_acc)
                    if output_tokens <= 0:
                        # 部分 Anthropic 兼容网关（如 Agnes）流式不发送 message_delta.usage，
                        # 用实际输出文本估算 output_tokens，避免 usage 输出恒为 0。
                        output_tokens = _estimate_output_tokens("".join(text_parts) + "".join(thinking_parts))
                    yield {
                        "type": "usage",
                        "input_tokens": input_tokens,
                        "cache_hit_tokens": cache_hit_tokens,
                        "cache_miss_tokens": cache_miss_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_tokens": reasoning_tokens,
                    }
                    # 记录模型本轮返回内容（From 方向，全部内容按到达顺序）
                    _record_model_response(self.model_name, "".join(resp_buf))
                    # LLM 流量存档：Anthropic 流式一轮成功响应
                    self._traffic_response(
                        text="".join(resp_buf),
                        input_tokens=input_tokens,
                        cache_hit_tokens=cache_hit_tokens,
                        cache_miss_tokens=cache_miss_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        message_type="tool_round" if tool_uses else "user_turn",
                        stream=True,
                        duration_ms=int(time.time() * 1000 - _round_start_ms),
                    )

                    if not tool_uses:
                        return

                    # 组装 assistant 消息（含 tool_use 块）继续工具循环
                    assistant_blocks: List[Dict[str, Any]] = []
                    if assistant_text:
                        assistant_blocks.append({"type": "text", "text": assistant_text})
                    for t in tool_uses:
                        assistant_blocks.append({"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["input"]})
                    api_messages.append({"role": "assistant", "content": assistant_blocks})

                    # 免费模型：工具调用轮次写回持久上下文（思考与工具结果进入上下文）
                    tool_uses_with_result: List[Dict[str, Any]] = []
                    for t in tool_uses:
                        ask_event = self._handle_ask_user_question(t["name"], t["id"], t["input"])
                        if ask_event is not None:
                            yield {"type": "tool_use", "id": t["id"], "name": "AskUserQuestion", "input": ask_event["input"]}
                            # 确保 assistant(tool_use) 写入 self.messages，供回答后续跑。
                            self._ensure_ask_user_assistant_message(t["id"], t["name"], ask_event["input"])
                            yield ask_event
                            break
                        yield {"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["input"]}
                        result = self._run_tool(t["name"], t["input"])
                        tool_content = _tool_result_content(result)
                        api_messages.append({
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": t["id"], "content": tool_content}],
                        })
                        tool_uses_with_result.append({
                            "id": t["id"], "name": t["name"], "input": t["input"], "_result": tool_content,
                        })
                        yield {"type": "tool_result", "tool_use_id": t["id"], "tool_name": t["name"], "content": tool_content}
                    if self._ask_user_paused:
                        # 已向用户提问：结束本轮生成，等待用户回答。
                        return
                    if tool_uses_with_result:
                        self._persist_tool_round(
                            assistant_blocks, tool_uses_with_result,
                            thinking_text="".join(thinking_parts),
                        )

                yield {"type": "error", "text": f"（达到工具调用轮数上限 {MAX_TOOL_ROUNDS} 轮，已停止）"}
        except Exception as e:
            yield {"type": "error", "text": f"Error: {str(e)}"}
    async def chat_nonstream_anthropic_async(
        self,
        user_message: str,
        use_tools: bool = True,
        attachments: Optional[List[Dict[str, Any]]] = None,
        continue_only: bool = False,
    ):
        """Anthropic Messages API 非流式对话（供声明 stream_supported=false 的模型使用）。

        事件形状与 chat_stream_anthropic_async 完全一致：
          text / thinking / tool_use / tool_result / usage / system / error
        """
        # 图片附件不再内联上传，统一走 Anthropic 原生工具循环（同流式路径）。

        if not continue_only:
            user_content = self._with_attachments_anthropic(user_message, attachments)
            self.messages.append(Message(
                role="user",
                content=user_content,
                timestamp=int(datetime.now().timestamp() * 1000)
            ))

        api_messages = self._build_anthropic_messages()
        system_text = self._anthropic_system_text()
        tool_defs = self._tool_definitions_anthropic() if use_tools else None
        base = self.base_url.rstrip("/")
        url = base + "/messages" if base.endswith("/v1") else base + "/v1/messages"
        headers = self._anthropic_headers()
        thinking = None
        if self.thinking_level != "off" and self.thinking_level in THINKING_BUDGETS:
            thinking = {"type": "enabled", "budget_tokens": THINKING_BUDGETS[self.thinking_level]}

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                for _round in range(MAX_TOOL_ROUNDS):
                    payload: Dict[str, Any] = {
                        "model": self.model_name,
                        "max_tokens": self.max_tokens,
                        "messages": api_messages,
                    }
                    if system_text:
                        payload["system"] = system_text
                    if tool_defs:
                        payload["tools"] = tool_defs
                    if thinking is not None:
                        payload["thinking"] = thinking
                    else:
                        payload["temperature"] = self.temperature

                    usage_acc: Dict[str, Any] = {}
                    _round_start_ms = time.time() * 1000
                    self._monitor_request_prompt(payload)
                    # 发送前读取流量库本会话最近 request（专供 UI 实时显示）
                    ctx = self._context_usage_event()
                    if ctx:
                        yield ctx
                    # 自动压缩：发送前本地估算上下文体积，超阈值则压缩并通知 UI
                    ac = self._maybe_auto_compact(payload.get("messages") or [], tools=payload.get("tools"), system_text=payload.get("system") or "")
                    if ac:
                        yield ac
                        self._refresh_system_message()
                        api_messages = self._build_anthropic_messages()
                        payload["messages"] = api_messages
                    _round_type = "tool_round" if api_messages and str(api_messages[-1].get("role")) == "tool" else "user_turn"
                    data: Optional[Dict[str, Any]] = None
                    async for retry_item in self._anthropic_post_with_retry(
                        client, url, headers, payload, round_type=_round_type,
                    ):
                        if isinstance(retry_item, dict) and retry_item.get("type") == "system":
                            yield retry_item
                            continue
                        data = retry_item
                    if data is None:
                        raise RuntimeError("Anthropic API 返回空响应")

                    usage_acc = data.get("usage") or {}

                    text_parts: List[str] = []
                    thinking_parts: List[str] = []
                    tool_uses: List[Dict[str, Any]] = []
                    resp_buf: List[str] = []
                    for block in data.get("content") or []:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "text":
                            frag = block.get("text", "")
                            if frag:
                                text_parts.append(frag)
                                resp_buf.append(frag)
                                yield {"type": "text", "text": frag}
                        elif btype == "thinking":
                            frag = block.get("thinking", "")
                            if frag:
                                thinking_parts.append(frag)
                                resp_buf.append(frag)
                                yield {"type": "thinking", "text": frag}
                        elif btype == "tool_use":
                            _inp = block.get("input") or {}
                            try:
                                _inp_str = json.dumps(_inp, ensure_ascii=False)
                            except Exception:
                                _inp_str = "{}"
                            tool_uses.append({
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "input": _inp,
                            })
                            resp_buf.append(f"\n[tool_use] {block.get('name') or '?'}: {_inp_str}")

                    assistant_text = "".join(text_parts)
                    if not (self._persist_tool_context and tool_uses):
                        # 写回上下文且有工具调用时，assistant 消息（含思考与 tool_use 块）
                        # 由 _persist_tool_round 统一写回，避免重复。
                        self.messages.append(Message(
                            role="assistant",
                            content=assistant_text,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))

                    input_tokens, cache_hit_tokens, cache_miss_tokens, output_tokens, reasoning_tokens = _usage_token_fields(usage_acc)
                    if output_tokens <= 0:
                        output_tokens = _estimate_output_tokens("".join(text_parts) + "".join(thinking_parts))
                    yield {
                        "type": "usage",
                        "input_tokens": input_tokens,
                        "cache_hit_tokens": cache_hit_tokens,
                        "cache_miss_tokens": cache_miss_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_tokens": reasoning_tokens,
                    }
                    # 记录模型本轮返回内容（From 方向，全部内容按到达顺序）
                    _record_model_response(self.model_name, "".join(resp_buf))
                    # LLM 流量存档：Anthropic 非流式一轮成功响应
                    self._traffic_response(
                        text="".join(resp_buf),
                        input_tokens=input_tokens,
                        cache_hit_tokens=cache_hit_tokens,
                        cache_miss_tokens=cache_miss_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        message_type="tool_round" if tool_uses else "user_turn",
                        stream=False,
                        duration_ms=int(time.time() * 1000 - _round_start_ms),
                    )

                    if not tool_uses:
                        return

                    assistant_blocks: List[Dict[str, Any]] = []
                    if assistant_text:
                        assistant_blocks.append({"type": "text", "text": assistant_text})
                    for t in tool_uses:
                        assistant_blocks.append({"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["input"]})
                    api_messages.append({"role": "assistant", "content": assistant_blocks})

                    # 免费模型：工具调用轮次写回持久上下文
                    tool_uses_with_result: List[Dict[str, Any]] = []
                    for t in tool_uses:
                        ask_event = self._handle_ask_user_question(t["name"], t["id"], t["input"])
                        if ask_event is not None:
                            yield {"type": "tool_use", "id": t["id"], "name": "AskUserQuestion", "input": ask_event["input"]}
                            self._ensure_ask_user_assistant_message(t["id"], t["name"], ask_event["input"])
                            yield ask_event
                            break
                        yield {"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["input"]}
                        result = self._run_tool(t["name"], t["input"])
                        tool_content = _tool_result_content(result)
                        api_messages.append({
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": t["id"], "content": tool_content}],
                        })
                        tool_uses_with_result.append({
                            "id": t["id"], "name": t["name"], "input": t["input"], "_result": tool_content,
                        })
                        yield {"type": "tool_result", "tool_use_id": t["id"], "tool_name": t["name"], "content": tool_content}
                    if self._ask_user_paused:
                        return
                    if tool_uses_with_result:
                        self._persist_tool_round(
                            assistant_blocks, tool_uses_with_result,
                            thinking_text="".join(thinking_parts),
                        )

                yield {"type": "error", "text": f"（达到工具调用轮数上限 {MAX_TOOL_ROUNDS} 轮，已停止）"}
        except Exception as e:
            yield {"type": "error", "text": f"Error: {str(e)}"}
