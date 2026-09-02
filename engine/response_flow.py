"""engine.response_flow — DeepSeek Responses API（/responses）无状态工具循环。
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    MAX_TOOL_ROUNDS,
    RETRY_MAX_ATTEMPTS,
    _is_openai_retryable_error,
    _retry_delays,
    _retry_event,
)
from .util import Message, _record_model_response, _usage_field

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

class ResponseFlowMixin:
    def _response_input_text(self, content: Any, block_type: str = "input_text") -> Any:
        """把 content 归一为 Responses API 的消息内容（字符串或内容块列表）。"""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")
        blocks: List[Dict[str, Any]] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                blocks.append({"type": block_type, "text": b.get("text", "")})
            elif bt in ("image", "image_url"):
                # 图片原始数据绝不进入上下文：替换为文本占位（与 anthropic 路径一致）
                blocks.append({"type": block_type, "text": "[图片附件已省略]"})
        if len(blocks) == 1 and blocks[0].get("type") == block_type:
            return blocks[0]["text"]
        return blocks
    def _to_response_input_items(self) -> List[Dict[str, Any]]:
        """把 self.messages 还原为 Responses API 的 input item 列表。

        约定：
        - system / user / assistant 消息以 message item 表达（system 放在正文里，与现有
          流程一致，不使用独立的 instructions 字段）；
        - assistant 的 thinking 以 reasoning item、tool_use 以 function_call item 表达；
        - 工具结果以 function_call_output item 表达，与 function_call 通过 call_id 配对；
        - assistant 纯文本以 output_text 块表达。
        """
        items: List[Dict[str, Any]] = []
        pending_output_ids: List[str] = []
        for msg in self.messages:
            role = msg.role
            if role == "system":
                items.append({"type": "message", "role": "system",
                              "content": self._response_input_text(msg.content)})
                continue
            if role == "user":
                if self._is_tool_result_message(msg):
                    c = msg.content
                    if isinstance(c, list):
                        for b in c:
                            if isinstance(b, dict) and b.get("type") == "tool_result":
                                items.append({"type": "function_call_output",
                                              "call_id": b.get("tool_use_id") or "",
                                              "output": str(b.get("content") or "")})
                    else:
                        # 纯文本工具结果：无显式 call_id，按出现顺序配对未消费的 function_call。
                        # 不能只取“最近一条 function_call”，否则同一 assistant 轮里多个工具调用
                        # 会把多个结果都回填到同一个 call_id，导致 DeepSeek 报
                        # "Duplicate tool output for call_id: ..."。用 FIFO 队列逐个消耗。
                        call_id = pending_output_ids.pop(0) if pending_output_ids else ""
                        items.append({"type": "function_call_output",
                                      "call_id": call_id, "output": str(c or "")})
                    continue
                items.append({"type": "message", "role": "user",
                              "content": self._response_input_text(msg.content)})
                continue
            if role == "assistant":
                text, thinking, tool_uses, _ = self._content_blocks(msg.content)
                # DeepSeek Responses API（/responses）官方要求：思考模式下 chain-of-thought
                # 以 reasoning item 返回，且必须排在 message item 之前（create-response：
                # "In thinking mode, the chain-of-thought is returned as a reasoning item
                # before the message item."）。顺序颠倒会在 tools + 思考模式下触发 400：
                # "The `reasoning_text` in the thinking mode must be passed back to the API."
                if thinking:
                    items.append({"type": "reasoning",
                                  "content": [{"type": "reasoning_text", "text": thinking}]})
                if text:
                    items.append({"type": "message", "role": "assistant",
                                  "content": [{"type": "output_text", "text": text}]})
                # web_search_call（服务端联网搜索）块原样回传，服务端自动恢复搜索结果
                if isinstance(msg.content, list):
                    wscs = [b for b in msg.content
                            if isinstance(b, dict) and b.get("type") == "web_search_call"]
                    for wsc in self._normalize_web_search_calls(wscs):
                        it = {k: v for k, v in wsc.items() if k != "type"}
                        it["type"] = "web_search_call"
                        items.append(it)
                for i, tu in enumerate(tool_uses):
                    _cid = tu.get("id") or f"call_{i}"
                    items.append({
                        "type": "function_call",
                        "call_id": _cid,
                        "name": tu.get("name") or "",
                        "arguments": json.dumps(tu.get("input") or {}, ensure_ascii=False),
                    })
                    pending_output_ids.append(_cid)
                continue
        return items
    def _response_tool_definitions(self) -> Optional[List[Dict[str, Any]]]:
        """Responses API 工具定义：扁平 function 工具 + 默认加一个服务端 web_search。

        注意 Responses API 的 function 工具是扁平结构（type/name/description/parameters），
        与 Chat Completions 的嵌套 function 结构不同，不能直接复用 _tool_definitions()。
        """
        defs: List[Dict[str, Any]] = []
        for tool in self._active_tools():
            defs.append({
                "type": "function",
                "name": tool.name,
                "parameters": tool.parameters,
                "description": tool.description,
            })
        if defs:
            defs[-1]["description"] += f"\n你的工作目录是 {self.cwd}。"
            # 默认给模型挂上服务端 web_search（官网要求：tools 里包含 web_search 工具）
            defs.append({"type": "web_search"})
        return defs
    @staticmethod
    def _response_usage(usage: Any) -> tuple:
        """从 Responses API usage 取 (cache_hit, cache_miss, output, reasoning_tokens)。

        input_tokens 为总输入 token；input_tokens_details.cached_tokens 为命中缓存数，
        因此未命中（全价计费）输入 = input_tokens - cached_tokens。
        """
        if usage is None:
            return 0, 0, 0, 0
        in_total = _usage_field(usage, "input_tokens")
        cached = _usage_field(usage, "input_tokens_details.cached_tokens")
        hit = cached
        miss = max(0, in_total - cached)
        out = _usage_field(usage, "output_tokens")
        reasoning = _usage_field(usage, "output_tokens_details.reasoning_tokens")
        return hit, miss, out, reasoning
    async def chat_stream_response_async(
        self,
        user_message: str,
        use_tools: bool = True,
        attachments: Optional[List[Dict[str, Any]]] = None,
        continue_only: bool = False,
    ):
        """DeepSeek Responses API（/responses）流式工具循环。

        与 Chat Completions 的差异：
        - 无状态：每次请求都完整回传 input（消息 item / function_call / function_call_output /
          reasoning / web_search_call）；
        - system 提示依旧放在 input 里的 system message（不使用独立的 instructions，与原流程一致）；
        - 默认给模型加一个内置 web_search 工具（服务端执行），模型有需时可联网后思考；
        - 用户标识用顶层 `user` 字段（DeepSeek 文档明确：Responses API 叫 user，Chat 叫 user_id）；
        - usages 里输入 token 需扣除 cached_tokens，未命中部分才是我们记的“入 tokens”。
        """
        if not continue_only:
            self.messages.append(Message(
                role="user",
                content=self._with_attachments(user_message, attachments),
                timestamp=int(datetime.now().timestamp() * 1000)
            ))
        try:
            client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            tool_defs = self._response_tool_definitions() if use_tools else None

            total_usage_in = 0
            total_usage_out = 0
            total_hit = 0
            total_miss = 0
            total_reasoning = 0
            # 无状态 API：一键生成当前历史，工具轮在其基础上追加 function_call / output，
            # 而不是每轮重建（否则收费聊天模式下不写回上下文会丢失本轮工具历史）。
            input_items = self._to_response_input_items()
            for _ in range(MAX_TOOL_ROUNDS):
                uid = self._session_user_id()
                response_params: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": input_items,
                    "max_output_tokens": self.max_tokens,
                    "stream": True,
                    # reasoning.effort: none=关闭思考；low/high/max=思考强度
                    "reasoning": {"effort": self._deepseek_effort()},
                }
                if uid:
                    response_params["user"] = uid
                if tool_defs:
                    response_params["tools"] = tool_defs
                # 思考模式下 temperature / top_p 不生效，官方明确忽略、不报错；思考时不发送
                if not self._is_thinking:
                    response_params["temperature"] = self.temperature

                # 缓存监控：只关心 input / tools / reasoning（剔除纯请求参数）
                self._monitor_request_prompt(
                    {"input": input_items, "reasoning": response_params["reasoning"]},
                    tools=tool_defs,
                )
                ctx = self._context_usage_event()
                if ctx:
                    yield ctx
                ac = self._maybe_auto_compact(
                    input_items, tools=tool_defs, system_text=self._build_system_prompt())
                if ac:
                    yield ac
                    self._refresh_system_message()
                    input_items = self._to_response_input_items()
                    response_params["input"] = input_items

                is_tool_round = bool(
                    tool_defs and any(it.get("type") == "function_call_output" for it in input_items))
                retry_delays = _retry_delays()
                retry_attempt = 0
                stream = None
                while True:
                    _req_start_ms = time.time() * 1000
                    self._traffic_request(
                        dict(response_params),
                        message_type="tool_round" if is_tool_round else "user_turn",
                        stream=True,
                        retry_attempt=retry_attempt,
                    )
                    try:
                        stream = await client.responses.create(**response_params)
                        break
                    except Exception as e:
                        if not _is_openai_retryable_error(e):
                            self._traffic_response(
                                text="", input_tokens=0, output_tokens=0,
                                message_type="error",
                                status_code=getattr(e, "status_code", None) or 0,
                                duration_ms=int(time.time() * 1000 - _req_start_ms),
                                error=str(e)[:300],
                            )
                            raise
                        if retry_attempt >= RETRY_MAX_ATTEMPTS:
                            raise RuntimeError(
                                f"DeepSeek Responses API 限流（已重试 {RETRY_MAX_ATTEMPTS} 次仍失败）: {str(e)[:300]}")
                        yield _retry_event(
                            attempt=retry_attempt + 1,
                            delay_ms=int(retry_delays[retry_attempt] * 1000),
                            error_status=getattr(e, "status_code", None) or 429,
                            error=str(e)[:200],
                        )
                        self._traffic_response(
                            text="", input_tokens=0, output_tokens=0,
                            message_type="retry",
                            status_code=getattr(e, "status_code", None) or 429,
                            duration_ms=int(time.time() * 1000 - _req_start_ms),
                            error=str(e)[:200],
                        )
                        await asyncio.sleep(retry_delays[retry_attempt])
                        retry_attempt += 1

                text_parts: List[str] = []
                thinking_parts: List[str] = []
                function_calls: Dict[str, Dict[str, Any]] = {}  # item_id -> {call_id,name,arguments}
                function_order: List[str] = []
                web_search_calls: List[Dict[str, Any]] = []
                final_response: Any = None
                final_status = ""
                resp_buf: List[str] = []
                async for event in stream:
                    evt_type = getattr(event, "type", "") or ""
                    if evt_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            text_parts.append(delta)
                            resp_buf.append(delta)
                            yield {"type": "text", "text": delta}
                    elif evt_type == "response.reasoning_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            thinking_parts.append(delta)
                            resp_buf.append(delta)
                            yield {"type": "thinking", "text": delta}
                    elif evt_type == "response.function_call_arguments.delta":
                        item_id = getattr(event, "item_id", "") or ""
                        delta = getattr(event, "delta", "") or ""
                        fc = function_calls.setdefault(item_id, {"call_id": "", "name": "", "arguments": ""})
                        fc["arguments"] += delta
                    elif evt_type == "response.output_item.added":
                        # web_search_call 只在 output_item.done 捕获（此时才带最终 action）。
                        # added 事件的中间态（in_progress、无 action）若被回传，DeepSeek /responses
                        # 会报 400 "missing field `action`"。这里忽略，避免把无 action 的中间态存入历史。
                        pass
                    elif evt_type == "response.output_item.done":
                        item = getattr(event, "item", None) or {}
                        it = self._item_field(item, "type", "")
                        item_id = self._item_field(item, "id", "") or self._item_field(item, "call_id", "")
                        if it == "function_call":
                            fc = function_calls.setdefault(item_id, {"call_id": "", "name": "", "arguments": ""})
                            fc["call_id"] = self._item_field(item, "call_id", "") or fc["call_id"] or item_id
                            fc["name"] = self._item_field(item, "name", "") or fc["name"] or ""
                            full_args = self._item_field(item, "arguments", None) or fc["arguments"]
                            if full_args:
                                fc["arguments"] = full_args
                            if item_id not in function_order:
                                function_order.append(item_id)
                        elif it == "web_search_call":
                            web_search_calls.append(self._as_dict(item))
                    elif evt_type in ("response.completed", "response.incomplete", "response.failed"):
                        final_response = getattr(event, "response", None)
                        final_status = evt_type
                        break

                web_search_calls = self._normalize_web_search_calls(web_search_calls)
                text = "".join(text_parts)
                reasoning = "".join(thinking_parts)
                _req_dur_ms = int(time.time() * 1000 - _req_start_ms)
                # usage（response.completed 的对象里带完整 usage）
                usage = getattr(final_response, "usage", None) if final_response is not None else None
                _usage_hit, _usage_miss, _usage_out, _usage_reason = self._response_usage(usage)
                total_hit += _usage_hit
                total_miss += _usage_miss
                total_usage_in += _usage_hit + _usage_miss
                total_usage_out += _usage_out
                total_reasoning += _usage_reason

                if final_status == "response.failed":
                    err = getattr(final_response, "error", None)
                    err_msg = (getattr(err, "message", "") if err else "") or "未知错误"
                    self._traffic_response(
                        text="", input_tokens=0, output_tokens=0,
                        message_type="error", status_code=0,
                        duration_ms=_req_dur_ms, error=str(err_msg)[:300],
                    )
                    yield {"type": "error", "text": f"DeepSeek 响应失败：{err_msg}"}
                    return

                if function_order:
                    # 需要本地执行 function 工具：把本轮 assistant 消息与 function_call 落盘
                    assistant_blocks: List[Dict[str, Any]] = []
                    if reasoning:
                        assistant_blocks.append({"type": "thinking", "thinking": reasoning})
                    if text:
                        assistant_blocks.append({"type": "text", "text": text})
                    for item_id in function_order:
                        fc = function_calls[item_id]
                        try:
                            fc_args = json.loads(fc["arguments"] or "{}")
                        except Exception:
                            fc_args = {}
                        assistant_blocks.append({
                            "type": "tool_use",
                            "id": fc["call_id"] or item_id,
                            "name": fc["name"],
                            "input": fc_args,
                        })
                    for wsc in web_search_calls:
                        assistant_blocks.append({"type": "web_search_call", **wsc})
                    if self._persist_tool_context:
                        self.messages.append(Message(
                            role="assistant", content=assistant_blocks,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    else:
                        self.messages.append(Message(
                            role="assistant", content=text,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    resp_full = f"{reasoning}\n\n{text}" if reasoning else text
                    for item_id in function_order:
                        fc = function_calls[item_id]
                        try:
                            _inp = json.dumps(json.loads(fc["arguments"] or "{}"), ensure_ascii=False)
                        except Exception:
                            _inp = "{}"
                        resp_buf.append(f"\n[tool_use] {fc['name']}: {_inp}")
                    _record_model_response(self.model_name, "".join(resp_buf))
                    self._traffic_response(
                        text="".join(resp_buf),
                        input_tokens=_usage_hit + _usage_miss if usage else total_usage_in,
                        cache_hit_tokens=_usage_hit,
                        cache_miss_tokens=_usage_miss,
                        output_tokens=_usage_out if usage else total_usage_out,
                        reasoning_tokens=_usage_reason,
                        message_type="tool_round",
                        stream=True,
                        duration_ms=_req_dur_ms,
                    )
                    # 把本轮 assistant 消息（含 reasoning）与 function_call / web_search_call
                    # 追加进 input，下一轮回传——Respones 无状态，必须带上本轮输出才连贯。
                    if reasoning:
                        input_items.append({
                            "type": "reasoning",
                            "content": [{"type": "reasoning_text", "text": reasoning}],
                        })
                    if text:
                        input_items.append({
                            "type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        })
                    for _wsc in web_search_calls:
                        input_items.append({"type": "web_search_call", **_wsc})
                    for item_id in function_order:
                        fc = function_calls[item_id]
                        input_items.append({
                            "type": "function_call",
                            "call_id": fc["call_id"] or item_id,
                            "name": fc["name"],
                            "arguments": fc["arguments"] or "{}",
                        })
                    for item_id in function_order:
                        fc = function_calls[item_id]
                        try:
                            args = json.loads(fc["arguments"] or "{}")
                        except Exception:
                            args = {}
                        yield {"type": "tool_use", "id": fc["call_id"] or item_id,
                               "name": fc["name"], "input": args}
                        result = self._run_tool(fc["name"], args)
                        if isinstance(result, dict) and result.get("__media__"):
                            result = f"[图片 × {len(result['__media__'])}]"
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": fc["call_id"] or item_id,
                            "output": str(result),
                        })
                        if self._persist_tool_context:
                            self.messages.append(Message(
                                role="user",
                                content=str(result),
                                timestamp=int(datetime.now().timestamp() * 1000),
                                tool_name=fc["name"],
                                tool_input=args,
                                tool_result=str(result),
                            ))
                        yield {"type": "tool_result", "tool_use_id": fc["call_id"] or item_id,
                               "tool_name": fc["name"], "content": result}
                    # 整轮 function_call（可能含多个）已全部写回 self.messages：此时才节流落盘，
                    # 避免在多调用轮中间写入“只有部分 function_call_output”的非对称状态。
                    if self._persist_tool_context:
                        self._maybe_persist_tool_round()
                    continue

                # No more tool calls: emit the final reply.
                final_blocks: Any = text
                if reasoning:
                    if text:
                        final_blocks = [
                            {"type": "thinking", "thinking": reasoning},
                            {"type": "text", "text": text},
                        ]
                    else:
                        final_blocks = [{"type": "thinking", "thinking": reasoning}]
                self.messages.append(Message(
                    role="assistant", content=final_blocks,
                    timestamp=int(datetime.now().timestamp() * 1000),
                ))
                resp_full = f"{reasoning}\n\n{text}" if reasoning else text
                _record_model_response(self.model_name, resp_full)
                self._traffic_response(
                    text=resp_full,
                    input_tokens=_usage_hit + _usage_miss if usage else total_usage_in,
                    cache_hit_tokens=_usage_hit,
                    cache_miss_tokens=_usage_miss,
                    output_tokens=_usage_out if usage else total_usage_out,
                    reasoning_tokens=_usage_reason,
                    message_type="user_turn",
                    stream=True,
                    duration_ms=_req_dur_ms,
                )
                yield {
                    "type": "usage",
                    "input_tokens": total_usage_in,
                    "cache_hit_tokens": total_hit,
                    "cache_miss_tokens": total_miss,
                    "output_tokens": total_usage_out,
                    "reasoning_tokens": total_reasoning,
                }
                return

            yield {"type": "error", "text": f"（达到工具调用轮数上限 {MAX_TOOL_ROUNDS} 轮，已停止）"}
        except Exception as e:
            yield {"type": "error", "text": f"Error: {str(e)}"}
    async def chat_nonstream_response_async(
        self,
        user_message: str,
        use_tools: bool = True,
        attachments: Optional[List[Dict[str, Any]]] = None,
        continue_only: bool = False,
    ):
        """DeepSeek Responses API（/responses）非流式工具循环，产出与流式路径一致的事件。

        供声明 stream_supported=false 的模型使用。除请求流式/非流式差异外，其余约束
        （无状态回传 input、system 放正文、默认挂 web_search、user 标识、usage 扣 cached）与
        chat_stream_response_async 完全一致。
        """
        if not continue_only:
            self.messages.append(Message(
                role="user",
                content=self._with_attachments(user_message, attachments),
                timestamp=int(datetime.now().timestamp() * 1000)
            ))
        try:
            client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            tool_defs = self._response_tool_definitions() if use_tools else None

            total_usage_in = 0
            total_usage_out = 0
            total_hit = 0
            total_miss = 0
            total_reasoning = 0
            input_items = self._to_response_input_items()
            for _ in range(MAX_TOOL_ROUNDS):
                uid = self._session_user_id()
                response_params: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": input_items,
                    "max_output_tokens": self.max_tokens,
                    "stream": False,
                    "reasoning": {"effort": self._deepseek_effort()},
                }
                if uid:
                    response_params["user"] = uid
                if tool_defs:
                    response_params["tools"] = tool_defs
                # 思考模式下 temperature / top_p 不生效，官方明确忽略、不报错；思考时不发送
                if not self._is_thinking:
                    response_params["temperature"] = self.temperature

                self._monitor_request_prompt(
                    {"input": input_items, "reasoning": response_params["reasoning"]},
                    tools=tool_defs,
                )
                ctx = self._context_usage_event()
                if ctx:
                    yield ctx
                ac = self._maybe_auto_compact(
                    input_items, tools=tool_defs, system_text=self._build_system_prompt())
                if ac:
                    yield ac
                    self._refresh_system_message()
                    input_items = self._to_response_input_items()
                    response_params["input"] = input_items

                is_tool_round = bool(
                    tool_defs and any(it.get("type") == "function_call_output" for it in input_items))
                retry_delays = _retry_delays()
                retry_attempt = 0
                while True:
                    _req_start_ms = time.time() * 1000
                    self._traffic_request(
                        dict(response_params),
                        message_type="tool_round" if is_tool_round else "user_turn",
                        stream=False,
                        retry_attempt=retry_attempt,
                    )
                    try:
                        response = await client.responses.create(**response_params)
                        _req_dur_ms = int(time.time() * 1000 - _req_start_ms)
                        break
                    except Exception as e:
                        if not _is_openai_retryable_error(e):
                            self._traffic_response(
                                text="", input_tokens=0, output_tokens=0,
                                message_type="error",
                                status_code=getattr(e, "status_code", None) or 0,
                                duration_ms=int(time.time() * 1000 - _req_start_ms),
                                error=str(e)[:300],
                            )
                            raise
                        if retry_attempt >= RETRY_MAX_ATTEMPTS:
                            raise RuntimeError(
                                f"DeepSeek Responses API 限流（已重试 {RETRY_MAX_ATTEMPTS} 次仍失败）: {str(e)[:300]}")
                        yield _retry_event(
                            attempt=retry_attempt + 1,
                            delay_ms=int(retry_delays[retry_attempt] * 1000),
                            error_status=getattr(e, "status_code", None) or 429,
                            error=str(e)[:200],
                        )
                        self._traffic_response(
                            text="", input_tokens=0, output_tokens=0,
                            message_type="retry",
                            status_code=getattr(e, "status_code", None) or 429,
                            duration_ms=int(time.time() * 1000 - _req_start_ms),
                            error=str(e)[:200],
                        )
                        await asyncio.sleep(retry_delays[retry_attempt])
                        retry_attempt += 1

                text_parts: List[str] = []
                thinking_parts: List[str] = []
                resp_buf: List[str] = []
                function_calls: Dict[str, Dict[str, Any]] = {}
                function_order: List[str] = []
                web_search_calls: List[Dict[str, Any]] = []
                for item in (response.output or []):
                    it = self._item_field(item, "type", "")
                    if it == "message":
                        if self._item_field(item, "role", "") == "assistant":
                            for cb in (self._item_field(item, "content", []) or []):
                                if self._item_field(cb, "type", "") == "output_text":
                                    txt = self._item_field(cb, "text", "") or ""
                                    if txt:
                                        text_parts.append(txt)
                                        resp_buf.append(txt)
                                        yield {"type": "text", "text": txt}
                    elif it == "reasoning":
                        for cb in (self._item_field(item, "content", []) or []):
                            if self._item_field(cb, "type", "") == "reasoning_text":
                                txt = self._item_field(cb, "text", "") or ""
                                if txt:
                                    thinking_parts.append(txt)
                                    resp_buf.append(txt)
                                    yield {"type": "thinking", "text": txt}
                    elif it == "function_call":
                        call_id = self._item_field(item, "call_id", "") or self._item_field(item, "id", "")
                        function_order.append(call_id)
                        function_calls[call_id] = {
                            "call_id": call_id,
                            "name": self._item_field(item, "name", "") or "",
                            "arguments": self._item_field(item, "arguments", "") or "{}",
                        }
                    elif it == "web_search_call":
                        web_search_calls.append(self._as_dict(item))

                web_search_calls = self._normalize_web_search_calls(web_search_calls)
                text = "".join(text_parts)
                reasoning = "".join(thinking_parts)
                usage = getattr(response, "usage", None)
                _usage_hit, _usage_miss, _usage_out, _usage_reason = self._response_usage(usage)
                total_hit += _usage_hit
                total_miss += _usage_miss
                total_usage_in += _usage_hit + _usage_miss
                total_usage_out += _usage_out
                total_reasoning += _usage_reason

                if self._item_field(response, "status", "") == "failed":
                    err = getattr(response, "error", None)
                    err_msg = (getattr(err, "message", "") if err else "") or "未知错误"
                    self._traffic_response(
                        text="", input_tokens=0, output_tokens=0,
                        message_type="error", status_code=0,
                        duration_ms=_req_dur_ms, error=str(err_msg)[:300],
                    )
                    yield {"type": "error", "text": f"DeepSeek 响应失败：{err_msg}"}
                    return

                if function_order:
                    # 需要本地执行 function 工具：把本轮 assistant 消息与 function_call 落盘
                    assistant_blocks: List[Dict[str, Any]] = []
                    if reasoning:
                        assistant_blocks.append({"type": "thinking", "thinking": reasoning})
                    if text:
                        assistant_blocks.append({"type": "text", "text": text})
                    for call_id in function_order:
                        fc = function_calls[call_id]
                        try:
                            fc_args = json.loads(fc["arguments"] or "{}")
                        except Exception:
                            fc_args = {}
                        assistant_blocks.append({
                            "type": "tool_use",
                            "id": fc["call_id"] or call_id,
                            "name": fc["name"],
                            "input": fc_args,
                        })
                    for wsc in web_search_calls:
                        assistant_blocks.append({"type": "web_search_call", **wsc})
                    if self._persist_tool_context:
                        self.messages.append(Message(
                            role="assistant", content=assistant_blocks,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    else:
                        self.messages.append(Message(
                            role="assistant", content=text,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    resp_full = f"{reasoning}\n\n{text}" if reasoning else text
                    for call_id in function_order:
                        fc = function_calls[call_id]
                        try:
                            _inp = json.dumps(json.loads(fc["arguments"] or "{}"), ensure_ascii=False)
                        except Exception:
                            _inp = "{}"
                        resp_buf.append(f"\n[tool_use] {fc['name']}: {_inp}")
                    _record_model_response(self.model_name, "".join(resp_buf))
                    self._traffic_response(
                        text="".join(resp_buf),
                        input_tokens=_usage_hit + _usage_miss if usage else total_usage_in,
                        cache_hit_tokens=_usage_hit,
                        cache_miss_tokens=_usage_miss,
                        output_tokens=_usage_out if usage else total_usage_out,
                        reasoning_tokens=_usage_reason,
                        message_type="tool_round",
                        stream=False,
                        duration_ms=_req_dur_ms,
                    )
                    # 把本轮 assistant 消息（含 reasoning）与 function_call / web_search_call
                    # 追加进 input，下一轮回传——Responses 无状态，必须带上本轮输出才连贯。
                    if reasoning:
                        input_items.append({
                            "type": "reasoning",
                            "content": [{"type": "reasoning_text", "text": reasoning}],
                        })
                    if text:
                        input_items.append({
                            "type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        })
                    for _wsc in web_search_calls:
                        input_items.append({"type": "web_search_call", **_wsc})
                    for call_id in function_order:
                        fc = function_calls[call_id]
                        input_items.append({
                            "type": "function_call",
                            "call_id": fc["call_id"] or call_id,
                            "name": fc["name"],
                            "arguments": fc["arguments"] or "{}",
                        })
                    for call_id in function_order:
                        fc = function_calls[call_id]
                        try:
                            args = json.loads(fc["arguments"] or "{}")
                        except Exception:
                            args = {}
                        yield {"type": "tool_use", "id": fc["call_id"] or call_id,
                               "name": fc["name"], "input": args}
                        result = self._run_tool(fc["name"], args)
                        if isinstance(result, dict) and result.get("__media__"):
                            result = f"[图片 × {len(result['__media__'])}]"
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": fc["call_id"] or call_id,
                            "output": str(result),
                        })
                        if self._persist_tool_context:
                            self.messages.append(Message(
                                role="user",
                                content=str(result),
                                timestamp=int(datetime.now().timestamp() * 1000),
                                tool_name=fc["name"],
                                tool_input=args,
                                tool_result=str(result),
                            ))
                        yield {"type": "tool_result", "tool_use_id": fc["call_id"] or call_id,
                               "tool_name": fc["name"], "content": result}
                    # 整轮 function_call（可能含多个）已全部写回 self.messages：此时才节流落盘，
                    # 避免在多调用轮中间写入“只有部分 function_call_output”的非对称状态。
                    if self._persist_tool_context:
                        self._maybe_persist_tool_round()
                    continue

                # No more tool calls: emit the final reply.
                final_blocks: Any = text
                if reasoning:
                    if text:
                        final_blocks = [
                            {"type": "thinking", "thinking": reasoning},
                            {"type": "text", "text": text},
                        ]
                    else:
                        final_blocks = [{"type": "thinking", "thinking": reasoning}]
                self.messages.append(Message(
                    role="assistant", content=final_blocks,
                    timestamp=int(datetime.now().timestamp() * 1000),
                ))
                resp_full = f"{reasoning}\n\n{text}" if reasoning else text
                _record_model_response(self.model_name, resp_full)
                self._traffic_response(
                    text=resp_full,
                    input_tokens=_usage_hit + _usage_miss if usage else total_usage_in,
                    cache_hit_tokens=_usage_hit,
                    cache_miss_tokens=_usage_miss,
                    output_tokens=_usage_out if usage else total_usage_out,
                    reasoning_tokens=_usage_reason,
                    message_type="user_turn",
                    stream=False,
                    duration_ms=_req_dur_ms,
                )
                yield {
                    "type": "usage",
                    "input_tokens": total_usage_in,
                    "cache_hit_tokens": total_hit,
                    "cache_miss_tokens": total_miss,
                    "output_tokens": total_usage_out,
                    "reasoning_tokens": total_reasoning,
                }
                return

            yield {"type": "error", "text": f"（达到工具调用轮数上限 {MAX_TOOL_ROUNDS} 轮，已停止）"}
        except Exception as e:
            yield {"type": "error", "text": f"Error: {str(e)}"}
    @staticmethod
    def _as_dict(item: Any) -> Dict[str, Any]:
        """把 SDK item 对象转 dict（尽量保留原始字段，便于 web_search_call 原样回传）。"""
        if isinstance(item, dict):
            return item
        try:
            if hasattr(item, "model_dump"):
                d = item.model_dump(exclude_none=True)
                if isinstance(d, dict):
                    return d
            out: Dict[str, Any] = {}
            for k in ("id", "type", "call_id", "name", "status", "role", "action"):
                v = getattr(item, k, None)
                if v is not None:
                    out[k] = v
            return out
        except Exception:
            return {}
    @staticmethod
    def _item_field(item: Any, key: str, default: Any = None) -> Any:
        """从 dict 或 SDK pydantic 对象安全读取字段。"""
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)
    @staticmethod
    def _normalize_web_search_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter/dedupe web_search_call items, keeping only those carrying an ``action``."""
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        anon: List[Dict[str, Any]] = []
        for c in calls or []:
            if not isinstance(c, dict) or c.get("action") is None:
                continue
            cid = c.get("id") or c.get("call_id")
            if cid:
                if cid not in merged:
                    order.append(cid)
                merged[cid] = c
            else:
                anon.append(c)
        return [merged[k] for k in order] + anon
