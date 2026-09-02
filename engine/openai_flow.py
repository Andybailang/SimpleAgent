"""engine.openai_flow — OpenAI Chat Completions（含 DeepSeek 思考/tools）流式与非流式工具循环。
"""
import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List, Optional

from .config import (
    MAX_TOOL_ROUNDS,
    RETRY_MAX_ATTEMPTS,
    THINKING_BUDGETS,
    _estimate_output_tokens,
    _is_openai_retryable_error,
    _retry_delays,
    _retry_event,
)
from .util import Message, _record_model_response, _usage_field

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

class OpenAIFlowMixin:
    def _call_llm(self, user_message: str, use_tools: bool = True) -> str:
        """调用 LLM（use_tools=False 时不携带工具定义，用于聊天模式）"""
        # 添加用户消息
        self.messages.append(Message(
            role="user",
            content=user_message,
            timestamp=int(datetime.now().timestamp() * 1000)
        ))

        try:
            # 构建 API 请求
            messages_to_send = []
            for msg in self.messages:
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content
                }
                messages_to_send.append(msg_dict)

            tool_defs = self._tool_definitions() if use_tools else None
            self._monitor_request_prompt(messages_to_send, tools=tool_defs)

            # 调用 LLM（第一次，不带 response_format，让 LLM 自由选择）
            payload_req = {
                "model": self.model_name,
                "messages": messages_to_send,
                "tools": tool_defs,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            _req_start_ms = time.time() * 1000
            self._traffic_request(payload_req, message_type="user_turn", stream=False)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_to_send,
                tools=tool_defs,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            _req_dur_ms = int(time.time() * 1000 - _req_start_ms)

            # 解析响应
            choice = response.choices[0]
            assistant_message = choice.message

            # 添加 assistant 消息
            assistant_msg = Message(
                role="assistant",
                content=assistant_message.content,
                timestamp=int(datetime.now().timestamp() * 1000)
            )
            self.messages.append(assistant_msg)

            # 记录模型本轮返回内容（From 方向，全部内容）
            _resp_text = assistant_message.content or ""
            if assistant_message.tool_calls:
                for tc in assistant_message.tool_calls:
                    try:
                        tc_args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        tc_args = {}
                    try:
                        tc_inp = json.dumps(tc_args, ensure_ascii=False)
                    except Exception:
                        tc_inp = "{}"
                    _resp_text = f"{_resp_text}\n[tool_use] {tc.function.name}: {tc_inp}"
            _record_model_response(self.model_name, _resp_text)
            # LLM 流量存档：第一次响应（request 已在上方记录）
            _uhit, _umiss, _uout, _ureason = self._openai_usage(response.usage)
            self._traffic_response(
                text=_resp_text,
                input_tokens=_uhit + _umiss,
                cache_hit_tokens=_uhit,
                cache_miss_tokens=_umiss,
                output_tokens=_uout,
                reasoning_tokens=_ureason,
                message_type="user_turn",
                duration_ms=_req_dur_ms,
            )

            # 检查是否有工具调用
            tool_calls = assistant_message.tool_calls
            if tool_calls:
                # 执行所有工具调用
                results = []
                for tool_call in tool_calls:
                    result = self._execute_tool({
                        "tool_name": tool_call.function.name,
                        "tool_input": json.loads(tool_call.function.arguments)
                    })
                    results.append(result)

                # 构建工具结果消息（纯文本格式）
                tool_result_text = "\n\n".join([
                    f"工具 {tc.function.name} 返回: {result}"
                    for tc, result in zip(tool_calls, results)
                ])

                # 构建第二次调用的消息列表
                second_messages = messages_to_send + [
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": tool_calls
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0].id,
                        "content": tool_result_text
                    }
                ]

                # 第二次调用 LLM，传入工具调用和结果，要求给出用户友好的回复
                _payload2 = {
                    "model": self.model_name,
                    "messages": second_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
                _req2_start_ms = time.time() * 1000
                self._traffic_request(_payload2, message_type="tool_round", stream=False)
                second_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=second_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                _req2_dur_ms = int(time.time() * 1000 - _req2_start_ms)

                final_message = second_response.choices[0].message
                _uhit2, _umiss2, _uout2, _ureason2 = self._openai_usage(second_response.usage)
                self._traffic_response(
                    text=final_message.content or "",
                    input_tokens=_uhit2 + _umiss2,
                    cache_hit_tokens=_uhit2,
                    cache_miss_tokens=_umiss2,
                    output_tokens=_uout2,
                    reasoning_tokens=_ureason2,
                    message_type="tool_round",
                    duration_ms=_req2_dur_ms,
                )
                self.messages.append(Message(
                    role="assistant",
                    content=final_message.content,
                    timestamp=int(datetime.now().timestamp() * 1000)
                ))

                return final_message.content

            # 没有工具调用，直接返回
            return assistant_message.content

        except Exception as e:
            # 添加错误消息
            error_msg = f"错误：{str(e)}"
            self.messages.append(Message(
                role="assistant",
                content=error_msg,
                timestamp=int(datetime.now().timestamp() * 1000)
            ))
            return error_msg
    def _with_attachments(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Build OpenAI-compatible content for a user turn with attachments.

        图片附件不再内联上传：仅把路径放进文本，模型通过
        extract_text_from_image 工具读取文字；其他文件同样只传路径，
        由模型用 Read 工具自行读取。始终返回普通字符串。
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
    def chat(self, user_message: str, use_tools: Optional[bool] = None) -> str:
        """
        与 Agent 对话

        Args:
            user_message: 用户输入的消息

        Returns:
            Agent 的回复
        """
        if use_tools is None:
            use_tools = (self.mode != "chat")
        return self._call_llm(user_message, use_tools=use_tools)
    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        流式对话（异步）

        Args:
            user_message: 用户输入的消息

        Yields:
            每次返回的文本片段
        """
        try:
            # 添加用户消息
            self.messages.append(Message(
                role="user",
                content=user_message,
                timestamp=int(datetime.now().timestamp() * 1000)
            ))

            # 构建 API 请求
            messages_to_send = []
            for msg in self.messages:
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content
                }
                messages_to_send.append(msg_dict)

            tool_defs = self._tool_definitions()
            self._monitor_request_prompt(messages_to_send, tools=tool_defs)

            # 调用 LLM（流式）
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_to_send,
                tools=tool_defs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True  # 开启流式
            )

            # 处理流式响应
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            # 添加 assistant 消息
            self.messages.append(Message(
                role="assistant",
                content="",  # 流式响应，稍后更新
                timestamp=int(datetime.now().timestamp() * 1000)
            ))

        except Exception as e:
            error_msg = f"错误：{str(e)}"
            yield error_msg
    def _to_openai_user_content(self, content: Any) -> Any:
        """Normalize stored content (str / Anthropic blocks / OpenAI blocks) to OpenAI-format content."""
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
                # 图片原始数据绝不进入上下文：统一替换为文本占位（与 anthropic 路径一致），
                # 需要看图时由模型调用工具（extract_text_from_image 等）在后端读取。
                blocks.append({"type": "text", "text": "[图片附件已省略]"})
        if len(blocks) == 1 and blocks[0].get("type") == "text":
            return blocks[0]["text"]
        return blocks
    def _content_blocks(self, content: Any):
        """把 Message.content 拆成 (text, thinking, tool_uses, tool_results)。

        thinking 块可能是 {"type":"thinking","thinking":...} 或带 text 字段；
        tool_use 块形如 {"type":"tool_use","id","name","input"}；
        tool_result 块形如 {"type":"tool_result","tool_use_id","content"}。
        """
        text = ""
        thinking = ""
        tool_uses: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    text += str(b.get("text") or "")
                elif bt == "thinking":
                    thinking += str(b.get("thinking") or b.get("text") or "")
                elif bt == "tool_use":
                    tool_uses.append(b)
                elif bt == "tool_result":
                    tool_results.append(b)
        return text, thinking, tool_uses, tool_results
    def _is_tool_result_message(self, msg: "Message") -> bool:
        """判断一条 user 消息是否为工具结果（用于还原 OpenAI tool 消息）。"""
        if msg.role != "user":
            return False
        if msg.tool_name:
            return True
        if isinstance(msg.content, list):
            return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in msg.content)
        return False
    def _to_openai_message(self, msg: "Message") -> Dict[str, Any]:
        """把一条 Message 转成 OpenAI Chat Completions 消息（含 reasoning_content / tool_calls）。"""
        role = msg.role
        if self._is_tool_result_message(msg):
            content = msg.content
            if isinstance(content, list):
                results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if results:
                    return {
                        "role": "tool",
                        "tool_call_id": results[0].get("tool_use_id") or "",
                        "content": str(results[0].get("content") or ""),
                    }
            # 纯文本 tool 结果（openai 路径持久化格式）：call_id 由 _build_openai_messages 回填
            return {"role": "tool", "tool_call_id": "", "content": str(content or "")}
        if role != "assistant":
            return {"role": role, "content": self._to_openai_user_content(msg.content)}
        text, thinking, tool_uses, _ = self._content_blocks(msg.content)
        out: Dict[str, Any] = {"role": "assistant", "content": text or None}
        # DeepSeek 思考模式 + tools 的“all-or-nothing”约束：历史里一旦任一 assistant
        # 消息带 reasoning_content，则所有 assistant 消息都必须带该字段（可为空字符串），
        # 缺失会让 DeepSeek 返回 400: "The `reasoning_content` in the thinking mode must
        # be passed back to the API."。这里只要处于思考模式就统一补齐，避免个别轮次缺失。
        if self._is_thinking:
            out["reasoning_content"] = thinking or ""
        if tool_uses:
            out["tool_calls"] = [
                {
                    "id": tu.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tu.get("name") or "",
                        "arguments": json.dumps(tu.get("input") or {}, ensure_ascii=False),
                    },
                }
                for i, tu in enumerate(tool_uses)
            ]
        return out
    def _build_openai_messages(self) -> List[Dict[str, Any]]:
        """把 self.messages 还原为 OpenAI 消息列表。

        工具结果以 tool 消息还原：优先用 tool_result 块里的 tool_use_id；纯文本工具结果
        （openai 路径持久化格式）按出现顺序回填最近一条 assistant 消息的 tool_call id。
        """
        out: List[Dict[str, Any]] = []
        pending_ids: List[str] = []
        for msg in self.messages:
            if self._is_tool_result_message(msg) and not (isinstance(msg.content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in msg.content)):
                call_id = pending_ids.pop(0) if pending_ids else ""
                item = self._to_openai_message(msg)
                item["tool_call_id"] = call_id
                out.append(item)
                continue
            item = self._to_openai_message(msg)
            # 顺带收集 assistant tool_calls 的 id，供后续纯文本工具结果回填
            for tc in (item.get("tool_calls") or []):
                pending_ids.append(tc["id"])
            out.append(item)
        return out
    @staticmethod
    def _openai_usage(usage: Any) -> tuple:
        """从 Chat Completions usage 取 (cache_hit, cache_miss, output, reasoning_tokens)。

        兼容多家字段：
        - DeepSeek / SiliconFlow 新式：prompt_cache_hit_tokens / prompt_cache_miss_tokens；
        - OpenAI 旧式 / Zhipu：prompt_tokens_details.cached_tokens（命中），
          未命中 = prompt_tokens - cached_tokens；
        - 均缺失时按未命中计（输入 = prompt_tokens）。
        """
        if usage is None:
            return 0, 0, 0, 0
        hit = _usage_field(usage, "prompt_cache_hit_tokens")
        miss = _usage_field(usage, "prompt_cache_miss_tokens")
        prompt_total = _usage_field(usage, "prompt_tokens")
        if hit or miss:
            # 命中/未命中都给了直接用；只有一侧时用 prompt_tokens 补另一侧
            if miss == 0 and hit > 0:
                miss = max(0, prompt_total - hit)
            if hit == 0 and miss > 0:
                hit = max(0, prompt_total - miss)
        else:
            # 旧式 / Zhipu：prompt_tokens_details.cached_tokens 为命中
            cached = _usage_field(usage, "prompt_tokens_details.cached_tokens")
            hit = cached
            miss = max(0, prompt_total - cached)
        if hit == 0 and miss == 0:
            miss = prompt_total
        out = _usage_field(usage, "completion_tokens")
        reasoning = _usage_field(usage, "completion_tokens_details.reasoning_tokens")
        return hit, miss, out, reasoning
    async def chat_stream_openai_async(self, user_message: str, use_tools: bool = True, attachments: Optional[List[Dict[str, Any]]] = None, continue_only: bool = False):
        """DeepSeek Chat Completions（OpenAI 兼容）流式工具循环，产出与 Anthropic 路径一致的事件。

        关键点（对齐 DeepSeek 官方 Chat Completions 定义）：
        - 思考模式：`thinking` 经 extra_body 传入；思考时 temperature/top_p 不发送（官方忽略）；
        - 工具调用时回传 reasoning_content（思考模式 + tools 下必须，否则 400）；
        - 终端用户标识用 `user_id`（DeepSeek 文档：Chat 叫 user_id，Responses 叫 user）。
        - 流式：正文走 `delta.content`，思考走 `delta.reasoning_content`，工具调用走
          `delta.tool_calls`（按 index 逐块累加）；末块经 `stream_options.include_usage` 带回 usage；
        - SiliconFlow 平台思考开关用 enable_thinking / thinking_budget（见 _deepseek_thinking_extra_body），
          流式路径与官方 DeepSeek 一样消费 reasoning_content。
        """
        if not continue_only:
            self.messages.append(Message(
                role="user",
                content=self._with_attachments(user_message, attachments),
                timestamp=int(datetime.now().timestamp() * 1000)
            ))
        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            # 用能保留 reasoning_content / tool_calls 的还原方法构建历史，
            # 满足 DeepSeek 在思考模式 + tools 下必须回传 reasoning_content 的要求。
            messages_to_send: List[Dict[str, Any]] = self._build_openai_messages()
            tool_defs = self._tool_definitions() if use_tools else None

            total_usage_in = 0
            total_usage_out = 0
            total_hit = 0
            total_miss = 0
            total_reasoning = 0
            for _ in range(MAX_TOOL_ROUNDS):
                self._monitor_request_prompt(messages_to_send, tools=tool_defs)
                retry_delays = _retry_delays()
                retry_attempt = 0
                while True:
                    _req_start_ms = time.time() * 1000
                    try:
                        # DeepSeek 思考模式：thinking 必须放 extra_body；思考模式下
                        # temperature / top_p 不生效（官方明确忽略），因此思考时不发送。
                        extra_body = self._deepseek_thinking_extra_body() or None
                        # 发送前读取流量库本会话最近 request（专供 UI 实时显示）
                        ctx = self._context_usage_event()
                        if ctx:
                            yield ctx
                        # 自动压缩：发送前本地估算上下文体积，超阈值则压缩并通知 UI
                        ac = self._maybe_auto_compact(
                            messages_to_send, tools=tool_defs,
                            system_text=self._build_system_prompt(),
                        )
                        if ac:
                            yield ac
                            self._refresh_system_message()
                            messages_to_send = self._build_openai_messages()
                        is_tool_round = bool(tool_defs and not messages_to_send[-1].get("role") == "user")
                        _payload_openai = {
                            "model": self.model_name,
                            "messages": messages_to_send,
                            "tools": tool_defs,
                            "max_tokens": self.max_tokens,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        }
                        if not self._is_thinking:
                            _payload_openai["temperature"] = self.temperature
                        if extra_body:
                            _payload_openai["extra_body"] = extra_body
                        self._traffic_request(
                            _payload_openai,
                            message_type="tool_round" if is_tool_round else "user_turn",
                            stream=True,
                            retry_attempt=retry_attempt,
                        )
                        stream = await client.chat.completions.create(**_payload_openai)
                        break
                    except Exception as e:
                        if not _is_openai_retryable_error(e):
                            # 非限流错误：记录一次失败的请求流量（响应侧 error）
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
                                f"OpenAI API 限流（已重试 {RETRY_MAX_ATTEMPTS} 次仍失败）: {str(e)[:300]}"
                            )
                        yield _retry_event(
                            attempt=retry_attempt + 1,
                            delay_ms=int(retry_delays[retry_attempt] * 1000),
                            error_status=getattr(e, "status_code", None) or 429,
                            error=str(e)[:200],
                        )
                        # 限流重试：也记录一次失败的请求流量（status=429）
                        self._traffic_response(
                            text="", input_tokens=0, output_tokens=0,
                            message_type="retry",
                            status_code=getattr(e, "status_code", None) or 429,
                            duration_ms=int(time.time() * 1000 - _req_start_ms),
                            error=str(e)[:200],
                        )
                        await asyncio.sleep(retry_delays[retry_attempt])
                        retry_attempt += 1
                # ---- 消费流式分块 ----
                content_parts: List[str] = []
                thinking_parts: List[str] = []
                tool_call_acc: Dict[int, Dict[str, str]] = {}
                usage: Any = None
                _req_dur_ms = int(time.time() * 1000 - _req_start_ms)
                try:
                    async for chunk in stream:
                        _req_dur_ms = int(time.time() * 1000 - _req_start_ms)
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            usage = chunk_usage
                        for choice in (chunk.choices or []):
                            delta = choice.delta
                            delta_content = delta.content
                            if delta_content:
                                content_parts.append(delta_content)
                                yield {"type": "text", "text": delta_content}
                            delta_reasoning = getattr(delta, "reasoning_content", None) or ""
                            if delta_reasoning:
                                thinking_parts.append(delta_reasoning)
                                yield {"type": "thinking", "text": delta_reasoning}
                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    tc_index = tc.index
                                    tc_id = tc.id or ""
                                    tc_func = tc.function
                                    tc_name = (tc_func.name if tc_func else None) or ""
                                    tc_args = (tc_func.arguments if tc_func else None) or ""
                                    # 部分厂商（deepseek-v4-flash）在真实工具块结束时补一个空哨兵块：
                                    # {"index":0,"id":"","type":"function","function":{"arguments":null}}
                                    # 全空即跳过，避免积累出幽灵工具调用。
                                    if not tc_id and not tc_name and not tc_args:
                                        continue
                                    acc = tool_call_acc.setdefault(
                                        tc_index, {"id": "", "name": "", "arguments": ""})
                                    if tc_id:
                                        acc["id"] = tc_id
                                    if tc_name:
                                        acc["name"] = tc_name
                                    if tc_args:
                                        acc["arguments"] += tc_args
                except Exception as e:
                    # 流式中途异常：记录失败响应并抛出，由外层统一收口为 error 事件
                    self._traffic_response(
                        text="", input_tokens=0, output_tokens=0,
                        message_type="error",
                        status_code=getattr(e, "status_code", None) or 0,
                        duration_ms=_req_dur_ms,
                        error=str(e)[:300],
                    )
                    raise
                content = "".join(content_parts)
                reasoning = "".join(thinking_parts)
                # 按 index 还原完整工具调用：用简单对象保持下游 attribute 访问兼容
                tool_calls = []
                for tc_index in sorted(tool_call_acc.keys()):
                    acc = tool_call_acc[tc_index]
                    if acc["id"] or acc["name"] or acc["arguments"]:
                        tool_calls.append(SimpleNamespace(
                            id=acc["id"] or f"call_{tc_index}",
                            function=SimpleNamespace(name=acc["name"], arguments=acc["arguments"] or "{}"),
                        ))
                resp_full = f"{reasoning}\n\n{content}" if reasoning else content
                _usage_hit, _usage_miss, _usage_out, _usage_reason = self._openai_usage(usage)
                if usage is None:
                    # 厂商未回传 usage（如网关忽略 stream_options 时）：输出按文本粗估
                    _usage_out = _estimate_output_tokens(content + reasoning)
                total_hit += _usage_hit
                total_miss += _usage_miss
                total_usage_in += _usage_hit + _usage_miss
                total_usage_out += _usage_out
                total_reasoning += _usage_reason

                if tool_calls:
                    record_tool_uses = []
                    for tc in tool_calls:
                        try:
                            record_args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            record_args = {}
                        record_tool_uses.append({"id": tc.id, "name": tc.function.name, "input": record_args})
                    assistant_msg = {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "{}",
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                    if self._is_thinking:
                        assistant_msg["reasoning_content"] = reasoning or ""
                    messages_to_send.append(assistant_msg)
                    if self._persist_tool_context:
                        # 写回上下文：assistant 消息携带 tool_use 块，工具结果随后写回
                        assistant_blocks: List[Dict[str, Any]] = []
                        if reasoning:
                            assistant_blocks.append({"type": "thinking", "thinking": reasoning})
                        if content:
                            assistant_blocks.append({"type": "text", "text": content})
                        for tc in tool_calls:
                            try:
                                tc_args = json.loads(tc.function.arguments or "{}")
                            except Exception:
                                tc_args = {}
                            assistant_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": tc_args,
                            })
                        self.messages.append(Message(
                            role="assistant",
                            content=assistant_blocks,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    else:
                        self.messages.append(Message(
                            role="assistant",
                            content=content,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    # 记录模型本轮返回内容（From 方向，全部内容）
                    _resp_text = resp_full
                    for _rt in record_tool_uses:
                        try:
                            _rt_inp = json.dumps(_rt.get("input") or {}, ensure_ascii=False)
                        except Exception:
                            _rt_inp = "{}"
                        _resp_text = f"{_resp_text}\n[tool_use] {_rt.get('name') or '?'}: {_rt_inp}"
                    _record_model_response(self.model_name, _resp_text)
                    # LLM 流量存档：工具轮响应（成功）
                    self._traffic_response(
                        text=_resp_text,
                        input_tokens=_usage_hit + _usage_miss if usage else total_usage_in,
                        cache_hit_tokens=_usage_hit,
                        cache_miss_tokens=_usage_miss,
                        output_tokens=_usage_out if usage else total_usage_out,
                        reasoning_tokens=_usage_reason,
                        message_type="tool_round",
                        stream=True,
                        duration_ms=_req_dur_ms,
                    )
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            args = {}
                        ask_event = self._handle_ask_user_question(tc.function.name, tc.id, args)
                        if ask_event is not None:
                            yield {"type": "tool_use", "id": tc.id, "name": "AskUserQuestion", "input": ask_event["input"]}
                            # 确保 assistant(tool_use) 写入 self.messages，供回答后续跑。
                            self._ensure_ask_user_assistant_message(tc.id, tc.function.name, ask_event["input"])
                            yield ask_event
                            break
                        yield {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args}
                        result = self._run_tool(tc.function.name, args)
                        if isinstance(result, dict) and result.get("__media__"):
                            result = f"[图片 × {len(result['__media__'])}]"
                        messages_to_send.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        if self._persist_tool_context:
                            self.messages.append(Message(
                                role="user",
                                content=str(result),
                                timestamp=int(datetime.now().timestamp() * 1000),
                                tool_name=tc.function.name,
                            tool_input=args,
                            tool_result=str(result),
                        ))
                        yield {"type": "tool_result", "tool_use_id": tc.id, "tool_name": tc.function.name, "content": result}
                    if self._ask_user_paused:
                        # 已向用户提问：结束本轮生成，等待用户回答（不继续工具循环、不落 final）。
                        return
                    # 整轮工具调用（可能含多个 tool_use）已全部写回 self.messages：
                    # 此时才节流落盘，避免在多工具轮中间写入“只有部分 tool_result”的非对称状态。
                    if self._persist_tool_context:
                        self._maybe_persist_tool_round()
                    continue

                # No more tool calls: emit the final reply.
                final_blocks: Any = content
                if reasoning:
                    if content:
                        final_blocks = [
                            {"type": "thinking", "thinking": reasoning},
                            {"type": "text", "text": content},
                        ]
                    else:
                        final_blocks = [{"type": "thinking", "thinking": reasoning}]
                self.messages.append(Message(
                    role="assistant",
                    content=final_blocks,
                    timestamp=int(datetime.now().timestamp() * 1000),
                ))
                # 记录模型最终回复（From 方向）
                _record_model_response(self.model_name, resp_full)
                # LLM 流量存档：最终回复响应（成功）
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

    async def chat_nonstream_openai_async(self, user_message: str, use_tools: bool = True, attachments: Optional[List[Dict[str, Any]]] = None, continue_only: bool = False):
        """DeepSeek Chat Completions（OpenAI 兼容）非流式工具循环，产出与流式路径一致的事件。

        供声明 stream_supported=false 的 OpenAI 兼容模型使用。思考开关（官方 thinking / SiliconFlow
        enable_thinking）、思考后工具调用必须回传 reasoning_content、终端标识 user_id 等约束与
        chat_stream_openai_async 完全一致。
        """
        if not continue_only:
            self.messages.append(Message(
                role="user",
                content=self._with_attachments(user_message, attachments),
                timestamp=int(datetime.now().timestamp() * 1000)
            ))
        try:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            # 用能保留 reasoning_content / tool_calls 的还原方法构建历史，
            # 满足 DeepSeek 在思考模式 + tools 下必须回传 reasoning_content 的要求。
            messages_to_send: List[Dict[str, Any]] = self._build_openai_messages()
            tool_defs = self._tool_definitions() if use_tools else None

            total_usage_in = 0
            total_usage_out = 0
            total_hit = 0
            total_miss = 0
            total_reasoning = 0
            for _ in range(MAX_TOOL_ROUNDS):
                self._monitor_request_prompt(messages_to_send, tools=tool_defs)
                retry_delays = _retry_delays()
                retry_attempt = 0
                while True:
                    _req_start_ms = time.time() * 1000
                    try:
                        # DeepSeek 思考模式：thinking 必须放 extra_body；思考模式下
                        # temperature / top_p 不生效（官方明确忽略），因此思考时不发送。
                        extra_body = self._deepseek_thinking_extra_body() or None
                        # 发送前读取流量库本会话最近 request（专供 UI 实时显示）
                        ctx = self._context_usage_event()
                        if ctx:
                            yield ctx
                        # 自动压缩：发送前本地估算上下文体积，超阈值则压缩并通知 UI
                        ac = self._maybe_auto_compact(
                            messages_to_send, tools=tool_defs,
                            system_text=self._build_system_prompt(),
                        )
                        if ac:
                            yield ac
                            self._refresh_system_message()
                            messages_to_send = self._build_openai_messages()
                        is_tool_round = bool(tool_defs and not messages_to_send[-1].get("role") == "user")
                        _payload_openai = {
                            "model": self.model_name,
                            "messages": messages_to_send,
                            "tools": tool_defs,
                            "max_tokens": self.max_tokens,
                            "stream": False,
                        }
                        if not self._is_thinking:
                            _payload_openai["temperature"] = self.temperature
                        if extra_body:
                            _payload_openai["extra_body"] = extra_body
                        self._traffic_request(
                            _payload_openai,
                            message_type="tool_round" if is_tool_round else "user_turn",
                            stream=False,
                            retry_attempt=retry_attempt,
                        )
                        response = await client.chat.completions.create(**_payload_openai)
                        _req_dur_ms = int(time.time() * 1000 - _req_start_ms)
                        break
                    except Exception as e:
                        if not _is_openai_retryable_error(e):
                            # 非限流错误：记录一次失败的请求流量（响应侧 error）
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
                                f"OpenAI API 限流（已重试 {RETRY_MAX_ATTEMPTS} 次仍失败）: {str(e)[:300]}"
                            )
                        yield _retry_event(
                            attempt=retry_attempt + 1,
                            delay_ms=int(retry_delays[retry_attempt] * 1000),
                            error_status=getattr(e, "status_code", None) or 429,
                            error=str(e)[:200],
                        )
                        # 限流重试：也记录一次失败的请求流量（status=429）
                        self._traffic_response(
                            text="", input_tokens=0, output_tokens=0,
                            message_type="retry",
                            status_code=getattr(e, "status_code", None) or 429,
                            duration_ms=int(time.time() * 1000 - _req_start_ms),
                            error=str(e)[:200],
                        )
                        await asyncio.sleep(retry_delays[retry_attempt])
                        retry_attempt += 1
                choice = response.choices[0]
                assistant_message = choice.message
                content = assistant_message.content or ""
                tool_calls = assistant_message.tool_calls
                usage = response.usage
                # DeepSeek 思考模式把思维链放在 reasoning_content，需显式取出
                reasoning = getattr(assistant_message, "reasoning_content", None) or ""
                if reasoning:
                    yield {"type": "thinking", "text": reasoning}
                if content:
                    yield {"type": "text", "text": content}
                resp_full = f"{reasoning}\n\n{content}" if reasoning else content
                _usage_hit, _usage_miss, _usage_out, _usage_reason = self._openai_usage(usage)
                if usage is None:
                    # 厂商未回传 usage：输出按文本粗估
                    _usage_out = _estimate_output_tokens(content + reasoning)
                total_hit += _usage_hit
                total_miss += _usage_miss
                total_usage_in += _usage_hit + _usage_miss
                total_usage_out += _usage_out
                total_reasoning += _usage_reason

                if tool_calls:
                    record_tool_uses = []
                    for tc in tool_calls:
                        try:
                            record_args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            record_args = {}
                        record_tool_uses.append({"id": tc.id, "name": tc.function.name, "input": record_args})
                    assistant_msg = {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "{}",
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                    if self._is_thinking:
                        assistant_msg["reasoning_content"] = reasoning or ""
                    messages_to_send.append(assistant_msg)
                    if self._persist_tool_context:
                        # 写回上下文：assistant 消息携带 tool_use 块，工具结果随后写回
                        assistant_blocks: List[Dict[str, Any]] = []
                        if reasoning:
                            assistant_blocks.append({"type": "thinking", "thinking": reasoning})
                        if content:
                            assistant_blocks.append({"type": "text", "text": content})
                        for tc in tool_calls:
                            try:
                                tc_args = json.loads(tc.function.arguments or "{}")
                            except Exception:
                                tc_args = {}
                            assistant_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": tc_args,
                            })
                        self.messages.append(Message(
                            role="assistant",
                            content=assistant_blocks,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    else:
                        self.messages.append(Message(
                            role="assistant",
                            content=content,
                            timestamp=int(datetime.now().timestamp() * 1000),
                        ))
                    # 记录模型本轮返回内容（From 方向，全部内容）
                    _resp_text = resp_full
                    for _rt in record_tool_uses:
                        try:
                            _rt_inp = json.dumps(_rt.get("input") or {}, ensure_ascii=False)
                        except Exception:
                            _rt_inp = "{}"
                        _resp_text = f"{_resp_text}\n[tool_use] {_rt.get('name') or '?'}: {_rt_inp}"
                    _record_model_response(self.model_name, _resp_text)
                    # LLM 流量存档：工具轮响应（成功）
                    self._traffic_response(
                        text=_resp_text,
                        input_tokens=_usage_hit + _usage_miss if usage else total_usage_in,
                        cache_hit_tokens=_usage_hit,
                        cache_miss_tokens=_usage_miss,
                        output_tokens=_usage_out if usage else total_usage_out,
                        reasoning_tokens=_usage_reason,
                        message_type="tool_round",
                        stream=False,
                        duration_ms=_req_dur_ms,
                    )
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            args = {}
                        ask_event = self._handle_ask_user_question(tc.function.name, tc.id, args)
                        if ask_event is not None:
                            yield {"type": "tool_use", "id": tc.id, "name": "AskUserQuestion", "input": ask_event["input"]}
                            self._ensure_ask_user_assistant_message(tc.id, tc.function.name, ask_event["input"])
                            yield ask_event
                            break
                        yield {"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args}
                        result = self._run_tool(tc.function.name, args)
                        if isinstance(result, dict) and result.get("__media__"):
                            result = f"[图片 × {len(result['__media__'])}]"
                        messages_to_send.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                        if self._persist_tool_context:
                            self.messages.append(Message(
                                role="user",
                                content=str(result),
                                timestamp=int(datetime.now().timestamp() * 1000),
                                tool_name=tc.function.name,
                            tool_input=args,
                            tool_result=str(result),
                        ))
                        yield {"type": "tool_result", "tool_use_id": tc.id, "tool_name": tc.function.name, "content": result}
                    if self._ask_user_paused:
                        return
                    # 整轮工具调用（可能含多个 tool_use）已全部写回 self.messages：
                    # 此时才节流落盘，避免在多工具轮中间写入“只有部分 tool_result”的非对称状态。
                    if self._persist_tool_context:
                        self._maybe_persist_tool_round()
                    continue

                # No more tool calls: emit the final reply.
                final_blocks: Any = content
                if reasoning:
                    if content:
                        final_blocks = [
                            {"type": "thinking", "thinking": reasoning},
                            {"type": "text", "text": content},
                        ]
                    else:
                        final_blocks = [{"type": "thinking", "thinking": reasoning}]
                self.messages.append(Message(
                    role="assistant",
                    content=final_blocks,
                    timestamp=int(datetime.now().timestamp() * 1000),
                ))
                # 记录模型最终回复（From 方向）
                _record_model_response(self.model_name, resp_full)
                # LLM 流量存档：最终回复响应（成功）
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

    @property
    def _is_siliconflow(self) -> bool:
        """当前是否走 SiliconFlow 平台。其 Chat Completions 的思考开关是
        `enable_thinking`（不是官方 DeepSeek 的 `thinking`），深度用 `thinking_budget`。"""
        return "siliconflow" in (self.base_url or "").lower()

    def _deepseek_thinking_extra_body(self) -> Dict[str, Any]:
        """Chat Completions 请求里通过 extra_body 传给模型的 thinking / 用户参数。

        DeepSeek 官方要求：OpenAI SDK 下 `thinking` 必须放 extra_body；
        `user_id`（Chat Completions 的终端用户标识）也走 extra_body。
        思考模式下 temperature / top_p 不生效（官方明确忽略、不报错），调用方据此不发送。

        SiliconFlow 平台差异：思考开关用 `enable_thinking`，深度用 `thinking_budget`
        （官方 DeepSeek 的 `thinking` 字段在 SiliconFlow 会被忽略，导致思考静默关闭），
        因此这里按平台分支处理，其余恒字段一致。
        """
        extra: Dict[str, Any] = {}
        if self._is_thinking:
            effort = self._deepseek_effort()
            if self._is_siliconflow:
                # SiliconFlow：enable_thinking 开启思考，thinking_budget 控制思维链长度；
                # reasoning_effort 该平台对 DeepSeek-V4-Flash 也有效（low/medium 会映射到 high）。
                extra["enable_thinking"] = True
                extra["thinking_budget"] = int(THINKING_BUDGETS.get(self.thinking_level, 8192))
                extra["reasoning_effort"] = effort
            else:
                # 官方 DeepSeek：create-chat-completion schema 把 reasoning_effort 放 thinking 里；
                # Thinking Mode 指南示例则在顶层传 reasoning_effort。两处都给同一值最稳。
                extra["thinking"] = {"type": "enabled", "reasoning_effort": effort}
                extra["reasoning_effort"] = effort
        else:
            if self._is_siliconflow:
                extra["enable_thinking"] = False
            else:
                extra["thinking"] = {"type": "disabled"}
        uid = self._session_user_id()
        if uid:
            extra["user_id"] = uid
        return extra
