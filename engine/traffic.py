"""engine.traffic — LLM 流量存档与上下文占用事件。
"""
from typing import Any, Dict, Optional

from .config import CJK_TOKENS_FACTOR
from traffic import TRAFFIC

class TrafficMixin:
    def _traffic_ctx(self, **overrides: Any) -> Dict[str, Any]:
        """合并当前会话流量上下文与临时覆盖，返回调用 TRAFFIC 时的公共属性。

        任一属性缺失时自动回退：model 用 self.model_name；
        provider 从 traffic_context 取；其余为空默认值。
        """
        ctx: Dict[str, Any] = dict(self.traffic_context)
        if "model" not in ctx or not ctx.get("model"):
            ctx["model"] = self.model_name
        ctx.update(overrides)
        return ctx
    def _traffic_request(self, payload: Any, message_type: str = "", stream: bool = False,
                         status_code: int = 0, retry_attempt: int = 0, error: str = "") -> None:
        """记录一条发往远端 LLM 的请求（入方向）。任何异常不影响主流程。"""
        try:
            ctx = self._traffic_ctx(
                message_type=message_type,
                stream_supported=1 if stream else 0,
                status_code=status_code,
                retry_attempt=retry_attempt,
                error=error,
            )
            TRAFFIC.log_request(payload=payload, **ctx)
        except Exception:
            pass
    def _traffic_response(self, text: str = "", input_tokens: int = 0,
                          cache_hit_tokens: int = 0, cache_miss_tokens: int = 0,
                          output_tokens: int = 0, reasoning_tokens: int = 0,
                          message_type: str = "", stream: bool = False, status_code: int = 200,
                          duration_ms: int = 0, error: str = "") -> None:
        """记录一条模型返回的响应（出方向）。任何异常不影响主流程。"""
        try:
            ctx = self._traffic_ctx(
                message_type=message_type,
                stream_supported=1 if stream else 0,
                status_code=status_code,
                duration_ms=duration_ms,
                error=error,
            )
            TRAFFIC.log_response(
                text=text,
                input_tokens=input_tokens,
                cache_hit_tokens=cache_hit_tokens,
                cache_miss_tokens=cache_miss_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                **ctx)
        except Exception:
            pass
    def _context_usage_event(self) -> Optional[Dict[str, Any]]:
        """发送前读取流量库中本会话最近一条 request 的字符数，折算为 context 事件（专供 UI 展示）。

        口径：取 llm_traffic 里 session_id 匹配、方向为 request 的最近一条记录的 chars
        （完整请求体的 JSON 字符数，含 system + messages + tools），按 1 字符 ≈ 0.6 token
        折算。与流量库同源，反映“当前上下文真实占用”，不做本地逐块拼接估算。

        本会话尚无 request 记录时返回 None（UI 保持 0）：首轮请求发出后才落库，
        turn 结束（server result 前补发）即可拿到本轮数值。
        """
        sid = (self.traffic_context or {}).get("session_id") or ""
        if not sid:
            return None
        limit = self._compact_limit_tokens
        try:
            chars = TRAFFIC.last_request_chars(sid)
            if chars is None or chars <= 0:
                return None
            est = int(chars * CJK_TOKENS_FACTOR)  # 1 字符 ≈ 0.6 token（与 /status 口径一致）
            # 埋点日志：核对与流量库口径
            try:
                print(
                    f"[CONTEXT-EST] session={sid} chars={chars} "
                    f"estimated_tokens={est} limit_tokens={limit}"
                )
            except Exception:
                pass
            return {
                "type": "context",
                "estimated_tokens": est,
                "limit_tokens": limit,
            }
        except Exception:
            return None
    def context_after_compact_event(self) -> Optional[Dict[str, Any]]:
        """压缩完成后基于压缩后的上下文本地估算 context 事件（专供 /compact 完成后补发）。

        压缩后流量库只有摘要请求的小 chars，不代表压缩后上下文大小，故直接用
        _estimate_context_tokens 对压缩后的 self.messages 估算（口径与自动压缩一致），
        保证 UI 圆环压缩后立刻回落。
        """
        try:
            messages = self._build_anthropic_messages()
            system_text = self._anthropic_system_text()
            tools = self._tool_definitions_anthropic()
            est = self._estimate_context_tokens(messages, tools=tools, system_text=system_text)
            limit = self._compact_limit_tokens
            return {
                "type": "context",
                "estimated_tokens": est,
                "limit_tokens": limit,
            }
        except Exception:
            return None
