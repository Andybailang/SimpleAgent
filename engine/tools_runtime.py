"""engine.tools_runtime — 工具执行、权限检查与工具定义。
"""
import os
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from path_util import strip_vermagic
from .config import (
    TOOL_RESULT_MAX_CHARS,
    WORK_MODE_TOOL_WHITELIST,
    WORK_MODE_ALLOWED_MCP_TOOLS,
)
from .util import Message, READ_ONLY_TOOLS, _maybe_attach_agent_note
from tools import tool_registry
from tools.base import Tool, ToolPermission
from tools.bash import is_readonly_command, READONLY_BLOCK_TOKENS
import mcp_manager

class ToolsRuntimeMixin:
    def _is_readonly_tool(self, tool_name: str) -> bool:
        """判断工具是否只读：注册表权限声明优先，READ_ONLY_TOOLS 集合兜底。"""
        tool = tool_registry.get_tool(tool_name)
        if tool and tool.permission_level == ToolPermission.READONLY:
            return True
        return tool_name in READ_ONLY_TOOLS
    def _find_tool(self, name: str) -> Optional[Tool]:
        """查找工具"""
        return tool_registry.get_tool(name)
    def _resolve_path(self, path: str, tool_name: str = "") -> str:
        """将相对/绝对路径解析到工作目录；非 full 权限下越界抛 ValueError。"""
        raw = os.path.expanduser(path or "")
        if not os.path.isabs(raw):
            raw = os.path.join(self.cwd, raw)
        resolved = strip_vermagic(os.path.realpath(raw))
        if self.permission_mode != "full":
            inside = resolved == self.cwd or resolved.startswith(self.cwd + os.sep)
            if not inside:
                cwd_norm = os.path.normcase(self.cwd)
                lexical = os.path.normcase(os.path.abspath(raw))
                inside_lexically = lexical == cwd_norm or lexical.startswith(cwd_norm + os.sep)
                if inside_lexically and self._is_readonly_tool(tool_name):
                    return resolved
                raise ValueError(f"路径 {path} 超出工作目录范围（{self.cwd}）")
        return resolved
    def _check_permission(self, tool_name: str, tool_input: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """只读权限下禁止会改变文件/执行写操作的工具；Bash 按命令内容放行只读命令。"""
        if self.permission_mode != "readonly":
            return None
        if self._is_readonly_tool(tool_name):
            return None
        if tool_name == "Bash":
            command = (tool_input or {}).get("command") or ""
            if is_readonly_command(command):
                return None
            hit = next((w for w in READONLY_BLOCK_TOKENS if re.search(
                r"(^|[\s|;&])" + re.escape(w) + r"($|[\s|;&])", command.lower())), None)
            hint = f"（检测到疑似写操作词：{hit}）" if hit else ""
            return (f"错误：当前会话为只读权限，命令可能修改文件或执行写操作{hint}，"
                    f"仅放行只读命令（如 git status/log/diff/show、ls、cat、rg 等）")
        return f"错误：当前会话为只读权限，禁止执行工具 {tool_name}"
    def _prepare_tool_args(self, tool: Tool, tool_input: Optional[Dict[str, Any]] = None) -> Any:
        """按工具 schema 过滤未知参数并检查必填参数，返回 (args, error)。
        模型偶尔会传 schema 之外的参数（如 Read 的 offset/length），过滤后可避免 TypeError。"""
        schema = tool.parameters or {}
        props = set(schema.get("properties", {}).keys())
        raw_args = dict(tool_input or {})
        # 别名归一化：主参数缺失但别名存在时映射过去（如 file_path 兼容 path）
        for main, alts in (tool.aliases or {}).items():
            if main not in raw_args:
                for alt in alts:
                    if alt in raw_args:
                        raw_args[main] = raw_args.pop(alt)
                        break
        args = {k: v for k, v in raw_args.items() if k in props}
        missing = [k for k in (schema.get("required") or []) if k not in args]
        if missing:
            return {}, f"错误：工具 {tool.name} 缺少必要参数：{', '.join(missing)}"
        return args, ""
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """执行工具调用"""
        tool_name = tool_call.get("tool_name")
        tool_input = tool_call.get("tool_input", {})

        if not tool_name:
            return "错误：缺少 tool_name 参数"

        blocked = self._check_permission(tool_name, tool_input)
        if blocked:
            return blocked

        # 查找工具
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            return f"错误：未知工具 '{tool_name}'"

        # 检查工具是否在当前模式下可用
        if not tool.is_available_in_mode(self.mode):
            return f"错误：工具 '{tool_name}' 在当前模式 '{self.mode}' 下不可用"

        # MCP 二进制/媒体读取拦截（OpenAI 模式失败；Anthropic 模式返回媒体占位）
        try:
            intercepted = mcp_manager.intercept_binary_read(
                getattr(self, "api_format", "openai"), tool_name, tool_input)
        except Exception:
            intercepted = None
        if intercepted is not None:
            if isinstance(intercepted, dict) and intercepted.get("__media__"):
                return f"[图片 × {len(intercepted['__media__'])}]"
            return str(intercepted)

        # 按 schema 过滤参数并检查必填项
        args, arg_err = self._prepare_tool_args(tool, tool_input)
        if arg_err:
            return arg_err
        # 执行工具
        try:
            result = tool_registry.execute_tool(self.tool_context, tool_name, **args)
            if isinstance(result, dict) and result.get("__media__"):
                return result
            return _maybe_attach_agent_note(tool_name, self._limit_tool_result(str(result)))
        except Exception as e:
            return f"工具执行失败: {str(e)}"
    def _persist_tool_round(self, assistant_blocks: List[Dict[str, Any]], tool_uses: List[Dict[str, Any]], thinking_text: str = "") -> None:
        """写回工具调用轮次到持久上下文 self.messages。

        内容：assistant 消息（含 tool_use 块）+ 每条 tool_use 对应的 user/tool_result
        消息。仅当 _persist_tool_context 为真（免费模型或编程模式）时写回；
        聊天模式收费模型不调用本方法（思考与工具调用仅在 turn 内参与，不写回，
        见 f4ccce5 契约）。
        """
        if not self._persist_tool_context:
            return
        try:
            if thinking_text:
                assistant_blocks = [
                    {"type": "thinking", "thinking": thinking_text},
                    *assistant_blocks,
                ]
            self.messages.append(Message(
                role="assistant",
                content=assistant_blocks,
                timestamp=int(datetime.now().timestamp() * 1000),
            ))
            for t in tool_uses:
                self.messages.append(Message(
                    role="user",
                    content=[{"type": "tool_result", "tool_use_id": t.get("id"), "content": t.get("_result", "")}],
                    timestamp=int(datetime.now().timestamp() * 1000),
                    tool_name=t.get("name"),
                    tool_input=t.get("input"),
                    tool_result=str(t.get("_result") or ""),
                ))
            # 本轮已完整写回 self.messages：立即节流落盘，让长任务中断时也能恢复已完成轮次上下文。
            self._maybe_persist_tool_round()
        except Exception:
            pass
    def _maybe_persist_tool_round(self) -> None:
        """每轮工具调用完整写回 self.messages 后调用（节流），把已提交的轮次立即写盘。

        仅在 server 注入过 _persist_hook 时生效；按 _persist_interval_ms 节流，
        避免工具链短时间内大量轮次触发频繁全量写盘。落盘失败（钩子内部异常）一律吞掉，
        不影响工具执行与流式输出。
        """
        hook = getattr(self, "_persist_hook", None)
        if not hook:
            return
        now = int(datetime.now().timestamp() * 1000)
        if now - self._last_persist_ms < self._persist_interval_ms:
            return
        self._last_persist_ms = now
        try:
            hook()
        except Exception:
            pass
    def _tool_definitions(self, only_read_file: bool = False) -> List[Dict[str, Any]]:
        """OpenAI-format tool definitions built from the active tools.

        实时读取注册表：MCP 等动态工具注册后无需重建 agent 即可生效；
        work 模式按 WORK_MODE_TOOL_WHITELIST 过滤（外置 MCP 一律不拼，省 token）。"""
        defs: List[Dict[str, Any]] = []
        for tool in self._active_tools():
            if only_read_file and tool.name != "Read":
                continue
            defs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "parameters": tool.parameters,
                    "description": tool.description,
                },
            })
        if defs:
            defs[-1]["function"]["description"] += f"\n你的工作目录是 {self.cwd}。"
        return defs
    def _tool_definitions_anthropic(self, only_read_file: bool = False) -> List[Dict[str, Any]]:
        """Anthropic-format tool definitions (name/description/input_schema)，按当前模式过滤。

        实时读取注册表；work 模式按 WORK_MODE_TOOL_WHITELIST 过滤。"""
        defs: List[Dict[str, Any]] = []
        for tool in self._active_tools():
            if only_read_file and tool.name != "Read":
                continue
            defs.append({
                "name": tool.name,
                "input_schema": tool.parameters,
                "description": tool.description,
            })
        if defs:
            defs[-1]["description"] += f"\n你的工作目录是 {self.cwd}。"
        return defs
    def _active_tools(self) -> List[Any]:
        """返回当前模式应拼接给模型的工具列表。

        work 模式：内置工具走 WORK_MODE_TOOL_WHITELIST，外置 MCP 走
        WORK_MODE_ALLOWED_MCP_TOOLS 精确名单（这里是 gitnexus 常用子集）；其余一律不拼，
        避免每次都把大量工具定义发给模型、增加输入 token；
        chat 模式：保持现状（可用的 chat 工具）。
        """
        tools: List[Any] = []
        for tool in tool_registry.get_all_tools():
            if not tool.is_available_in_mode(self.mode):
                continue
            if self.mode == "work" and not self._work_mode_allows(tool):
                continue
            tools.append(tool)
        return tools

    @staticmethod
    def _work_mode_allows(tool: Tool) -> bool:
        """work 模式是否放行该工具：内置白名单或显式声明的 MCP 工具名。"""
        if tool.name in WORK_MODE_TOOL_WHITELIST:
            return True
        return tool.name in WORK_MODE_ALLOWED_MCP_TOOLS
    def _limit_tool_result(self, text: str) -> str:
        """工具结果文本封顶：防止超大输出打爆上下文；截断时提示模型改用更精确的查询。

        图片等媒体结果（{"__media__": [...]}）按图片 token 计费，不在此限制内。
        """
        if len(text) <= TOOL_RESULT_MAX_CHARS:
            return text
        return (text[:TOOL_RESULT_MAX_CHARS] +
                "\n...[工具结果过长（原始 %d 字符），已截断。如需更多信息，请用更精确的查询参数"
                "（如更具体的路径、目录层级、搜索条件或读取范围）重试。]" % len(text))
    def _run_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name without touching self.messages history.

        返回值：普通字符串；或媒体结构 {"__media__": [...]}（Anthropic 图片块）。
        """
        blocked = self._check_permission(tool_name, args)
        if blocked:
            return blocked
        tool = tool_registry.get_tool(tool_name)
        if tool is None:
            return f"错误：未知工具 '{tool_name}'"
        if not tool.is_available_in_mode(self.mode):
            return f"错误：工具 '{tool_name}' 在当前模式 '{self.mode}' 下不可用"
        args, arg_err = self._prepare_tool_args(tool, args)
        if arg_err:
            return arg_err
        # MCP 二进制/媒体读取拦截：媒体原始数据绝不进入模型上下文，一律转文本占位
        try:
            intercepted = mcp_manager.intercept_binary_read(
                getattr(self, "api_format", "openai"), tool_name, args)
        except Exception:
            intercepted = None
        if intercepted is not None:
            if isinstance(intercepted, dict) and intercepted.get("__media__"):
                return f"[图片 × {len(intercepted['__media__'])}]"
            return str(intercepted)
        try:
            result = tool_registry.execute_tool(self.tool_context, tool_name, **args)
            if isinstance(result, dict) and result.get("__media__"):
                return f"[图片 × {len(result['__media__'])}]"
            return _maybe_attach_agent_note(tool_name, self._limit_tool_result(str(result)))
        except Exception as e:
            return f"工具执行失败: {str(e)}"

    # ---------------------------------------------------------------------------
    # AskUserQuestion 交互工具：暂停 -> 前端问题卡片 -> 用户回答 -> tool_result 续跑
    # ---------------------------------------------------------------------------
    def _is_ask_user_question(self, tool_name: str) -> bool:
        """判断是否是 AskUserQuestion 交互工具。"""
        return tool_name == "AskUserQuestion"

    @staticmethod
    def _normalize_ask_user_questions(tool_input: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把模型传入的不同参数形态归一为 questions 数组（前端 QuestionCard 消费的形状）。

        兼容两种形态：
        - 多问题：{"questions": [{question, header?, options?, multiSelect?}, ...]}
        - 单问题：{"question": str, "header"?: str, "options"?: [{label, description?}], "multiSelect"?: bool}
        """
        inp = tool_input or {}
        qs = inp.get("questions")
        if isinstance(qs, list):
            norm = []
            for q in qs:
                if not isinstance(q, dict):
                    continue
                if not q.get("question"):
                    continue
                norm.append({
                    "question": str(q.get("question")),
                    "header": str(q.get("header") or "") or None,
                    "options": q.get("options") or [],
                    "multiSelect": bool(q.get("multiSelect", q.get("multi_select", False))),
                })
            if norm:
                return norm
        single = inp.get("question") or inp.get("message")
        if single:
            return [{
                "question": str(single),
                "header": str(inp.get("header") or inp.get("title") or "") or None,
                "options": inp.get("options") or [],
                "multiSelect": bool(inp.get("multiSelect", inp.get("multi_select", False))),
            }]
        if inp:
            return [{
                "question": json.dumps(inp, ensure_ascii=False)[:500],
                "header": None,
                "options": [],
            }]
        return []

    @staticmethod
    def _answers_to_text(questions: List[Dict[str, Any]], answers: Optional[Dict[str, Any]]) -> str:
        """把用户回答转为喂给模型的 tool_result 文本。"""
        answers = answers or {}
        parts: List[str] = []
        for i, q in enumerate(questions):
            q_text = str(q.get("question") or "").strip()
            ans = answers.get(str(i), answers.get(i, ""))
            if ans is None:
                ans = ""
            ans_text = str(ans).strip()
            if q_text or ans_text:
                parts.append(f"{q_text or f'问题{i+1}'}: {ans_text or '（用户未回答）'}")
        if not parts:
            # 模型只回传了答案字符串等情况：直接拼接所有回答值
            vals = [str(v).strip() for v in answers.values() if v is not None and str(v).strip()]
            parts = vals or ["（用户未回答）"]
        return "\n".join(parts)

    def _handle_ask_user_question(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """拦截 AskUserQuestion：登记 pending 问题并把事件回给流式循环；
        非 AskUserQuestion 返回 None，走正常工具执行。"""
        if not self._is_ask_user_question(tool_name):
            return None
        inp = tool_input or {}
        questions = self._normalize_ask_user_questions(inp)
        # 统一把归一后的 questions 注入 input，前端 QuestionCard 只消费 input.questions。
        enriched_input = dict(inp)
        if questions:
            enriched_input["questions"] = questions
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        self._pending_question = {
            "request_id": request_id,
            "tool_use_id": tool_use_id,
            "input": enriched_input,
            "questions": questions,
        }
        self._pending_question_event = {
            "type": "tokenicode_permission_request",
            "tool_name": "AskUserQuestion",
            "request_id": request_id,
            "tool_use_id": tool_use_id,
            "input": enriched_input,
            "questions": questions,
        }
        self._ask_user_paused = True
        return self._pending_question_event

    def _ensure_ask_user_assistant_message(self, tool_use_id: str, tool_name: str, tool_input: Any) -> None:
        """确保 self.messages 里记录带 tool_use 的 assistant 消息，供回答后继续。

        若 self.messages 最后一条 assistant 已含该 tool_use（如 _persist_tool_context
        时的正常写回）则不重复追加；若最后一条 assistant 只有纯文本则把 tool_use 合并进去，
        保证回答后续跑时能正确还原 tool 上下文，且不产生重复的 assistant 消息。
        """
        tool_block = {"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input or {}}
        if self.messages and self.messages[-1].role == "assistant":
            last_content = self.messages[-1].content
            if isinstance(last_content, list):
                if any(
                    isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") == tool_use_id and b.get("name") == tool_name
                    for b in last_content
                ):
                    return
                self.messages[-1].content = last_content + [tool_block]
                return
            # 纯文本 assistant：合并为 [text?, tool_use]
            merged = []
            if last_content and str(last_content).strip():
                merged.append({"type": "text", "text": str(last_content)})
            merged.append(tool_block)
            self.messages[-1].content = merged
            return
        self.messages.append(Message(
            role="assistant",
            content=[tool_block],
            timestamp=int(datetime.now().timestamp() * 1000),
        ))
