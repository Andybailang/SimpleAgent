"""server_state — 会话内存态、桥接会话、持久化与 LLM 轮次编排。
"""
import os
import json
import uuid
import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from path_util import realpath_clean
from engine import SimpleAgent, Message
from server_config import (
    COST_ESTIMATE_ENABLED,
    _create_agent,
    _estimate_cost_usd,
    _is_local_model,
    _is_shared_temp_cwd,
    _read_config_env,
    _resolve_preset_model,
)
import mcp_manager
import skills
from local_handlers import handle_local_turn
from roles import get_role

# ==================== 会话管理 ====================

sessions: Dict[str, Dict[str, Any]] = {}


def create_session(session_id: str) -> Dict[str, Any]:
    """创建新会话"""
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:8]}"

    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "agent": None,
            "messages": [],
            "created_at": datetime.now().isoformat()
        }

    return sessions[session_id]


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """获取会话"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return sessions[session_id]


# ==================== 初始化 Agent ====================

agents: Dict[str, SimpleAgent] = {}


def get_or_create_agent(session_id: str, api_key: str, base_url: str, model_name: str) -> SimpleAgent:
    """获取或创建 Agent 实例"""
    if session_id not in agents:
        agents[session_id] = SimpleAgent(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )
    return agents[session_id]
# ==================== 启动服务 ====================

# ==================== bridge endpoints (browser UI -> FastAPI) ====================

# ---------------------------------------------------------------------------
# Streaming bridge sessions
#
# The browser UI (Tokenicore frontend) expects a persistent Claude CLI process
# that emits NDJSON events over a Tauri event channel. FastAPI cannot push
# events over HTTP, so we emulate the channel with an asyncio queue per
# session plus an SSE endpoint that drains it. Event shapes follow the Claude
# SDK stream format consumed by useStreamProcessor.ts.
# ---------------------------------------------------------------------------

# stdin_id (desk_xxx) -> bridge session state
bridge_sessions: Dict[str, Dict[str, Any]] = {}
# canonical CLI session id (cli_xxx) -> shared message list (for list/messages)
bridge_message_stores: Dict[str, List[Dict[str, Any]]] = {}
# canonical CLI session id -> stdin_id (for resume lookup)
bridge_session_by_cli: Dict[str, str] = {}
# canonical CLI session id -> agent (kept across resume spawns)
bridge_agents_by_cli: Dict[str, Any] = {}
# normalized project path -> tool permission mode (default/readonly/full)
project_permissions: Dict[str, str] = {}
# 磁盘持久化的会话：session_id -> 会话数据（对齐原版 tokenicode 的会话文件）
persisted_sessions: Dict[str, Dict[str, Any]] = {}


def _data_dir() -> str:
    """持久化数据目录（默认 ~/.bigcodex，可用 BIGCODEX_DATA_DIR 覆盖）。"""
    d = os.environ.get("BIGCODEX_DATA_DIR") or os.path.join(os.path.expanduser("~"), ".bigcodex")
    os.makedirs(os.path.join(d, "sessions"), exist_ok=True)
    return d


def _session_file(session_id: str) -> str:
    return os.path.join(_data_dir(), "sessions", f"{session_id}.json")


def _persist_bs(bs: Dict[str, Any]) -> None:
    """将会话（前端历史 + engine 上下文）写盘；失败不影响对话。"""
    try:
        agent = bs.get("agent")
        engine_messages = [m.to_dict() for m in agent.messages] if agent is not None else []
        data = {
            "id": bs["session_id"],
            "stdin_id": bs["stdin_id"],
            "cwd": bs.get("cwd"),
            "model": bs.get("model"),
            "provider_id": bs.get("provider_id"),
            "thinking_level": bs.get("thinking_level"),
            "tool_permission": bs.get("tool_permission"),
            "chat_mode": bool(bs.get("chat_mode")),
            "role": bs.get("role"),
            "created_at": bs.get("created_at"),
            "last_activity": bs.get("last_activity"),
            "messages": bs["messages"],
            "engine_messages": engine_messages,
            "stats": bs.get("stats") or {},
        }
        persisted_sessions[bs["session_id"]] = data
        path = _session_file(bs["session_id"])
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


def _delete_session_file(session_id: str) -> None:
    persisted_sessions.pop(session_id, None)
    try:
        path = _session_file(session_id)
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _load_persisted_state() -> None:
    """启动时从磁盘恢复项目权限与会话索引。"""
    global project_permissions
    perm_path = os.path.join(_data_dir(), "permissions.json")
    try:
        with open(perm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            project_permissions = {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    sessions_dir = os.path.join(_data_dir(), "sessions")
    try:
        for name in os.listdir(sessions_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(sessions_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("id"):
                    persisted_sessions[data["id"]] = data
            except Exception:
                continue
    except Exception:
        pass


def _save_permissions() -> None:
    try:
        path = os.path.join(_data_dir(), "permissions.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(project_permissions, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


# 启动时恢复磁盘状态（模块加载即执行）
_load_persisted_state()



def _project_key(path: str) -> str:
    """Normalize a project path for permission lookup (handles ~/ style keys)."""
    raw = (path or "").strip()
    if raw.startswith("~"):
        raw = os.path.join(os.path.expanduser("~"), raw[1:].lstrip("/\\"))
    return os.path.normcase(realpath_clean(raw))


def _now_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


ATTACH_MARKER = "@@BIGCODEX_ATTACH@@"


def _parse_attachments(text: str):
    """Split a user message into (clean_text, attachments).

    The frontend appends a JSON marker block listing attachment metadata
    (name / path / isImage). Returns (text, []) when no marker is present.
    """
    if ATTACH_MARKER not in text:
        return text, []
    clean, _, json_part = text.partition(ATTACH_MARKER)
    try:
        attachments = json.loads(json_part.strip())
        if not isinstance(attachments, list):
            return text, []
    except Exception:
        return text, []
    return clean.rstrip(), attachments


def _new_bridge_session(stdin_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    existing = bridge_sessions.get(stdin_id)
    if existing is not None:
        return existing

    resume_id = body.get("resume_session_id") or ""
    old_stdin = bridge_session_by_cli.get(resume_id) if resume_id else None
    old_bs = bridge_sessions.get(old_stdin) if old_stdin else None
    restore_data = None
    if old_bs is None and resume_id:
        restore_data = persisted_sessions.get(resume_id)
    old_agent = bridge_agents_by_cli.get(resume_id) if old_bs is not None else None
    # 前端 spawn 时总会带上当前选择的 model；resume 会话以 body.model 为准，
    # 避免切换模型后仍沿用旧模型（缓存监控也会因此串到旧 model 名下）。
    requested_model = body.get("model") or ""
    if old_bs is not None:
        messages = old_bs["messages"]
        agent = old_agent
        cli_id = resume_id
        bridge_message_stores.setdefault(cli_id, messages)
    elif restore_data is not None:
        # 磁盘恢复：内存中没有该会话，从持久化文件重建
        messages = restore_data.get("messages") or []
        bridge_message_stores[resume_id] = messages
        cli_id = resume_id
        agent = None
    else:
        messages = []
        cli_id = f"cli_{uuid.uuid4().hex[:12]}"
        agent = None
        bridge_message_stores[cli_id] = messages

    # 模型切换：不能复用旧模型的 agent，改为按新模型重建（_ensure_agent 懒创建），
    # 并把旧 agent 的引擎上下文通过 _restore 延续给新 agent。
    model_switched = bool(requested_model) and old_agent is not None and old_agent.model_name != requested_model
    if model_switched:
        agent = None

    old_perm = old_bs.get("tool_permission") if old_bs is not None else (restore_data.get("tool_permission") if restore_data is not None else None)

    restore_stats = (restore_data or {}).get("stats") or {}
    restore_cwd = (restore_data or {}).get("cwd")
    if restore_data is not None:
        restore_payload = restore_data
    elif model_switched and old_agent is not None:
        restore_payload = {"engine_messages": [m.to_dict() for m in old_agent.messages]}
    else:
        restore_payload = None
    bs_cwd = restore_cwd or body.get("cwd") or os.getcwd()
    # 会话角色：请求显式指定的 role 优先，其次沿用恢复/旧会话的 role；
    # 都没有时兼容旧 chat_mode（true -> 聊天），再按共享临时目录默认聊天判定。
    restored_role = str((restore_data or old_bs or {}).get("role") or "").strip()
    requested_role = str(body.get("role") or "").strip() or restored_role
    if requested_role and get_role(requested_role) is None:
        requested_role = ""
    if not requested_role:
        chat_mode = bool((restore_data or old_bs or {}).get("chat_mode")) or bool(body.get("chat_mode"))
        if not chat_mode and restore_data is None and old_bs is None and _is_shared_temp_cwd(bs_cwd):
            chat_mode = True
        requested_role = "chat" if chat_mode else "programming"
    chat_mode = requested_role == "chat"
    # 收费模型不允许进入非编程（chat）模式：编程模式才需要长链路工具上下文写回、可断点续跑；
    # 聊天模式用收费模型会白白烧 token。新会话与切换模型都从这里拦截（免费模型不受限）。
    if chat_mode:
        _pay_fm = _resolve_preset_model(
            requested_model or (restore_data or old_bs or {}).get("model") or "")
        if _pay_fm and str(_pay_fm.get("billing") or "paid").lower() == "paid":
            raise HTTPException(
                status_code=400,
                detail="收费模型只能在编程模式使用，请切换到编程模式或选择免费模型",
            )
    bs = {
        "stdin_id": stdin_id,
        "session_id": cli_id,
        "cwd": bs_cwd,
        "model": requested_model or (restore_data or old_bs or {}).get("model") or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
        "provider_id": body.get("provider_id") or (restore_data or {}).get("provider_id"),
        "thinking_level": (restore_data or {}).get("thinking_level") or body.get("thinking_level"),
        "permission_mode": body.get("permission_mode"),
        "tool_permission": body.get("tool_permission") or old_perm or project_permissions.get(_project_key(restore_cwd or body.get("cwd") or os.getcwd())) or "default",
        "chat_mode": chat_mode,
        "role": requested_role,
        "messages": messages,
        "events": deque(),
        "generating": False,
        "pending": [],
        "cancelled": False,
        "task": None,
        "agent": agent,
        "created_at": (restore_data or {}).get("created_at") or _now_ms(),
        "last_activity": _now_ms(),
        "stats": {
            "turns": int(restore_stats.get("turns") or 0),
            "input_tokens": int(restore_stats.get("input_tokens") or 0),
            "output_tokens": int(restore_stats.get("output_tokens") or 0),
            "duration_ms": int(restore_stats.get("duration_ms") or 0),
            "cost_usd": float(restore_stats.get("cost_usd") or 0.0),
        },
        "_restore": restore_payload,
    }
    bridge_sessions[stdin_id] = bs
    bridge_session_by_cli[cli_id] = stdin_id
    if agent is not None:
        bridge_agents_by_cli[cli_id] = agent
    return bs


def _emit_bridge_event(bs: Dict[str, Any], event: Dict[str, Any]) -> None:
    bs["events"].append(event)


def _release_mcp(agent: Any) -> None:
    """释放 Agent 持有的 MCP filesystem 实例引用（引用计数归零时停止对应实例）。"""
    try:
        mcp_manager.release(getattr(agent, "cwd", None) or os.getcwd())
    except Exception:
        pass


def _ensure_agent(bs: Dict[str, Any]) -> Any:
    """确保会话有可用的 agent：懒创建并恢复持久化上下文（跳过 system）。

    首次使用时创建 agent（含默认系统提示），然后把上次落盘的
    engine_messages 载入作为对话上下文；恢复只做一次。
    """
    agent = bs.get("agent")
    role_cfg = get_role(str(bs.get("role") or "")) if bs.get("role") else None
    role_prompt = (role_cfg or {}).get("prompt") or None
    # engine 把 None 归一存成 ""，比较前统一归一，避免空角色提示词被误判为“角色变更”而每次重建 agent
    if agent is not None and (getattr(agent, "role_prompt", None) or None) != role_prompt:
        # 角色变更（仅空会话发生）时重建 agent，确保新的 role_prompt 生效
        prev = bridge_agents_by_cli.get(bs["session_id"])
        if prev is not None:
            _release_mcp(prev)
        bridge_agents_by_cli.pop(bs["session_id"], None)
        bs["agent"] = None
        agent = None
    if agent is None:
        agent = _create_agent(
            bs["model"],
            cwd=bs.get("cwd"),
            thinking_level=bs.get("thinking_level") or "off",
            plain_chat=bool(bs.get("chat_mode")),
            role_prompt=role_prompt,
        )
        mcp_manager.acquire(getattr(agent, "cwd", None))
        bs["agent"] = agent
        bridge_agents_by_cli[bs["session_id"]] = agent
    else:
        agent.thinking_level = bs.get("thinking_level") or agent.thinking_level
    agent.set_permission_mode(bs.get("tool_permission") or "default")
    # 注入工具轮次落盘钩子：引擎在每轮工具调用完整写回 self.messages 后（节流）调用它，
    # 把当前会话（含已完成的工具/思考上下文）立即写盘。这样长任务（大量工具轮/长时间思考）
    # 在进程被意外终止（崩溃、外部安全策略强杀等）时，磁盘始终保留到最近一轮完成后的现场，
    # 重启后可在这个会话里延续上下文继续工作，而不是整轮上下文丢失。失败不影响对话。
    agent._persist_hook = lambda: _persist_bs(bs)
    restore = bs.get("_restore")
    if restore:
        for m in (restore.get("engine_messages") or []):
            if not isinstance(m, dict) or m.get("role") == "system":
                continue
            try:
                agent.messages.append(Message.from_dict(m))
            except Exception:
                continue
        bs["_restore"] = None
    return agent


def _emit_turn_complete(bs: Dict[str, Any], content_text: str, usage_in: int = 0, usage_out: int = 0, files_written: bool = False, references: Optional[list] = None) -> None:
    """在不调用 LLM 的情况下发出一个完整轮次的事件序列（斜杠命令/本地回复）。"""
    _emit_bridge_event(bs, {
        "type": "stream_event",
        "event": {"type": "message_start", "message": {"usage": {"input_tokens": usage_in}}},
    })
    _emit_bridge_event(bs, {
        "type": "stream_event",
        "event": {"type": "content_block_start", "index": 0,
                  "content_block": {"type": "text", "text": ""}},
    })
    if content_text:
        _emit_bridge_event(bs, {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "index": 0,
                      "delta": {"type": "text_delta", "text": content_text}},
        })
    assistant_event = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": content_text}]},
        "session_id": bs["session_id"],
    }
    if references:
        assistant_event["references"] = references
    _emit_bridge_event(bs, assistant_event)
    _emit_bridge_event(bs, {
        "type": "stream_event",
        "event": {"type": "message_delta", "usage": {"output_tokens": usage_out}},
    })
    stats = bs.get("stats") or {}
    _emit_bridge_event(bs, {
        "type": "result",
        "subtype": "success",
        "result": content_text,
        "files_written": files_written,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
        "num_turns": stats.get("turns", 0),
        "duration_ms": 0,
        "total_cost_usd": stats.get("cost_usd") if COST_ESTIMATE_ENABLED else None,
        "total_input_tokens": stats.get("input_tokens", 0),
        "total_output_tokens": stats.get("output_tokens", 0),
        "total_duration_ms": stats.get("duration_ms", 0),
        "session_id": bs["session_id"],
    })


async def _handle_local_turn(bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> None:
    """LocalLLM 模式：本地处理器执行，伪造与正常 LLM 完全一致的事件流。"""
    try:
        # 清空上一轮可能残留的引用元数据，由本轮的处理器按需写入
        bs.pop("local_references", None)
        result = await run_in_threadpool(handle_local_turn, bs, text, attachments)
    except Exception as e:
        result = f"本地处理失败：{e}"
    stats = bs.setdefault("stats", {"turns": 0, "input_tokens": 0, "output_tokens": 0, "duration_ms": 0, "cost_usd": 0.0})
    stats["turns"] = stats.get("turns", 0) + 1
    message_record = {"role": "assistant", "content": result, "timestamp": _now_ms()}
    local_refs = bs.get("local_references")
    if local_refs:
        message_record["references"] = local_refs
    bs["messages"].append(message_record)
    _persist_bs(bs)
    _emit_turn_complete(bs, result, files_written=True, references=bs.get("local_references"))


async def _generate_turn(bs: Dict[str, Any], text: str, attachments: Optional[List[Dict[str, Any]]] = None, continue_only: bool = False) -> None:
    """Run one LLM turn and emit Claude-SDK-shaped NDJSON events."""
    try:
        # 斜杠命令：目前仅 /compact 有实做（压缩上下文），其余暂接空响应，后续逐步实现
        if isinstance(text, str) and text.strip().startswith("/"):
            cmd_line = text.strip()
            cmd = cmd_line.split()[0].lower()
            if cmd == "/compact":
                try:
                    agent = bs.get("agent")
                    restore = bs.get("_restore") or {}
                    has_context = bool(
                        (agent is not None and any(m.role != "system" for m in agent.messages))
                        or any(
                            isinstance(m, dict) and m.get("role") != "system" and str(m.get("content") or "").strip()
                            for m in (restore.get("engine_messages") or [])
                        )
                        or any(str(m.get("content") or "").strip() for m in (bs.get("messages") or []))
                    )
                    if not has_context:
                        feedback = "当前会话还没有对话内容，无需压缩"
                    else:
                        agent = _ensure_agent(bs)
                        feedback = agent.compact_context()
                        _persist_bs(bs)
                        # 压缩完成后补发 context 事件（基于压缩后上下文本地估算），
                        # 让 UI 圆环立即回落，而不是停留在压缩前的红色。
                        try:
                            ctx = agent.context_after_compact_event()
                            if ctx:
                                _emit_bridge_event(bs, {**ctx, "session_id": bs["session_id"]})
                        except Exception:
                            pass
                except Exception as e:
                    feedback = f"压缩失败：{e}"
                _emit_turn_complete(bs, feedback)
            else:
                # 技能调用：/<技能名> [参数] —— 命中后读取 SKILL.md 注入本轮，走正常生成
                skill_hit = None
                try:
                    for sk in skills.list_skills(bs.get("cwd")):
                        if cmd == "/" + sk["name"].lower():
                            skill_hit = sk
                            break
                except Exception:
                    skill_hit = None
                if skill_hit is None:
                    bs["messages"].append({"role": "assistant", "content": "", "timestamp": _now_ms()})
                    _persist_bs(bs)
                    _emit_turn_complete(bs, "")
                    return
                try:
                    skill_content = skills.read_skill(skill_hit["path"], bs.get("cwd"))
                except Exception as e:
                    _emit_turn_complete(bs, f"技能读取失败：{e}")
                    return
                if len(skill_content) > SKILL_CONTENT_MAX_CHARS:
                    skill_content = skill_content[:SKILL_CONTENT_MAX_CHARS] + "\n...[技能内容过长，已截断]"
                rest = cmd_line[len(cmd):].strip()
                bs["messages"].append({"role": "user", "content": cmd_line, "timestamp": _now_ms()})
                _persist_bs(bs)
                text = (
                    "用户要求使用以下技能来指导完成后续任务，请严格遵循技能内容执行：\n\n"
                    f"【技能：{skill_hit['name']}】\n"
                    f"{skill_content}\n\n"
                    f"【用户请求】\n{rest}"
                )
            # 非技能斜杠命令（/compact /stats /chat /未知）在此返回；
            # 技能调用已改写 text（不再等于 cmd_line），继续走正常生成流程
            if text == cmd_line:
                return

        # LocalLLM 模式：不走远端 LLM，由本地处理器匹配并伪造流式回复
        if _is_local_model(bs):
            await _handle_local_turn(bs, text, attachments)
            return

        agent = _ensure_agent(bs)
        # 新的生成轮开始：清掉上一轮遗留/超时的待答问题，避免陈旧 pending_question 误续跑。
        bs["pending_question"] = None
        # 注入 LLM 流量存档上下文（请求/响应属性共同字段）
        try:
            fm = _resolve_preset_model(bs.get("model") or "")
            agent.traffic_context = {
                "session_id": bs.get("session_id") or "",
                "stdin_id": bs.get("stdin_id") or "",
                "cwd": bs.get("cwd") or "",
                "role": bs.get("role") or "",
                "thinking_level": bs.get("thinking_level") or "",
                "provider": (fm or {}).get("id", ""),
            }
            # 注入自动压缩阈值（发送前本地估算对比用；来源 config.env）
            _auto_cfg = _read_config_env()
            try:
                agent.context_max_tokens = int(_auto_cfg.get("CONTEXT_MAX_TOKENS") or 1000000)
            except (TypeError, ValueError):
                agent.context_max_tokens = 1000000
            try:
                agent.compact_threshold = float(_auto_cfg.get("COMPACT_THRESHOLD") or 0.9)
            except (TypeError, ValueError):
                agent.compact_threshold = 0.9
        except Exception:
            pass
        turn_start = _now_ms()

        _emit_bridge_event(bs, {
            "type": "stream_event",
            "event": {"type": "message_start", "message": {"usage": {"input_tokens": 0}}},
        })
        _emit_bridge_event(bs, {
            "type": "stream_event",
            "event": {"type": "content_block_start", "index": 0,
                      "content_block": {"type": "text", "text": ""}},
        })

        full: List[str] = []
        usage_in = 0
        usage_out = 0
        peak_input_tokens = 0
        paused_for_question = False

        if getattr(agent, "api_format", "openai") == "anthropic":
            block_index = 0
            async for evt in agent.chat_stream_async(text, use_tools=True, attachments=attachments, continue_only=continue_only):
                if bs["cancelled"]:
                    break
                t = evt.get("type") if isinstance(evt, dict) else None
                if t == "text":
                    chunk = evt.get("text") or ""
                    if chunk:
                        full.append(chunk)
                        _emit_bridge_event(bs, {
                            "type": "stream_event",
                            "event": {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "text_delta", "text": chunk}},
                        })
                elif t == "thinking":
                    th = evt.get("text") or ""
                    if th:
                        _emit_bridge_event(bs, {
                            "type": "stream_event",
                            "event": {"type": "content_block_delta", "index": 1,
                                      "delta": {"type": "thinking_delta", "thinking": th}},
                        })
                elif t == "tool_use":
                    block_index += 1
                    tool_block = {
                        "type": "tool_use",
                        "id": evt.get("id"),
                        "name": evt.get("name"),
                        "input": evt.get("input") or {},
                    }
                    _emit_bridge_event(bs, {
                        "type": "stream_event",
                        "event": {"type": "content_block_start", "index": block_index,
                                  "content_block": tool_block},
                    })
                    _emit_bridge_event(bs, {
                        "type": "assistant",
                        "message": {"content": [tool_block]},
                        "session_id": bs["session_id"],
                    })
                elif t == "tool_result":
                    _emit_bridge_event(bs, {
                        "type": "tool_result",
                        "tool_use_id": evt.get("tool_use_id"),
                        "tool_name": evt.get("tool_name"),
                        "content": evt.get("content") or "",
                        "session_id": bs["session_id"],
                    })
                elif t == "usage":
                    usage_in += evt.get("input_tokens") or 0
                    usage_out += evt.get("output_tokens") or 0
                    peak_input_tokens = max(peak_input_tokens, evt.get("input_tokens") or 0)
                elif t == "context":
                    # 发送前本地估算的上下文占用（专供 UI 实时显示，不参与计费/统计）
                    _emit_bridge_event(bs, {**evt, "session_id": bs["session_id"]})
                elif t == "system":
                    _emit_bridge_event(bs, {**evt, "session_id": bs["session_id"]})
                elif t == "tokenicode_permission_request" and evt.get("tool_name") == "AskUserQuestion":
                    bs["pending_question"] = {
                        "request_id": evt.get("request_id"),
                        "tool_use_id": evt.get("tool_use_id"),
                        "input": evt.get("input") or {},
                        "questions": evt.get("questions") or [],
                    }
                    _emit_bridge_event(bs, evt)
                    paused_for_question = True
                    break
                elif t == "error":
                    raise RuntimeError(evt.get("text") or "Agent error")
        else:
            block_index = 0
            async for chunk in agent.chat_stream_async(text, use_tools=True, attachments=attachments, continue_only=continue_only):
                if bs["cancelled"]:
                    break
                if isinstance(chunk, dict):
                    t = chunk.get("type")
                    if t == "usage":
                        usage_in += chunk.get("input_tokens") or 0
                        usage_out += chunk.get("output_tokens") or 0
                        peak_input_tokens = max(peak_input_tokens, chunk.get("input_tokens") or 0)
                    elif t == "context":
                        # 发送前本地估算的上下文占用（专供 UI 实时显示，不参与计费/统计）
                        _emit_bridge_event(bs, {**chunk, "session_id": bs["session_id"]})
                    elif t == "system":
                        _emit_bridge_event(bs, {**chunk, "session_id": bs["session_id"]})
                    elif t == "thinking":
                        th = chunk.get("text") or ""
                        if th:
                            _emit_bridge_event(bs, {
                                "type": "stream_event",
                                "event": {"type": "content_block_delta", "index": 1,
                                          "delta": {"type": "thinking_delta", "thinking": th}},
                            })
                    elif t == "tool_use":
                        block_index += 1
                        tool_block = {
                            "type": "tool_use",
                            "id": chunk.get("id"),
                            "name": chunk.get("name"),
                            "input": chunk.get("input") or {},
                        }
                        _emit_bridge_event(bs, {
                            "type": "stream_event",
                            "event": {"type": "content_block_start", "index": block_index,
                                      "content_block": tool_block},
                        })
                        _emit_bridge_event(bs, {
                            "type": "assistant",
                            "message": {"content": [tool_block]},
                            "session_id": bs["session_id"],
                        })
                    elif t == "tool_result":
                        _emit_bridge_event(bs, {
                            "type": "tool_result",
                            "tool_use_id": chunk.get("tool_use_id"),
                            "tool_name": chunk.get("tool_name"),
                            "content": chunk.get("content") or "",
                            "session_id": bs["session_id"],
                        })
                    elif t == "tokenicode_permission_request" and chunk.get("tool_name") == "AskUserQuestion":
                        bs["pending_question"] = {
                            "request_id": chunk.get("request_id"),
                            "tool_use_id": chunk.get("tool_use_id"),
                            "input": chunk.get("input") or {},
                            "questions": chunk.get("questions") or [],
                        }
                        _emit_bridge_event(bs, chunk)
                        paused_for_question = True
                        break
                    elif t == "error":
                        raise RuntimeError(chunk.get("text") or "Agent error")
                    continue
                if chunk:
                    full.append(chunk)
                    _emit_bridge_event(bs, {
                        "type": "stream_event",
                        "event": {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": chunk}},
                    })

        if paused_for_question:
            # 已向用户提问：持久化一个 assistant(tool_use AskUserQuestion) 消息，
            # 供会话历史/断点续跑还原问题卡片；不落最终 result。
            pending_q = bs.get("pending_question") or {}
            question_use = {
                "type": "tool_use",
                "id": pending_q.get("tool_use_id") or "",
                "name": "AskUserQuestion",
                "input": pending_q.get("input") or {},
            }
            bs["messages"].append({
                "role": "assistant",
                "content": [question_use],
                "timestamp": _now_ms(),
            })
            _persist_bs(bs)
            return

        content_text = "".join(full)
        bs["messages"].append({"role": "assistant", "content": content_text, "timestamp": _now_ms()})
        stats = bs.setdefault("stats", {"turns": 0, "input_tokens": 0, "output_tokens": 0, "duration_ms": 0, "cost_usd": 0.0})
        turn_ms = max(0, _now_ms() - turn_start)
        stats["turns"] = stats.get("turns", 0) + 1
        stats["input_tokens"] = stats.get("input_tokens", 0) + usage_in
        stats["output_tokens"] = stats.get("output_tokens", 0) + usage_out
        stats["duration_ms"] = stats.get("duration_ms", 0) + turn_ms
        turn_cost = _estimate_cost_usd(usage_in, usage_out)
        if turn_cost is not None:
            stats["cost_usd"] = stats.get("cost_usd", 0.0) + turn_cost
        _persist_bs(bs)
        _emit_bridge_event(bs, {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": content_text}]},
            "session_id": bs["session_id"],
        })
        _emit_bridge_event(bs, {
            "type": "stream_event",
            "event": {"type": "message_delta", "usage": {"output_tokens": usage_out}},
        })
        # 本轮已发出请求并落库，补发一次 context 事件，确保单轮会话结束
        # 后圆环也有数值（不再停留在 0）。
        try:
            ctx = agent._context_usage_event()
            if ctx:
                _emit_bridge_event(bs, {**ctx, "session_id": bs["session_id"]})
        except Exception:
            pass
        _emit_bridge_event(bs, {
            "type": "result",
            "subtype": "success",
            "result": content_text,
            "usage": {"input_tokens": usage_in, "output_tokens": usage_out, "peak_input_tokens": peak_input_tokens},
            "num_turns": stats.get("turns", 0),
            "duration_ms": turn_ms,
            "total_cost_usd": stats.get("cost_usd") if COST_ESTIMATE_ENABLED else None,
            "total_input_tokens": stats.get("input_tokens", 0),
            "total_output_tokens": stats.get("output_tokens", 0),
            "total_duration_ms": stats.get("duration_ms", 0),
            "session_id": bs["session_id"],
        })
    except asyncio.CancelledError:
        # 用户主动停止（signal 端点同时置 cancelled 并 cancel task）：
        # 流式 await 直接收到 CancelledError，break 后的正常落盘代码不会执行。
        # 这里与正常 turn 结束一样兜底落盘——把本轮已产出的部分正文加入前端历史，
        # 并连同 engine 上下文（已完成的工具轮/思考块）一起写盘，保证停止后
        # 磁盘会话文件可还原现场，程序重启后能在这个会话里延续上下文继续工作。
        try:
            partial_text = "".join(full) if "full" in locals() else ""
            if partial_text:
                bs["messages"].append({"role": "assistant", "content": partial_text, "timestamp": _now_ms()})
                stats = bs.setdefault("stats", {"turns": 0, "input_tokens": 0, "output_tokens": 0, "duration_ms": 0, "cost_usd": 0.0})
                turn_ms = max(0, _now_ms() - turn_start) if "turn_start" in locals() else 0
                stats["turns"] = stats.get("turns", 0) + 1
                stats["duration_ms"] = stats.get("duration_ms", 0) + turn_ms
                if "usage_in" in locals():
                    stats["input_tokens"] = stats.get("input_tokens", 0) + usage_in
                if "usage_out" in locals():
                    stats["output_tokens"] = stats.get("output_tokens", 0) + usage_out
            _persist_bs(bs)
        except Exception:
            pass
        raise
    except Exception as e:
        error_text = f"Error: {str(e)}"
        _emit_bridge_event(bs, {
            "type": "system", "subtype": "error", "message": error_text, "error": error_text,
        })
        err_stats = bs.get("stats") or {}
        _emit_bridge_event(bs, {
            "type": "result", "subtype": "error", "result": error_text, "error": error_text,
            "num_turns": err_stats.get("turns", 0),
            "duration_ms": 0,
            "total_cost_usd": err_stats.get("cost_usd") if COST_ESTIMATE_ENABLED else None,
            "total_input_tokens": err_stats.get("input_tokens", 0),
            "total_output_tokens": err_stats.get("output_tokens", 0),
            "total_duration_ms": err_stats.get("duration_ms", 0),
        })


async def _pump_generation(bs: Dict[str, Any]) -> None:
    """Drain queued user texts one turn at a time."""
    if bs["generating"]:
        return
    bs["generating"] = True
    try:
        while bs["pending"] and not bs["cancelled"]:
            item = bs["pending"].pop(0)
            if isinstance(item, dict):
                text = item.get("text", "")
                attachments = item.get("attachments") or []
                continue_only = bool(item.get("continue_only", False))
            else:
                text = item
                attachments = []
                continue_only = False
            await _generate_turn(bs, text, attachments, continue_only=continue_only)
    finally:
        bs["generating"] = False


def _schedule_pump(bs: Dict[str, Any]) -> None:
    if bs["generating"]:
        return
    try:
        bs["task"] = asyncio.create_task(_pump_generation(bs))
    except RuntimeError:
        pass

# 斜杠技能调用时单次注入的 SKILL.md 最大字符数（防止超大技能正文撑爆上下文）
SKILL_CONTENT_MAX_CHARS = 20000

_BUILTIN_COMMANDS: List[Dict[str, Any]] = [
    {"name": "/ask", "description": "Ask a question without making changes", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
    {"name": "/bypass", "description": "Switch to bypass mode (skip all permission prompts)", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
    {"name": "/clear", "description": "Clear conversation history", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
    {"name": "/code", "description": "Switch to code mode (default)", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
    {"name": "/compact", "description": "Compact conversation to reduce context", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "session"},
    {"name": "/export", "description": "Export conversation to markdown", "source": "builtin", "category": "builtin", "has_args": True, "immediate": True, "execution": "ui"},
    {"name": "/help", "description": "Show available commands", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
    {"name": "/plan", "description": "Enter plan mode for complex tasks", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
    {"name": "/print", "description": "Print the last N assistant replies (-md saves as markdown)", "source": "builtin", "category": "builtin", "has_args": True, "immediate": True, "execution": "ui"},
    {"name": "/speech", "description": "Generate speech (TTS) from text or the last assistant reply (-v voice)", "source": "builtin", "category": "builtin", "has_args": True, "immediate": True, "execution": "ui"},
    {"name": "/rewind", "description": "Rewind conversation to a previous turn", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
    {"name": "/status", "description": "Query this session's token usage from the traffic DB", "source": "builtin", "category": "builtin", "has_args": False, "immediate": True, "execution": "ui"},
]
