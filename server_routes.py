"""server_routes — FastAPI 端点（按领域分组）。

所有 HTTP 端点统一登记在 APIRouter 上，由 server.py
装配到应用。依赖的会话状态/推送与模型配置分别来自 server_state / server_config。
"""
from fastapi import APIRouter, Body, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import json
import base64
import tempfile
import mimetypes
import os
import uuid
import asyncio
import httpx
from datetime import datetime
from pathlib import Path as FsPath

from server_config import (
    PRESET_MODELS,
    _create_agent,
    _read_config_env,
    _repo_root,
    _resolve_preset_model,
    _shared_temp_dir,
)
from server_state import (
    sessions,
    agents,
    bridge_sessions,
    bridge_message_stores,
    bridge_session_by_cli,
    bridge_agents_by_cli,
    project_permissions,
    persisted_sessions,
    create_session,
    get_or_create_agent,
    _persist_bs,
    _delete_session_file,
    _parse_attachments,
    _new_bridge_session,
    _emit_bridge_event,
    _schedule_pump,
    _ensure_agent,
    _project_key,
    _save_permissions,
    _release_mcp,
    _session_file,
    _now_ms,
    _BUILTIN_COMMANDS,
)
from engine import Message
from cache_monitor import CACHE_MONITOR
from traffic import TRAFFIC
import mcp_manager
import skills
from roles import get_roles, get_role, create_role, update_role, delete_role
from image_config import IMAGE_MODELS
from video_config import VIDEO_MODELS
from speech_config import DEFAULT_VOICE, get_speech_voices, get_active_voice_name
import multimodal_config
import user_models

router = APIRouter()

class MessageRequest(BaseModel):
    """单条消息请求"""
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    timestamp: Optional[int] = None


class ChatRequest(BaseModel):
    """聊天请求"""
    messages: List[MessageRequest]
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    messages: List[MessageRequest]
    new_message: MessageRequest


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    agent_status: str
    version: str


class SetProjectPermissionRequest(BaseModel):
    """项目工具权限设置请求"""
    project: str
    mode: str


class SessionTitleRequest(BaseModel):
    """智能会话命名请求"""
    user_message: str = ""
    assistant_message: str = ""
    provider_id: Optional[str] = None


def _clean_session_title(text: str) -> str:
    """清洗 LLM 生成的会话标题：去空白/首尾引号，截断到 15 字以内（英文字符 2 个折算 1 字）。"""
    title = (text or "").strip().replace("\r", " ").replace("\n", " ").strip()
    for prefix in ("标题：", "标题:", "Title：", "Title:", "标题为", "标题是"):
        if title.startswith(prefix):
            title = title[len(prefix):].strip()
            break
    for quote in ('"', "'", '“', '”', '‘', '’', '「', '」', '『', '』'):
        title = title.strip(quote)
    # 截断规则：ASCII 字符（英文字母/数字等）每 2 个折算 1 字，非 ASCII 字符（中文等）1 个折算 1 字，
    # 累计折算宽度不超过 15 字
    limit = 15.0
    width = 0.0
    result = []
    for ch in title:
        w = 0.5 if ch.isascii() else 1.0
        if width + w > limit:
            break
        width += w
        result.append(ch)
    return "".join(result)
@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    agent_status = "initialized" if len(agents) > 0 else "not_initialized"

    return HealthResponse(
        status="healthy",
        agent_status=agent_status,
        version="1.0.0"
    )


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送消息端点"""
    try:
        # 获取或创建会话
        session_id = request.session_id or uuid.uuid4().hex[:8]
        session = create_session(session_id)

        # 初始化 Agent（如果还没有）
        if session["agent"] is None:
            from env_util import load_env
            load_env()

            agent = get_or_create_agent(
                session_id,
                os.getenv("OPENAI_API_KEY", ""),
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
            )
            session["agent"] = agent

        # 调用 Agent
        agent = session["agent"]
        last_message_content = request.messages[-1].content
        response = agent.chat(last_message_content)

        # 添加响应消息
        response_message = MessageRequest(
            role="assistant",
            content=response,
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        session["messages"].append(response_message)

        return ChatResponse(
            session_id=session_id,
            messages=session["messages"],
            new_message=response_message
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@router.get("/api/conversations")
async def get_conversations():
    """获取会话列表"""
    return {
        "conversations": list(sessions.keys()),
        "total": len(sessions)
    }


@router.post("/api/conversations")
async def create_conversation():
    """创建新会话"""
    session_id = uuid.uuid4().hex[:8]
    session = create_session(session_id)

    return {
        "session_id": session_id,
        "status": "created"
    }


@router.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """删除会话"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # 删除会话
    del sessions[session_id]

    # 删除 Agent
    if session_id in agents:
        del agents[session_id]

    return {
        "status": "deleted",
        "session_id": session_id
    }


# ==================== Mock 端点 ====================

@router.post("/api/mock/not_implemented")
async def mock_not_implemented():
    """Mock 端点 - 用于暂不支持的功能"""
    return {
        "status": "not_implemented",
        "message": "This feature is not yet implemented"
    }


@router.get("/api/mock/sse-test")
async def sse_test():
    """SSE 测试端点"""
    async def event_generator():
        for i in range(10):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """聊天流式响应端点"""
    try:
        # 获取或创建会话
        session_id = request.session_id or uuid.uuid4().hex[:8]
        session = create_session(session_id)

        # 初始化 Agent（如果还没有）
        if session["agent"] is None:
            from env_util import load_env
            load_env()

            agent = get_or_create_agent(
                session_id,
                os.getenv("OPENAI_API_KEY", ""),
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
            )
            session["agent"] = agent

        # 定义流式生成器
        async def event_generator():
            """生成 SSE 事件"""
            try:
                agent = session["agent"]
                last_message_content = request.messages[-1].content

                # 调用流式聊天
                async for chunk in agent.chat_stream(last_message_content):
                    # 发送 SSE 事件
                    yield f"data: {json.dumps({'content': chunk, 'type': 'content'})}\n\n"

                # 发送完成事件
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                # 发送错误事件
                error_data = {
                    "type": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 禁用 Nginx 缓存
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stream chat failed: {str(e)}"
        )
@router.get("/api/commands")
async def api_commands(cwd: Optional[str] = None):
    """内置斜杠命令 + 可用技能列表（对齐原版 list_all_commands）。

    技能以 /技能名 形式并入命令框：category=skill、path 指向 SKILL.md，
    调用时由 _generate_turn 读取技能内容注入本轮。
    """
    commands = list(_BUILTIN_COMMANDS)
    try:
        for sk in skills.list_skills(cwd or None):
            commands.append({
                "name": "/" + sk["name"],
                "description": sk.get("description") or "",
                "source": sk.get("scope") or "global",
                "category": "skill",
                "has_args": True,
                "path": sk.get("path"),
                "immediate": False,
            })
    except Exception:
        pass
    return commands


@router.get("/api/config")
async def api_config():
    """返回前端可调用的应用配置（当前为上下文窗口与自动压缩阈值）。"""
    cfg = _read_config_env()
    try:
        max_tokens = int(cfg.get("CONTEXT_MAX_TOKENS") or 1000000)
    except (TypeError, ValueError):
        max_tokens = 1000000
    try:
        threshold = float(cfg.get("COMPACT_THRESHOLD") or 0.9)
    except (TypeError, ValueError):
        threshold = 0.9
    return {
        "context_max_tokens": max_tokens,
        "compact_threshold": threshold,
    }


@router.get("/api/cache_monitor/stats")
async def api_cache_monitor_stats():
    """返回各 LLM 的 Prompt 缓存命中估算统计（进程内存内，自服务启动以来）。"""
    return {"models": CACHE_MONITOR.stats()}


@router.get("/api/cache_monitor/records")
async def api_cache_monitor_records(model: str = "", page: int = 1, page_size: int = 20):
    """分页返回指定 LLM 当前内存中的 Prompt 原文记录（新记录在后）。"""
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    if not model:
        return {"model": "", "total": 0, "page": page, "page_size": page_size, "records": []}
    total, items = CACHE_MONITOR.records(model, page, page_size)
    return {"model": model, "total": total, "page": page, "page_size": page_size, "records": items}


@router.get("/api/cache_monitor/responses")
async def api_cache_monitor_responses(model: str = "", page: int = 1, page_size: int = 20):
    """分页返回指定 LLM 的响应记录（From 方向，新记录在后）。"""
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    if not model:
        return {"model": "", "total": 0, "page": page, "page_size": page_size, "responses": []}
    total, items = CACHE_MONITOR.responses(model, page, page_size)
    return {"model": model, "total": total, "page": page, "page_size": page_size, "responses": items}


@router.delete("/api/cache_monitor/records")
async def api_cache_monitor_clear(model: str = ""):
    """清除指定 LLM（或全部）的缓存监控记录与统计；model 为空表示全部清除。"""
    if model:
        cleared = CACHE_MONITOR.clear(model)
        return {"cleared": cleared, "model": model}
    cleared = CACHE_MONITOR.clear(None)
    disk_cleared = CACHE_MONITOR.clear_snapshot_file()
    return {"cleared": cleared, "disk_cleared": disk_cleared, "model": ""}


@router.post("/api/cache_monitor/save")
async def api_cache_monitor_save():
    """把当前缓存监控数据（记录与统计）保存到临时快照文件，供重启后预置。"""
    return CACHE_MONITOR.save_to_file()


# ===========================================================================
# LLM 流量统计 API（SQLite 持久化，LocalLLM 除外）
# ===========================================================================


@router.get("/api/traffic/models")
async def api_traffic_models():
    """出现过 LLM 流量的模型列表（按最近使用时间倒序；排除 local）。"""
    return {"models": TRAFFIC.models()}


@router.get("/api/traffic/session")
async def api_traffic_session(session_id: str = ""):
    """按会话聚合的流量摘要（供 /status 查看，与是否活跃无关，直接查流量库）。"""
    return TRAFFIC.session_stats(session_id)


@router.get("/api/traffic/session/cost")
async def api_traffic_session_cost(session_id: str = ""):
    """本会话费用估算（单位：元）。

    筛选范围 = 本会话所有收费模型产生的费用：把会话内流量按模型拆开，
    逐模型套用与流量统计页完全相同的单价算法（免费模型与未配置单价按 0），
    求和后返回。与 /api/traffic/summary 的 cost_estimate 同算法，只是
    聚合范围从时间窗口换成会话（session_id 即流量库 llm_traffic.session_id）。
    """
    return _compute_traffic_cost(TRAFFIC.session_per_model(session_id))


@router.get("/api/traffic/stats")
async def api_traffic_stats(model: str = "", granularity: str = "hour", n: int = 24):
    """柱状图聚合数据：model 为空=全部（按模型堆叠），granularity=hour/day/week。"""
    return TRAFFIC.stats(model=model, granularity=granularity, n=n)


def _compute_traffic_cost(per: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按模型单价估算流量费用（单位：元）。

    入参 per 为按模型聚合的流量统计（与 TRAFFIC.per_model /
    TRAFFIC.session_per_model 返回结构一致）。

    公式（单价单位：元/百万 tokens）：
      费用 = 命中tokens×命中单价 + 未命中tokens×未命中单价 + 输出tokens×输出单价

    返回 {total, per_model: [{model, billable, free, cost}], priced: 是否有有效单价}。
    免费模型（billing=free）与未配置单价的模型按 0。
    """
    priced_any = False
    total = 0.0
    per_model: List[Dict[str, Any]] = []
    for item in per:
        m = item["model"]
        fm = _resolve_preset_model(m)
        billing = "free" if (fm and fm.get("billing") == "free") else "paid"
        miss = float(fm.get("price_cache_miss") or 0) if fm else 0.0
        hit = float(fm.get("price_cache_hit") or 0) if fm else 0.0
        out = float(fm.get("price_output") or 0) if fm else 0.0
        if billing == "free" or (miss <= 0 and hit <= 0 and out <= 0):
            cost = 0.0
        else:
            cost = (
                item.get("tokens_in_hit", 0) / 1_000_000.0 * hit
                + item.get("tokens_in_miss", 0) / 1_000_000.0 * miss
                + item.get("tokens_out", 0) / 1_000_000.0 * out
            )
        if cost > 0:
            priced_any = True
        total += cost
        per_model.append({
            "model": m,
            "billing": billing,
            "free": billing == "free",
            "cost": round(cost, 4),
        })
    return {"total": round(total, 4), "per_model": per_model, "priced": priced_any}


def _estimate_traffic_cost(
    model: str = "",
    granularity: str = "hour",
    n: int = 24,
) -> Dict[str, Any]:
    """按模型单价估算流量费用（单位：元，时间范围内）。

    与 _compute_traffic_cost 共用同一算法：先取时间范围内按模型聚合的流量，
    再逐模型按单价估算后求和（免费模型与未配置单价按 0）。
    """
    per = TRAFFIC.per_model(granularity=granularity, n=n, model=model)
    return _compute_traffic_cost(per)


@router.get("/api/traffic/summary")
async def api_traffic_summary(model: str = "", granularity: str = "hour", n: int = 24):
    """汇总卡片：总次数/入出字符数/命中与未命中 tokens/出 tokens/平均耗时/失败数。

    额外返回费用估算（按模型单价，单位：元）：
    - 单模型筛选时直接对该模型估算；
    - 全部模型时按模型分别估算后相加（免费模型与未配置单价按 0）。
    估算公式（单价单位：元/百万 tokens）：
      费用 = 命中tokens×命中单价 + 未命中tokens×未命中单价 + 输出tokens×输出单价
    """
    data = TRAFFIC.summary(model=model, granularity=granularity, n=n)
    data["cost_estimate"] = _estimate_traffic_cost(
        model=model, granularity=granularity, n=n)
    return data


@router.get("/api/traffic/records")
async def api_traffic_records(model: str = "", direction: str = "", page: int = 1, page_size: int = 20):
    """分页明细（最近在前）。model/direction 可空=不筛选。"""
    return TRAFFIC.records(model=model, direction=direction, page=page, page_size=page_size)


@router.delete("/api/traffic/records")
async def api_traffic_clear(model: str = ""):
    """清空指定模型（或全部）的流量记录。model 为空表示全部清除。"""
    return {"cleared": TRAFFIC.clear(model), "model": model}


@router.post("/api/session/start")
async def api_session_start(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Maps to Tauri start_claude_session."""
    body = payload or {}
    session_id = body.get("session_id") or f"desk_{uuid.uuid4().hex[:8]}"
    bs = _new_bridge_session(session_id, body)
    prompt = body.get("prompt")
    if prompt:
        clean_text, attachments = _parse_attachments(prompt)
        if clean_text.strip().startswith("/"):
            # 与 stdin 路径一致：斜杠命令不写入会话历史
            bs["pending"].append({"text": clean_text, "attachments": attachments})
        else:
            msg = {"role": "user", "content": clean_text, "timestamp": _now_ms()}
            if attachments:
                msg["attachments"] = attachments
            bs["messages"].append(msg)
            bs["pending"].append({"text": clean_text, "attachments": attachments})
        _schedule_pump(bs)
        _persist_bs(bs)
    return {
        "stdin_id": bs["stdin_id"],
        "cli_session_id": bs["session_id"],
        "pid": 0,
        "cli_path": "",
    }


@router.post("/api/project/permission")
async def api_project_set_permission(payload: SetProjectPermissionRequest):
    """Set tool permission for a project; applies to all its sessions."""
    if payload.mode not in ("default", "readonly", "full"):
        raise HTTPException(status_code=400, detail=f"未知权限模式 {payload.mode}")
    key = _project_key(payload.project)
    project_permissions[key] = payload.mode
    for bs in bridge_sessions.values():
        if bs.get("cwd") and _project_key(bs["cwd"]) == key:
            bs["tool_permission"] = payload.mode
            if bs["agent"] is not None:
                bs["agent"].set_permission_mode(payload.mode)
            _persist_bs(bs)
    _save_permissions()
    return {"status": "ok", "project": payload.project, "mode": payload.mode}


@router.get("/api/project/permission")
async def api_project_get_permission(project: str = ""):
    """Get a project's current tool permission mode (default if never set)."""
    key = _project_key(project or os.getcwd())
    return {"project": project, "mode": project_permissions.get(key, "default")}


@router.post("/api/session/track")
async def api_session_track(payload: Optional[Dict[str, Any]] = Body(default=None)):
    body = payload or {}
    if body.get("session_id"):
        sid = body["session_id"]
        if sid not in bridge_session_by_cli:
            # No stdin mapping yet — create a lightweight placeholder store.
            bridge_message_stores.setdefault(sid, [])
    return {"status": "ok"}


@router.post("/api/session/title")
async def api_session_title(payload: SessionTitleRequest):
    """智能会话命名：用预置免费 Agnes 模型根据对话内容生成简短标题（≤15 字）。"""
    user_text = (payload.user_message or "").strip()
    assistant_text = (payload.assistant_message or "").strip()
    if not user_text and not assistant_text:
        return JSONResponse(content="")
    prompt = (
        "你是会话标题命名助手。请根据下面的对话内容，为这个会话起一个简洁、准确的标题，"
        "直接输出标题本身，不要引号、标点、解释或多余文字，控制在 15 个字以内。\n\n"
        f"用户：{user_text}\n"
        f"助手：{assistant_text}\n\n"
        "标题："
    )
    try:
        # 强制使用预置免费 Agnes（anthropic Messages API），非流式、无工具、无思考。
        agent = _create_agent("agnes", plain_chat=True, thinking_level="off")
        title = await run_in_threadpool(agent._anthropic_summarize, prompt)
        return JSONResponse(content=_clean_session_title(title))
    except Exception as e:
        print(f"[title] 生成失败：{e}")
        return JSONResponse(content="")


@router.get("/api/session/active")
async def api_session_active():
    """Maps to Tauri list_active_processes."""
    # Web mode keeps sessions alive across refreshes; returning [] prevents
    # the frontend from killing sessions it cannot see after a page reload.
    return []


@router.get("/api/sessions")
async def api_sessions():
    """Maps to Tauri list_sessions."""
    items = []
    seen: set = set()
    # Bridge sessions (real conversations with at least one message)
    for stdin_id, bs in bridge_sessions.items():
        if not bs["messages"]:
            continue
        sid = bs["session_id"]
        if sid in seen:
            continue
        seen.add(sid)
        first = bs["messages"][0]["content"] if bs["messages"] else ""
        items.append({
            "id": sid,
            "path": f"/sessions/{sid}",
            "project": bs.get("cwd") or os.getcwd(),
            "projectDir": bs.get("cwd") or os.getcwd(),
            "modifiedAt": bs.get("last_activity", _now_ms()),
            "preview": first[:20],
            "cliResumeId": sid,
            "toolPermission": bs.get("tool_permission") or "default",
            "chatMode": bool(bs.get("chat_mode")),
            "role": bs.get("role") or ("chat" if bool(bs.get("chat_mode")) else "programming"),
            "model": bs.get("model") or "",
        })
    # 磁盘持久化的会话（重启后从文件恢复）
    for sid, data in persisted_sessions.items():
        if sid in seen:
            continue
        msgs = data.get("messages") or []
        if not msgs:
            continue
        seen.add(sid)
        first = msgs[0].get("content") or ""
        items.append({
            "id": sid,
            "path": f"/sessions/{sid}",
            "project": data.get("cwd") or os.getcwd(),
            "projectDir": data.get("cwd") or os.getcwd(),
            "modifiedAt": data.get("last_activity") or data.get("created_at") or 0,
            "preview": first[:20],
            "cliResumeId": sid,
            "toolPermission": data.get("tool_permission") or "default",
            "chatMode": bool(data.get("chat_mode")),
            "role": data.get("role") or ("chat" if bool(data.get("chat_mode")) else "programming"),
            "model": data.get("model") or "",
        })
    # Legacy /api/chat sessions
    for sid, session in sessions.items():
        if sid in seen or not session["messages"]:
            continue
        seen.add(sid)
        first = session["messages"][0]["content"] if session["messages"] else ""
        items.append({
            "id": sid,
            "path": f"/sessions/{sid}",
            "project": os.getcwd(),
            "projectDir": os.getcwd(),
            "modifiedAt": int(datetime.now().timestamp() * 1000),
            "preview": first[:20],
            "cliResumeId": None,
        })
    return items


@router.get("/api/session/messages")
async def api_session_messages(path: str = ""):
    """Maps to Tauri load_session (raw message array)."""
    sid = path.rstrip("/").split("/")[-1]
    for bs in bridge_sessions.values():
        if bs["session_id"] == sid:
            return bs["messages"]
    store = bridge_message_stores.get(sid)
    if store is not None:
        return store
    data = persisted_sessions.get(sid)
    if data is not None:
        return data.get("messages") or []
    session = sessions.get(sid)
    if session is None:
        return []
    return session["messages"]


@router.get("/api/sessions/search")
async def api_sessions_search(q: str = ""):
    if not q:
        return []
    results = []
    for bs in bridge_sessions.values():
        for m in bs["messages"]:
            content = m.get("content", "") or ""
            if q.lower() in content.lower():
                results.append({
                    "session_id": bs["session_id"],
                    "snippet": content[:120],
                    "match_count": 1,
                    "match_role": m.get("role", "user"),
                })
                break
    for sid, data in persisted_sessions.items():
        for m in (data.get("messages") or []):
            content = m.get("content", "") or ""
            if q.lower() in content.lower():
                results.append({
                    "session_id": sid,
                    "snippet": content[:120],
                    "match_count": 1,
                    "match_role": m.get("role", "user"),
                })
                break
    return results


@router.post("/api/session/{stdin_id}/stdin")
async def api_session_stdin(stdin_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    body = payload or {}
    text = body.get("text", "")
    bs = _new_bridge_session(stdin_id, body)
    bs["last_activity"] = _now_ms()
    if text:
        clean_text, attachments = _parse_attachments(text)
        if clean_text.strip().startswith("/"):
            # 斜杠命令是控制指令，不写入会话历史（如 /compact 压缩上下文）
            bs["pending"].append({"text": clean_text, "attachments": attachments})
        else:
            msg = {"role": "user", "content": clean_text, "timestamp": _now_ms()}
            if attachments:
                msg["attachments"] = attachments
            bs["messages"].append(msg)
            bs["pending"].append({"text": clean_text, "attachments": attachments})
        _schedule_pump(bs)
    _persist_bs(bs)
    return {"status": "ok"}


@router.post("/api/session/{stdin_id}/question")
async def api_session_question(stdin_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    """用户回答 AskUserQuestion 后，把答案以 tool_result 写回 agent 上下文并续跑。

    前端 QuestionCard 确认后调用；answer 记录以"问题下标 -> 答案文本"的方式传入，
    服务端转成 tool_result，注入引擎上下文，并追加"请继续"用户轮驱动模型产出最终回复。
    """
    body = payload or {}
    request_id = body.get("request_id")
    tool_use_id = body.get("tool_use_id")
    answers = body.get("answers") or {}

    bs = bridge_sessions.get(stdin_id)
    if bs is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    pending = bs.get("pending_question")
    if not pending:
        raise HTTPException(status_code=400, detail="当前没有待回答的问题")
    if request_id and pending.get("request_id") and request_id != pending.get("request_id"):
        raise HTTPException(status_code=409, detail="问题已过期，请重新发起提问")

    agent = _ensure_agent(bs)
    tid = tool_use_id or pending.get("tool_use_id")
    questions = pending.get("questions") or []
    answer_text = agent._answers_to_text(questions, answers)

    # 追加 tool_result 到引擎上下文（OpenAI role=tool / Anthropic tool_result 块）。
    agent.messages.append(Message(
        role="user",
        content=[{"type": "tool_result", "tool_use_id": tid, "content": answer_text}],
        timestamp=_now_ms(),
        tool_name="AskUserQuestion",
        tool_input=pending.get("input") or {},
        tool_result=answer_text,
    ))
    bs["pending_question"] = None
    _persist_bs(bs)

    # 续跑：以 continue_only 复用工具轮（messages 已含 tool_result），模型直接继续。
    bs["pending"].append({"text": "", "attachments": [], "continue_only": True})
    _schedule_pump(bs)
    return {"status": "ok", "request_id": request_id, "tool_use_id": tid}


@router.get("/api/session/{stdin_id}/debug")
async def api_session_debug(stdin_id: str):
    bs = bridge_sessions.get(stdin_id)
    if bs is None:
        return {"exists": False}
    return {
        "exists": True,
        "queue_size": len(bs["events"]),
        "generating": bs["generating"],
        "pending": list(bs["pending"]),
        "cancelled": bs["cancelled"],
        "messages": len(bs["messages"]),
        "task_done": bs["task"].done() if bs["task"] else None,
        "queued": list(bs["events"])[:10],
    }


@router.get("/api/session/{stdin_id}/stream")
async def api_session_stream(stdin_id: str):
    """SSE endpoint draining the NDJSON event queue for a bridge session.

    The frontend connects its stream listener before /session/start resolves,
    so an unknown session is tolerated briefly while it is being created.
    """
    bs = bridge_sessions.get(stdin_id)

    async def event_generator():
        nonlocal bs
        if bs is None:
            for _ in range(300):
                await asyncio.sleep(0.1)
                bs = bridge_sessions.get(stdin_id)
                if bs is not None:
                    break
        if bs is None:
            yield f"data: {json.dumps({'type': 'process_exit', 'code': 1})}\n\n"
            return
        # system:init on every (re)connect so the frontend marks stdin ready.
        yield f"data: {json.dumps({'type': 'system', 'subtype': 'init', 'model': bs['model'], 'session_id': bs['session_id']}, ensure_ascii=False)}\n\n"
        # Poll the event deque. The upstream LLM may burst its tokens, so a
        # simple polling loop is far more robust than cross-task queue wakeups
        # (which stalled under load in earlier builds).
        while True:
            if bs["events"]:
                event = bs["events"].popleft()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "process_exit":
                    break
                continue
            await asyncio.sleep(0.05)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/session/{stdin_id}/signal")
async def api_session_signal(stdin_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    body = payload or {}
    bs = bridge_sessions.get(stdin_id)
    if bs is not None:
        bs["cancelled"] = True
        if bs["task"] and not bs["task"].done():
            bs["task"].cancel()
        _emit_bridge_event(bs, {"type": "process_exit", "code": 0})
    return {"status": "ok"}


@router.delete("/api/session/{session_id}")
async def api_session_delete(session_id: str):
    # The list UI passes the canonical CLI id (cli_xxx), while bridge sessions
    # are keyed by stdin id (desk_xxx) — match either.
    bs = bridge_sessions.get(session_id)
    if bs is None:
        for candidate in bridge_sessions.values():
            if candidate["session_id"] == session_id:
                bs = candidate
                break
    if bs is not None:
        bs["cancelled"] = True
        if bs["task"] and not bs["task"].done():
            bs["task"].cancel()
        _emit_bridge_event(bs, {"type": "process_exit", "code": 0})
        bridge_sessions.pop(bs["stdin_id"], None)
        bridge_session_by_cli.pop(bs["session_id"], None)
        bridge_message_stores.pop(bs["session_id"], None)
        _deleted_agent = bridge_agents_by_cli.pop(bs["session_id"], None)
        if _deleted_agent is not None:
            _release_mcp(_deleted_agent)
        _delete_session_file(bs["session_id"])
    if session_id in sessions:
        del sessions[session_id]
    if session_id in agents:
        del agents[session_id]
    _delete_session_file(session_id)
    return {"status": "deleted"}


@router.put("/api/session/{session_id}/model")
async def api_session_set_model(session_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    """切换会话使用的模型：更新内存与磁盘落盘，使会话列表 / 恢复时都能读到新模型。"""
    body = payload or {}
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model 不能为空")
    # 内存会话（活跃）
    for stdin_id, bs in bridge_sessions.items():
        if bs.get("session_id") == session_id:
            bs["model"] = model
            _persist_bs(bs)
            return {"status": "ok", "model": model}
    # 磁盘持久化会话（重启后恢复）
    data = persisted_sessions.get(session_id)
    if data is not None:
        data["model"] = model
        path = _session_file(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return {"status": "ok", "model": model}
    raise HTTPException(status_code=404, detail="会话不存在")


# ---- filesystem ----

# 文件树忽略目录名单（与原版 tokenicode read_dir_recursive 一致）
_IGNORED_TREE_DIRS = {
    "node_modules", "target", "__pycache__", ".git", ".DS_Store",
    "Thumbs.db", ".venv", "venv", ".env", "dist", "build", ".next",
    ".nuxt", ".parcel-cache", "coverage", ".turbo", ".svelte-kit",
}


def _read_dir_recursive(dir_path: str, current_depth: int, max_depth: int) -> list:
    """按深度递归读取目录树（对应原版 read_file_tree），跳过忽略名单。"""
    try:
        entries = list(os.scandir(dir_path))
    except OSError:
        return []
    meta = []
    for entry in entries:
        try:
            is_dir = entry.is_dir()
        except OSError:
            is_dir = False
        meta.append((entry, is_dir, entry.name.lower()))
    meta.sort(key=lambda m: (not m[1], m[2]))
    nodes = []
    for entry, is_dir, _lower in meta:
        if entry.name in _IGNORED_TREE_DIRS:
            continue
        if is_dir and current_depth < max_depth:
            children = _read_dir_recursive(entry.path, current_depth + 1, max_depth)
        elif is_dir:
            children = []
        else:
            children = None
        nodes.append({
            "name": entry.name,
            "path": entry.path,
            "is_dir": is_dir,
            "children": children,
        })
    return nodes


@router.get("/api/files/tree")
async def api_files_tree(path: str = "", depth: int = 5):
    root = path or os.path.expanduser("~")
    if not os.path.isdir(root):
        return []
    return _read_dir_recursive(root, 0, max(1, min(depth, 10)))


@router.get("/api/files/content")
async def api_files_content(path: str = ""):
    try:
        return FsPath(path).read_text(encoding="utf-8")
    except Exception:
        return ""


@router.post("/api/files/content")
async def api_files_write(payload: Optional[Dict[str, Any]] = Body(default=None)):
    body = payload or {}
    path = body.get("path", "")
    content = body.get("content", "")
    if path:
        FsPath(path).parent.mkdir(parents=True, exist_ok=True)
        FsPath(path).write_text(content, encoding="utf-8")
    return {"status": "ok"}


@router.get("/api/files/size")
async def api_files_size(path: str = ""):
    try:
        return FsPath(path).stat().st_size
    except Exception:
        return 0


@router.post("/api/files/temp")
async def api_files_temp(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Maps to Tauri save_temp_file — persist uploaded bytes and return a path."""
    body = payload or {}
    name = os.path.basename(body.get("name") or "upload.bin") or "upload.bin"
    b64 = body.get("base64") or ""
    cwd = body.get("cwd") or None
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raw = b""
    if cwd:
        upload_dir = os.path.join(cwd, ".bigcodex_uploads")
    else:
        upload_dir = os.path.join(tempfile.gettempdir(), "bigcodex_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, f"{uuid.uuid4().hex[:8]}_{name}")
    with open(path, "wb") as f:
        f.write(raw)
    return path


@router.post("/api/files/print-md")
async def api_files_print_md(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Maps to Tauri save_print_markdown — archive /print -md output under <cwd>/.bigcodex_uploads."""
    body = payload or {}
    cwd = str(body.get("cwd") or "").strip()
    content = str(body.get("content") or "")
    filename = str(body.get("filename") or "").strip()
    if not cwd:
        raise HTTPException(status_code=400, detail="cwd is required")
    if not content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    upload_dir = os.path.join(cwd, ".bigcodex_uploads")
    try:
        os.makedirs(upload_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"mkdir failed: {e}")
    if filename:
        filename = os.path.basename(filename)
        if not filename.lower().endswith(".md"):
            filename += ".md"
    else:
        filename = f"print_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = os.path.join(upload_dir, filename)
    if os.path.exists(path):
        path = os.path.join(upload_dir, f"{uuid.uuid4().hex[:8]}_{filename}")
    try:
        FsPath(path).write_text(content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")
    return {"status": "ok", "path": path}


@router.get("/api/files/base64")
async def api_files_base64(path: str = ""):
    """Maps to Tauri read_file_base64 — full base64 data URL for previews/lightbox."""
    try:
        raw = FsPath(path).read_bytes()
    except Exception:
        return ""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


# ---- export ----

def _find_session_messages(sid: str):
    """按会话 id 定位消息列表（bridge / 持久化 / legacy 会话）。"""
    for bs in bridge_sessions.values():
        if bs["session_id"] == sid:
            return bs["messages"]
    store = bridge_message_stores.get(sid)
    if store is not None:
        return store
    data = persisted_sessions.get(sid)
    if data is not None:
        return data.get("messages") or []
    session = sessions.get(sid)
    if session is not None:
        return session["messages"]
    return None


def _export_markdown(messages, source_path, conversation_only):
    """按原版 tokenicode 思路生成会话 Markdown（头部 + User/Assistant 小节）。"""
    md = "# Claude Code Session\n\n"
    md += f"*Exported from: {source_path}*\n\n---\n\n"
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        content = m.get("content") or ""
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(str(block.get("text") or ""))
                elif not conversation_only and btype == "tool_use":
                    parts.append(f"**Tool: {block.get('name') or 'Tool'}**\n")
                    inp = block.get("input")
                    if inp is not None:
                        parts.append("```json\n" + json.dumps(inp, ensure_ascii=False, indent=2) + "\n```")
            text = "\n\n".join(p for p in parts if p)
        else:
            text = str(content)
        text = text.rstrip()
        if role in ("user", "human"):
            if not conversation_only or text.strip():
                md += f"## User\n\n{text}\n\n"
        elif role == "assistant":
            if not conversation_only or text.strip():
                md += f"## Assistant\n\n{text}\n\n"
    return md


@router.post("/api/export/markdown")
async def api_export_markdown(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Maps to Tauri export_session_markdown."""
    body = payload or {}
    sid = str(body.get("session_id") or "").rstrip("/").split("/")[-1]
    output_path = str(body.get("output_path") or "").strip()
    conversation_only = bool(body.get("conversation_only"))
    if not sid or not output_path:
        raise HTTPException(status_code=400, detail="session_id and output_path are required")
    messages = _find_session_messages(sid)
    if messages is None:
        raise HTTPException(status_code=404, detail="session not found")
    md = _export_markdown(messages, f"/sessions/{sid}", conversation_only)
    try:
        FsPath(output_path).parent.mkdir(parents=True, exist_ok=True)
        FsPath(output_path).write_text(md, encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")
    return {"status": "ok", "path": output_path}


@router.post("/api/export/json")
async def api_export_json(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """Maps to Tauri export_session_json."""
    body = payload or {}
    sid = str(body.get("session_id") or "").rstrip("/").split("/")[-1]
    output_path = str(body.get("output_path") or "").strip()
    if not sid or not output_path:
        raise HTTPException(status_code=400, detail="session_id and output_path are required")
    messages = _find_session_messages(sid)
    if messages is None:
        raise HTTPException(status_code=404, detail="session not found")
    data = json.dumps(messages, ensure_ascii=False, indent=2)
    try:
        FsPath(output_path).parent.mkdir(parents=True, exist_ok=True)
        FsPath(output_path).write_text(data, encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")
    return {"status": "ok", "path": output_path}


# ---- system ----

@router.get("/api/system/cwd")
async def api_system_cwd():
    """Current working directory of the backend process (default project dir)."""
    return os.getcwd()


@router.get("/api/system/drives")
async def api_system_drives():
    """本机可用盘符（Windows 返回 C:、D: 等；非 Windows 返回 ["/"]）。"""
    if os.name == "nt":
        import string
        drives = []
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if os.path.isdir(root):
                drives.append(root)
        return {"drives": drives}
    return {"drives": ["/"]}


@router.get("/api/system/shared-temp")
async def api_system_shared_temp():
    """共享临时目录：仓库根下，不存在则创建；供聊天等非开发工作会话一键使用。"""
    return {"path": _shared_temp_dir()}


@router.get("/api/system/home")

async def api_system_home():
    return os.path.expanduser("~")


@router.get("/api/system/app-dir")
async def api_system_app_dir():
    """程序目录（安装/项目根），供前端读写程序内配置（如 mcp.json、skills）。"""
    return _repo_root()


# ---- MCP ----

@router.get("/api/mcp/status")
async def api_mcp_status():
    """MCP 服务器连接状态（内置 filesystem/fetch + mcp.json 用户配置）。"""
    return {
        "enabled": True,
        "servers": mcp_manager.get_status(),
    }


@router.get("/api/system/decode")
async def api_system_decode(path: str = ""):
    return path


@router.get("/api/projects/recent")
async def api_projects_recent():
    return []


@router.get("/api/models")
async def api_models():
    """设置页「默认模型」下拉框的数据源（与前端 PRESET_MODELS 同步）。"""
    return [
        {
            "id": m["id"],
            "name": m["name"],
            "api_url": m["api_url"],
            "model": m["model"],
            "max_tokens": m["max_tokens"],
            "billing": m.get("billing", "paid"),
            "price_cache_miss": m.get("price_cache_miss", 0),
            "price_cache_hit": m.get("price_cache_hit", 0),
            "price_output": m.get("price_output", 0),
        }
        for m in PRESET_MODELS
    ]


@router.get("/api/image-models")
async def api_image_models():
    """图片生成模型配置（内置工具 generate_image 的数据源；不含 API key）。"""
    return [
        {
            "id": m.get("id", ""),
            "name": m.get("name", ""),
            "api_url": m.get("api_url", ""),
            "model": m.get("model", ""),
            "kind": m.get("kind", "kolors"),
            "image_size": m.get("image_size", "1024x1024"),
            "ratio": m.get("ratio", "1:1"),
            "batch_size": m.get("batch_size", 1),
        }
        for m in IMAGE_MODELS
    ]


@router.get("/api/video-models")
async def api_video_models():
    """视频生成模型配置（内置工具 generate_video 的数据源；不含 API key）。"""
    return [
        {
            "id": m.get("id", ""),
            "name": m.get("name", ""),
            "model": m.get("model", ""),
            "kind": m.get("kind", "agnes-v2.0"),
            "size": m.get("size", "720P"),
            "aspect_ratio": m.get("aspect_ratio", "16:9"),
            "seconds": m.get("seconds", "5"),
            "video_width": m.get("video_width", 832),
            "video_height": m.get("video_height", 448),
            "num_frames": m.get("num_frames", 81),
            "frame_rate": m.get("frame_rate", 24),
        }
        for m in VIDEO_MODELS
    ]


@router.get("/api/user-models")
async def api_user_models_get():
    """自定义模型原始数据源（设置页「模型设定」加载）。"""
    return user_models.load_user_models()


@router.put("/api/user-models")
async def api_user_models_set(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """校验并保存自定义模型到 user_models.json。

    校验必填项（api_format / api_key_env / billing 等）不通过时返回 400 + 错误列表；
    通过后落盘。模型预置在进程启动时导入，因此改动需重启后端生效。
    """
    body = payload or {}
    errors = user_models.validate_user_models(body)
    if errors:
        raise HTTPException(status_code=400, detail=errors)
    user_models.save_user_models(body)
    return {"status": "ok", "restart_required": True}


async def _speech_bytes(text: str, voice: str) -> bytes:
    """调用 edge-tts 在内存中合成音频字节。"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            audio.extend(chunk.get("data") or b"")
    if not audio:
        raise RuntimeError("edge-tts 未返回音频数据")
    return bytes(audio)


def _sanitize_speech_name(filename, default: str) -> str:
    name = os.path.basename(str(filename or "").strip()) or default
    if "." not in name:
        name = f"{name}.mp3"
    return name


@router.get("/api/speech-voices")
async def api_speech_voices():
    """语音人声列表（来自 edge-tts --list-voices），供设置页「语音人声」下拉。"""
    return await get_speech_voices()


@router.post("/api/speech")
async def api_speech(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """生成语音并以 base64 data URL 返回（内存中，不落盘），供会话朗读按钮播放。"""
    body = payload or {}
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    voice = str(body.get("voice") or get_active_voice_name() or DEFAULT_VOICE).strip()
    try:
        audio = await _speech_bytes(text, voice)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"语音合成失败：{e}")
    return {"dataUrl": f"data:audio/mpeg;base64,{base64.b64encode(audio).decode('ascii')}"}


@router.post("/api/speech/save")
async def api_speech_save(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """生成语音并保存 mp3 到 cwd/.bigcodex_uploads/，返回路径（供 /speech 命令）。"""
    body = payload or {}
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    voice = str(body.get("voice") or get_active_voice_name() or DEFAULT_VOICE).strip()
    cwd = str(body.get("cwd") or os.getcwd())
    default_name = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    filename = _sanitize_speech_name(body.get("filename"), default_name)
    out_dir = os.path.join(cwd, ".bigcodex_uploads")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建保存目录失败：{e}")
    path = os.path.join(out_dir, filename)
    try:
        import edge_tts
        await edge_tts.Communicate(text, voice).save(path)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"语音合成失败：{e}")
    return {"status": "ok", "path": path}


@router.get("/api/multimodal-config")
async def api_multimodal_config_get():
    """当前激活的图片/视频模型（设置页「多模态模型」数据源）。"""
    return {
        "active_image_model": multimodal_config.get_active_image_model(),
        "active_video_model": multimodal_config.get_active_video_model(),
        "active_voice": multimodal_config.get_active_voice(),
    }


@router.put("/api/multimodal-config")
async def api_multimodal_config_set(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """保存激活的图片/视频模型（写入 models.json 顶层字段）。"""
    body = payload or {}
    if body.get("active_image_model") is not None:
        multimodal_config.set_active_image_model(str(body["active_image_model"]))
    if body.get("active_video_model") is not None:
        multimodal_config.set_active_video_model(str(body["active_video_model"]))
    if body.get("active_voice") is not None:
        multimodal_config.set_active_voice(str(body["active_voice"]))
    return {
        "active_image_model": multimodal_config.get_active_image_model(),
        "active_video_model": multimodal_config.get_active_video_model(),
        "active_voice": multimodal_config.get_active_voice(),
    }


def _find_model_entry(provider_id: str, model: str) -> Optional[Dict[str, Any]]:
    """按 provider_id + model 匹配预置模型条目（兼容只传 model）。"""
    for m in PRESET_MODELS:
        if provider_id and m.get("id") == provider_id and m.get("model") == model:
            return m
    for m in PRESET_MODELS:
        if m.get("model") == model:
            return m
    return None


async def _generate_role_prompt(entry: Dict[str, Any], role_name: str) -> str:
    """调用指定模型生成 <role_name>助手 的角色指令文本（openai / anthropic 两种格式）。"""
    api_key = str(entry.get("api_key") or "").strip()
    if not api_key or api_key in ("your api key", "sk-local"):
        raise HTTPException(status_code=400, detail=f"模型 {entry.get('name', '')} 未配置 API key")
    api_format = entry.get("api_format") or "openai"
    api_auth = entry.get("api_auth") or ("x-api-key" if api_format == "anthropic" else "bearer")
    base = str(entry.get("api_url") or "").rstrip("/")
    prompt_text = (
        "你是一名角色设定助手。请为角色名「{name}」生成一段用于 AI 助手的角色扮演指令。"
        "要求：以“你是一位{name}助手”开头；描述该角色的身份定位、专长、语气与回答风格；"
        "正文控制在 100-200 字；直接输出指令正文，不要输出任何前后缀说明。"
    ).format(name=role_name)
    headers: Dict[str, str] = {"content-type": "application/json"}
    if api_format == "anthropic":
        url = base + "/messages" if base.endswith("/v1") else base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if api_auth == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
        payload: Dict[str, Any] = {
            "model": entry["model"],
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.7,
        }
    else:
        url = base + "/chat/completions" if not base.endswith("/chat/completions") else base
        if api_auth == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
        payload = {
            "model": entry["model"],
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.7,
        }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0)) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"调用 {entry.get('name', '')} 失败：{e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"{entry.get('name', '')} 返回 {resp.status_code}：{resp.text[:200]}")
    try:
        data = resp.json()
        if api_format == "anthropic":
            text = "".join(
                b.get("text", "")
                for b in (data.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail=f"{entry.get('name', '')} 响应解析失败")
    return str(text or "").strip()


@router.post("/api/roles/generate-prompt")
async def api_roles_generate_prompt(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """用指定模型生成 <角色名>助手 的预设角色指令文本。"""
    body = payload or {}
    role_name = str(body.get("role_name") or "").strip()
    provider_id = str(body.get("provider_id") or "").strip()
    model = str(body.get("model") or "").strip()
    if not role_name:
        raise HTTPException(status_code=400, detail="角色名不能为空")
    if not model:
        raise HTTPException(status_code=400, detail="请选择用于生成预设文本的模型")
    entry = _find_model_entry(provider_id, model)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"未找到模型 {provider_id}:{model}")
    text = await _generate_role_prompt(entry, role_name)
    return {"prompt": text}


@router.get("/api/roles")
async def api_roles_list():
    """角色列表（内置编程/聊天 + 用户自定义）。"""
    return get_roles()


@router.post("/api/roles")
async def api_roles_create(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """新增自定义角色。"""
    body = payload or {}
    name = str(body.get("name") or "").strip()
    prompt = str(body.get("prompt") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="角色名不能为空")
    return create_role(name, prompt)


@router.put("/api/roles/{role_id}")
async def api_roles_update(role_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    """更新自定义角色。"""
    body = payload or {}
    updated = update_role(role_id, name=body.get("name"), prompt=body.get("prompt"))
    if updated is None:
        raise HTTPException(status_code=404, detail="角色不存在或为内置角色")
    return updated


@router.delete("/api/roles/{role_id}")
async def api_roles_delete(role_id: str):
    """删除自定义角色。"""
    if not delete_role(role_id):
        raise HTTPException(status_code=404, detail="角色不存在或为内置角色")
    return {"status": "ok"}


@router.post("/api/session/{session_id}/role")
async def api_session_set_role(session_id: str, payload: Optional[Dict[str, Any]] = Body(default=None)):
    """空会话切换角色：更新会话角色并释放旧 agent（下次惰性重建，role_prompt 生效）。"""
    bs = bridge_sessions.get(session_id)
    if bs is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    role = str((payload or {}).get("role") or "").strip()
    if not role or get_role(role) is None:
        raise HTTPException(status_code=400, detail="未知角色")
    if role == "chat":
        _pay_fm = _resolve_preset_model(str(bs.get("model") or ""))
        if _pay_fm and str(_pay_fm.get("billing") or "paid").lower() == "paid":
            raise HTTPException(
                status_code=400,
                detail="收费模型只能在编程模式使用，请切换到编程模式或选择免费模型",
            )
    bs["role"] = role
    bs["chat_mode"] = role == "chat"
    if bs.get("agent") is not None:
        try:
            _release_mcp(bs["agent"])
        except Exception:
            pass
        bs["agent"] = None
        bridge_agents_by_cli.pop(bs["session_id"], None)
    _persist_bs(bs)
    return {"status": "ok", "role": role}
