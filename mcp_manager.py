"""
MCP 服务器管理：内置 Filesystem（按 Agent cwd 多实例）+ Fetch / Git / 程序目录 mcp.json 用户配置服务器（全局单例，脱离 ~/.claude.json）。

职责：
- 在后台 asyncio 事件循环中维护每个服务器的长驻 stdio 会话；
- Filesystem 服务器按 Agent 的 cwd 多实例化：同一 cwd 的 Agent 共享一个实例（引用计数），
  不同 cwd 各自启动独立实例，允许目录绑定到各自的 cwd；Fetch / Git 与用户配置服务器保持全局单例；
- 把每个 MCP 工具包装成 Tool 动态注册进全局 ToolRegistry（engine 全链路复用），
  工具名全局唯一，执行时按 ToolContext.agent.cwd 路由到对应的 filesystem 实例；
- 提供同步 call_tool 桥接（engine 的 execute 是同步的）；
- 暴露 /api/mcp/status 所需的连接状态（聚合全局服务器 + filesystem 多实例）。

模式归属：
- 只读工具（readOnlyHint / 名单兜底）→ 聊天 + 工作模式，READONLY 权限；
- 写类工具 → 仅工作模式。
"""
import asyncio
import json
import os
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from tools import Tool, ToolMode, tool_registry
from tools.base import ToolPermission
from path_util import realpath_clean, strip_vermagic

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - SDK 缺失时降级为不可用
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    MCP_SDK_AVAILABLE = False

# ---- fetch 工具 max_length 上限 ----

# 与 engine/config.py 的 TOOL_RESULT_MAX_CHARS 保持一致：环境变量 BIGCODEX_TOOL_RESULT_MAX_CHARS，
# 默认 30000。fetch 单次返回字符数被钳制在此上限内，避免返回内容超过工具结果封顶
# 又触发 engine 的二次截断；LLM 主动传 max_length 时也不允许超过该值。
FETCH_MAX_LENGTH_CAP = int(os.environ.get("BIGCODEX_TOOL_RESULT_MAX_CHARS", "30000") or 30000)
# mcp_server_fetch 的 Fetch.max_length 硬上限（lt=1000000），额外保险
_FETCH_HARD_CAP = 1_000_000

# ---- 内置服务器定义 ----


def _repo_root() -> str:
    """仓库根目录（本文件位于 <root>/src/agent/）。"""
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _user_path_base() -> str:
    """相对路径解析基准：用户视角的"项目根目录"（安装版即安装目录）。

    后端通常以装配区布局运行（<根>/app-run/src/agent/server.py），此时
    _repo_root() 返回的是装配区 app-run，而用户填写的相对路径（如
    .\\app-run\\mcptool\\...）是以安装目录（app-run 的上一级）为基准的；
    源码直跑时 _repo_root() 本身就是项目根目录。通过目录名与装配区特征
    （含 runtime/ 且同级存在 mcp.json）区分两种布局。
    """
    root = _repo_root()
    if os.path.basename(root) == "app-run" and os.path.isdir(os.path.join(root, "runtime")):
        parent = os.path.dirname(root)
        if os.path.isfile(os.path.join(parent, "app-run", "mcp.json")):
            return parent
    return root


def _is_likely_relative_path(value: str) -> bool:
    """粗略判断配置值是否像相对路径（需要按项目根目录解析）。

    命中以下任一情况才视为相对路径：
    - 以 . 开头（./ .\\ ../ ..\\，含 .env 这类点开头文件）；
    - 含路径分隔符且既不是 URL（://）也不是命令行开关/根相对路径（- / \\ 开头）。

    其余（绝对路径、裸命令名如 cmd/gitnexus/python、开关、URL）原样保留，
    交由 PATH / shell 语义处理。
    """
    if not value:
        return False
    if os.path.isabs(value):
        return False
    if len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/"):
        return False  # Windows 盘符绝对路径（C:\... / C:/...）
    if value.startswith("."):
        return True
    if value.startswith(("-", "/", "\\")):
        return False  # 命令行开关 / 根相对路径
    if "://" in value:
        return False  # URL
    return ("\\" in value) or ("/" in value)


def _resolve_user_path(value: str) -> str:
    """把用户配置中的相对路径按项目根目录解析为绝对路径；非路径值原样返回。"""
    if not _is_likely_relative_path(value):
        return value
    return os.path.abspath(os.path.join(_user_path_base(), value))


def _filesystem_cfg(root: str) -> Dict[str, Any]:
    """Filesystem 服务器配置：本地 Node 入口（mcptool/filesystem_server），允许目录为给定 root。"""
    fs_entry = os.path.join(_repo_root(), "mcptool", "filesystem_server", "index.js")
    return {
        "name": "filesystem",
        "builtin": True,
        "command": _node_command(),
        "args": [fs_entry, root],
        "env": {},
        "description": "文件系统访问（mcptool/filesystem_server）",
    }


def _node_command() -> str:
    """Node 命令解析：系统 PATH 里的 node 优先，否则用打包的便携 node（<app>/runtime/node/node.exe）。"""
    which = shutil.which("node")
    if which:
        which = strip_vermagic(which)
        if os.path.isfile(which):
            return which
    bundled = strip_vermagic(os.path.join(_repo_root(), "runtime", "node", "node.exe"))
    if os.path.isfile(bundled):
        return bundled
    return "node"


def _builtin_servers() -> List[Dict[str, Any]]:
    """内置服务器：Fetch / Git（mcptool 本地入口）。Filesystem 按 cwd 多实例，不在此列。"""
    fetch_entry = os.path.join(_repo_root(), "mcptool", "fetch_server", "fetch_server.py")
    git_entry = os.path.join(_repo_root(), "mcptool", "git_server", "git_server.py")
    return [
        {
            "name": "fetch",
            "builtin": True,
            "command": sys.executable,
            "args": [fetch_entry],
            "env": {"PYTHONIOENCODING": "utf-8"},
            "description": "网页抓取（mcptool/fetch_server）",
        },
        {
            "name": "git",
            "builtin": True,
            "command": sys.executable,
            "args": [git_entry],
            "env": {"PYTHONIOENCODING": "utf-8"},
            "description": "Git 版本控制（mcptool/git_server）",
        },
    ]


def _load_user_servers(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """读取程序目录 mcp.json 的 mcpServers 作为可选额外服务器（沿用原版字段格式）。

    disabled: true 表示"保留配置但暂时禁用"——默认不返回（不启动进程、不注册工具，
    模型完全看不到其工具）；include_disabled=True 时全部返回并携带 enabled 标志，
    供状态面板展示禁用态（设置页开关保留配置，重启后生效）。
    """
    _migrate_legacy_mcp()
    path = _user_mcp_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    mcp_servers = (data or {}).get("mcpServers") or {}
    if not isinstance(mcp_servers, dict):
        return []
    servers: List[Dict[str, Any]] = []
    for name, cfg in mcp_servers.items():
        if name in ("filesystem", "fetch", "git"):
            continue  # 内置服务器优先，避免重复注册
        if not isinstance(cfg, dict):
            continue
        command = str(cfg.get("command") or "").strip()
        if not command:
            continue
        enabled = not bool(cfg.get("disabled"))
        if not enabled and not include_disabled:
            continue  # 已禁用服务器：不启动、不注册任何工具
        servers.append({
            "name": name,
            "builtin": False,
            # 相对路径（.\app-run\mcptool\... / mcptool\...）统一按安装目录（项目根目录）解析为绝对路径
            "command": _resolve_user_path(command),
            "args": [_resolve_user_path(str(a)) for a in (cfg.get("args") or [])],
            "env": {str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
            "description": "用户配置的 MCP 服务器（mcp.json）",
            "enabled": enabled,
        })
    return servers


def _user_mcp_path() -> str:
    """用户 MCP 配置路径：程序目录内 mcp.json（本程序自包含，不依赖 ~/.claude.json）。"""
    return os.path.join(_repo_root(), "mcp.json")


def _migrate_legacy_mcp() -> None:
    """一次性迁移旧 ~/.claude.json 的 mcpServers 到程序目录 mcp.json（幂等）。"""
    new_path = _user_mcp_path()
    if os.path.exists(new_path):
        return
    legacy = os.path.join(os.path.expanduser("~"), ".claude.json")
    try:
        if not os.path.isfile(legacy):
            return
        with open(legacy, "r", encoding="utf-8") as f:
            data = json.load(f)
        mcp_servers = (data or {}).get("mcpServers") or {}
        if isinstance(mcp_servers, dict) and mcp_servers:
            with open(new_path, "w", encoding="utf-8") as f:
                json.dump({"mcpServers": mcp_servers}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _win_safe_command(command: str, args: List[str]) -> Tuple[str, List[str]]:
    """Windows 下把 npx/uvx 等 .cmd/.ps1 命令包装成 cmd /c，保证可被 spawn。"""
    if os.name != "nt" or not command:
        return command, args
    base = os.path.basename(command).lower()
    if base in ("cmd", "cmd.exe"):
        return command, args
    exe = shutil.which(command)
    if exe:
        ext = os.path.splitext(exe)[1].lower()
        if ext in ("", ".exe"):
            return command, args
    return "cmd", ["/c", command] + args


def _collect_servers() -> List[Dict[str, Any]]:
    """Fetch / Git + 用户配置服务器，统一做 Windows 命令包装（filesystem 走按 cwd 多实例，不在此列）。"""
    servers = _builtin_servers()
    servers.extend(_load_user_servers())
    for s in servers:
        s["command"], s["args"] = _win_safe_command(s["command"], s["args"])
    return servers


# ---- 只读分类 ----

# Filesystem 服务器只读工具（annotations.readOnlyHint 缺失时的兜底名单）
_FS_READ_TOOLS = {
    "read_file", "read_text_file", "read_media_file", "read_multiple_files",
    "list_directory", "list_directory_with_sizes", "directory_tree",
    "search_files", "get_file_info", "list_allowed_directories",
}
# Fetch 服务器全部工具只读
_FETCH_READ_TOOLS = {"fetch"}


# ---- 二进制/媒体读取拦截 ----

# 非文本文件扩展名：OpenAI 模式下不允许经 MCP 文件服务按文本读入上下文；
# 一律禁止读取（见 intercept_binary_read），媒体原始数据绝不进入模型上下文。
_BINARY_EXTS = {
    # 图片
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".avif", ".heic", ".tiff",
    # 音频 / 视频
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".mp4", ".mkv", ".avi", ".mov", ".webm",
    # 文档（压缩/二进制格式）
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp",
    # 归档 / 二进制
    ".zip", ".gz", ".tar", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".db", ".sqlite",
}


def _is_binary_path(path: str) -> bool:
    """按扩展名判断是否为二进制/媒体文件（大小写不敏感）。"""
    ext = os.path.splitext(str(path or ""))[1].lower()
    return ext in _BINARY_EXTS


def _extract_paths(tool_name: str, arguments: Dict[str, Any]) -> List[str]:
    """从 filesystem 读类工具参数里提取目标路径列表。"""
    if not isinstance(arguments, dict):
        return []
    paths: List[str] = []
    raw = arguments.get("paths")
    if isinstance(raw, list):
        paths.extend(str(p) for p in raw if p)
    p = arguments.get("path")
    if isinstance(p, str) and p.strip():
        paths.append(p.strip())
    return paths


def intercept_binary_read(api_format: str, full_name: str, arguments: Dict[str, Any]) -> Any:
    """拦截 MCP 文件服务对二进制/媒体文件的读取：一律拒绝，且不读取文件内容。

    媒体原始数据绝不进入模型上下文；需要看图 / 读图文字时使用独立工具
    （generate_image / generate_video / extract_text_from_image）在后端读取。
    返回 None 表示不拦截；返回字符串表示拒绝读取的错误文本。
    """
    if not isinstance(full_name, str) or not full_name.startswith("mcp_filesystem_"):
        return None
    tool_name = full_name[len("mcp_filesystem_"):]
    if tool_name not in _FS_READ_TOOLS:
        return None
    paths = _extract_paths(tool_name, arguments)
    if not paths:
        return None
    binary = [p for p in paths if _is_binary_path(p)]
    if not binary:
        return None
    return ("错误：MCP 文件服务禁止读取二进制/媒体文件（%s）——媒体原始数据不会进入上下文，"
            "请勿再调用读文件工具读取该文件；图生图 / 图生视频请直接传入图片路径，由工具内部读取。"
            % binary[0])


def _is_readonly(server_name: str, mcp_tool: Any) -> bool:
    """MCP 工具是否只读：优先 readOnlyHint，缺失时按服务器名单兜底。"""
    annotations = getattr(mcp_tool, "annotations", None)
    hint = getattr(annotations, "readOnlyHint", None)
    if hint is not None:
        return bool(hint)
    if server_name == "filesystem":
        return mcp_tool.name in _FS_READ_TOOLS
    if server_name == "fetch":
        return mcp_tool.name in _FETCH_READ_TOOLS
    return False


def _tool_schema(mcp_tool: Any) -> Dict[str, Any]:
    """mcp 1.x 用 inputSchema，2.x 用 input_schema；统一为 engine 期望的 JSON schema。"""
    return getattr(mcp_tool, "input_schema", None) or getattr(mcp_tool, "inputSchema", None) or {}


# ---- 全局状态 ----

# 全局单例服务器记录（fetch/git + 用户配置服务器）：name -> rec
_GLOBAL_RECORDS: Dict[str, Dict[str, Any]] = {}
# Filesystem 多实例：规范化 cwd -> rec
_FS_INSTANCES: Dict[str, Dict[str, Any]] = {}
# Filesystem 实例引用计数：规范化 cwd -> 引用数（当前持有该 cwd 的 Agent 数量）
_FS_REFCOUNTS: Dict[str, int] = {}
# 默认 filesystem 实例（仓库根，启动即连接，常驻）
_FS_DEFAULT_KEY: Optional[str] = None
_LOOP: Optional[asyncio.AbstractEventLoop] = None
_STARTED = False
_SHUTDOWN_EVENT: Optional[asyncio.Event] = None
_INIT_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()  # 保护 _FS_INSTANCES / _FS_REFCOUNTS / _GLOBAL_RECORDS
_TOOL_CALL_TIMEOUT = 180.0
_FS_CONNECT_TIMEOUT = 60.0  # 首次调用时等待实例连接的最长时间
_FS_TOOLS_REGISTERED = False  # filesystem 工具只全局注册一次（执行时按 cwd 路由）


def _norm_cwd(cwd: Optional[str]) -> str:
    """规范化 cwd（与 engine.SimpleAgent 的 cwd 归一逻辑一致）。"""
    return os.path.normcase(realpath_clean(cwd or os.getcwd()))


def _new_record(cfg: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "builtin": bool(cfg.get("builtin")),
        "description": cfg.get("description", ""),
        "status": "error" if error else "connecting",
        "tool_count": 0,
        "error": error,
        "cfg": cfg,
    }


def _new_fs_record(key: str, keepalive: bool = False) -> Dict[str, Any]:
    return {
        "name": "filesystem",
        "cwd": key,
        "keepalive": keepalive,
        "builtin": True,
        "cfg": _filesystem_cfg(key),
        "status": "connecting",
        "tool_count": 0,
        "error": None,
        "session": None,
        "task": None,
    }


def initialize() -> None:
    """启动后台 MCP 事件循环并异步连接全局服务器（不阻塞 server 启动）。

    Filesystem 默认实例（仓库根）随启动预连接；其他 cwd 的实例由 acquire 按需启动。
    """
    global _STARTED
    with _INIT_LOCK:
        if _STARTED:
            if _LOOP is not None and _LOOP.is_running():
                return
            # 循环异常退出后允许重启
            _STARTED = False
        _STARTED = True
    threading.Thread(target=_run_loop, name="mcp-loop", daemon=True).start()


def acquire(cwd: Optional[str] = None) -> None:
    """Agent 创建时调用：为其 cwd 的 filesystem 实例增加引用，不存在则启动实例。"""
    key = _norm_cwd(cwd)
    loop = _LOOP
    if loop is None or not loop.is_running():
        return
    need_start = False
    with _STATE_LOCK:
        _FS_REFCOUNTS[key] = _FS_REFCOUNTS.get(key, 0) + 1
        if key not in _FS_INSTANCES:
            need_start = True
    if need_start:
        asyncio.run_coroutine_threadsafe(_start_fs_instance(key), loop)


def release(cwd: Optional[str] = None) -> None:
    """Agent 销毁时调用：为其 cwd 的 filesystem 实例释放引用，归零则停止实例。

    默认常驻实例（仓库根）只减引用不停止。
    """
    key = _norm_cwd(cwd)
    with _STATE_LOCK:
        count = _FS_REFCOUNTS.get(key, 0)
        if count > 1:
            _FS_REFCOUNTS[key] = count - 1
            return
        _FS_REFCOUNTS.pop(key, None)
        rec = _FS_INSTANCES.get(key)
        if rec is not None and rec.get("keepalive"):
            return  # 默认常驻实例不停止
        rec = _FS_INSTANCES.pop(key, None)
    if rec is not None and _LOOP is not None and _LOOP.is_running():
        asyncio.run_coroutine_threadsafe(_stop_fs_instance(rec), _LOOP)


def get_status() -> List[Dict[str, Any]]:
    """返回所有服务器/实例的连接状态（供 /api/mcp/status 使用）。

    全局服务器（fetch/git + 用户配置）按名字列出；filesystem 每个 cwd 实例一条，
    默认实例沿用名字 filesystem，其余以 filesystem@<路径> 区分。后端启动后新保存的
    配置会以未连接状态列出，提示重启后生效；被禁用的服务器（disabled: true）以
    disabled 状态列出，说明配置已保留但不会启动/注册工具。
    """
    merged: Dict[str, Dict[str, Any]] = dict(_GLOBAL_RECORDS)
    for key, rec in _FS_INSTANCES.items():
        name = "filesystem"
        if key != _FS_DEFAULT_KEY:
            name = "filesystem@%s" % _shorten(key)
        merged[name] = rec
    for cfg in _load_user_servers(include_disabled=True):
        if cfg["name"] not in merged:
            disabled = not cfg.get("enabled", True)
            merged[cfg["name"]] = {
                "builtin": False,
                "description": cfg["description"],
                "status": "disabled" if disabled else "error",
                "tool_count": 0,
                "error": None if disabled else "配置已保存，重启后端后生效",
            }
    return [
        {
            "name": name,
            "builtin": bool(rec.get("builtin")),
            "description": rec.get("description", ""),
            "status": rec.get("status", "idle"),
            "tool_count": int(rec.get("tool_count") or 0),
            "tools": list(rec.get("tools") or []),
            "error": rec.get("error"),
            "cwd": rec.get("cwd"),
        }
        for name, rec in sorted(merged.items(), key=lambda kv: (not kv[1].get("builtin"), kv[0]))
    ]


def _shorten(path: str, limit: int = 40) -> str:
    """把长路径缩短为中间省略的展示串（状态面板用）。"""
    if len(path) <= limit:
        return path
    head = path[: limit // 2 - 1]
    tail = path[-(limit // 2):]
    return "%s…%s" % (head, tail)


def get_registered_tool_names() -> List[str]:
    """返回已注册的 MCP 工具名（调试用）。"""
    return sorted(
        t.name for t in tool_registry.get_all_tools()
        if t.name.startswith("mcp_")
    )


# ---- 后台事件循环 ----

def _run_loop() -> None:
    global _LOOP
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _LOOP = loop
    try:
        loop.run_until_complete(_main())
    except Exception as e:  # pragma: no cover - 循环异常兜底
        for rec in list(_GLOBAL_RECORDS.values()) + list(_FS_INSTANCES.values()):
            if rec.get("status") == "connecting":
                rec["status"] = "error"
                rec["error"] = f"循环异常：{e}"
    finally:
        _LOOP = None


async def _main() -> None:
    global _SHUTDOWN_EVENT, _FS_DEFAULT_KEY
    _SHUTDOWN_EVENT = asyncio.Event()
    if not MCP_SDK_AVAILABLE:
        err = "MCP Python SDK 未安装（pip install mcp）"
        for cfg in _collect_servers():
            _GLOBAL_RECORDS[cfg["name"]] = _new_record(cfg, err)
        return
    # 全局单例：fetch/git + 用户配置服务器
    for cfg in _collect_servers():
        _GLOBAL_RECORDS[cfg["name"]] = _new_record(cfg)
    for name in list(_GLOBAL_RECORDS):
        _GLOBAL_RECORDS[name]["task"] = asyncio.create_task(_connect_one(name))
    # Filesystem 默认实例：仓库根（保持设置页立即可见已连接；Agent cwd=仓库根时直接复用）
    _FS_DEFAULT_KEY = _norm_cwd(_repo_root())
    await _start_fs_instance(_FS_DEFAULT_KEY, keepalive=True)
    # 保持循环运行；进程退出时 daemon 线程随之终止
    await _SHUTDOWN_EVENT.wait()
    for rec in list(_GLOBAL_RECORDS.values()) + list(_FS_INSTANCES.values()):
        task = rec.get("task")
        if task is not None:
            task.cancel()


async def _start_fs_instance(key: str, keepalive: bool = False) -> None:
    """在事件循环内创建 filesystem 实例并启动连接任务（重复调用幂等）。"""
    with _STATE_LOCK:
        if key in _FS_INSTANCES:
            return
        rec = _new_fs_record(key, keepalive)
        _FS_INSTANCES[key] = rec
    rec["task"] = asyncio.create_task(_connect_fs_instance(key))


async def _connect_one(name: str) -> None:
    """连接单个全局服务器（fetch/git / 用户配置）并注册其工具；长驻会话直到关闭。"""
    rec = _GLOBAL_RECORDS[name]
    cfg = rec["cfg"]
    try:
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg["args"],
            env=cfg["env"] or None,
            cwd=_user_path_base(),  # 子进程以项目根目录（安装版为安装目录）为当前目录
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                mcp_tools = result.tools
                _register_tools(name, cfg, mcp_tools)
                rec["status"] = "connected"
                rec["tool_count"] = len(mcp_tools)
                rec["tools"] = [t.name for t in mcp_tools]
                rec["error"] = None
                rec["session"] = session
                # 会话长驻：挂起直到关闭
                await _SHUTDOWN_EVENT.wait()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        rec["status"] = "error"
        rec["tool_count"] = 0
        rec["error"] = str(e)[:500]
        rec["session"] = None


async def _connect_fs_instance(key: str) -> None:
    """连接指定 cwd 的 filesystem 实例；工具只注册一次，长驻直到被 release 停止。"""
    global _FS_TOOLS_REGISTERED
    rec = _FS_INSTANCES.get(key)
    if rec is None:
        return
    cfg = rec["cfg"]
    try:
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg["args"],
            env=cfg["env"] or None,
            cwd=_user_path_base(),  # 子进程以项目根目录（安装版为安装目录）为当前目录
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                mcp_tools = result.tools
                if not _FS_TOOLS_REGISTERED:
                    _register_tools("filesystem", cfg, mcp_tools)
                    _FS_TOOLS_REGISTERED = True
                rec["status"] = "connected"
                rec["tool_count"] = len(mcp_tools)
                rec["tools"] = [t.name for t in mcp_tools]
                rec["error"] = None
                rec["session"] = session
                # 会话长驻：直到任务被取消（release 或关闭）
                await asyncio.Event().wait()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        rec["status"] = "error"
        rec["tool_count"] = 0
        rec["error"] = str(e)[:500]
        rec["session"] = None


async def _stop_fs_instance(rec: Dict[str, Any]) -> None:
    """停止一个 filesystem 实例（取消其连接任务，触发 stdio 清理）。"""
    task = rec.get("task")
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    rec["status"] = "idle"
    rec["session"] = None


# ---- 动态注册 ----

# MCP 工具使用说明附加（防止模型组合乱用：fetch 结果直接阅读、读文件后直接阅读、写文件匹配扩展名）
# 文件内容读取类工具（读取后直接阅读，不要组合 SemanticSearch）
_FS_FILE_READ_TOOLS = {"read_file", "read_text_file", "read_media_file", "read_multiple_files"}
_FS_WRITE_TOOLS = {"write_file"}


def _with_usage_note(server_name: str, tool_name: str, description: str) -> str:
    """按服务器/工具在描述末尾附加使用说明；无附加时原样返回。"""
    if server_name == "fetch":
        return (description
                + " 默认返回网页转 Markdown 的正文（会丢弃图片 <img> 的 src）；"
                  "需要提取页面图片或 HTML 结构时，请使用 raw=true 获取原始 HTML，"
                  "从 <img src=\"...\"> 中提取图片链接。"
                  "可传 max_length（字符数）控制单次返回长度，上限为工具结果封顶值 "
                  "（BIGCODEX_TOOL_RESULT_MAX_CHARS，默认 30000），超长内容用 start_index 分段获取。"
                  "直接阅读回答，不要后续调用 SemanticSearch。")
    if server_name == "git":
        return description + " repo_path 填当前工作目录（Agent 的 cwd）对应的 Git 仓库根路径。"
    if server_name == "filesystem":
        if tool_name in _FS_FILE_READ_TOOLS:
            return description + " 读取后直接阅读回答，不要立即调用 SemanticSearch。"
        if tool_name in _FS_WRITE_TOOLS:
            return description + " 写入纯文本请用 .txt/.md/.py 等文本扩展名，不要用 .docx/.jpg 等二进制扩展名。"
    return description


def _make_executor(server_name: str, mcp_tool_name: str, full_name: str):
    def execute(context: Any, **kwargs: Any) -> str:
        cwd = None
        agent = getattr(context, "agent", None)
        if agent is not None:
            cwd = getattr(agent, "cwd", None)
        return _call_tool_sync(server_name, mcp_tool_name, full_name, kwargs, cwd)
    return execute


def _register_tools(server_name: str, cfg: Dict[str, Any], mcp_tools: List[Any]) -> None:
    """把服务器的 MCP 工具包装成 Tool 注册进全局注册表。

    命名：mcp_<server>_<tool>，避免与内置工具（如 delete_file）冲突。
    只读工具聊天/工作模式均可用；写类工具仅工作模式。
    工具名全局唯一，filesystem 调用时按执行上下文的 cwd 路由到对应实例。
    """
    for mcp_tool in mcp_tools:
        tool_name = mcp_tool.name
        full_name = f"mcp_{server_name}_{tool_name}"
        readonly = _is_readonly(server_name, mcp_tool)
        tool = Tool(
            name=full_name,
            description=_with_usage_note(
                server_name, tool_name,
                f"[{server_name}] {(mcp_tool.description or f'MCP 工具 {tool_name}').strip()}"),
            parameters=_tool_schema(mcp_tool),
            modes=[ToolMode.WORK, ToolMode.CHAT] if readonly else [ToolMode.WORK],
            permission_level=ToolPermission.READONLY if readonly else ToolPermission.DEFAULT,
        )
        tool.execute = _make_executor(server_name, tool_name, full_name)
        tool_registry.register_dynamic(tool)


# ---- 工具调用桥接 ----

def _resolve_record(server_name: str, cwd: Optional[str]) -> Optional[Dict[str, Any]]:
    """按服务器名解析调用目标：filesystem 按 cwd 路由到实例，其余走全局记录。"""
    if server_name == "filesystem":
        return _FS_INSTANCES.get(_norm_cwd(cwd))
    return _GLOBAL_RECORDS.get(server_name)


def _call_tool_sync(server_name: str, mcp_tool_name: str, full_name: str,
                    arguments: Dict[str, Any], cwd: Optional[str] = None) -> str:
    """同步桥接：把 engine 的同步 execute 转发到后台 MCP 事件循环。"""
    if server_name == "fetch" and mcp_tool_name == "fetch":
        # 注入/钳制 max_length：LLM 未传时用我们的工具结果上限，传了也不允许超限，
        # 避免单次返回内容超过 engine 的 TOOL_RESULT_MAX_CHARS 被二次截断。
        try:
            current = int(arguments.get("max_length") or 0)
        except (TypeError, ValueError):
            current = 0
        cap = min(FETCH_MAX_LENGTH_CAP, _FETCH_HARD_CAP)
        if current <= 0:
            arguments["max_length"] = cap
        else:
            arguments["max_length"] = min(current, cap)
    loop = _LOOP
    if loop is None or not loop.is_running():
        return f"错误：MCP 服务未启动，无法调用 {full_name}"
    rec = _resolve_record(server_name, cwd)
    if rec is None or rec.get("session") is None:
        # Filesystem：目标 cwd 实例缺失（如未经 acquire 的旧路径 Agent），补启动并等待连接
        if server_name == "filesystem":
            return _lazy_ensure_and_call(loop, server_name, mcp_tool_name, full_name, arguments, cwd)
        return f"错误：MCP 服务器 {server_name} 未连接，无法调用 {full_name}"
    future = asyncio.run_coroutine_threadsafe(
        _call_tool_async(server_name, mcp_tool_name, arguments, cwd), loop)
    try:
        return future.result(timeout=_TOOL_CALL_TIMEOUT)
    except Exception as e:
        return f"错误：MCP 工具 {full_name} 执行失败：{e}"


def call_tool(server_name: str, tool_name: str, arguments: Dict[str, Any],
              cwd: Optional[str] = None) -> str:
    """同步调用指定服务器的 MCP 工具（供本地处理器等模块直接使用）。"""
    return _call_tool_sync(server_name, tool_name, f"mcp_{server_name}_{tool_name}", arguments, cwd)


def _lazy_ensure_and_call(loop: asyncio.AbstractEventLoop, server_name: str,
                          mcp_tool_name: str, full_name: str, arguments: Dict[str, Any],
                          cwd: Optional[str]) -> str:
    """filesystem 实例缺失时补启动并轮询等待连接，然后转发调用。"""
    rec = _resolve_record(server_name, cwd)
    if rec is None:
        acquire(cwd)
        rec = _resolve_record(server_name, cwd)
    deadline = time.monotonic() + _FS_CONNECT_TIMEOUT
    while rec is not None and rec.get("session") is None and time.monotonic() < deadline:
        time.sleep(0.2)
        rec = _resolve_record(server_name, cwd)
    if rec is None or rec.get("session") is None:
        return f"错误：MCP 文件服务器尚未就绪，无法调用 {full_name}（实例连接超时）"
    future = asyncio.run_coroutine_threadsafe(
        _call_tool_async(server_name, mcp_tool_name, arguments, cwd), loop)
    try:
        return future.result(timeout=_TOOL_CALL_TIMEOUT)
    except Exception as e:
        return f"错误：MCP 工具 {full_name} 执行失败：{e}"


async def _call_tool_async(server_name: str, mcp_tool_name: str, arguments: Dict[str, Any],
                           cwd: Optional[str] = None) -> str:
    rec = _resolve_record(server_name, cwd)
    session = rec.get("session") if rec else None
    if session is None:
        return f"错误：MCP 服务器 {server_name} 未连接"
    try:
        result = await session.call_tool(mcp_tool_name, arguments=arguments)
    except Exception as e:
        return f"错误：MCP 工具调用失败：{e}"
    return _format_result(result)


def _format_result(result: Any) -> Any:
    """把 MCP CallToolResult 转成文本；ImageContent 块保留为媒体结构。

    - 含图片块（ImageContent：data + mimeType）时返回 {"__media__": [...], "text": ...}，
      由 engine 在 Anthropic 路径转成 tool_result 图片块；
    - 否则返回纯文本。
    """
    media: List[Dict[str, Any]] = []
    parts: List[str] = []
    for block in (result.content or []):
        data = getattr(block, "data", None)
        mime = getattr(block, "mimeType", None)
        if data is not None and mime is not None:
            media.append({"media_type": str(mime), "data": str(data)})
            continue
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(str(block))
    text = "\n".join(p for p in parts if p).strip()
    is_error = getattr(result, "isError", None)
    if is_error is None:
        is_error = getattr(result, "is_error", False)
    if is_error:
        return f"错误：{text or 'MCP 工具返回错误'}"
    if media:
        out: Dict[str, Any] = {"__media__": media}
        if text:
            out["text"] = text
        return out
    return text or "(空结果)"
