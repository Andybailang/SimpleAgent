"""
Bash 工具：在项目目录执行 shell 命令
"""
import os
import re
import subprocess
import sys
from typing import List
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext

COMMAND_TIMEOUT = 60
COMMAND_OUTPUT_LIMIT = 20000

# 只读模式下 Bash 命令内容级放行（防误伤，非安全边界）：
# 白名单首命令本身只读；git 需再匹配只读子命令；解释器仅放行版本查询。
READONLY_CMD_ALLOWLIST = {
    "ls", "dir", "cat", "type", "find", "rg", "grep", "findstr",
    "where", "where.exe", "which", "tree", "more",
    "get-content", "get-childitem", "select-string",
}
READONLY_GIT_SUBCOMMANDS = {
    "status", "log", "diff", "show", "branch", "remote", "ls-files",
    "grep", "blame", "tag", "rev-list", "shortlog", "describe", "help", "version",
}
READONLY_INTERPRETERS = {"python", "python3", "py", "node", "npm"}
READONLY_VERSION_FLAGS = {"--version", "-version", "-v", "-V"}
# 白名单命令内仍禁止的危险参数（如 find -exec/-delete 会写文件）
READONLY_CMD_BLOCK_ARGS = {
    "find": {"-exec", "-execdir", "-delete", "-ok"},
}
# 常见写操作词：仅用于拒绝时的提示信息
READONLY_BLOCK_TOKENS = {
    "rm", "del", "erase", "mv", "move", "ren", "rename", "mkdir", "md",
    "rd", "rmdir", "touch", "copy", "xcopy", "robocopy", "icacls", "attrib",
    "chmod", "chown", "mklink", "add", "commit", "push", "pull", "reset",
    "checkout", "clean", "restore", "stash", "merge", "rebase", "init",
    "clone", "apply", "cherry-pick", "revert", "switch", "gc", "prune",
    "fetch", "install", "uninstall", "upgrade", "update", "pip", "npm",
    "pnpm", "yarn", "cargo", "brew", "conda", "curl", "wget", "certutil",
    "set-content", "add-content", "out-file", "new-item", "copy-item",
    "move-item", "remove-item", "taskkill", "shutdown", "format", "reg", "tee",
}


def _env_list(name: str, default: str = "") -> List[str]:
    """读取逗号分隔的环境变量列表（用于扩展只读放行配置）。"""
    raw = os.environ.get(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def is_readonly_command(command: str) -> bool:
    """只读模式下判断 Bash 命令是否可放行：白名单命令 + 无输出重定向。
    启发式防误伤，不是安全边界（解释器 eval、编码混淆等仍可绕过）。"""
    if not command or not command.strip():
        return False
    low = command.strip().lower()
    # 输出重定向（> 前不是 < 或 -，排除 << 输入重定向与 -> 箭头）一律禁止
    if re.search(r"(?<!<)(?<!-)>", low):
        return False
    extra_heads = set(_env_list("BIGCODEX_READONLY_ALLOW_COMMANDS"))
    block_extra = set(_env_list("BIGCODEX_READONLY_BLOCK_TOKENS"))
    for seg in re.split(r"[|;&]", low):
        seg = seg.strip().strip('"')
        if not seg:
            continue
        parts = seg.split()
        head = parts[0].strip('"\'`')
        if not head:
            return False
        if head in READONLY_CMD_ALLOWLIST or head in extra_heads:
            if any(a in parts for a in READONLY_CMD_BLOCK_ARGS.get(head, ())):
                return False
            continue
        if head == "git":
            if len(parts) == 1 or parts[1] in ("--version", "version"):
                continue
            if parts[1] in READONLY_GIT_SUBCOMMANDS:
                continue
            if parts[1] == "config" and any(f in parts for f in ("--get", "--list", "-l", "--get-regexp")):
                continue
            return False
        if head in READONLY_INTERPRETERS:
            if len(parts) == 2 and parts[1] in READONLY_VERSION_FLAGS:
                continue
            return False
        return False
    # 用户自定义黑名单词兜底（对白名单段也生效）
    if block_extra:
        all_tokens = set()
        for seg in re.split(r"[|;&]", low):
            all_tokens.update(seg.strip().strip('"').split())
        if all_tokens & block_extra:
            return False
    return True


def _decode_output(raw: bytes) -> str:
    """Decode command output bytes, trying common encodings."""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """终止命令进程（Windows 需要连带杀掉子进程树）"""
    try:
        if sys.platform == "win32":
            subprocess.run(
                f"taskkill /PID {proc.pid} /T /F",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class BashTool(BaseTool):
    """Bash 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="Bash",
            description=(
                "在项目目录执行 shell 命令（如 git、npm、python）。参数：command（要执行的命令，字符串）、"
                "timeout（可选，超时毫秒，默认 60000，最大 300000）。命令非交互执行，输出过长会截断。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时毫秒（默认 60000，最大 300000）"
                    }
                },
                "required": ["command"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.DEFAULT,
        )

    @classmethod
    def execute(cls, context: ToolContext, command: str, timeout: int = COMMAND_TIMEOUT) -> str:
        """执行 shell 命令（在项目目录内，非交互，含超时与输出截断）"""
        if not command or not command.strip():
            return "错误：command 不能为空"
        # timeout 按 Claude Code 习惯为毫秒；兼容旧调用传秒（<1000 视为秒）
        try:
            t = int(timeout)
        except (TypeError, ValueError):
            t = COMMAND_TIMEOUT
        t_ms = t if t >= 1000 else t * 1000
        timeout_s = max(1, min(t_ms, 300000) // 1000)
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=context.cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                out_bytes, err_bytes = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc)
                return f"命令超时（>{timeout_s} 秒），已终止：{command}"
            out = _decode_output(out_bytes)
            err = _decode_output(err_bytes)
            combined = out if not err else (f"{out}\n[stderr]\n{err}" if out else err)
            if len(combined) > COMMAND_OUTPUT_LIMIT:
                combined = combined[:COMMAND_OUTPUT_LIMIT] + "\n...[输出过长，已截断]"
            return f"命令: {command}\n退出码: {proc.returncode}\n输出:\n{combined}"
        except FileNotFoundError:
            return f"错误：找不到命令或 shell：{command}"
        except Exception as e:
            return f"执行命令出错: {str(e)}"
