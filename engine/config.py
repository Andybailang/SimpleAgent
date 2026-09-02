"""engine.config — 引擎配置常量与辅助函数。

集中存放 SimpleAgent 各流程共享的纯模块级常量、环境变量解析与重试/预算/工具白名单等。
"""
import os
from typing import Any, Dict, List, Optional, Tuple

def _env_int(name: str, default: int) -> int:
    """读取环境变量并转换为正整数，解析失败时回退到默认值。"""
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default
def _env_float(name: str, default: float) -> float:
    """读取环境变量并转换为正浮点数，解析失败时回退到默认值。"""
    try:
        return max(0.1, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default
# 中文字符→token 折算系数：全中文 1 字符 ≈ 0.6 token（供 /status 与上下文用量估算）
CJK_TOKENS_FACTOR = 0.6
def _estimate_output_tokens(text: str) -> int:
    """网关未回传 output_tokens 时，按输出文本粗估 token 数（中文按 1 字≈0.6 token、ASCII 约 4 字符/token）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    ascii_chars = len(text) - cjk
    return max(1, round(cjk * CJK_TOKENS_FACTOR + ascii_chars / 4.0))
# 单条用户消息内允许的最大工具调用轮数（每轮 LLM 决策一次，可包含多个工具调用）。
# 可通过环境变量 BIGCODEX_MAX_TOOL_ROUNDS 覆盖。
MAX_TOOL_ROUNDS = _env_int("BIGCODEX_MAX_TOOL_ROUNDS", 200)

# 工具轮次落盘的最小间隔（毫秒）：每轮工具调用完整写回 self.messages 后，按此间隔节流写盘，
# 保证长任务（大量工具轮 / 长时间思考构成的多轮工具链）在进程被意外终止时，也能从磁盘
# 恢复已完成轮次的上下文，而不是整轮上下文丢失。可通过环境变量 BIGCODEX_PERSIST_INTERVAL_MS 覆盖。
PERSIST_INTERVAL_MS = _env_int("BIGCODEX_PERSIST_INTERVAL_MS", 3000)

# 限流重试配置（参照 Claude Code）：默认最多重试 10 次，间隔 1s→3s→5s→15s→…封顶 180s。
# 智谱错误码 1302（账户速率限制）、1305（平台服务过载）均返回 HTTP 429，429 为通用限流信号；
# 1302/1305 是智谱特有业务码，部分网关可能只带业务码不带 429，故判定时同时按状态码与响应体文本识别。
RETRY_MAX_ATTEMPTS = _env_int("BIGCODEX_MAX_RETRIES", 10)
RETRY_BASE_DELAY = _env_float("BIGCODEX_RETRY_BASE_DELAY", 1.0)
RETRY_MAX_DELAY = 180.0
_RETRY_DELAY_TEMPLATE = [1.0, 3.0, 5.0, 15.0, 30.0, 60.0, 120.0, 180.0]
# 缓存监控序列化时剔除的纯请求参数（非 prompt 内容，不参与前缀缓存比较）
_CACHE_PARAM_KEYS = frozenset({"model", "max_tokens", "stream", "temperature", "thinking"})
def _retry_delays() -> List[float]:
    """返回每次重试前的等待秒数序列，长度等于 RETRY_MAX_ATTEMPTS，封顶 RETRY_MAX_DELAY。"""
    delays = [min(d * RETRY_BASE_DELAY, RETRY_MAX_DELAY) for d in _RETRY_DELAY_TEMPLATE]
    if len(delays) < RETRY_MAX_ATTEMPTS:
        delays.extend([RETRY_MAX_DELAY] * (RETRY_MAX_ATTEMPTS - len(delays)))
    return delays[:RETRY_MAX_ATTEMPTS]
def _is_rate_limited(status_code: int, body: str) -> bool:
    """判断响应是否属于可重试的限流/过载错误（HTTP 429 通用；智谱业务码 1302/1305）。"""
    if status_code == 429:
        return True
    return ("1302" in body) or ("1305" in body)
def _is_openai_retryable_error(e: Exception) -> bool:
    """判断 OpenAI SDK 抛出的异常是否属于可重试的限流/过载。"""
    status = getattr(e, "status_code", None)
    if status == 429:
        return True
    detail = ""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            detail += (resp.text if hasattr(resp, "text") else str(resp)) + " "
        except Exception:
            pass
    detail += str(e)
    return ("1302" in detail) or ("1305" in detail)
def _retry_event(attempt: int, delay_ms: int, error_status: int, error: str) -> Dict[str, Any]:
    """构造限流重试事件（契约与 tokenicode CLI 的 system/api_retry 一致，前端直接消费）。"""
    return {
        "type": "system",
        "subtype": "api_retry",
        "attempt": attempt,
        "max_retries": RETRY_MAX_ATTEMPTS,
        "retry_delay_ms": delay_ms,
        "error_status": error_status,
        "error": error,
    }
# 根目录 AGENTS.md 自动加载进系统提示，单次最大字符数
AGENTS_MD_MAX_CHARS = 20000

# 思考深度 -> Anthropic thinking budget_tokens 映射（off 不传 thinking 参数）
THINKING_BUDGETS = {
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}

# /compact 压缩时保留的最近消息条数（可通过环境变量 BIGCODEX_COMPACT_KEEP_LAST 覆盖）
COMPACT_KEEP_LAST = _env_int("BIGCODEX_COMPACT_KEEP_LAST", 6)

# /compact 保留窗口的字符体积预算（超出后从最旧一端丢弃；图片按 data URL/base64 长度计入）
COMPACT_RECENT_MAX_CHARS = _env_int("BIGCODEX_COMPACT_RECENT_MAX_CHARS", 60000)

# 付费模型本地裁剪保留窗口的预估 token 预算：从最近一条开始累计，最多
# COMPACT_KEEP_LAST 条，累计 token（= 各消息字符数 × CJK_TOKENS_FACTOR，本地估算）
# 不超过该值时持续并入；某条会使累计超过该值时立即停止（该条不并入）。
# 极端情况下最近一条本身已超过预算，则一条都不保留。
PAID_COMPACT_RECENT_MAX_TOKENS = _env_int("BIGCODEX_PAID_COMPACT_RECENT_MAX_TOKENS", 150000)

# /compact 压缩后保留上下文的硬性上限：即使最新一条被保护（如超大图片），
# 超过该上限时仍会把超大图片块降级为文本占位，确保压缩后不再超限。
COMPACT_RECENT_HARD_CAP_CHARS = _env_int("BIGCODEX_COMPACT_RECENT_HARD_CAP_CHARS", 300000)

# /compact 生成摘要的最大字符数（超出截断）
COMPACT_MAX_SUMMARY_CHARS = 8000

# 付费模型本地裁剪的占位说明（不调用 LLM 生成摘要，避免一次近乎全量的 prompt
# 造成缓存完全未命中、token 浪费；直接交给模型一个“旧消息已裁剪”的简短说明）。
LOCAL_COMPACT_NOTICE = ("[付费模型省 token：较早的 {n} 条对话已本地裁剪，"
                        "当前仅保留最近对话，请基于最近内容继续]")

# 自动压缩（发送前检测）的默认上下文窗口与触发阈值。
# 由 server.py 从 config.env（CONTEXT_MAX_TOKENS / COMPACT_THRESHOLD）注入
# 到 agent.context_max_tokens / agent.compact_threshold；这里仅提供程序兜底默认值。
# 触发时本地估算完整请求体的 token 数（不等服务端 usage，防缓存命中率误导）。
AUTO_COMPACT_CONTEXT_MAX_TOKENS = _env_int("CONTEXT_MAX_TOKENS", 1000000)
AUTO_COMPACT_THRESHOLD = _env_float("COMPACT_THRESHOLD", 0.9)

# 工具结果文本封顶（字符数）：防止 directory_tree 等超大输出打爆上下文；
# 截断时提示模型改用更精确的查询。图片等媒体结果按图片 token 计费，不在此限制内。
TOOL_RESULT_MAX_CHARS = _env_int("BIGCODEX_TOOL_RESULT_MAX_CHARS", 30000)

# 编程模式（work）下只拼接给模型的内置工具白名单：文件操作 + Bash + TodoWrite。
# 其余内置工具（语义搜索 / 图片视频 / 网页提取等）与外置 MCP 工具一律不拼，节省每轮请求 token。
# 注意：此白名单只影响「拼接给模型看的 tool 定义」，不影响工具本身的注册与执行。
WORK_MODE_TOOL_WHITELIST = {
    "Read", "Write", "Edit", "LS", "Grep", "Glob",
    "delete_file", "create_directory", "rename_file",
    "Bash", "TestRunner", "TodoWrite", "extract_text_from_image",
    "AskUserQuestion",
}

# 编程模式（work）额外放行的外置 MCP 工具：默认外置 MCP 一律不拼以省 token，
# 仅显式列出需要的能力。这是 gitnexus 代码图谱工具里对「单项目编码」最常用的子集；
# 刻意排除 explain / pdg_query（需 analyze --pdg 才有结果）、group_*（多仓库）、
# cypher / check / shape_check / api_impact（低频或偏冷门）；tool_map 仅用于查看
# 工具定义，编码场景低频，同样不拼。
# list_repos 列出已索引仓库，多项目时用于定位当前工作项目（配合 repo 参数使用）。
WORK_MODE_ALLOWED_MCP_TOOLS = {
    "mcp_gitnexus_query",
    "mcp_gitnexus_context",
    "mcp_gitnexus_impact",
    "mcp_gitnexus_detect_changes",
    "mcp_gitnexus_rename",
    "mcp_gitnexus_trace",
    "mcp_gitnexus_list_repos",
    "mcp_gitnexus_route_map",
}
