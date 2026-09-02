"""
Local Deepseek 配置：网页版 DeepSeek（chat.deepseek.com）的 OpenAI 兼容本地代理
=============================================================================
本地代理是转发器：只把最新提问转发给网页版 DeepSeek，上下文由网页版维护
（single_turn=True）。api_url 基本地址不可达时，chat 处理器视为无法处理。
"""
import os

LOCAL_DEEPSEEK_CONFIG = {
    "name": "Local Deepseek",
    "api_key": os.environ.get("LocalDeepseek_API_KEY", "sk-deepseek-proxy"),
    "api_url": "http://localhost:9527/v1/chat/completions",
    "model": "deepseek-chat",
    "max_tokens": 32768,
    "stream": False,
    # 要求代理返回网页版 SSE 原始流（chat.py 自行解析正文；为将来透传引用/完整下游流预留）
    "raw_sse": True,
    # 本地代理是转发器：只把最新提问转发给网页版 DeepSeek，上下文由网页版维护
    "single_turn": True,
}

# Local Deepseek 允许调用的本地工具白名单（文本协议教学用）。
# 新增工具只需在此列表追加工具名（工具定义来自 tool_registry，教学文本自动带上）。
# 内置工具直接用工具名（如 generate_image / generate_video）；
# MCP 工具必须填完整注册名：mcp_<server名>_<工具名>，例如
#   "mcp_filesystem_list_directory"、"mcp_fetch_fetch_url"。
# server 名取自 mcp.json 的键名，工具名是该服务器暴露的原始工具名；
# 可运行 `python -c "from tools import tool_registry; import mcp_manager;
# tool_registry.load_all_tools(); print([t.name for t in
# tool_registry.get_all_tools() if t.name.startswith('mcp_')])"` 查看全部已注册 MCP 工具。
LOCAL_DEEPSEEK_ALLOWED_TOOLS = ["generate_image", 
                                "generate_video", 
                                "extract_from_urls",
                                "search_and_extract",
                                "mcp_12306车票查询_query_tickets", 
                                "mcp_12306车票查询_query_schedule",
                                "mcp_12306车票查询_search_station"]

# 透传给网页版 DeepSeek 的技能白名单：命中这些技能时，把技能描述全文（SKILL.md）
# 连同用户请求一起透传，让网页版按技能规范执行（如 image-builder / video-builder
# 的提示词润色、展示规范，news-media-digest 的新闻主内容图片提取等）；
# 未命中时仍只透传真实用户输入。
LOCAL_DEEPSEEK_PASSTHROUGH_SKILLS = ["image-builder", "video-builder", "news-media-digest"]
# 是否在程序启动后第一次使用 LocalLLM 时自动发送工具使用教学（静默，不展示回复）
LOCAL_DEEPSEEK_TEACH_ON_START = True
# 单轮对话中工具调用循环的最大轮数（防止网页版无限调工具）
LOCAL_DEEPSEEK_MAX_TOOL_ROUNDS = 3

# 单次对话请求超时（秒）：网页版代理响应可能较慢
LOCAL_DEEPSEEK_TIMEOUT = 120
# 连通性探测超时（秒）：can_handle 用短超时快速判断代理是否可达
LOCAL_DEEPSEEK_PROBE_TIMEOUT = 1.5

# 请求代理返回 SSE 原始流时使用的自定义请求头
LOCAL_DEEPSEEK_RAW_SSE_HEADER = "X-BigCodex-Raw-SSE"
