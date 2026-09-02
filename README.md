# 简易 AI Agent 核心

基于 OpenAI API 的简易 AI 编程助手，支持文件操作和代码搜索。

## 功能

- **对话交互**：自然的语言对话
- **文件读取**：读取任何文本文件内容
- **文件写入**：创建或覆盖文件
- **目录列表**：列出目录下的文件和子目录
- **代码搜索**：递归搜索代码中的匹配行

## 安装

### 1. 安装依赖

```bash
cd src/agent
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 创建 .env 并填写（变量说明见根目录 README「配置」一节）

# 编辑 .env 文件，填入你的 API 密钥
# OPENAI_API_KEY=your-api-key
# OPENAI_BASE_URL=your-base-url
# OPENAI_MODEL_NAME=your-model
```

## 使用方式

### 命令行模式

```bash
# Use launcher script (recommended for Windows)
run_cli.bat

# Or run directly
python cli.py
```

### 基本命令

- `/exit` 或 `/quit` - 退出程序
- `/clear` - 清空对话历史

### Python 代码中使用

```python
from engine import SimpleAgent
import os

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 初始化 Agent
agent = SimpleAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model_name=os.getenv("OPENAI_MODEL_NAME")
)

# 开始对话
response = agent.chat("帮我列出当前目录的文件")
print(response)

# 获取消息历史
history = agent.get_messages()
```

## 工具说明

Agent 内置 12 个工具：文件读写、目录操作、正则/语义搜索、`Bash` 命令执行、任务规划等。所有相对路径都基于工作目录（`cwd`，默认进程当前目录）；绝对路径必须在工作目录范围内，否则会被拒绝：

1. **Read(file_path, offset?, limit?)** - 读取文件
   - 返回：文件内容或错误信息
2. **Write(file_path, content)** - 创建或覆盖写入文件（自动创建父目录）
   - 返回：成功或错误信息
3. **Edit(file_path, old_string, new_string, replace_all?)** - 精确替换文件中的文本片段（默认仅第一处，replace_all=true 替换全部）
   - 返回：成功或错误信息
4. **delete_file(path)** - 删除文件（拒绝删除目录）
   - 返回：成功或错误信息
5. **LS(path?)** - 列出目录下的文件和子目录
   - 返回：目录列表
6. **create_directory(path)** - 创建目录（可一次创建多级）
   - 返回：成功或错误信息
7. **rename_file(src, dest)** - 重命名或移动文件/目录
   - 返回：成功或错误信息
8. **Grep(pattern, path?)** - 搜索代码
   - 参数：pattern（正则表达式），path（可选，搜索路径）
   - 返回：匹配的代码行列表
9. **Bash(command, timeout?)** - 在项目目录执行 shell 命令（git、npm、python 等）
   - 参数：command（命令字符串），timeout（可选，超时秒数，默认 60，最大 300）
   - 返回：退出码与输出（输出超过 20000 字符会截断）

## 语义搜索（SemanticSearch）

Agent 内置 `SemanticSearch` 工具：用自然语言描述要找的内容，通过嵌入模型把 query 与工作区文本分块向量化，按余弦相似度返回 Top-K 相关片段（含文件路径与行号）。适合做“语义版 Grep”，例如查询“处理附件上传的逻辑”这类无法用正则表达的需求。

- 参数：`query`（必填）、`path`（可选，默认当前目录）、`glob`（可选，如 `*.md`）、`top_k`（可选，默认 5）。
- 嵌入模型：默认 `BAAI/bge-m3`（SiliconFlow），可在 `src/agent/semantic.env` 中修改。
- 配置：复制 `semantic.env.example` 为 `semantic.env` 后按需修改；优先级为进程环境变量 > `semantic.env` > 内置默认值。关键项：
  - `SEMANTIC_SEARCH_API_KEY`：API key（留空则回退环境变量 `SILICONFLOW_API_KEY`）
  - `SEMANTIC_SEARCH_MODEL`：嵌入模型名
  - `SEMANTIC_SEARCH_ENABLED`：是否启用（默认 true）
  - `SEMANTIC_SEARCH_TOP_K` / `SEMANTIC_SEARCH_CHUNK_SIZE` 等：检索与分块参数
- 未配置 key 或未启用时，工具会返回明确的错误提示，不影响其他功能。

## 权限模式

Agent 支持 3 种工具权限模式（默认 `default`；通过 `POST /api/project/permission` 设置，body 为 `{"project": "<项目目录>", "mode": "default|readonly|full"}`，对该项目下所有会话生效）：

- `default`（默认目录内）：文件工具只能访问任务目录及子目录，越界路径被拒绝。
- `readonly`（只读）：只允许 `Read` / `LS` / `Grep` / `Glob` 等只读工具，其余工具（写入/编辑/删除/Bash 写命令等）一律拒绝。
- `full`（完全访问）：可读写任意目录，不做路径限制。

前端项目右键菜单「权限设定」可在三个模式间切换，设定对项目下所有会话生效。

## 项目结构

```
src/agent/
├── engine/            # Agent core engine (split into a package: config/util/traffic/thinking/prompt/tools_runtime/context/openai_flow/anthropic_flow/response_flow/stream/agent)
├── local_handlers/    # LocalLLM 本地处理器（见下方「LocalLLM 本地处理」）
├── tools/             # Tool system (one file per tool + registry/base/config, auto-loaded)
├── cli.py            # Command-line interface
├── run_cli.bat       # Windows launcher script
├── test_agent.py     # Test script
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## LocalLLM 本地处理（local_handlers）

LocalLLM 是一种**不调用远端 LLM** 的本地处理模式：用户选择 `LocalLLM` 模型后，
会话不再走 `engine.SimpleAgent` 的 OpenAI/Anthropic 请求，而是由
`server._handle_local_turn → local_handlers.handle_local_turn` 在本地匹配并执行处理器，
最后把结果伪造为标准事件流返回前端。

### 注册与匹配机制

`local_handlers/registry.py` 提供 `LocalHandlerRegistry`：

- 每个处理器继承 `base.BaseLocalHandler`，实现 `can_handle()`（是否接管本次输入）
  与 `handle()`（执行并返回文本）；
- 处理器在各自模块底部 `local_handler_registry.register(...)` **导入即注册**，
  按注册顺序依次匹配，第一个 `can_handle()` 返回 True 的执行；
- 都不匹配时注册表返回兜底提示（能力清单）；`__init__.py` 按固定顺序导入
  `pdf → ocr → text → chat`，因此优先级：**PDF > 图片 OCR > 文本透传 > DeepSeek 对话**；
- 每次回复末尾统一附加 `> ⚡ 以上内容由本地处理生成（LocalLLM）` 标记。

### 各处理器实现细节

#### PDF 处理器（pdf.py）

用户上传 `.pdf` 附件时接管：

1. 用 PyMuPDF 把每页渲染成 PNG（DPI 200），存入 `<cwd>/.bigcodex_uploads/pdf_pages/`；
2. **文字类 PDF**（整篇可提取文本 ≥ 100 字符）：直接返回 PyMuPDF 纯文本，不走 OCR；
3. **扫描类 PDF**（整篇文本过少）：走 PaddleOCR 逐页识别；
4. **混合型 PDF**：文本不足的页单独回退 OCR；
5. 回复中列出每页生成的图片路径（`pdf_<uuid>_pN.png`）。

#### 图片识别处理器（ocr.py）

用户附图（`isImage=true` 且带 `path`）时接管，对每张图调用 PaddleOCR
（`ocr_engine.get_ocr`，tiny 模型），按图片名分段返回识别文字。

#### 文本文件处理器（text.py）

用户上传或消息中带**文本类文件路径**时接管，把文件内容透传给 UI 展示：

- 支持常见文本/代码扩展名与 `Makefile`、`README` 等无扩展名文件名；
- `.doc/.docx` 用 `aspose-words-foss` 转 Markdown；
- 代码类文件用对应语言的 Markdown 围栏包裹（便于前端高亮）；
- 单文件上限 `MAX_TEXT_CHARS = 100_000` 字符，超出截断并提示；
- 图片/PDF 扩展名不在此列（分别由 OCR / PDF 处理器负责）。

#### DeepSeek 对话处理器（chat.py，Local Deepseek 兜底大脑）

所有其他输入（普通对话、搜索、技能调用等）的兜底处理器。它把用户消息转发给
**本地 Deepseek 代理**（`CodexWebProxy`，把 chat.deepseek.com 网页版包装成
OpenAI 兼容 API），并把代理回复原样回传。详见下方「与 Local Deepseek 的交互」。

### 与 Local Deepseek 的交互

#### 代理与配置（deepseek_config.py）

`LOCAL_DEEPSEEK_CONFIG` 定义代理接入参数：

```python
{
    "api_url": "http://localhost:9527/v1/chat/completions",  # CodexWebProxy
    "api_key": os.environ.get("LocalDeepseek_API_KEY", "sk-deepseek-proxy"),
    "model": "deepseek-chat",
    "max_tokens": 32768,
    "stream": False,
    "raw_sse": True,       # 要求代理返回网页版 SSE 原始流
    "single_turn": True,   # 只转发最新提问，上下文由网页版维护
}
```

关键设计：代理是**转发器**，网页版 DeepSeek 自己维护对话上下文（single_turn），
BigCodeX 侧**从不回传历史**，每次只发最新一条文本。

`LOCAL_DEEPSEEK_ALLOWED_TOOLS`（白名单工具）与
`LOCAL_DEEPSEEK_PASSTHROUGH_SKILLS`（透传技能）等常量也在此文件配置。

#### 请求与 SSE 原始流解析（chat.py + deepseek_sse.py）

`ChatHandler._post()` 向代理发送单条文本：

- 请求带 `Authorization: Bearer` 与 `X-BigCodex-Raw-SSE: 1` 自定义头；
- 代理识别该头后返回**网页版 SSE 原始流**（`event:`/`data:` 行），否则返回纯文本兜底；
- `deepseek_sse.parse_sse_content()` 解析 SSE 提取可见正文（初始 envelope 的
  RESPONSE 片段、`APPEND` 增量、`BATCH` 补丁、裸 `{"v":...}` 增量）；
- `deepseek_sse.parse_sse_references()` 解析引用元数据（SEARCH results /
  fragment references），写入 `bs["local_references"]` 供前端渲染 `[citation:N]`。

#### 工具白名单与 MCP 工具命名

`LOCAL_DEEPSEEK_ALLOWED_TOOLS` 是允许网页版通过文本协议调用的本地工具白名单，
教学文本自动从 `tool_registry` 取每个工具的定义（说明 + 参数 JSON Schema）：

- **内置工具**：直接用工具名，如 `generate_image` / `generate_video`；
- **MCP 工具**：必须填完整注册名 `mcp_<server名>_<工具名>`，例如
  `mcp_filesystem_list_directory`、`mcp_fetch_fetch_url`；
  - `server名` 取自 `mcp.json` 的键名；
  - `工具名` 是该 MCP 服务器暴露的原始工具名；
  - 查看当前全部已注册 MCP 工具：

    ```bash
    cd src/agent
    python -c "from tools import tool_registry; import mcp_manager; \
    tool_registry.load_all_tools(); print([t.name for t in \
    tool_registry.get_all_tools() if t.name.startswith('mcp_')])"
    ```

> ⚠️ 安全注意：MCP 写类工具默认权限为 `DEFAULT`，而 LocalLLM 的
> `_LocalToolContext.check_permission()` 直接返回 None（默认放行），权限校验比
> engine 远端路径宽松。放行写类 MCP 工具给 LocalLLM 前请确认该工具的行为可接受。

#### 工具使用教学（文本协议）

网页版 DeepSeek 不是真的函数调用，因此采用**文本协议**教它声明本地工具调用：

- `_build_tool_teaching_prompt()` 从 `tool_registry` 动态取白名单工具定义
  （说明 + 参数 JSON Schema），教网页版在需要时输出：
  `<tool_call>{"tool": "generate_image", "arguments": {...}}</tool_call>`
- **首次教学**：程序启动后第一次使用 LocalLLM 时静默发送一次（进程级标记
  `_TOOL_TEACHING_DONE`，回复不展示给用户）；
- **强制重教**：用户手动在网页版新开对话后，使用 `/teach-tool-usage` 技能
  重新发送教学（chat.py 识别技能名后直接发教学文本）。

#### 本地工具微循环

`ChatHandler.handle()` 对每轮回复执行：

1. `_post()` 发送用户文本（或上一轮工具结果）给代理；
2. `_parse_tool_calls()` 解析回复中的 `<tool_call>` 标记（容错：坏 JSON 跳过）；
3. 无工具调用 → 直接返回该回复；
4. 有工具调用 → 逐个执行 `_run_tool()`（走 `tool_registry`，复用
   `generate_image` / `generate_video` 的完整实现），并把结果以
   `【工具结果】...` 文本回传代理，等待其组织最终回复；
5. `_emit_tool_events()` 向会话事件队列发射 `tool_use` / `tool_result` 事件
   （与远端 LLM 路径形状一致，前端无需改动即可渲染工具卡片）；
6. 单轮最多 `LOCAL_DEEPSEEK_MAX_TOOL_ROUNDS`（默认 3）轮，防止无限循环。

工具结果回传文本内置「展示规则」：本地路径不是网址，展示必须用标准 Markdown
（图片 `![](路径)`、视频 `![video](路径)`），禁止转 https 链接 / URL 编码 /
反斜杠转义——避免网页版把 `D:\...\xxx.png` 编码成 `https://D:%5C...` 乱码。

#### 技能透传

命中 `LOCAL_DEEPSEEK_PASSTHROUGH_SKILLS`（当前 image-builder / video-builder /
news-media-digest）时，
把技能描述全文（SKILL.md）+ 用户请求一起透传给网页版，让网页版按技能规范执行
（提示词润色、先确认后生成、台词逐字保留、新闻主内容图片提取等）；
未命中时只透传真实用户输入。

#### 连通性与超时

- `can_handle()` 用短超时（1.5s）TCP 探测代理基本地址，不可达时不接管；
- 单次对话请求超时 `LOCAL_DEEPSEEK_TIMEOUT = 120s`（网页版响应可能较慢）。

#### 调试日志（local_llm_trace.log）

每次 `_post()` 会把 **REQ（发出原文）/ RESP_RAW（代理返回的原始 SSE）/
RESP_PARSED（解析后正文）** 追加写入仓库根 `local_llm_trace.log`
（已被 `.gitignore` 忽略），用于排查路径转义、URL 编码等显示链路问题。

## 配置示例

`.env` 文件示例：

```bash
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4
DEFAULT_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=4096
```
