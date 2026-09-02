"""engine.prompt — 系统提示词构建与会话相关内容加载。
"""
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import AGENTS_MD_MAX_CHARS
from .util import Message
from tools import tool_registry
from tools.base import Tool
import skills

class PromptMixin:
    def _build_system_prompt(self) -> str:
        """按当前模式构建系统提示：固定模板在前，会话相关（AGENTS.md）在最后，利于前缀缓存命中。"""
        if self.mode == "chat":
            return self._build_chat_prompt()
        if self.role_prompt:
            head = self.role_prompt
        else:
            head = self._work_role_prompt()
        return f"{head}\n\n{self._build_work_tail()}"
    def _work_role_prompt(self) -> str:
        """编程角色提示词（身份描述）。"""
        return (
            "你是一个 AI 编程助手。你可以帮助用户：\n"
            "- 读取、创建、编辑和删除文件\n"
            "- 列出目录内容、创建目录\n"
            "- 重命名或移动文件\n"
            "- 搜索代码\n"
            "- 执行 shell 命令（git、npm、python 等）"
        )
    def _build_work_tail(self) -> str:
        """工作模式后续：路径/权限约束、工具说明、使用规范、流程规范、AGENTS.md。

        不拼接「可用技能」清单（技能列表仅聊天模式注入，精简编程模式每轮 token 费用）。
        工具定义只随原生 `tools` 数组下发（见 tools_runtime._tool_definitions），
        这里不再重复拼一份纯文本清单，避免同一批工具被写入两次、白耗输入 token。
        """
        tail = f"""所有文件路径都相对于工作目录（也支持绝对路径，但必须在工作目录范围内）。shell 命令同样在项目目录下执行。当需要执行文件操作或命令时，调用相应的工具。
当前工具权限由会话设定控制：默认只能在项目目录内操作；只读模式下禁止任何修改类工具，Bash 仅放行只读命令（如 git status/log/diff/show、ls、cat、rg 等查询类），修改文件或写操作类命令会被拒绝；完全访问模式下可操作任意路径。项目内的软链接若指向项目外，只读工具（Read/LS/Grep/Glob）仍可访问，写入类工具一律禁止。

工具使用规范：
- 代码类查询与理解（定位符号、找调用关系、评估改动影响、梳理流程串接）：若本轮请求提供了 `mcp_gitnexus_*` 工具，优先用 gitnexus 的 query / context / impact / detect_changes 完成，而不是先 grep / rg 等文本检索；grep 等仅在没有 gitnexus 工具或结果不足时兜底
- 每个工具只做自己的事，不要组合乱用
- Write 写入文件时注意扩展名和内容类型要匹配（纯文本用 .txt/.md/.py，不要用 .docx/.jpg 等二进制扩展名）
- 当用户提供图片时，必须调用工具 extract_text_from_image 提取文字，不要尝试直接理解图片内容
- 图片/媒体文件不要用 Read、read_media_file 等读文件工具读取其内容（读取会占用上下文且被后端拦截）
"""
        tail += """   示例：
     # 简单任务分解
     {"todos": [
       {"content": "分析需求文档", "status": "completed"},
       {"content": "设计 API 接口", "status": "in_progress", "activeForm": "设计 API"},
       {"content": "实现服务层", "status": "pending"},
       {"content": "编写单元测试", "status": "pending"}
     ]}

     # 复杂任务分解（嵌套）
     {"todos": [
       {"content": "重构前端架构", "status": "in_progress", "activeForm": "重构前端"},
       {"content": "更新路由配置", "status": "completed"},
       {"content": "迁移组件库", "status": "pending"}
     ]}

开发流程规范（多步骤任务必须遵守）：
1. 规划：新建项目、重构、调试、编写或修复测试等多步骤任务，开始前先调用 TodoWrite 创建待办列表，并随进度更新各项状态（pending/in_progress/completed）。
2. 实现：按步骤完成代码或文件修改，避免一次性大段输出未经校验的代码。
3. 验证：每完成关键步骤必须运行验证——运行程序、语法检查（如 python -m py_compile）或执行相关测试，确认通过后再继续下一步。
4. 提交：提交前先 git status / git diff 检查改动、确认没有临时文件；只有测试与验证全部通过后才允许 git commit，并用简短中文或英文写提交信息。

工具调用：请使用平台提供的原生工具调用机制（tool_use），直接调用本轮请求中提供的工具列表里的工具，不要输出自定义 JSON 格式。

重要：
- 你可以调用 0 到多个工具
- 多步骤任务请先调用 TodoWrite 规划，再开始实现
- 工具调用完成后，需要给出用户友好的总结
- 如果工具执行失败，直接告诉用户发生了什么
"""
        agents_md = self._load_agents_md()
        if agents_md:
            tail += (
                "\n\n==================== 项目规范（根目录 AGENTS.md） ====================\n"
                + agents_md
                + "\n==================== AGENTS.md 结束 ===================="
            )
        # 编程模式不拼接「可用技能」清单：技能列表只在聊天模式注入，
        # 避免每轮请求白付这段常驻 token（技能具体内容仍在用户 /技能名 时注入）。
        return tail
    def _build_chat_prompt(self) -> str:
        """聊天模式系统提示：普通助手 + 该模式下可用（只读）工具。"""
        tools_desc = self._format_tool_list(tool_registry.get_tools_by_mode(self.mode))
        prompt = f"""你是一个乐于助人的 AI 助手。你可以帮助用户：
- 回答各类问题
- 借助只读工具查阅工作区内容：读取文件、浏览目录、正则搜索、语义搜索

当前可用的辅助工具（均为只读，不会修改任何文件，也不会执行命令）：
{tools_desc}

注意：
- 每个工具只做自己的事，不要组合乱用
- fetch 拿到网页内容后，直接阅读并回答，不要再用 SemanticSearch 处理
- SemanticSearch 只搜本地文件，不支持网址和网页内容
- 你不能修改文件或执行命令，只能使用上述只读工具辅助回答问题
- 当用户提供图片时，必须调用工具 extract_text_from_image 提取文字，不要尝试直接理解图片内容
- 图片/媒体文件不要用 Read、read_media_file 等读文件工具读取其内容；图生图 / 图生视频时直接把图片路径传给 generate_image / generate_video
- 工具执行失败时，直接告诉用户发生了什么
"""
        skills_summary = self._load_skills_summary()
        if skills_summary:
            prompt += (
                "\n\n==================== 可用技能 ====================\n"
                + skills_summary
                + "\n==================== 技能结束 ===================="
            )
        return prompt
    def _format_tool_list(self, tools: List[Tool]) -> str:
        """把工具列表格式化为系统提示里的编号清单（固定内容，随模式稳定）。"""
        lines = []
        for i, tool in enumerate(tools, 1):
            sig = self._format_tool_signature(tool)
            desc = self._strip_param_section(tool.description)
            lines.append(f"{i}. {tool.name}({sig}): {desc}")
        return "\n".join(lines)
    @staticmethod
    def _format_tool_signature(tool: Tool) -> str:
        """按工具 schema 生成参数签名（必填参数不带 ?，可选参数带 ?）。"""
        schema = tool.parameters or {}
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required") or [])
        return ", ".join(name if name in required else f"{name}?" for name in props)
    @staticmethod
    def _strip_param_section(desc: str) -> str:
        """去掉描述里的 “参数:...” / “Parameters:...” 清单（签名已体现），保留其余说明。"""
        desc = (desc or "").strip()
        m = re.search(r"(?:参数|Parameters?|Params)\s*[:：]", desc)
        if not m:
            return desc
        return desc[:m.start()].rstrip("。；;，, .")
    def _add_system_message(self):
        """添加系统提示"""
        self.messages.append(Message(
            role="system",
            content=self._build_system_prompt(),
            timestamp=int(datetime.now().timestamp() * 1000)
        ))
    def _load_agents_md(self) -> str:
        """读取工作目录根部的 AGENTS.md（或 agents.md），不存在时返回空字符串。"""
        for name in ("AGENTS.md", "agents.md"):
            path = os.path.join(self.cwd, name)
            if not os.path.isfile(path):
                continue
            try:
                raw = open(path, "rb").read()
            except Exception:
                return ""
            content = ""
            for enc in ("utf-8", "gbk", "latin-1"):
                try:
                    content = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if len(content) > AGENTS_MD_MAX_CHARS:
                content = content[:AGENTS_MD_MAX_CHARS] + "\n...[内容过长，已截断]"
            return content
        return ""
    def _load_skills_summary(self) -> str:
        """汇总已启用技能（未标记 disable-model-invocation）的名称与描述。

        只列清单不列正文：具体内容在用户输入 /<技能名> 时注入，避免常驻上下文。
        """
        try:
            items = skills.list_skills(self.cwd)
        except Exception:
            return ""
        enabled = [s for s in items if not s.get("disable_model_invocation")]
        if not enabled:
            return ""
        lines = [
            "以下技能可帮助用户更高效地完成任务，可通过输入 /技能名 参数 调用；"
            "当任务匹配时应主动建议用户使用对应技能："
        ]
        for s in enabled:
            scope_label = "项目" if s.get("scope") == "project" else "全局"
            lines.append(f"- /{s['name']}（{scope_label}）：{s.get('description') or ''}")
        return "\n".join(lines)
    def _refresh_system_message(self):
        """重新构建系统提示，让 AGENTS.md 的修改在下一轮对话生效。"""
        prompt = self._build_system_prompt()
        for msg in self.messages:
            if msg.role == "system":
                msg.content = prompt
                return
    def _anthropic_system_text(self) -> str:
        """Concatenate stored system messages into the Anthropic top-level system field."""
        parts = [str(msg.content) for msg in self.messages if msg.role == "system" and msg.content]
        return "\n\n".join(parts)
