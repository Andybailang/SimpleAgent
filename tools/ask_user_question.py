"""AskUserQuestion — 内置交互工具（向用户提问，等待回答后继续）。

某些专为编程 Agent 训练的模型会主动调用 AskUserQuestion 来澄清需求。本工具把它
声明为内置工具，使模型"知道"可以调用；但它的执行走「暂停 -> 前端问题卡片 -> 用户回答
-> 以 tool_result 续跑」的交互通道，而不是在本进程内做任何本地操作。
"""
from typing import Any, Dict, List

from .base import BaseTool, Tool, ToolContext, ToolMode, ToolPermission


class AskUserQuestionTool(BaseTool):
    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="AskUserQuestion",
            description=(
                "向用户提出一个或多个问题以澄清需求。当任务需要用户决策、路径选择、"
                "技术选型，或存在多个合理方案需要用户确认时使用。调用后会暂停执行，"
                "等用户回答后自动继续。questions 可含多个问题；每个问题可有 "
                "question(必填)、header(分组标题，可选)、options(选项数组，每项含 "
                "label/description)、multiSelect(是否多选)。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "单个问题的正文（与 questions 二选一；两者都传时 questions 优先）。",
                    },
                    "header": {
                        "type": "string",
                        "description": "可选的问题分组标题。",
                    },
                    "options": {
                        "type": "array",
                        "description": "可选项列表；每项为 {label, description?}。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["label"],
                        },
                    },
                    "questions": {
                        "type": "array",
                        "description": "多个问题的数组；每项含 question/header/options/multiSelect。",
                        "items": {"type": "object"},
                    },
                    "multiSelect": {"type": "boolean"},
                },
                "anyOf": [
                    {"required": ["question"]},
                    {"required": ["questions"]},
                ],
            },
            # 两种模式都可声明；但本工具在执行流里会被 AskUserQuestion 拦截器处理，
            # 本地 execute 仅作为兜底（例如非流式/非标准路径误调用时）。
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
        )

    @classmethod
    def execute(cls, context: ToolContext, **kwargs: Any) -> str:
        # 正常路径由流式引擎的拦截器转为"提问卡片"，不在此执行；
        # 兜底返回一个说明文本，避免未知工具错误。
        questions = kwargs.get("questions") or []
        if not questions and kwargs.get("question"):
            questions = [{"question": kwargs.get("question"), "options": kwargs.get("options") or []}]
        return (
            "[AskUserQuestion] 已向用户提问，等待回答后将继续执行。"
            "（此消息出现在非交互路径的兜底，正常桌面/网页版不会看到。）"
            if questions
            else "[AskUserQuestion] 参数为空，未生成有效提问。"
        )
