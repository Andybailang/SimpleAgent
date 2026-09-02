"""
工具注册和加载模块
"""
import importlib
import os
import pkgutil
from typing import Dict, List, Optional, Type
from .base import Tool, ToolMode, BaseTool, ToolContext


class ToolRegistry:
    """工具注册器"""

    _tools: Dict[str, Tool] = {}
    _tool_classes: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_class: Type[BaseTool]):
        """注册工具类"""
        tool_def = tool_class.get_tool_definition()
        tool_def.execute = tool_class.execute
        cls._tools[tool_def.name] = tool_def
        cls._tool_classes[tool_def.name] = tool_class
        return tool_def

    @classmethod
    def get_tool(cls, name: str) -> Optional[Tool]:
        """获取工具定义"""
        return cls._tools.get(name)

    @classmethod
    def get_tool_class(cls, name: str) -> Optional[Type[BaseTool]]:
        """获取工具类"""
        return cls._tool_classes.get(name)

    @classmethod
    def get_all_tools(cls) -> List[Tool]:
        """获取所有工具（按固定展示顺序排序）"""
        return sorted(cls._tools.values(), key=_tool_sort_key)

    @classmethod
    def get_tools_by_mode(cls, mode: str) -> List[Tool]:
        """获取指定模式下的工具（按固定展示顺序排序）"""
        try:
            mode_enum = ToolMode(mode)
        except ValueError:
            return []
        tools = [t for t in cls._tools.values() if mode_enum in t.modes]
        return sorted(tools, key=_tool_sort_key)

    @classmethod
    def register_dynamic(cls, tool: Tool) -> Tool:
        """注册动态工具实例（MCP 等外部来源），保留 execute 闭包。"""
        cls._tools[tool.name] = tool
        cls._tool_classes.pop(tool.name, None)
        return tool

    @classmethod
    def execute_tool(cls, context: ToolContext, tool_name: str, **kwargs) -> str:
        """执行工具：优先类工具，其次动态工具实例。"""
        tool_class = cls.get_tool_class(tool_name)
        if tool_class:
            return tool_class.execute(context, **kwargs)
        tool = cls.get_tool(tool_name)
        if tool is not None and tool.execute is not None:
            return tool.execute(context, **kwargs)
        return f"错误：未知工具 '{tool_name}'"

    @classmethod
    def load_all_tools(cls, tools_dir: str = None):
        """加载所有工具"""
        if tools_dir is None:
            tools_dir = os.path.dirname(__file__)

        skip_modules = {"base", "config", "semantic_config", "template"}
        for _, module_name, _ in pkgutil.iter_modules([tools_dir]):
            if module_name in skip_modules or module_name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f"{__name__}.{module_name}")
                # 查找模块中继承 BaseTool 的类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                            issubclass(attr, BaseTool) and
                            attr != BaseTool):
                        cls.register(attr)
            except Exception as e:
                print(f"加载工具 {module_name} 失败: {e}")


# 工具在提示词 / API 定义中的固定展示顺序（与重构前 engine.py 保持一致）
TOOL_ORDER = [
    "Read", "Write", "Edit", "delete_file", "LS", "create_directory",
    "rename_file", "Grep", "Glob", "Bash", "TestRunner", "TodoWrite", "SemanticSearch",
]


def _tool_sort_key(tool: Tool) -> int:
    """返回工具在 TOOL_ORDER 中的下标；未列出的工具排到最后（保持注册顺序）。"""
    if tool.name in TOOL_ORDER:
        return TOOL_ORDER.index(tool.name)
    return len(TOOL_ORDER)


# 创建全局注册器实例并自动加载全部工具
tool_registry = ToolRegistry()
tool_registry.load_all_tools()


def get_tool_definitions(mode: str = "work") -> List[Dict[str, object]]:
    """
    获取指定模式下的工具定义（用于 LLM API）
    """
    tools = tool_registry.get_tools_by_mode(mode)
    definitions = []
    for tool in tools:
        definitions.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "parameters": tool.parameters,
                "description": tool.description,
            },
        })
    return definitions
