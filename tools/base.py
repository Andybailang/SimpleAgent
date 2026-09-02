"""
工具基类和权限控制
"""
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum


class ToolPermission(str, Enum):
    """工具权限级别"""
    READONLY = "readonly"      # 只读权限
    DEFAULT = "default"        # 默认权限（工作目录内）
    FULL = "full"              # 完全访问


class ToolMode(str, Enum):
    """工具可用模式"""
    WORK = "work"              # 工作模式
    CHAT = "chat"              # 聊天模式
    BOTH = "both"              # 两种模式都可用


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Optional[Callable] = None  # 由注册器绑定到工具类的 execute 类方法

    # 工具元数据
    modes: List[ToolMode] = field(default_factory=lambda: [ToolMode.WORK])
    permission_level: ToolPermission = ToolPermission.DEFAULT
    requires_semantic_key: bool = False  # 是否需要语义搜索API key

    # 参数别名（主参数名 -> 兼容别名列表）
    aliases: Dict[str, List[str]] = field(default_factory=dict)

    # 工具依赖的其他工具（如 SemanticSearch 依赖嵌入API）
    dependencies: List[str] = field(default_factory=list)

    def is_available_in_mode(self, mode: str) -> bool:
        """检查工具是否在指定模式下可用"""
        try:
            mode_enum = ToolMode(mode)
        except ValueError:
            return False
        return mode_enum in self.modes


class ToolContext:
    """
    工具执行上下文
    包含 Agent 的引用、权限配置、工作目录等
    """

    def __init__(self, agent, cwd: str, permission_mode: str = "default"):
        self.agent = agent
        self.cwd = cwd
        self.permission_mode = permission_mode
        self.mode = "work"  # work 或 chat

    def resolve_path(self, path: str, tool_name: str = "") -> str:
        """路径解析（权限检查），复用 engine/tools_runtime.py 中的路径解析逻辑。"""
        return self.agent._resolve_path(path, tool_name)

    def check_permission(self, tool_name: str, tool_input: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """权限检查：READONLY 工具直接放行，其余委托给 engine。"""
        tool = self.agent._find_tool(tool_name)
        if tool and tool.permission_level == ToolPermission.READONLY:
            return None
        return self.agent._check_permission(tool_name, tool_input)

    def set_mode(self, mode: str):
        """设置当前模式"""
        self.mode = mode


class BaseTool:
    """
    所有工具的基类
    提供统一的接口和辅助方法
    """

    @classmethod
    def get_tool_definition(cls) -> Tool:
        """返回工具定义"""
        raise NotImplementedError

    @classmethod
    def execute(cls, context: ToolContext, **kwargs) -> str:
        """执行工具逻辑"""
        raise NotImplementedError
