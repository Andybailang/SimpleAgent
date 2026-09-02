"""
工具配置管理
"""
import os
from typing import Dict, Any


class ToolConfig:
    """工具配置管理器"""

    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def load_from_env(cls, tool_name: str, prefix: str = "TOOL_") -> Dict[str, Any]:
        """从环境变量加载工具配置"""
        config = {}
        prefix_upper = f"{prefix}{tool_name.upper()}_"
        for key, value in os.environ.items():
            if key.startswith(prefix_upper):
                config_key = key[len(prefix_upper):].lower()
                config[config_key] = value
        return config

    @classmethod
    def get_config(cls, tool_name: str) -> Dict[str, Any]:
        """获取工具配置"""
        if tool_name not in cls._configs:
            cls._configs[tool_name] = cls.load_from_env(tool_name)
        return cls._configs[tool_name]

    @classmethod
    def set_config(cls, tool_name: str, config: Dict[str, Any]):
        """设置工具配置"""
        cls._configs[tool_name] = config
