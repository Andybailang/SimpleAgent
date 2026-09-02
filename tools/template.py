"""
新工具创建模板

在 tools/ 目录下复制本文件为 <tool_name>.py，实现 get_tool_definition 与 execute 两个类方法，
工具会自动被 ToolRegistry 扫描注册（无需修改 __init__.py 或 engine/）。

示例（以 MyTool 为例）：

    from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext

    class MyTool(BaseTool):
        \"\"\"MyTool 工具实现\"\"\"

        @classmethod
        def get_tool_definition(cls) -> Tool:
            return Tool(
                name="MyTool",
                description="工具描述",
                parameters={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "参数说明"},
                    },
                    "required": ["param1"],
                },
                modes=[ToolMode.WORK, ToolMode.CHAT],
                permission_level=ToolPermission.DEFAULT,
                aliases={},
            )

        @classmethod
        def execute(cls, context: ToolContext, param1: str, **kwargs) -> str:
            try:
                # 工具实现
                return f"成功执行，参数：{param1}"
            except Exception as e:
                return f"执行失败：{str(e)}"
"""  # noqa: W605
