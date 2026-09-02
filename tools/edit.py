"""
Edit 工具：精确替换文件中的文本片段
"""
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext


class EditTool(BaseTool):
    """Edit 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="Edit",
            description=(
                "精确替换文件中的文本片段。参数：file_path（文件路径）、old_string（要替换的原文）、"
                "new_string（替换后的文本）、replace_all（可选，true 时替换所有匹配，默认仅第一处）"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要替换的原文"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "替换后的文本"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "true 时替换所有匹配；默认只替换第一处"
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.DEFAULT,
            aliases={"file_path": ["path"]},
        )

    @classmethod
    def execute(cls, context: ToolContext, file_path: str, old_string: str,
                new_string: str, replace_all: bool = False) -> str:
        """编辑文件：替换文本片段（replace_all=true 时替换所有匹配）"""
        try:
            path = context.resolve_path(file_path, "Edit")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not old_string:
                return "错误：old_string 不能为空"
            if old_string not in content:
                return f"错误：在 {path} 中没有找到要替换的文本片段，请先 Read 确认内容"
            if replace_all:
                content = content.replace(old_string, new_string)
            else:
                content = content.replace(old_string, new_string, 1)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"成功更新文件 {path}"
        except FileNotFoundError:
            return f"错误：文件 {file_path} 不存在"
        except PermissionError:
            return f"错误：没有权限修改文件 {file_path}"
        except Exception as e:
            return f"编辑文件 {file_path} 时出错: {str(e)}"
