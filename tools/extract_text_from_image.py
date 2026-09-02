"""
extract_text_from_image 工具：识别图片中的文字（PaddleOCR）
==========================================================
图片附件不再内联上传，模型看到图片路径后调用本工具获取文字。
"""
import os
from typing import List
from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext
from ocr_engine import get_ocr, extract_rec_texts


class ExtractTextFromImageTool(BaseTool):
    """extract_text_from_image 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="extract_text_from_image",
            description=(
                "识别图片中的文字，返回识别结果文本。参数：path（必填，图像文件路径）。"
                "当用户提供图片时必须调用本工具提取文字，不要尝试直接理解图片内容。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "图像文件路径"
                    }
                },
                "required": ["path"]
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.READONLY,
            aliases={"path": ["file_path", "image_path"]},
        )

    @classmethod
    def execute(cls, context: ToolContext, path: str) -> str:
        """执行图片文字识别"""
        try:
            resolved = context.resolve_path(path, "extract_text_from_image")
        except Exception as e:
            return f"错误：{e}"
        if not os.path.isfile(resolved):
            return f"错误：图片文件不存在（{resolved}）"
        try:
            ocr = get_ocr()
            result = ocr.predict(resolved)
            texts: List[str] = extract_rec_texts(result)
            if not texts:
                return f"图片 {resolved} 未识别到文字"
            return f"图片 {resolved} 识别结果：\n" + "\n".join(texts)
        except Exception as e:
            return f"图片识别失败（{resolved}）：{e}"
