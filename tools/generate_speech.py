"""
generate_speech 工具：使用 edge-tts（Microsoft Edge TTS）把文字合成语音 mp3
=====================================================================
- text（必填）：待朗读的文本；
- output（可选）：保存文件名，默认 output.mp3，存到 <cwd>/.bigcodex_uploads/；
- voice（可选）：音色 ShortName，默认取激活音色（models.json active_voice）或
  zh-CN-XiaoxiaoNeural；
- rate / volume / pitch（可选）：语速 / 音量 / 音调（如 "+10%" / "-10%" / "+5Hz"）。
依赖公网 Microsoft Edge TTS 服务，失败时返回错误说明。
edge-tts 为异步库，这里通过独立线程跑 asyncio.run，避免在当前事件循环线程内调用冲突。
"""
import asyncio
import os
import threading
from datetime import datetime
from typing import Optional

from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext
from speech_config import DEFAULT_VOICE, get_active_voice_name

SAVE_DIR_NAME = ".bigcodex_uploads"


def _save_audio(text: str, voice: str, path: str, rate, volume, pitch) -> None:
    """在独立线程里保存语音文件。"""
    import edge_tts
    kwargs = {}
    if rate:
        kwargs["rate"] = str(rate)
    if volume:
        kwargs["volume"] = str(volume)
    if pitch:
        kwargs["pitch"] = str(pitch)
    asyncio.run(edge_tts.Communicate(text, voice, **kwargs).save(path))


def _await_async(fn, *args, **kwargs):
    """在独立线程跑 asyncio.run，避免工具在事件循环线程内调用。"""
    out: dict = {}

    def runner():
        try:
            out["value"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out.get("value")


class GenerateSpeechTool(BaseTool):
    """generate_speech 工具实现"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="generate_speech",
            description=(
                "使用 edge-tts 把文字合成为语音 mp3（依赖公网 Microsoft Edge TTS）。"
                "参数：text（必填，要朗读的文本）；output（可选，保存文件名，默认 output.mp3，"
                "存到 .bigcodex_uploads/）；voice（可选，音色 ShortName，默认当前激活人声）；"
                "rate/volume/pitch（可选，语速/音量/音调，如 +10% / -10% / +5Hz）。"
                "生成结果保存到 .bigcodex_uploads/ 并返回本地路径。注意：非必要不要连续生成。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要朗读的文本"},
                    "output": {"type": "string", "description": "保存文件名（默认 output.mp3）"},
                    "voice": {"type": "string", "description": "音色 ShortName（如 zh-CN-XiaoxiaoNeural，默认当前激活人声）"},
                    "rate": {"type": "string", "description": "语速（如 +10% / -10%）"},
                    "volume": {"type": "string", "description": "音量（如 +10% / -10%）"},
                    "pitch": {"type": "string", "description": "音调（如 +5Hz / -5Hz）"},
                },
                "required": ["text"],
            },
            modes=[ToolMode.WORK, ToolMode.CHAT],
            permission_level=ToolPermission.DEFAULT,
            aliases={"speech": ["text_to_speech", "tts"]},
        )

    @classmethod
    def execute(
        cls,
        context: ToolContext,
        text: str,
        output: str = "output.mp3",
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> str:
        """合成语音并保存到 .bigcodex_uploads/，返回本地路径。"""
        text = (text or "").strip()
        if not text:
            return "错误：text 不能为空"
        voice_id = (voice or get_active_voice_name() or DEFAULT_VOICE).strip()
        out_name = cls._safe_filename(output, "mp3")
        save_dir = os.path.join(context.cwd, SAVE_DIR_NAME)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            return "错误：创建保存目录失败"
        save_path = os.path.join(save_dir, out_name)
        try:
            _await_async(_save_audio, text, voice_id, save_path, rate, volume, pitch)
        except Exception as e:
            return f"错误：语音合成失败：{e}"
        return f"已生成语音，保存在 .bigcodex_uploads/：\n- {save_path}"

    @classmethod
    def _safe_filename(cls, output, fallback_ext: str) -> str:
        """把输出文件名规范化：只取文件名，无扩展名时补默认扩展名。"""
        name = os.path.basename(str(output or "").strip()) or f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if "." not in name:
            name = f"{name}.{fallback_ext}"
        return name
