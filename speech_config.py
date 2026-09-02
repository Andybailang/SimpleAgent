"""
语音（edge-tts）配置
====================
- 音色列表来自 edge_tts.list_voices()（等价于 CLI `edge-tts --list-voices`），
  中文优先 + 常用英文，做内存缓存；联网失败时回退内置常见人声。
- 激活音色存于 models.json 顶层 active_voice（默认 zh-CN-XiaoxiaoNeural），
  由设置页「多模态 → 语音人声」读写，generate_speech 与朗读按钮共用。
"""
import time
from typing import Any, Dict, List

from multimodal_config import get_active_voice

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

# 常见人声的展示名（ShortName -> 中文名/英文名）
_ZH_NAMES = {
    "zh-CN-XiaoxiaoNeural": "晓晓",
    "zh-CN-XiaoyiNeural": "晓伊",
    "zh-CN-YunjianNeural": "云健",
    "zh-CN-YunxiNeural": "云希",
    "zh-CN-YunxiaNeural": "云夏",
    "zh-CN-YunyangNeural": "云扬",
    "zh-CN-liaoning-XiaobeiNeural": "晓北",
    "zh-CN-shaanxi-XiaoniNeural": "晓妮",
    "zh-HK-HiuGaaiNeural": "曉佳",
    "zh-HK-HiuMaanNeural": "曉曼",
    "zh-HK-WanLungNeural": "雲龍",
    "zh-TW-HsiaoChenNeural": "曉臻",
    "zh-TW-HsiaoYuNeural": "曉雨",
    "zh-TW-YunJheNeural": "雲哲",
    "en-US-AriaNeural": "Aria",
    "en-US-JennyNeural": "Jenny",
    "en-US-GuyNeural": "Guy",
    "en-GB-SoniaNeural": "Sonia",
    "ja-JP-NanamiNeural": "Nanami",
    "ko-KR-SunHiNeural": "SunHi",
    "fr-FR-DeniseNeural": "Denise",
    "de-DE-KatjaNeural": "Katja",
    "es-ES-ElviraNeural": "Elvira",
}

# 性格/风格英文 -> 中文展示
_PERSONALITY_ZH = {
    "warm": "温柔", "natural": "自然", "passionate": "热情", "calm": "沉稳",
    "gentle": "温和", "lively": "活泼", "cheerful": "开朗", "serious": "严肃",
    "friendly": "亲切", "soft": "柔和", "bright": "明亮", "mature": "成熟",
    "energetic": "有活力", "deep": "深沉",
}
_CATEGORY_ZH = {
    "news": "新闻", "novel": "小说", "documentary": "纪录片",
    "poetry": "诗歌", "storytelling": "讲故事", "narration": "旁白",
}

_CACHE: Dict[str, Any] = {"ts": 0.0, "voices": []}
_CACHE_TTL = 3600  # 秒


def get_active_voice_name() -> str:
    return get_active_voice() or DEFAULT_VOICE


def _gender_zh(g: str) -> str:
    return "男声" if str(g).lower() == "male" else "女声"


def _style_tag(voice: dict) -> str:
    tag = voice.get("VoiceTag") or {}
    for p in tag.get("VoicePersonalities") or []:
        key = str(p).strip().lower()
        if key in _PERSONALITY_ZH:
            return _PERSONALITY_ZH[key]
    for c in tag.get("ContentCategories") or []:
        key = str(c).strip().lower()
        if key in _CATEGORY_ZH:
            return _CATEGORY_ZH[key]
    return "自然"


def _display_name(voice: dict) -> str:
    short = str(voice.get("ShortName") or "")
    if short in _ZH_NAMES:
        return _ZH_NAMES[short]
    friendly = str(voice.get("FriendlyName") or "")
    name = friendly.replace("Microsoft ", "").split("Online")[0].strip()
    if "(" in name:
        name = name.split("(")[0].strip()
    return name or short


def build_voice_entry(voice: dict) -> Dict[str, Any]:
    short = str(voice.get("ShortName") or "")
    gender = _gender_zh(voice.get("Gender") or "")
    style = _style_tag(voice)
    name = _display_name(voice)
    return {
        "id": short,
        "name": name,
        "gender": gender,
        "style": style,
        "locale": str(voice.get("Locale") or ""),
        "display_name": f"{name}（{gender}，{style}）",
        "label": f"{name}（{gender}，{style}）{short}",
    }


def _curated_fallback() -> List[Dict[str, Any]]:
    """联网失败时回退的常见人声。"""
    seeds = [
        ("zh-CN-XiaoxiaoNeural", "Female", "zh-CN", {"VoicePersonalities": ["Warm"]}),
        ("zh-CN-YunxiNeural", "Male", "zh-CN", {"VoicePersonalities": ["Passionate"]}),
        ("zh-CN-YunjianNeural", "Male", "zh-CN", {"VoicePersonalities": ["Calm"]}),
        ("zh-CN-XiaoyiNeural", "Female", "zh-CN", {"VoicePersonalities": ["Lively"]}),
        ("zh-CN-YunyangNeural", "Male", "zh-CN", {"VoicePersonalities": ["Mature"]}),
        ("zh-CN-YunxiaNeural", "Female", "zh-CN", {"VoicePersonalities": ["Gentle"]}),
        ("zh-CN-liaoning-XiaobeiNeural", "Female", "zh-CN-liaoning", {}),
        ("zh-CN-shaanxi-XiaoniNeural", "Female", "zh-CN-shaanxi", {}),
        ("zh-HK-HiuGaaiNeural", "Female", "zh-HK", {}),
        ("zh-TW-HsiaoChenNeural", "Female", "zh-TW", {}),
        ("en-US-AriaNeural", "Female", "en-US", {"VoicePersonalities": ["Warm"]}),
        ("en-US-JennyNeural", "Female", "en-US", {"VoicePersonalities": ["Gentle"]}),
        ("en-US-GuyNeural", "Male", "en-US", {"VoicePersonalities": ["Friendly"]}),
        ("ja-JP-NanamiNeural", "Female", "ja-JP", {}),
        ("ko-KR-SunHiNeural", "Female", "ko-KR", {}),
    ]
    return [build_voice_entry({
        "ShortName": s, "Gender": g, "Locale": loc, "VoiceTag": tag,
    }) for s, g, loc, tag in seeds]


def _rank(e: dict) -> int:
    loc = str(e.get("locale") or "")
    if loc.startswith("zh-CN"):
        return 0
    if loc.startswith("zh"):
        return 1
    if loc.startswith("en"):
        return 2
    return 3


async def get_speech_voices(refresh: bool = False) -> List[Dict[str, Any]]:
    """抓取音色（含缓存）；联网失败返回内置常见人声。"""
    import edge_tts
    now = time.time()
    if not refresh and _CACHE["voices"] and now - _CACHE["ts"] < _CACHE_TTL:
        return _CACHE["voices"]
    try:
        raw = await edge_tts.list_voices()
        voices = [build_voice_entry(v) for v in (raw or []) if v.get("ShortName")]
        voices.sort(key=_rank)
    except Exception:
        voices = _curated_fallback()
    _CACHE["ts"] = now
    _CACHE["voices"] = voices
    return voices
