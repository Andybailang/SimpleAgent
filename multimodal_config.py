"""
多模态模型配置（激活的图片/视频模型）
=====================================
激活模型 id 存于 models.json 顶层字段（active_image_model / active_video_model），
由设置页「多模态模型」读写；生成工具按激活模型选择，未配置时回退第一个预置。
"""
import json
import os
import shutil
from typing import Any, Dict

from path_util import strip_vermagic

ACTIVE_IMAGE_FIELD = "active_image_model"
ACTIVE_VIDEO_FIELD = "active_video_model"
ACTIVE_SPEECH_FIELD = "active_voice"


def _repo_root() -> str:
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _strip_json_comments(text: str) -> str:
    """移除 JSONC 注释（// 与 /* */）。"""
    out: list = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if ch == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _models_path() -> str:
    return os.path.join(_repo_root(), "models.json")


def _ensure_models_file() -> None:
    """models.json 不存在时先从 models.example.json 复制生成，避免只写 active 字段丢失模型预置。"""
    if os.path.exists(_models_path()):
        return
    example = os.path.join(_repo_root(), "models.example.json")
    try:
        if os.path.exists(example):
            shutil.copyfile(example, _models_path())
    except Exception:
        pass


def _read_models_data() -> Dict[str, Any]:
    """读取 models.json 顶层（不存在或解析失败返回空 dict）。"""
    _ensure_models_file()
    try:
        with open(_models_path(), "r", encoding="utf-8-sig") as f:
            raw = f.read()
        data = json.loads(_strip_json_comments(raw))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_models_data(data: Dict[str, Any]) -> None:
    """把数据写回 models.json（JSONC 注释会被规范化移除）。"""
    _ensure_models_file()
    with open(_models_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_active_image_model() -> str:
    return str(_read_models_data().get(ACTIVE_IMAGE_FIELD) or "").strip()


def get_active_video_model() -> str:
    return str(_read_models_data().get(ACTIVE_VIDEO_FIELD) or "").strip()


def get_active_voice() -> str:
    return str(_read_models_data().get(ACTIVE_SPEECH_FIELD) or "").strip()


def set_active_image_model(model_id: str) -> None:
    data = _read_models_data()
    data[ACTIVE_IMAGE_FIELD] = str(model_id or "").strip()
    _write_models_data(data)


def set_active_video_model(model_id: str) -> None:
    data = _read_models_data()
    data[ACTIVE_VIDEO_FIELD] = str(model_id or "").strip()
    _write_models_data(data)


def set_active_voice(voice: str) -> None:
    data = _read_models_data()
    data[ACTIVE_SPEECH_FIELD] = str(voice or "").strip()
    _write_models_data(data)
