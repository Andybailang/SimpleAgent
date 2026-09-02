"""
视频生成模型配置（models.json 的 video_models 段）
=================================================
与 image_config 保持同一套解析规则：
- 数据源：仓库根目录 models.json（不存在时回退 models.example.json）
- API key 解析：进程环境变量 > src/agent/.env > semantic.env
- 激活模型：models.json 顶层 active_video_model（未配置回退第一个预置）
解析结果供内置工具 generate_video 与 GET /api/video-models 共用。
"""
import json
import os
from typing import Any, Dict, List, Optional

from path_util import strip_vermagic
import user_models
from multimodal_config import get_active_video_model


def _load_env_file(path: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return result


_SEMANTIC_ENV = _load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic.env"))


def _env_setting(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    return _SEMANTIC_ENV.get(name, "").strip()


def _repo_root() -> str:
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _strip_json_comments(text: str) -> str:
    out: List[str] = []
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


def _resolve_api_key(entry: Dict[str, Any]) -> str:
    names = [entry.get("api_key_env", "")] + list(entry.get("api_key_fallback_envs") or [])
    for name in names:
        name = str(name or "").strip()
        if not name:
            continue
        val = os.environ.get(name, "").strip()
        if not val:
            val = _env_setting(name)
        if val:
            return val
    return str(entry.get("api_key") or "")


def _build_video_entries(items: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """把 video_models 条目列表标准化为视频模型条目。"""
    entries: List[Dict[str, Any]] = []
    for entry in items or []:
        if not isinstance(entry, dict) or not entry.get("model"):
            continue
        entries.append({
            "id": str(entry.get("id") or ""),
            "name": str(entry.get("name") or entry.get("id") or "?"),
            "api_key": _resolve_api_key(entry),
            "api_url": str(entry.get("api_url") or "https://apihub.agnes-ai.com/v1/videos"),
            "query_url": str(entry.get("query_url") or ""),
            "api_auth": str(entry.get("api_auth") or "bearer"),
            "model": str(entry["model"]),
            "kind": str(entry.get("kind") or "agnes-v2.0"),
            "size": str(entry.get("size") or ""),
            "aspect_ratio": str(entry.get("aspect_ratio") or ""),
            "seconds": str(entry.get("seconds") or ""),
            "video_width": int(entry.get("video_width") or 832),
            "video_height": int(entry.get("video_height") or 448),
            "num_frames": int(entry.get("num_frames") or 81),
            "frame_rate": int(entry.get("frame_rate") or 24),
            "num_inference_steps": int(entry.get("num_inference_steps") or 20),
        })
    return entries


def _load_video_models_config() -> List[Dict[str, Any]]:
    root = _repo_root()
    raw = ""
    for path in (os.path.join(root, "models.json"), os.path.join(root, "models.example.json")):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                raw = f.read()
            break
        except OSError:
            continue
    entries: List[Dict[str, Any]] = []
    if raw.strip():
        try:
            data = json.loads(_strip_json_comments(raw))
        except Exception:
            data = None
        items = data.get("video_models") if isinstance(data, dict) else None
        entries += _build_video_entries(items)
    try:
        user = user_models.load_user_models()
        entries += _build_video_entries(user.get("video_models") or [])
    except Exception as e:
        print(f"[video_config] 读取 user_models.json 失败：{e}")
    return entries


VIDEO_MODELS: List[Dict[str, Any]] = _load_video_models_config()


def resolve_video_model(name: str = "") -> Optional[Dict[str, Any]]:
    """按 id/name/model 匹配视频模型；未指定时优先激活模型，未命中回退第一个配置。"""
    if not VIDEO_MODELS:
        return None
    if name:
        for m in VIDEO_MODELS:
            if name in (m["model"], m["id"], m["name"]):
                return m
    active = get_active_video_model()
    if active:
        for m in VIDEO_MODELS:
            if active in (m["model"], m["id"], m["name"]):
                return m
    return VIDEO_MODELS[0]
