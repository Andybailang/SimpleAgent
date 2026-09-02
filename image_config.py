"""
图片生成模型配置（models.json 的 image_models 段）
==================================================
与 server.py 的文本模型预置解析规则保持一致：
- 数据源：仓库根目录 models.json（不存在时回退 models.example.json）
- API key 解析：进程环境变量 > src/agent/.env（由 server/cli 加载）> semantic.env
解析结果供内置工具 generate_image 与 GET /api/image-models 共用。
"""
import json
import os
from typing import Any, Dict, List, Optional

from path_util import strip_vermagic
import user_models
from multimodal_config import get_active_image_model


_SEMANTIC_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic.env")


def _load_env_file(path: str) -> Dict[str, str]:
    """极简 .env 解析（KEY=VALUE、# 注释），与 semantic_config 保持一致。"""
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


_SEMANTIC_ENV = _load_env_file(_SEMANTIC_ENV_PATH)


def _env_setting(name: str) -> str:
    """读取配置：进程环境变量优先，其次 src/agent/semantic.env。"""
    val = os.environ.get(name, "").strip()
    if val:
        return val
    return _SEMANTIC_ENV.get(name, "").strip()


def _repo_root() -> str:
    """仓库根目录（本文件位于 <root>/src/agent/）。"""
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _strip_json_comments(text: str) -> str:
    """移除 JSONC 注释（// 与 /* */），与 server.py 保持一致。"""
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
    """按配置解析图片模型 API key：api_key_env 主变量 → api_key_fallback_envs
    依次回退 → api_key 字面值。每个变量先查进程/OS 环境，再查 semantic.env。"""
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


def _build_image_entries(items: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """把 image_models 条目列表标准化为图片模型条目。"""
    entries: List[Dict[str, Any]] = []
    for entry in items or []:
        if not isinstance(entry, dict) or not entry.get("model"):
            continue
        entries.append({
            "id": str(entry.get("id") or ""),
            "name": str(entry.get("name") or entry.get("id") or "?"),
            "api_key": _resolve_api_key(entry),
            "api_url": str(entry.get("api_url") or "https://api.siliconflow.cn/v1/images/generations"),
            "api_auth": str(entry.get("api_auth") or "bearer"),
            "model": str(entry["model"]),
            "kind": str(entry.get("kind") or "kolors"),
            "image_size": str(entry.get("image_size") or "1024x1024"),
            "ratio": str(entry.get("ratio") or "1:1"),
            "batch_size": int(entry.get("batch_size") or 1),
            "num_inference_steps": int(entry.get("num_inference_steps") or 20),
            "guidance_scale": float(entry.get("guidance_scale") or 7.5),
        })
    return entries


def _load_image_models_config() -> List[Dict[str, Any]]:
    """读取 models.json（回退 models.example.json）的 image_models 段，标准化为条目列表。"""
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
        items = data.get("image_models") if isinstance(data, dict) else None
        entries += _build_image_entries(items)
    try:
        user = user_models.load_user_models()
        entries += _build_image_entries(user.get("image_models") or [])
    except Exception as e:
        print(f"[image_config] 读取 user_models.json 失败：{e}")
    return entries


IMAGE_MODELS: List[Dict[str, Any]] = _load_image_models_config()


def resolve_image_model(name: str = "") -> Optional[Dict[str, Any]]:
    """按 id/name/model 匹配图片模型；未指定时优先激活模型，未命中回退第一个配置。"""
    if not IMAGE_MODELS:
        return None
    if name:
        for m in IMAGE_MODELS:
            if name in (m["model"], m["id"], m["name"]):
                return m
    active = get_active_image_model()
    if active:
        for m in IMAGE_MODELS:
            if active in (m["model"], m["id"], m["name"]):
                return m
    return IMAGE_MODELS[0]
