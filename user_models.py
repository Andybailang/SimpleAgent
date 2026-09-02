"""user_models — 自定义模型配置（user_models.json）。

设置页「模型设定」用：允许用户在当前预置 models.json 之外，通过界面新增/编辑
自定义模型（语言/图像/视频三大类），保存到仓库根目录 user_models.json。

结构（与 models.json 同构，但只含用户新增的条目）::

    {
      "models": [        # 语言模型（供应商级，每个供应商可挂多个模型）
        {
          "id": "...", "name": "...", "api_key_env": "...",
          "api_url": "...", "api_format": "anthropic", "api_auth": "x-api-key",
          "models": [
            {"model": "...", "max_tokens": 4096, "billing": "free"},
            {"model": "...", "max_tokens": 8192, "billing": "paid",
             "price_cache_miss": 1.5, "price_cache_hit": 0.05, "price_output": 4.5}
          ]
        }
      ],
      "image_models": [ ... ],   # 图像生成模型
      "video_models": [ ... ]    # 视频生成模型
    }

程序加载模型时（server_config / image_config / video_config）会额外导入本文件，
未填的可选属性由各 *_config 的标准化逻辑自动补默认值，因此这里保存时只做必填项校验，
其余属性按用户输入原样保留。
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

from path_util import strip_vermagic


def _repo_root() -> str:
    """仓库根目录（本文件位于 <root>/src/agent/）。"""
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _user_models_path() -> str:
    return os.path.join(_repo_root(), "user_models.json")


def _strip_json_comments(text: str) -> str:
    """去掉 JSON 中的 // 行注释与 /* */ 块注释（保留字符串内的 //）。"""
    out: List[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
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


def load_user_models() -> Dict[str, Any]:
    """读取仓库根目录 user_models.json，不存在/解析失败返回空 dict。"""
    path = _user_models_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
        data = json.loads(_strip_json_comments(raw))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[user_models] 读取 user_models.json 失败：{e}")
        return {}


def _make_id(name: str, existing_ids: Optional[set] = None) -> str:
    """由名称生成唯一的 id（slug；仅保留字母数字，其余用 '-' 替换）。"""
    base = re.sub(r"[^A-Za-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    if not base:
        base = "user-model"
    existing = existing_ids or set()
    candidate = base
    i = 2
    while candidate in existing:
        candidate = f"{base}-{i}"
        i += 1
    return candidate


def _clean_entry(entry: Any) -> Dict[str, Any]:
    """把条目统一为 dict，裁剪字符串两端空白，去掉空字符串字段。

    空字符串视为未填写，交由各 *_config 的标准化逻辑补默认值，故落盘时丢弃。
    """
    if not isinstance(entry, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in entry.items():
        if isinstance(value, str):
            value = value.strip()
        if value == "":
            continue
        out[key] = value
    return out


def _ensure_id(entry: Dict[str, Any], used_ids: set) -> None:
    """若条目缺 id，按名称生成唯一 id；并登记到 used_ids。"""
    if entry.get("id"):
        entry["id"] = str(entry["id"]).strip()
        used_ids.add(entry["id"])
        return
    entry["id"] = _make_id(entry.get("name") or entry.get("model") or "model", used_ids)
    used_ids.add(str(entry["id"]))


def _normalize_language_models(providers: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """标准化语言模型供应商列表（含嵌套 models），供应商 id 保证唯一。"""
    used_ids: set = set()
    out: List[Dict[str, Any]] = []
    for p in providers or []:
        if not isinstance(p, dict):
            continue
        models = p.get("models")
        norm_models: List[Dict[str, Any]] = []
        if isinstance(models, list):
            for m in models:
                norm_models.append(_clean_entry(m))
        if not norm_models:
            continue
        norm_p = _clean_entry({k: v for k, v in p.items() if k != "models"})
        norm_p["models"] = norm_models
        _ensure_id(norm_p, used_ids)
        out.append(norm_p)
    return out


def _normalize_simple_models(items: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """标准化平铺的图像/视频模型列表，id 保证唯一。"""
    used_ids: set = set()
    out: List[Dict[str, Any]] = []
    for e in items or []:
        norm = _clean_entry(e)
        if not norm:
            continue
        _ensure_id(norm, used_ids)
        out.append(norm)
    return out


def save_user_models(data: Dict[str, Any]) -> Dict[str, Any]:
    """校验并保存 user_models.json（归一化 id/空白字段后落盘），返回归一化后的数据。"""
    if not isinstance(data, dict):
        data = {}
    body: Dict[str, Any] = {
        "models": _normalize_language_models(data.get("models")),
        "image_models": _normalize_simple_models(data.get("image_models")),
        "video_models": _normalize_simple_models(data.get("video_models")),
    }
    with open(_user_models_path(), "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)
    return body


# --- 校验 -------------------------------------------------------------------

ALLOWED_API_FORMATS = {"anthropic", "openai", "response"}
ALLOWED_API_AUTHS = {"bearer", "x-api-key"}
ALLOWED_BILLING = {"free", "paid"}
ALLOWED_IMAGE_KINDS = {"kolors", "agnes-image"}
ALLOWED_VIDEO_KINDS = {"agnes-v2.0", "agnes-v2.5", "agnes-v2.5-flash"}


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _num_error(label: str, value: Any) -> Optional[str]:
    """值为空时返回 None（空由 required 判断）；非数字返回错误文案。"""
    if _is_blank(value):
        return None
    try:
        import math
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            raise ValueError
    except (TypeError, ValueError):
        return f"{label} 必须是有效数字"
    return None


def _check_text(entry: Dict[str, Any], key: str, label: str, ctx: str, errors: List[str]) -> None:
    if _is_blank(entry.get(key)):
        errors.append(f"{ctx}: 必填 {label}（{key}）")


def _check_enum(value: Any, allowed: set, label: str, ctx: str, errors: List[str]) -> None:
    if _is_blank(value):
        return
    v = str(value).strip().lower()
    if v not in allowed:
        errors.append(f"{ctx}: {label} 必须是 {'/'.join(sorted(allowed))} 之一")


def _validate_language_provider(index: int, p: Any, errors: List[str]) -> None:
    ctx = f"语言模型 #{index + 1}"
    if not isinstance(p, dict):
        errors.append(f"{ctx}: 必须是一个对象")
        return
    _check_text(p, "name", "名称", ctx, errors)
    _check_text(p, "api_key_env", "API key 环境变量名", ctx, errors)
    _check_text(p, "api_url", "API 地址", ctx, errors)
    _check_text(p, "api_format", "API 协议", ctx, errors)
    _check_text(p, "api_auth", "认证方式", ctx, errors)
    _check_enum(p.get("api_format"), ALLOWED_API_FORMATS, "api_format", ctx, errors)
    _check_enum(p.get("api_auth"), ALLOWED_API_AUTHS, "api_auth", ctx, errors)

    models = p.get("models")
    if not isinstance(models, list) or not models:
        errors.append(f"{ctx}: 至少需要一个模型")
        return
    for j, m in enumerate(models):
        mctx = f"{ctx} 模型 #{j + 1}"
        if not isinstance(m, dict):
            errors.append(f"{mctx}: 必须是一个对象")
            continue
        _check_text(m, "model", "模型名", mctx, errors)
        _check_text(m, "max_tokens", "最大 token", mctx, errors)
        err = _num_error(f"{mctx} max_tokens", m.get("max_tokens"))
        if err:
            errors.append(err)
        billing = str(m.get("billing") or "").strip().lower()
        _check_text(m, "billing", "计费类型", mctx, errors)
        _check_enum(m.get("billing"), ALLOWED_BILLING, "billing", mctx, errors)
        if billing == "paid":
            for key in ("price_cache_miss", "price_cache_hit", "price_output"):
                _check_text(m, key, f"收费单价 {key}", mctx, errors)
                err = _num_error(f"{mctx} {key}", m.get(key))
                if err:
                    errors.append(err)
        else:
            for key in ("price_cache_miss", "price_cache_hit", "price_output"):
                err = _num_error(f"{mctx} {key}", m.get(key))
                if err:
                    errors.append(err)


def _validate_simple_model(kind_label: str, allowed_kinds: set, index: int, p: Any, errors: List[str]) -> None:
    ctx = f"{kind_label} #{index + 1}"
    if not isinstance(p, dict):
        errors.append(f"{ctx}: 必须是一个对象")
        return
    _check_text(p, "name", "名称", ctx, errors)
    _check_text(p, "api_url", "API 地址", ctx, errors)
    _check_text(p, "model", "模型名", ctx, errors)
    # API key：允许 api_key_env 或 api_key 字面值，二者至少其一
    if _is_blank(p.get("api_key_env")) and _is_blank(p.get("api_key")):
        errors.append(f"{ctx}: 必填 API key 环境变量名（api_key_env）或 API key（api_key）")
    _check_enum(p.get("api_auth"), ALLOWED_API_AUTHS, "api_auth", ctx, errors)
    _check_enum(p.get("kind"), allowed_kinds, "kind", ctx, errors)
    for key in (
        "batch_size", "num_inference_steps", "guidance_scale",
        "video_width", "video_height", "num_frames", "frame_rate",
    ):
        err = _num_error(f"{ctx} {key}", p.get(key))
        if err:
            errors.append(err)


def validate_user_models(data: Any) -> List[str]:
    """校验自定义模型必填项；返回错误文案列表（空列表表示通过）。"""
    if not isinstance(data, dict):
        return ["自定义模型数据必须是 JSON 对象"]
    errors: List[str] = []
    for i, p in enumerate(data.get("models") or []):
        _validate_language_provider(i, p, errors)
    for i, p in enumerate(data.get("image_models") or []):
        _validate_simple_model("图像模型", ALLOWED_IMAGE_KINDS, i, p, errors)
    for i, p in enumerate(data.get("video_models") or []):
        _validate_simple_model("视频模型", ALLOWED_VIDEO_KINDS, i, p, errors)
    return errors
