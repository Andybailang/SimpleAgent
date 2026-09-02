"""server_config — 仓库/路径、模型预置加载、费用估算与 Agent 创建。
"""
import os
import json
from typing import Any, Dict, List, Optional

import user_models
from path_util import realpath_clean, strip_vermagic
from tools.semantic_config import _semantic_setting
from engine import SimpleAgent

def _repo_root() -> str:
    """仓库根目录（本文件位于 <root>/src/agent/）。"""
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


SHARED_TEMP_DIR_NAME = "共享临时目录"


def _shared_temp_dir() -> str:
    """共享临时目录（程序目录内），不存在则创建；
    供聊天等非开发工作会话使用。"""
    path = os.path.join(_repo_root(), SHARED_TEMP_DIR_NAME)
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def _is_shared_temp_cwd(cwd: Optional[str]) -> bool:
    """cwd 是否为共享临时目录（新会话默认聊天模式判定）。"""
    if not cwd:
        return False
    try:
        return os.path.normcase(realpath_clean(str(cwd))) == os.path.normcase(realpath_clean(_shared_temp_dir()))
    except Exception:
        return False


def _strip_json_comments(text: str) -> str:
    """去掉 JSON 中的 // 行注释与 /* */ 块注释（保留字符串内的 //）。"""
    out = []
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


def _resolve_preset_api_key(p: Dict[str, Any]) -> str:
    """按配置解析预置的 API key：api_key_env 主变量 → api_key_fallback_envs
    依次回退 → api_key 字面值。每个变量先查进程/OS 环境，再查 semantic.env。"""
    names = [p.get("api_key_env", "")] + list(p.get("api_key_fallback_envs") or [])
    for name in names:
        name = str(name or "").strip()
        if not name:
            continue
        val = os.environ.get(name, "").strip()
        if not val:
            val = _semantic_setting(name, "")
        if val:
            return val
    return str(p.get("api_key") or "your api key")


def _build_language_entries(providers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把供应商列表（含嵌套 models）标准化为扁平的语言模型条目列表。

    条目结构（与旧 PRESET_MODELS 兼容，额外支持 api_auth）：
    {id, name, api_key, api_url, model, max_tokens, api_format, api_auth, stream_supported,
     billing, price_cache_miss, price_cache_hit, price_output}
    """
    entries: List[Dict[str, Any]] = []
    for p in providers or []:
        if not isinstance(p, dict):
            continue
        pname = str(p.get("name") or p.get("id") or "?")
        models = p.get("models") or [{
            "model": p.get("model", ""),
            "max_tokens": p.get("max_tokens", 4096),
        }]
        api_key = _resolve_preset_api_key(p)
        for m in models:
            if not isinstance(m, dict) or not m.get("model"):
                continue
            entries.append({
                "id": str(p.get("id") or ""),
                "name": pname,
                "api_key": api_key,
                "api_url": str(p.get("api_url") or ""),
                "model": str(m["model"]),
                "max_tokens": int(m.get("max_tokens") or p.get("max_tokens") or 4096),
                "api_format": str(m.get("api_format") or p.get("api_format") or "openai"),
                "api_auth": str(p.get("api_auth") or "x-api-key"),
                "billing": str(m.get("billing") or p.get("billing") or "paid"),
                "stream_supported": bool(m.get("stream_supported", True)),
                # 计费单价（元 / 百万 tokens），设置页「流量统计」费用估算用；未设置按 0
                "price_cache_miss": float(m.get("price_cache_miss") if m.get("price_cache_miss") is not None else (p.get("price_cache_miss") or 0)),
                "price_cache_hit": float(m.get("price_cache_hit") if m.get("price_cache_hit") is not None else (p.get("price_cache_hit") or 0)),
                "price_output": float(m.get("price_output") if m.get("price_output") is not None else (p.get("price_output") or 0)),
            })
    return entries


def _load_models_config() -> List[Dict[str, Any]]:
    """读取仓库根目录 models.json，返回标准化的语言模型条目列表。

    除预置外，还会导入用户自定义模型（user_models.json 的 models 段），
    使设置页「默认模型」、_create_agent / _resolve_preset_model 等一律可见。
    """
    models_path = os.path.join(_repo_root(), "models.json")
    example_path = os.path.join(_repo_root(), "models.example.json")
    try:
        if not os.path.exists(models_path) and os.path.exists(example_path):
            with open(example_path, "r", encoding="utf-8-sig") as src:
                raw = src.read()
            with open(models_path, "w", encoding="utf-8") as dst:
                dst.write(raw)
        if os.path.exists(models_path):
            with open(models_path, "r", encoding="utf-8-sig") as f:
                raw = f.read()
            data = json.loads(_strip_json_comments(raw))
            providers = data.get("models") if isinstance(data, dict) else data
        else:
            providers = []
    except Exception as e:
        print(f"[models] 读取 models.json 失败，回退到内置默认：{e}")
        providers = []
    providers = providers or []
    entries = _build_language_entries(providers)
    try:
        user = user_models.load_user_models()
        entries += _build_language_entries(user.get("models") or [])
    except Exception as e:
        print(f"[models] 读取 user_models.json 失败：{e}")
    return entries


PRESET_MODELS: List[Dict[str, Any]] = _load_models_config()


# 会话统计费用估算（USD / 每 1M tokens；默认 0 表示不估算，费用显示 —）
COST_INPUT_PER_1M = float(os.environ.get("BIGCODEX_COST_INPUT_PER_1M", "0") or 0)
COST_OUTPUT_PER_1M = float(os.environ.get("BIGCODEX_COST_OUTPUT_PER_1M", "0") or 0)
COST_ESTIMATE_ENABLED = COST_INPUT_PER_1M > 0 or COST_OUTPUT_PER_1M > 0


def _estimate_cost_usd(in_tok: int, out_tok: int) -> Optional[float]:
    """按配置单价估算一轮费用；未配置单价返回 None。"""
    if not COST_ESTIMATE_ENABLED:
        return None
    return in_tok / 1_000_000.0 * COST_INPUT_PER_1M + out_tok / 1_000_000.0 * COST_OUTPUT_PER_1M


def _resolve_preset_model(model_name: str) -> Optional[Dict[str, Any]]:
    """按模型名（或 provider_id:model / id / name）匹配预置的模型。"""
    if not model_name:
        return None
    for m in PRESET_MODELS:
        if model_name in (m["model"], m["id"], m["name"]):
            return m
        if m.get("id") and model_name == f"{m['id']}:{m['model']}":
            return m
    return None


def _is_local_model(bs: Dict[str, Any]) -> bool:
    """判断当前会话是否为 LocalLLM 模式（兼容预置 name/id/model 三种匹配）。"""
    model = bs.get("model") or ""
    if model in ("LocalLLM", "local"):
        return True
    fm = _resolve_preset_model(model) if model else None
    return bool(fm and (fm.get("id") == "local" or fm.get("name") == "LocalLLM"))


def _create_agent(model_name: str = "", cwd: Optional[str] = None, thinking_level: str = "off", plain_chat: bool = False, role_prompt: Optional[str] = None) -> SimpleAgent:
    """创建 Agent：优先用匹配到的预置模型，否则回退到 .env。cwd 用于文件工具的工作目录。"""
    fm = _resolve_preset_model(model_name) if model_name else None
    if fm is not None:
        base_url = fm["api_url"].rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        return SimpleAgent(
            api_key=fm["api_key"],
            base_url=base_url,
            model_name=fm["model"],
            max_tokens=fm["max_tokens"],
            cwd=cwd,
            api_format=fm.get("api_format", "openai"),
            api_auth=fm.get("api_auth", "x-api-key"),
            billing=fm.get("billing", "paid"),
            stream_supported=fm.get("stream_supported", True),
            thinking_level=thinking_level,
            plain_chat=plain_chat,
            mode="chat" if plain_chat else "work",
            role_prompt=role_prompt,
        )
    from env_util import load_env
    load_env()
    return SimpleAgent(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model_name=model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
        cwd=cwd,
        api_format=os.getenv("OPENAI_API_FORMAT", "openai"),
        thinking_level=thinking_level,
        plain_chat=plain_chat,
        mode="chat" if plain_chat else "work",
        role_prompt=role_prompt,
    )
def _read_config_env() -> Dict[str, str]:
    """从仓库根目录 config.env 读取配置项（端口、上下文窗口、压缩阈值等）。"""
    result: Dict[str, str] = {}
    try:
        with open(os.path.join(_repo_root(), "config.env"), "rb") as f:
            raw_bytes = f.read()
        for line in _decode_config_text(raw_bytes).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    except Exception:
        pass
    return result


def _decode_config_text(raw_bytes: bytes) -> str:
    """Decode config.env bytes; handles UTF-8 (BOM or not), UTF-16, GBK."""
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw_bytes.decode("utf-16")
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return raw_bytes.decode("utf-8-sig")
    if b"\x00" in raw_bytes:
        try:
            enc = "utf-16-be" if raw_bytes[0] == 0 else "utf-16-le"
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, ValueError):
            pass
    for enc in ("utf-8", "gbk"):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return raw_bytes.decode("utf-8", errors="replace")
