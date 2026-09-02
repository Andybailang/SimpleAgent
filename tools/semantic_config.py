"""
SemanticSearch 语义搜索配置（独立 env：src/agent/semantic.env，便于修改）
优先级：进程环境变量 > semantic.env > 内置默认值。
"""
import os
from typing import Dict


SEMANTIC_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "semantic.env"
)


def _load_semantic_env(path: str) -> Dict[str, str]:
    """极简 .env 解析（KEY=VALUE、# 注释），用于读取 semantic.env。"""
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


_SEMANTIC_ENV = _load_semantic_env(SEMANTIC_CONFIG_PATH)


def _semantic_setting(name: str, default: str = "") -> str:
    """读取语义搜索配置：进程环境变量优先，其次 semantic.env，最后默认值。"""
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    return _SEMANTIC_ENV.get(name, default).strip() or default


def _semantic_int(name: str, default: int) -> int:
    """读取语义搜索整数配置，解析失败时回退默认值。"""
    try:
        return max(1, int(_semantic_setting(name, str(default))))
    except (TypeError, ValueError):
        return default


SEMANTIC_SEARCH_ENABLED = _semantic_setting("SEMANTIC_SEARCH_ENABLED", "true").lower() not in (
    "0", "false", "no", "off",
)
SEMANTIC_SEARCH_API_BASE = _semantic_setting(
    "SEMANTIC_SEARCH_API_BASE", "https://api.siliconflow.cn/v1").rstrip("/")
SEMANTIC_SEARCH_API_KEY = _semantic_setting(
    "SEMANTIC_SEARCH_API_KEY", "") or _semantic_setting("SILICONFLOW_API_KEY", "")
SEMANTIC_SEARCH_MODEL = _semantic_setting("SEMANTIC_SEARCH_MODEL", "BAAI/bge-m3")
SEMANTIC_SEARCH_TOP_K = _semantic_int("SEMANTIC_SEARCH_TOP_K", 5)
SEMANTIC_SEARCH_CHUNK_SIZE = _semantic_int("SEMANTIC_SEARCH_CHUNK_SIZE", 800)
SEMANTIC_SEARCH_CHUNK_OVERLAP = _semantic_int("SEMANTIC_SEARCH_CHUNK_OVERLAP", 80)
SEMANTIC_SEARCH_MAX_FILES = _semantic_int("SEMANTIC_SEARCH_MAX_FILES", 100)
SEMANTIC_SEARCH_MAX_CHUNKS = _semantic_int("SEMANTIC_SEARCH_MAX_CHUNKS", 200)
SEMANTIC_SEARCH_MAX_FILE_BYTES = _semantic_int("SEMANTIC_SEARCH_MAX_FILE_BYTES", 200000)
SEMANTIC_SEARCH_TIMEOUT = _semantic_int("SEMANTIC_SEARCH_TIMEOUT", 60)
