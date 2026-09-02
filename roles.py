"""
角色扮演配置（全局共享）
======================
角色定义存储于仓库根目录 roles.json（与 models.json 同级、同样模式，
roles.json 被 git 忽略，首次启动自动从 roles.example.json 复制生成）。

- 内置角色（编程 / 聊天）硬编码在后端，只读不可删改，提示词为空
  （走原 work / chat 系统提示词模板）；
- 用户自定义角色可增删改，提示词作为 system prompt 前缀，
  再拼接编程模式的工具说明等提示词。
"""
import json
import os
import uuid
from path_util import strip_vermagic
from typing import Any, Dict, List, Optional

ROLES_FILENAME = "roles.json"
ROLES_EXAMPLE_FILENAME = "roles.example.json"

BUILTIN_ROLE_PROGRAMMING: Dict[str, Any] = {
    "id": "programming",
    "name": "编程",
    "prompt": "",
    "builtin": True,
}

BUILTIN_ROLE_CHAT: Dict[str, Any] = {
    "id": "chat",
    "name": "聊天",
    "prompt": "",
    "builtin": True,
}


def _repo_root() -> str:
    """仓库根目录（本文件位于 <root>/src/agent/）。"""
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _roles_path() -> str:
    return os.path.join(_repo_root(), ROLES_FILENAME)


def _strip_json_comments(raw: str) -> str:
    """去掉 // 行注释（配置模板可读性，字符串内不含 // 的简单处理）。"""
    out = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        idx = line.find("//")
        out.append(line if idx < 0 else line[:idx])
    return "\n".join(out)


def _load_custom_roles() -> List[Dict[str, Any]]:
    """读取 roles.json（不存在则从 roles.example.json 生成），返回自定义角色列表。"""
    path = _roles_path()
    example = os.path.join(_repo_root(), ROLES_EXAMPLE_FILENAME)
    try:
        if not os.path.exists(path) and os.path.exists(example):
            with open(example, "r", encoding="utf-8-sig") as src:
                raw = src.read()
            with open(path, "w", encoding="utf-8") as dst:
                dst.write(raw)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.loads(_strip_json_comments(f.read()))
            items = data.get("roles") if isinstance(data, dict) else data
            if isinstance(items, list):
                return [it for it in items if isinstance(it, dict) and it.get("id")]
    except Exception as e:
        print(f"[roles] 读取 roles.json 失败：{e}")
    return []


def _save_custom_roles(items: List[Dict[str, Any]]) -> None:
    path = _roles_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"roles": items}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def get_roles() -> List[Dict[str, Any]]:
    """内置角色 + 自定义角色（内置在前）。"""
    return [dict(BUILTIN_ROLE_PROGRAMMING), dict(BUILTIN_ROLE_CHAT)] + _load_custom_roles()


def get_role(role_id: str) -> Optional[Dict[str, Any]]:
    """按 id 查角色；内置或自定义，找不到返回 None。"""
    if role_id == BUILTIN_ROLE_PROGRAMMING["id"]:
        return dict(BUILTIN_ROLE_PROGRAMMING)
    if role_id == BUILTIN_ROLE_CHAT["id"]:
        return dict(BUILTIN_ROLE_CHAT)
    for r in _load_custom_roles():
        if r.get("id") == role_id:
            return dict(r)
    return None


def create_role(name: str, prompt: str) -> Dict[str, Any]:
    """新增自定义角色。"""
    items = _load_custom_roles()
    role = {
        "id": f"role_{uuid.uuid4().hex[:8]}",
        "name": str(name or "").strip(),
        "prompt": str(prompt or "").strip(),
        "builtin": False,
    }
    items.append(role)
    _save_custom_roles(items)
    return dict(role)


def update_role(role_id: str, name: Optional[str] = None, prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """更新自定义角色；内置角色不可修改，返回 None。"""
    if role_id in (BUILTIN_ROLE_PROGRAMMING["id"], BUILTIN_ROLE_CHAT["id"]):
        return None
    items = _load_custom_roles()
    for r in items:
        if r.get("id") == role_id:
            if name is not None:
                r["name"] = str(name).strip()
            if prompt is not None:
                r["prompt"] = str(prompt).strip()
            _save_custom_roles(items)
            return dict(r)
    return None


def delete_role(role_id: str) -> bool:
    """删除自定义角色；内置角色不可删除。"""
    if role_id in (BUILTIN_ROLE_PROGRAMMING["id"], BUILTIN_ROLE_CHAT["id"]):
        return False
    items = _load_custom_roles()
    kept = [r for r in items if r.get("id") != role_id]
    if len(kept) == len(items):
        return False
    _save_custom_roles(kept)
    return True
