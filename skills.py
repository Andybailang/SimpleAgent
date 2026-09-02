"""Skills 管理：全局 <程序目录>/skills + 项目 <cwd>/.bigcodex/skills 两级。

SKILL.md 约定与 Claude Code / 原版 tokenicode（Tauri Rust 实现）一致：
- 目录名 = 技能名，目录内必须有 SKILL.md；
- 文件头可选 YAML frontmatter（--- 包裹）：description / disable-model-invocation /
  user-invocable / allowed-tools / argument-hint / model / context / agent / version；
- description 缺省时取正文第一行（去掉 #）或目录名。

路径安全：技能文件只允许位于 <root>/<技能名>/SKILL.md，技能名为单层目录，
防止路径穿越到任意位置。
"""
import os
import shutil
from typing import Any, Dict, List, Optional

from path_util import strip_vermagic
from fastapi import APIRouter, Body, HTTPException

try:
    import yaml
    YAML_AVAILABLE = True
except Exception:  # pragma: no cover - pyyaml 缺失时降级为不解析 frontmatter
    yaml = None
    YAML_AVAILABLE = False

router = APIRouter(prefix="/api/skills", tags=["skills"])

SKILL_FILE = "SKILL.md"


def _repo_root() -> str:
    """程序根目录（本文件位于 <root>/src/agent/）。"""
    return strip_vermagic(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))


def _global_skills_root() -> str:
    _migrate_legacy_skills()
    return os.path.join(_repo_root(), "skills")


def _migrate_legacy_skills() -> None:
    """一次性迁移旧 ~/.claude/skills 到程序目录 skills/（同名技能不覆盖，幂等）。"""
    try:
        legacy = os.path.join(os.path.expanduser("~"), ".claude", "skills")
        target = os.path.join(_repo_root(), "skills")
        if not os.path.isdir(legacy) or not os.path.isdir(target):
            return
        for entry in os.listdir(legacy):
            src = os.path.join(legacy, entry)
            dst = os.path.join(target, entry)
            if (
                os.path.isdir(src)
                and os.path.isfile(os.path.join(src, SKILL_FILE))
                and not os.path.isdir(dst)
            ):
                shutil.copytree(src, dst)
    except Exception:
        pass


def _project_skills_root(cwd: Optional[str]) -> str:
    return os.path.join(str(cwd or ""), ".bigcodex", "skills")


def _allowed_roots(cwd: Optional[str]) -> List[str]:
    roots = [_global_skills_root()]
    if cwd:
        roots.append(_project_skills_root(cwd))
    return roots


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def is_skill_path(path: str, cwd: Optional[str]) -> bool:
    """path 必须是 <root>/<技能名>/SKILL.md，技能名为单层目录。"""
    if not path:
        return False
    try:
        real = _norm(path)
    except Exception:
        return False
    for root in _allowed_roots(cwd):
        root_real = _norm(root)
        if not (real == root_real or real.startswith(root_real + os.sep)):
            continue
        rel = os.path.relpath(real, root_real)
        parts = rel.split(os.sep)
        if len(parts) == 2 and parts[1].lower() == SKILL_FILE.lower() and parts[0] not in ("", ".", ".."):
            return True
    return False


def parse_frontmatter(content: str) -> tuple:
    """解析 SKILL.md 头部 YAML frontmatter。

    返回 (frontmatter dict, body str)；无 frontmatter 或解析失败时返回 ({}, 原内容)。
    """
    if not YAML_AVAILABLE:
        return {}, content
    trimmed = content.lstrip()
    if not trimmed.startswith("---"):
        return {}, content
    rest = trimmed[3:]
    marker = rest.find("\n---")
    if marker < 0:
        return {}, content
    yaml_str = rest[:marker]
    body = rest[marker + 4:]
    body = body.lstrip("\n")
    try:
        fm = yaml.safe_load(yaml_str) or {}
    except Exception:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, body


def _description_from(fm: Dict[str, Any], body: str, name: str) -> str:
    desc = fm.get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    for line in body.splitlines():
        s = line.strip()
        if s:
            return s.lstrip("#").strip()
    return name


def set_frontmatter_field(content: str, field: str, value: Optional[str]) -> str:
    """设置/移除 frontmatter 中的某个字段（行级编辑，保留其余内容与正文）。

    value 为 None 时移除该字段；若 frontmatter 因此为空则整个移除。
    无 frontmatter 且 value 非 None 时自动创建。
    """
    field_prefix = field + ":"
    trimmed = content.lstrip()
    if trimmed.startswith("---"):
        after_open = trimmed[3:]
        close_idx = after_open.find("\n---")
        if close_idx >= 0:
            yaml_section = after_open[:close_idx]
            body = after_open[close_idx + 4:]
            lines = [
                line for line in yaml_section.splitlines()
                if not line.strip().startswith(field_prefix)
            ]
            if value is not None:
                lines.append(f"{field}: {value}")
            if not any(line.strip() for line in lines):
                return body.lstrip("\n")
            result = "---\n" + "\n".join(lines) + "\n---" + body
            return result
    if value is None:
        return content
    return f"---\n{field}: {value}\n---\n{content}"


def _read_file_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def scan_skill_dir(root: str, scope: str) -> List[Dict[str, Any]]:
    """扫描单个技能目录（<root>/<名称>/SKILL.md），返回 SkillInfo 列表。"""
    found: List[Dict[str, Any]] = []
    try:
        entries = sorted(os.listdir(root))
    except Exception:
        return found
    for entry in entries:
        skill_dir = os.path.join(root, entry)
        skill_file = os.path.join(skill_dir, SKILL_FILE)
        if not (os.path.isdir(skill_dir) and os.path.isfile(skill_file)):
            continue
        try:
            content = _read_file_text(skill_file)
        except Exception:
            content = ""
        fm, body = parse_frontmatter(content)
        found.append({
            "name": entry,
            "description": _description_from(fm, body, entry),
            "path": skill_file,
            "scope": scope,
            "disable_model_invocation": fm.get("disable-model-invocation"),
            "user_invocable": fm.get("user-invocable"),
            "allowed_tools": fm.get("allowed-tools"),
            "argument_hint": fm.get("argument-hint"),
            "model": fm.get("model"),
            "context": fm.get("context"),
            "agent": fm.get("agent"),
            "version": fm.get("version"),
        })
    return found


def list_skills(cwd: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出全部技能：全局 + 项目（cwd 为空时只列全局）。"""
    skills = scan_skill_dir(_global_skills_root(), "global")
    if cwd:
        skills += scan_skill_dir(_project_skills_root(cwd), "project")
    return skills


def read_skill(path: str, cwd: Optional[str] = None) -> str:
    if not is_skill_path(path, cwd):
        raise ValueError("技能路径不在允许范围内")
    try:
        return _read_file_text(path)
    except Exception as e:
        raise ValueError(f"读取技能失败：{e}")


def write_skill(path: str, content: str, cwd: Optional[str] = None) -> None:
    if not is_skill_path(path, cwd):
        raise ValueError("技能路径不在允许范围内")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise ValueError(f"写入技能失败：{e}")


def delete_skill(path: str, cwd: Optional[str] = None) -> None:
    if not is_skill_path(path, cwd):
        raise ValueError("技能路径不在允许范围内")
    skill_dir = os.path.dirname(path)
    try:
        shutil.rmtree(skill_dir)
    except Exception as e:
        raise ValueError(f"删除技能失败：{e}")


def toggle_skill_enabled(path: str, enabled: bool, cwd: Optional[str] = None) -> None:
    """切换 disable-model-invocation 开关：enabled=True 表示允许模型自动调用（移除标记）。"""
    if not is_skill_path(path, cwd):
        raise ValueError("技能路径不在允许范围内")
    try:
        content = _read_file_text(path)
    except Exception as e:
        raise ValueError(f"读取技能失败：{e}")
    new_content = set_frontmatter_field(content, "disable-model-invocation", None if enabled else "true")
    write_skill(path, new_content, cwd)


# ---- HTTP 接口 ----

@router.get("")
async def api_skills_list(cwd: Optional[str] = ""):
    """技能列表（全局 + 项目）。"""
    return list_skills(cwd or None)


@router.get("/content")
async def api_skills_read(path: str = "", cwd: Optional[str] = ""):
    try:
        return read_skill(path, cwd or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/content")
async def api_skills_write(payload: Dict[str, Any] = Body(default=None)):
    body = payload or {}
    try:
        write_skill(body.get("path", ""), body.get("content", ""), body.get("cwd"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@router.delete("/content")
async def api_skills_delete(path: str = "", cwd: Optional[str] = ""):
    try:
        delete_skill(path, cwd or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}


@router.post("/toggle")
async def api_skills_toggle(payload: Dict[str, Any] = Body(default=None)):
    body = payload or {}
    enabled = bool(body.get("enabled"))
    try:
        toggle_skill_enabled(body.get("path", ""), enabled, body.get("cwd"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "disable_model_invocation": None if enabled else True}
