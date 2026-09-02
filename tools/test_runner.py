"""
TestRunner 工具：检测并运行指定单元测试，返回 通过/失败 与错误信息。

与 Bash 不同，本工具聚焦「跑测试」场景：
- 自动识别测试框架（pytest / unittest / vitest / jest / mocha / npm test 等）；
- 以子进程在项目目录内执行测试命令，支持超时与输出截断；
- 返回结构化的 通过/失败 判定 + 测试汇总 + 失败/错误明细，便于模型据此修复代码。
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

from .base import BaseTool, Tool, ToolPermission, ToolMode, ToolContext

COMMAND_TIMEOUT = 300  # 默认超时秒数（测试可能较慢），上限 600 秒
COMMAND_OUTPUT_LIMIT = 30000  # 原始输出保留上限（截断保留尾部，保住汇总与失败信息）
ERROR_DETAIL_LIMIT = 8000  # 失败/错误明细展示上限

SUPPORTED_FRAMEWORKS = {"auto", "pytest", "unittest", "vitest", "jest", "mocha", "npm", "node"}
JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
# 常见虚拟环境目录名（用于为被测项目挑选正确的 Python 解释器）
VENV_DIR_NAMES = {".venv", "venv", "env"}

# 测试目标名安全性：目标是文件路径或 pytest 节点 id（如 tests/test_a.py::TestC::test_m）。
# 白名单字符可防 shell 注入（Windows 下用 shell=True 时的二次防御）。
_SAFE_TARGET_RE = re.compile(r"^[A-Za-z0-9_./\\:@\[\]()\-]+$")


def _is_safe_target(target: str) -> bool:
    return bool(_SAFE_TARGET_RE.match(target))


def _pytest_available(python_exe: Optional[str] = None) -> bool:
    """判断指定解释器里是否有 pytest；对当前解释器用 find_spec（快），对其它解释器用子进程探测。"""
    py = python_exe or sys.executable
    try:
        if os.path.normcase(py) == os.path.normcase(sys.executable):
            return importlib.util.find_spec("pytest") is not None
    except Exception:
        return False
    try:
        r = subprocess.run([py, "-c", "import pytest"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def _venv_python(venv_dir: str) -> Optional[str]:
    """返回虚拟环境根目录下的 Python 可执行文件路径（不存在则 None）。"""
    if os.name == "nt":
        cand = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        cand = os.path.join(venv_dir, "bin", "python")
    return cand if os.path.isfile(cand) else None


def _discover_venv_python(cwd: str, target: str) -> Optional[str]:
    """从测试目标向上（到 cwd）查找项目虚拟环境的 Python 解释器。"""
    base = target.split("::")[0]
    start = base if base not in (".", "") else cwd
    if not os.path.isabs(start):
        start = os.path.join(cwd, start)
    cwd_abs = os.path.abspath(cwd)
    dirs = []
    cur = os.path.abspath(start if os.path.isdir(start) else os.path.dirname(start))
    while True:
        dirs.append(cur)
        if cur == cwd_abs or os.path.dirname(cur) == cur:
            break
        parent = os.path.dirname(cur)
        # 不越过 cwd（项目根）向上乱找，避免抓到无关的系统目录
        if not (parent == cwd_abs or parent.startswith(cwd_abs + os.sep)):
            break
        cur = parent
    if cwd_abs not in dirs:
        dirs.append(cwd_abs)
    seen = set()
    for d in dirs:
        for name in VENV_DIR_NAMES:
            venv_dir = os.path.join(d, name)
            if venv_dir in seen:
                continue
            seen.add(venv_dir)
            py = _venv_python(venv_dir)
            if py:
                return py
    return None


def _runtime_venv_label() -> str:
    """当前 agent 运行时解释器是否本身处于虚拟环境。"""
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return "虚拟环境"
    return "系统 Python"


def _decode_output(raw: bytes) -> str:
    """解码子进程输出，尝试常见编码。"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """终止子进程（Windows 需要连带杀掉子进程树）。"""
    try:
        if sys.platform == "win32":
            subprocess.run(
                f"taskkill /PID {proc.pid} /T /F",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _quote_win(arg: str) -> str:
    """Windows cmd 下把单个参数包成安全双引号（cmd 内双引号转义为两个双引号）。"""
    return '"' + arg.replace('"', '""') + '"'


def _run_cmd(cmd: List[str], cwd: str, timeout_s: int) -> Tuple[int, str]:
    """执行命令列表，返回 (退出码, stdout+stderr 合并文本)。"""
    try:
        if os.name == "nt":
            # Windows 下 npm/npx 是 .cmd shim，需 shell=True；逐参数加引号防注入
            command = " ".join(_quote_win(a) for a in cmd)
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                shell=False,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        try:
            out_bytes, err_bytes = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            return -1, f"命令超时（>{timeout_s} 秒），已终止"
        out = _decode_output(out_bytes)
        err = _decode_output(err_bytes)
        combined = out if not err else (f"{out}\n[stderr]\n{err}" if out else err)
        return proc.returncode, combined
    except FileNotFoundError:
        return -1, "错误：找不到命令或可执行文件"
    except Exception as e:
        return -1, f"执行命令出错: {str(e)}"


def _find_package_json(start: str, cwd: str) -> Optional[str]:
    """从 start 向上（到 cwd 为止）查找最近的 package.json。"""
    start = os.path.abspath(start)
    cwd = os.path.abspath(cwd)
    cur = start if os.path.isdir(start) else os.path.dirname(start)
    while True:
        candidate = os.path.join(cur, "package.json")
        if os.path.isfile(candidate):
            return candidate
        if cur == cwd or os.path.dirname(cur) == cur:
            break
        cur = os.path.dirname(cur)
    # 若不在 cwd 内，再查一次 cwd 本身
    root_candidate = os.path.join(cwd, "package.json")
    if os.path.isfile(root_candidate):
        return root_candidate
    return None


def _detect_node_framework(start_path: str, cwd: str) -> Optional[str]:
    """从最近 package.json 探测 JS 测试框架：vitest > jest > mocha；有 test 脚本则回退 npm。"""
    pkg_path = _find_package_json(start_path, cwd)
    if not pkg_path:
        return None
    try:
        import json

        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg = json.load(f)
    except Exception:
        return None
    deps = {}
    for section in ("dependencies", "devDependencies"):
        deps.update(pkg.get(section, {}) or {})
    for name in ("vitest", "jest", "mocha"):
        if name in deps:
            return name
    if pkg.get("scripts", {}).get("test"):
        return "npm"
    return None


def _resolve_python(cwd: str, target: str) -> Tuple[str, str]:
    """挑选用来跑测试的 Python 解释器，返回 (python_exe, 环境说明文本)。

    优先级：目标项目里发现的虚拟环境 > 激活的 VIRTUAL_ENV > 当前运行时解释器。
    若选中的解释器与 agent 运行时不同（说明存在环境差异），会在说明里提示。
    """
    discovered = _discover_venv_python(cwd, target)
    if discovered:
        if os.path.normcase(discovered) != os.path.normcase(sys.executable):
            note = (f"项目虚拟环境：{discovered}\n"
                    f"注意：检测到项目 Python 环境与 agent 运行时解释器 "
                    f"{sys.executable}（{_runtime_venv_label()}）不同，测试将使用项目环境的 Python 运行。")
        else:
            note = f"项目虚拟环境：{discovered}（与当前运行时一致）"
        return discovered, note
    ve = os.environ.get("VIRTUAL_ENV")
    if ve:
        py = _venv_python(ve)
        if py and os.path.normcase(py) != os.path.normcase(sys.executable):
            return py, f"激活的虚拟环境 VIRTUAL_ENV={ve}，使用其 Python：{py}"
        if py:
            return py, f"激活的虚拟环境 VIRTUAL_ENV={ve}（与当前运行时一致）"
    return sys.executable, f"运行时解释器：{sys.executable}（{_runtime_venv_label()}）"


def _detect_framework(target: str, hint: str, cwd: str,
                      python_exe: Optional[str] = None) -> Tuple[Optional[str], str]:
    """返回 (框架, 错误信息)。框架为 None 时表示无法判定。"""
    hint = (hint or "auto").strip().lower()
    if hint != "auto":
        if hint not in SUPPORTED_FRAMEWORKS:
            return None, (f"错误：不支持的测试框架 '{hint}'，"
                          f"可选：{', '.join(sorted(SUPPORTED_FRAMEWORKS - {'auto'}))}")
        return ("npm" if hint == "node" else hint), ""

    # 目标可能带 pytest 节点 id（:: 分隔），取首个路径段判断
    base = target.split("::")[0]
    ext = os.path.splitext(base)[1].lower() if "." in os.path.basename(base) else ""

    # 1. 文件扩展名优先
    if ext in JS_EXTS:
        node_fw = _detect_node_framework(os.path.join(cwd, base), cwd)
        if node_fw:
            return node_fw, ""
        return None, "错误：无法识别 JS 测试框架（未找到含 vitest/jest/mocha/test 脚本的 package.json）"
    if ext == ".py":
        return ("pytest" if _pytest_available(python_exe) else "unittest"), ""

    # 2. 目录 / 根目标：优先按所在目录的 package.json 判定，否则回退 Python
    abs_base = os.path.join(cwd, base)
    if os.path.isdir(abs_base) or base in (".", ""):
        node_fw = _detect_node_framework(abs_base if os.path.isdir(abs_base) else cwd, cwd)
        if node_fw:
            return node_fw, ""
        if _pytest_available(python_exe):
            return "pytest", ""
        return "unittest", ""

    return None, "错误：未知的测试框架，请显式传 framework（pytest/unittest/vitest/jest/mocha/npm）"


def _which(name: str) -> str:
    """返回可执行文件路径（Windows 解析 .cmd shim），找不到则回退名字。"""
    try:
        found = shutil.which(name)
        if found:
            return found
    except Exception:
        pass
    return name


def _node_to_unittest(target: str) -> str:
    """把 pytest 风格节点 id 转成 unittest 点分模块路径（如 tests/a.py::C::m -> tests.a.C.m）。"""
    parts = target.split("::")
    path_part = parts[0]
    if path_part.endswith(".py"):
        path_part = path_part[:-3]
    mod = path_part.replace("\\", ".").replace("/", ".")
    mod = mod.lstrip(".")
    if mod.startswith(".."):
        mod = mod[2:]
    suffix = parts[1:]
    return ".".join([mod] + suffix) if suffix else mod


def _build_command(framework: str, target: str, verbose: bool, cwd: str,
                   python_exe: Optional[str] = None) -> Tuple[List[str], str]:
    """返回 (命令列表, 人类可读命令文本)。"""
    t = target or "."
    py = python_exe or sys.executable
    if framework == "pytest":
        cmd = [py, "-m", "pytest", t, "-q", "-p", "no:cacheprovider"]
        cmd += ["-v", "--tb=long"] if verbose else ["--tb=short"]
        return cmd, "python -m pytest " + t
    if framework == "unittest":
        if "::" in t:
            module = _node_to_unittest(t)
            cmd = [py, "-m", "unittest"] + (["-v"] if verbose else []) + [module]
            return cmd, f"python -m unittest {module}"
        if t.endswith(".py") and os.path.isfile(os.path.join(cwd, t)):
            cmd = [py, "-m", "unittest"] + (["-v"] if verbose else []) + [t]
            return cmd, f"python -m unittest {t}"
        if t == ".":
            cmd = [py, "-m", "unittest", "discover", "-s", "."] + (["-v"] if verbose else [])
            return cmd, "python -m unittest discover"
        abs_t = os.path.join(cwd, t)
        if os.path.isdir(abs_t):
            cmd = [py, "-m", "unittest", "discover", "-s", t] + (["-v"] if verbose else [])
            return cmd, f"python -m unittest discover -s {t}"
        module = _node_to_unittest(t)
        cmd = [py, "-m", "unittest"] + (["-v"] if verbose else []) + [module]
        return cmd, f"python -m unittest {module}"
    if framework == "vitest":
        cmd = [_which("npx"), "vitest", "run", t] + (["--reporter=verbose"] if verbose else [])
        return cmd, "npx vitest run " + t
    if framework == "jest":
        cmd = [_which("npx"), "jest", t] + (["--verbose"] if verbose else [])
        return cmd, "npx jest " + t
    if framework == "mocha":
        cmd = [_which("npx"), "mocha", t]
        return cmd, "npx mocha " + t
    if framework == "npm":
        cmd = [_which("npm"), "test"] + (["--", t] if t != "." else [])
        return cmd, "npm test" + (" " + t if t != "." else "")
    return [py, "-m", "unittest", t], "python -m unittest " + t


def _pick_summary(framework: str, text: str) -> str:
    """从输出中抽取测试汇总文本（如 '3 passed, 1 failed in 0.52s'）。"""
    low = text.lower()
    if re.search(r"\bno tests ran\b", low) or re.search(r"\bno test files found\b", low) \
            or re.search(r"\bno tests collected\b", low):
        return "未收集到任何测试（no tests collected/ran）"
    parts: List[str] = []
    if framework in ("pytest", "unittest"):
        for pattern, label in (
            (r"(\d+)\s+passed\b", "passed"),
            (r"(\d+)\s+failed\b", "failed"),
            (r"(\d+)\s+errors?\b", "error"),
            (r"(\d+)\s+skipped\b", "skipped"),
            (r"(\d+)\s+xfailed\b", "xfailed"),
            (r"(\d+)\s+xpassed\b", "xpassed"),
        ):
            m = re.search(pattern, low)
            if m:
                parts.append(f"{m.group(1)} {label}")
        ran = re.search(r"\bran (\d+) tests?\b", low)
        if ran:
            parts.append(f"Ran {ran.group(1)} tests")
        if framework == "unittest" and re.search(r"\bok\b", low) and "OK" not in parts:
            parts.append("OK")
    elif framework in ("vitest", "jest", "mocha"):
        suites = re.search(r"\btest suites?[:\s]+(\d+)\s+passed[^,]*,\s*(\d+)\s+failed", low)
        tests = re.search(r"\btests?[:\s]+(\d+)\s+passed[^,]*,\s*(\d+)\s+failed", low)
        if suites:
            parts.append(f"Test Suites: {suites.group(1)} passed, {suites.group(2)} failed")
        elif mt := re.search(r"\btest suites?[:\s]+(\d+)\s+(passed|failed)", low):
            parts.append(f"Test Suites: {mt.group(1)} {mt.group(2)}")
        if tests:
            parts.append(f"Tests: {tests.group(1)} passed, {tests.group(2)} failed")
        elif mt := re.search(r"\btests?[:\s]+(\d+)\s+(passed|failed)", low):
            parts.append(f"Tests: {mt.group(1)} {mt.group(2)}")
    if not parts:
        for m in re.finditer(r"(\d+)\s+(passed|failed|errors?|skipped)\b", low):
            parts.append(f"{m.group(1)} {m.group(2)}")
    return "，".join(parts) if parts else "（未能解析测试汇总）"


def _extract_error_details(framework: str, text: str) -> str:
    """抽取失败/错误明细（从首个失败特征行到结尾，截断到上限）。"""
    lines = text.splitlines()
    if framework == "pytest":
        rx = re.compile(r"^_{5,}")
    elif framework == "unittest":
        rx = re.compile(r"^(FAIL|ERROR|FAILED):")
    elif framework in ("vitest", "jest", "mocha"):
        rx = re.compile(r"^\s*[●✕✓×]\s|^Expected\b|^Received\b|AssertionError|^FAIL\b|^\s*at\b")
    else:
        rx = re.compile(r"^\s*[●✕✓×]\s|^_{5,}|FAIL|ERROR|AssertionError|^Expected\b|^Received\b")

    start = 0
    for i, line in enumerate(lines):
        if rx.search(line):
            start = i
            break
    if start == 0:
        start = max(0, len(lines) - 60)
    detail = "\n".join(lines[start:])
    if len(detail) > ERROR_DETAIL_LIMIT:
        detail = detail[-ERROR_DETAIL_LIMIT:]
    return detail.strip()


class TestRunnerTool(BaseTool):
    """TestRunner 工具实现：运行指定单元测试并返回通过/失败与错误信息。"""

    @classmethod
    def get_tool_definition(cls) -> Tool:
        return Tool(
            name="TestRunner",
            description=(
                "Detect the test framework and run specified unit tests, returning pass/fail result "
                "and error details. Parameters: target (file/dir/test node such as "
                "tests/test_a.py::TestC::test_m, or '.' for all tests), framework (optional: "
                "auto/pytest/unittest/vitest/jest/mocha/npm), verbose (optional bool), timeout "
                "(optional ms, default 300000, max 600000). Runs from the project root and "
                "auto-detects the project's virtual environment (.venv/venv/env) to pick the right Python "
                "interpreter, and auto-detects pytest/unittest for Python and vitest/jest/mocha/npm for JS."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "测试目标：文件/目录路径，或测试节点 id（如 tests/test_a.py::TestC::test_m）；传 '.' 运行全部"
                    },
                    "framework": {
                        "type": "string",
                        "description": "测试框架（auto/pytest/unittest/vitest/jest/mocha/npm），默认自动识别"
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "true 时输出更详细的执行信息"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "超时毫秒（默认 300000，最大 600000）"
                    }
                },
                "required": ["target"]
            },
            modes=[ToolMode.WORK],
            permission_level=ToolPermission.DEFAULT,
        )

    @classmethod
    def execute(cls, context: ToolContext, target: str,
                framework: Optional[str] = None, verbose: bool = False,
                timeout: Optional[int] = None) -> str:
        """执行指定单元测试。"""
        if not target or not str(target).strip():
            return "错误：target 不能为空"
        target = str(target).strip()
        if not _is_safe_target(target):
            return (f"错误：target 含非法字符（当前：{target}）。"
                    f"仅允许文件路径/测试节点 id（字母、数字、-_.:/\\[]()）,请使用合法的测试目标。")

        try:
            t = int(timeout) if timeout is not None else COMMAND_TIMEOUT
        except (TypeError, ValueError):
            t = COMMAND_TIMEOUT
        t_ms = t if t >= 1000 else t * 1000
        timeout_s = max(1, min(t_ms, 600000) // 1000)

        cwd = getattr(context, "cwd", os.getcwd())
        python_exe, env_note = _resolve_python(cwd, target)
        fw, err = _detect_framework(target, framework, cwd, python_exe)
        if err:
            return err
        cmd, cmd_text = _build_command(fw, target, bool(verbose), cwd, python_exe)

        exit_code, combined = _run_cmd(cmd, cwd, timeout_s)
        if len(combined) > COMMAND_OUTPUT_LIMIT:  # 截断保留尾部（汇总与失败信息通常在后段）
            combined = "..." + combined[-COMMAND_OUTPUT_LIMIT:]
        summary = _pick_summary(fw, combined)
        details = _extract_error_details(fw, combined)

        if exit_code == 0:
            status = "通过 (PASSED)"
        elif exit_code == 5 and fw == "pytest":
            status = "未通过 (NO TESTS COLLECTED)"
        else:
            status = "失败 (FAILED)"

        result_lines = [
            f"测试结果: {status}",
            f"框架: {fw}",
        ]
        if fw in ("pytest", "unittest"):
            result_lines += [f"Python 环境: {python_exe}", env_note]
        result_lines += [f"运行命令: {cmd_text}", f"退出码: {exit_code}", f"测试汇总: {summary}"]
        if exit_code != 0 and details:
            result_lines.append(f"失败/错误明细:\n{details}")
        elif details:
            result_lines.append(f"输出:\n{details}")
        else:
            result_lines.append(f"输出:\n{combined.strip()[:ERROR_DETAIL_LIMIT]}")
        return "\n".join(result_lines)
