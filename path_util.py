"""路径工具：Windows 扩展路径前缀（\\?\\）统一剥离。

背景：Tauri 桌面壳在 Windows 上通过 GetFinalPathNameByHandle / resource_dir
拿到资源目录时，可能返回带 ``\\?\\`` 扩展路径前缀的路径（如
``\\?\\D:\\BigCodex\\app-run``）。该前缀一路传播到后端进程：Python 的
``__file__``、``os.getcwd()``、``os.path.abspath/realpath/join`` 结果都会
保留它。对绝大多数 Windows 原生 API 与 Python 依然可用，但对子进程里跑的
Node（如内置 filesystem MCP server）来说，``\\?\\`` 前缀会让 Node 的模块
加载器把 ``D:`` 当作一个目录而报 ``EISDIR: lstat 'D:'``，表现为 MCP 服务器
启动失败（TaskGroup 1 sub-exception）。因此在这些路径参与拼接/回显/调用前
统一剥离该前缀。

本模块只依赖标准库 os，可被后端任意模块安全导入（无循环依赖）。
"""

import os
from typing import Optional


_EXTLEN_PREFIX = "\\\\?\\"
_UNC_PREFIX = "\\\\?\\UNC\\"


def strip_vermagic(p: Optional[str]) -> Optional[str]:
    """去掉 Windows 扩展路径前缀 ``\\?\\``，返回常规化字符串路径。

    - ``\\?\\D:\\...`` -> ``D:\\...``
    - ``\\?\\UNC\\server\\share`` -> ``\\\\server\\share``
    - 无前缀的路径原样返回；非字符串输入原样返回。

    注意：只处理前导前缀，不做反斜杠/大小写/换行等其它规范化，避免改变
    原有语义；对非扩展路径是安全的 no-op。
    """
    if not isinstance(p, str):
        return p
    if p.startswith(_UNC_PREFIX):
        return "\\\\" + p[len(_UNC_PREFIX):]
    if p.startswith(_EXTLEN_PREFIX):
        return p[len(_EXTLEN_PREFIX):]
    return p


def abspath_clean(p: Optional[str]) -> str:
    """abspath 前先剥离扩展路径前缀。

    os.path.abspath 对带 ``\\?\\`` 字符串结果仍带前缀（Windows Python），
    因此先剥离再绝对化，保证返回值是普通盘符路径。
    """
    if not p:
        return p or ""
    return os.path.abspath(strip_vermagic(p))


def realpath_clean(p: Optional[str]) -> str:
    """realpath 前先剥离扩展路径前缀。

    os.path.realpath 在 Windows 上对已存在的目录可能经 GetFinalPathNameByHandle
    返回带 ``\\?\\`` 的结果；先剥离输入前缀可避免该结果被前缀传染。对不存在的
    路径回退为 abspath 后的常规化路径。
    """
    if not p:
        return p or ""
    return os.path.normpath(os.path.realpath(strip_vermagic(p)))
