"""load_dotenv 的 pyc 安全封装。

python-dotenv 的 find_dotenv() 依赖调用栈查找 .env 位置，在直接运行
字节码（.pyc，BigCodexApp 装配形态）时会因 frame.f_back is None 崩溃
（AssertionError）。这里改为显式定位：<仓库根>/src/agent/../.. == 仓库根，
即 config.env 同级目录下的 .env。文件不存在时静默跳过（与无参行为一致）。
"""
import os


def load_env(dotenv_path=None, override=False):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    if not dotenv_path:
        # 本文件位于 <root>/src/agent/ 下，上一级上一级即仓库根
        dotenv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"
        )
    return load_dotenv(dotenv_path, override=override)