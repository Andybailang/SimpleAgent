"""
简易 AI Agent 命令行入口
支持流式输出和基本命令
"""

import sys
import os
import signal
import subprocess
from typing import Optional

# 设置 UTF-8 编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_util import load_env
from engine import SimpleAgent


class CLI:
    """命令行界面"""

    def __init__(self):
        """初始化 CLI"""
        # 加载环境变量
        load_env()

        # 初始化 Agent
        self.agent = SimpleAgent(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
        )

        # 运行状态
        self.running = True

        # 绑定信号处理
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """处理退出信号"""
        self.running = False
        print("\n\n正在退出...")

    def _print_streaming(self, text: str):
        """流式打印文本（类似效果）"""
        # 分词后逐字打印
        words = text.split()
        for i, word in enumerate(words):
            print(word, end=" ")
            sys.stdout.flush()
            if i < len(words) - 1:
                print(" ", end="")
        print()

    def _print_header(self):
        """打印标题"""
        print("\n" + "="*60)
        print("  简易 AI Agent 命令行工具")
        print("  按 /exit 或 /quit 退出")
        print("  按 /clear 清空历史")
        print("="*60 + "\n")

    def _print_history(self):
        """打印消息历史"""
        history = self.agent.get_history_text()
        print("\n" + "-"*60)
        print("消息历史：")
        print("-"*60 + history + "\n")

    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "-"*60)
        print("可用命令：")
        print("  /exit       - 退出程序")
        print("  /quit       - 退出程序")
        print("  /clear      - 清空对话历史")
        print("  /history    - 查看对话历史")
        print("  /help       - 显示此帮助信息")
        print("  /model      - 显示当前模型")
        print("  /temperature - 显示当前温度设置")
        print("-"*60 + "\n")

    def _print_status(self):
        """显示状态信息"""
        print("\n" + "-"*60)
        print("状态信息：")
        print(f"  模型: {os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')}")
        print(f"  温度: {os.getenv('DEFAULT_TEMPERATURE', '0.7')}")
        print(f"  最大 Token: {os.getenv('DEFAULT_MAX_TOKENS', '4096')}")
        print(f"  消息数量: {len(self.agent.get_messages())}")
        print("-"*60 + "\n")

    def _process_command(self, command: str) -> bool:
        """
        处理命令

        Args:
            command: 命令字符串

        Returns:
            是否继续运行
        """
        cmd = command.strip().lower()

        if cmd in ["/exit", "/quit"]:
            self.running = False
            return False
        elif cmd == "/clear":
            self.agent.clear_history()
            print("\n[OK] 对话历史已清空\n")
            return True
        elif cmd == "/history":
            self._print_history()
            return True
        elif cmd == "/help":
            self._print_help()
            return True
        elif cmd == "/model":
            print(f"\n当前模型: {os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')}\n")
            return True
        elif cmd == "/temperature":
            print(f"\n当前温度: {os.getenv('DEFAULT_TEMPERATURE', '0.7')}\n")
            return True
        elif cmd == "/status":
            self._print_status()
            return True
        elif cmd.startswith("/"):
            print(f"\n未知命令: {command}\n")
            self._print_help()
            return True
        else:
            return True

    def run(self):
        """运行 CLI"""
        self._print_header()

        print("正在初始化 Agent...")

        # 检查 API Key
        if not os.getenv("OPENAI_API_KEY"):
            print("\n错误：未设置 OPENAI_API_KEY")
            print("请在 src/agent/.env 文件中配置你的 API Key\n")
            return

        print("[OK] Agent 初始化成功！\n")

        # 主循环
        while self.running:
            try:
                # 获取用户输入
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # 处理命令
                if not self._process_command(user_input):
                    break

                # 发送给 Agent
                print("\nAgent: ", end="")
                response = self.agent.chat(user_input)
                self._print_streaming(response)

                # 刷新输出缓冲区
                sys.stdout.flush()
                sys.stderr.flush()

            except KeyboardInterrupt:
                print("\n\n检测到中断信号，准备退出...")
                self.running = False
            except EOFError:
                print("\n\n检测到 EOF，准备退出...")
                self.running = False
            except Exception as e:
                print(f"\n错误: {str(e)}\n")
                self.running = False


def main():
    """主函数"""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
