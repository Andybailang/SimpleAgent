#!/bin/bash

# 启动简易 AI Agent CLI

echo "=========================================="
echo "  简易 AI Agent CLI"
echo "=========================================="
echo ""

# 进入 agent 目录
cd "$(dirname "$0")"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 运行 CLI
echo ""
echo "启动 Agent CLI..."
echo "按 /exit 或 /quit 退出"
echo "按 /clear 清空历史"
echo ""
echo "=========================================="
echo ""

python cli.py
