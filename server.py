"""API 服务层 - 连接前端 UI 和 Agent 核心（入口：装配 app + 启动）。
"""
import os
import sys
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 添加 agent 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载 .env（模块加载时即生效，避免首个会话拿到默认模型名）
# 用 env_util.load_env 显式路径加载：find_dotenv() 在 pyc 形态下会崩溃
try:
    from env_util import load_env
    load_env()
except ImportError:
    pass

from server_config import _repo_root, _read_config_env
from server_routes import router as api_router
import mcp_manager
import skills
from ocr_engine import warmup_ocr


# 创建 FastAPI 应用
app = FastAPI(
    title="BigCodex API",
    description="AI 编程助手 API 服务",
    version="1.0.0",
)
app.include_router(skills.router)
app.include_router(api_router)

# CORS 配置 - 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_mcp():
    """后台连接 MCP 服务器并动态注册工具（不阻塞 server 启动）。"""
    try:
        mcp_manager.initialize()
    except Exception as e:
        print(f"[MCP] 初始化失败: {e}")
    # OCR 模型加载较重，放后台线程预热，首次识别不再卡顿（失败不影响服务）
    try:
        threading.Thread(target=warmup_ocr, daemon=True).start()
    except Exception as e:
        print(f"[OCR] 预热失败: {e}")


# ==================== 前端静态托管（生产/安装包模式） ====================
# 前端构建产物 src/ui/dist 存在时，由后端单进程托管页面：
# - /assets/* 精确文件直接返回；
# - 其余非 /api 路径回退 index.html（SPA 路由）；
# - /api/* 未匹配路由保持 404（API 路由优先注册，不会被此兜底抢占）。
_DIST_DIR = os.path.join(_repo_root(), "src", "ui", "dist")
if os.path.isdir(_DIST_DIR) and os.path.isfile(os.path.join(_DIST_DIR, "index.html")):
    from fastapi.responses import FileResponse as _FileResponse

    _DIST_NORM = os.path.normpath(_DIST_DIR)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_static(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        if full_path:
            candidate = os.path.normpath(os.path.join(_DIST_NORM, full_path))
            if os.path.isfile(candidate) and candidate.startswith(_DIST_NORM):
                return _FileResponse(candidate)
        return _FileResponse(os.path.join(_DIST_NORM, "index.html"))


if __name__ == "__main__":
    import uvicorn

    _port_cfg = _read_config_env()
    try:
        port = int(_port_cfg.get("BACKEND_PORT") or 8000)
    except (TypeError, ValueError):
        port = 8000

    print("=" * 60)
    print("BigCodex API Server")
    print("=" * 60)
    print(f"API: http://localhost:{port}")
    print(f"Health: http://localhost:{port}/api/health")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port)
