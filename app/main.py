"""RAG 问答系统后端入口"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import documents, qa, models

# 创建 FastAPI 实例
app = FastAPI(
    title="Ollama RAG 问答系统",
    description="基于 Ollama 的本地知识库问答系统 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加 CORS 中间件（允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(models.router)

# 健康检查（根路径 "/" 留给下方的前端静态页面）
@app.get("/health", tags=["健康检查"])
async def health():
    return {
        "status": "running",
        "service": "Ollama RAG 问答系统",
        "version": "1.0.0",
        "docs": "/docs"
    }

# 启动事件
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("Ollama RAG 问答系统启动成功！")
    print(f"服务地址: http://localhost:8000")
    print(f"API 文档: http://localhost:8000/docs")
    print("=" * 60)

# 生产模式：若前端已构建（frontend/dist 存在），挂载静态页面到根路径
# API 路由（/qa、/documents、/docs）已在上面注册，优先匹配，不会被子页面覆盖
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

# 运行服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)