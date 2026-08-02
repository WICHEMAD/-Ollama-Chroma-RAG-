"""RAG 问答系统后端入口"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import documents, qa

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

# 根路径健康检查
@app.get("/", tags=["健康检查"])
async def root():
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

# 运行服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)