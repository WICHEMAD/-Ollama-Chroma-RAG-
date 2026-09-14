# 🧠 Ollama RAG 知识库问答系统

基于 **RAG（检索增强生成）** 技术的本地离线知识库问答系统。上传 PDF / TXT 文档构建专属知识库，提问时走「查询改写 → 混合检索 → 可选重排 → 云端生成」链路，由大模型生成带来源引用的回答。

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 📤 **文档入库** | 支持 PDF、TXT 上传，自动切分、向量化，内容哈希去重 |
| 🔍 **混合检索** | 稠密向量 + BM25 稀疏检索 + RRF 融合，补强专有名词 / 编号等关键词型召回 |
| 🤖 **AI 问答** | 查询改写 + RAG 检索增强生成，基于知识库内容给出准确回答（流式 / 非流式） |
| 🧭 **Agent 问答** | 工具调用式问答（文档检索 / 计算器）+ 多轮记忆 |
| 📊 **来源追溯** | 每个回答标注参考文档来源 |
| ☁️ **云端模型** | 支持 OpenAI 兼容云端模型注册、持久化、按名调用 |
| 🎨 **现代 UI** | React 前端：上传 / 文档管理 / 问答 / 模型选择 / 首屏配置 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  前端 (React + Vite)                 │
│         问答界面 │ 文档管理 │ 模型配置 │ 统计          │
└──────────────────────┬──────────────────────────────┘
                       │ REST API（SSE 流式）
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI 后端服务                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐      │
│  │ 问答路由  │  │ 文档路由  │  │  模型路由     │      │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘      │
│  ┌────▼──────────────▼──────────────▼──────────┐    │
│  │               RAG 业务层                     │    │
│  │   查询改写 → 混合检索 → 重排 → 生成           │    │
│  └───┬──────────────┬───────────────┬──────────┘    │
└──────┼──────────────┼───────────────┼───────────────┘
       │              │               │
┌──────▼─────┐  ┌─────▼──────┐  ┌─────▼───────────┐
│   Chroma   │  │ 本地 Ollama │  │   云端 LLM API   │
│  (向量库)   │  │ (嵌入模型)  │  │ (OpenAI 兼容)    │
└────────────┘  └────────────┘  └─────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Ollama（本地运行嵌入模型）

### 1. 安装依赖

```bash
# 后端
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 2. 准备嵌入模型

```bash
ollama pull qwen3-embedding:0.6b
```

### 3. 配置

在项目根目录新建 `.env`，常用配置项（未设置时用默认值）：

| 键 | 默认值 | 用途 |
|---|---|---|
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | 本地嵌入模型 |
| `ENABLE_HYBRID` | `true` | 混合检索总开关 |
| `ENABLE_RERANK` | `false` | 是否启用重排（首次启用需联网下载模型） |
| `CHUNK_SIZE` | `500` | 文档切分块大小 |

> 对话生成与查询改写走**云端模型**（OpenAI 兼容），无需在 `.env` 配置——首次启动后在首屏配置弹窗里注册即可。

### 4. 启动

```bash
# 后端（Windows）
venv\Scripts\activate          # cmd / PowerShell
source venv/Scripts/activate   # Git Bash
uvicorn app.main:app --reload --port 8000

# 前端（开发模式）
cd frontend
npm run dev
```

- 后端：http://localhost:8000（API 文档 http://localhost:8000/docs）
- 前端开发：http://localhost:5173

生产部署：`cd frontend && npm run build` 构建后，后端自动把 `frontend/dist` 挂载到根路径。

### 5. 使用

1. 打开前端，在首屏弹窗注册一个云端模型（provider / base_url / api_key / model）
2. 上传 PDF / TXT 文档构建知识库
3. 输入问题，获得带来源引用的回答

## 📁 项目结构

```
app/
├── main.py                  # FastAPI 入口
├── config.py                # 配置中心（单例）
├── routers/                 # API 层
│   ├── qa.py                # 问答接口（/qa）
│   ├── documents.py         # 文档上传 / 统计 / 列表
│   └── models.py            # 云端模型增删查
├── rag/                     # RAG 业务层
│   ├── chain.py             # 主链（编排）
│   ├── retriever.py         # 混合检索 + RRF 融合
│   ├── sparse_retriever.py  # BM25 稀疏检索
│   ├── query_rewriter.py    # 查询改写
│   └── memory.py            # 对话记忆
├── document/                # 文档加载 / 切分 / 入库
├── vectordb/                # Chroma 向量库封装
├── models/                  # 模型客户端 + 云端注册表
└── agent/                   # Agent 执行器 + 工具
frontend/                    # React 前端（Vite）
```

## 🔧 API 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/qa/ask` | 提交问题，获取回答（非流式） |
| POST | `/qa/stream` | 流式问答（SSE） |
| POST | `/qa/agent` | Agent 问答（工具调用） |
| POST | `/documents/upload` | 上传文档（PDF / TXT） |
| GET | `/documents/stats` | 文档统计 |
| GET | `/documents/list` | 文档列表 |
| GET | `/models` | 列出云端模型 |
| POST | `/models` | 添加云端模型 |
| DELETE | `/models/{name}` | 删除云端模型 |

## 🛠️ 技术栈

- **后端框架**：FastAPI + Uvicorn
- **编排框架**：LangChain（langchain-ollama / langchain-openai / langchain-chroma / langgraph）
- **对话生成**：外部云端模型（OpenAI 兼容）
- **嵌入模型**：本地 Ollama `qwen3-embedding:0.6b`
- **向量数据库**：Chroma（本地持久化）
- **检索**：混合检索（稠密向量 + BM25 + RRF），可选 bce-reranker 重排
- **前端**：React 18 + Vite
