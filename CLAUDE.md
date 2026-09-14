# ollama_rag_system — 本地离线 RAG 知识库系统

FastAPI + LangChain + Ollama + Chroma 的离线 RAG 问答系统，前端 React（Vite）。内部自用工具。

问答链路：**查询改写 → 检索/嵌入 → （可选）重排 → LLM 生成**；另有 Agent 模式（工具调用 + 多轮记忆）。

## 运行命令

### 后端
```bash
# 激活虚拟环境（Windows）
venv\Scripts\activate          # cmd / PowerShell
source venv/Scripts/activate   # Git Bash

# 启动服务
uvicorn app.main:app --reload --port 8000
```
服务地址 http://localhost:8000，API 文档 http://localhost:8000/docs

### 前端
```bash
cd frontend
npm install        # 首次
npm run dev        # 开发（vite）
npm run build      # 构建（产物 frontend/dist，后端自动挂载到根路径）
```

### 测试
```bash
python test_full_system.py
```

## 架构约定

### 模型分工（对话与改写均已走云端）
| 能力 | 模型 | 位置 |
|---|---|---|
| 查询改写 | 云端模型（`.env` 的 `QUERY_REWRITE_MODEL`，默认空） | 云端 |
| 嵌入 | `qwen3-embedding:0.6b` | 本地 Ollama |
| 对话生成（普通问答 + Agent） | 用户配置的云端模型 | 云端 |

本地**只常驻嵌入**这一个能力。改写与生成都走云端；`build_chat_client` / `get_agent_executor` 拿不到云端模型时抛「请先配置云端模型」，不做本地兜底（改写在 `query_rewriter.py` 内捕获后回退原问题）。

### 键名映射（易错）
`.env` 里是 `LLM_MODEL` / `EMBEDDING_MODEL`，`config.py` 读取后暴露为 `config.CHAT_MODEL` / `config.EMBED_MODEL`。改模型名要改 `.env` 的 `LLM_MODEL`，不是 `CHAT_MODEL`。

## 关键陷阱与约定

- **换嵌入模型 = 维度变 = 需重建向量库**，别轻易改 `EMBED_MODEL`。
- rerank（bce-reranker）首次加载需联网下载，失败自动降级为基础检索，不会中断链路。
- SSE 流式格式：`data:{"chunk":…}` → `data:{"type":"sources"}` → `data:[DONE]`，前端 `api.js` 按此解析，**改格式必须同步前端**。
- 路由层延迟导入，避免循环依赖。
- Agent 记忆用 `InMemorySaver` + `thread_id`，**后端重启即清空**。
- `calculator` 用 `eval`，但有白名单 + 长度限制（≤100），别去掉这层防护。
- `models.json` 存明文 `api_key`（已 gitignore），注意不外泄。
- 查询改写**默认不生效**：`.env` 未设 `QUERY_REWRITE_MODEL` 时 `config.QUERY_REWRITE_MODEL=None`，`rewrite()` 直接透传原问题。要让改写生效需设 `QUERY_REWRITE_MODEL` 为已注册的云端模型名。改写固定温度 `0.3`、超时 `5s`（`query_rewriter.py` 类常量，不走 `config.QUERY_REWRITE_TEMPERATURE`）。

## 环境/模型运行坑（本机 16GB RAM，Windows）

- deepseek-r1 系列作对话模型会崩：`1.5b` 崩 `peg-native format`、`7b` 检索阶段 OOM（`std::bad_alloc`）。稳定用 `qwen2.5:1.5b`。
- 多个本地模型同时常驻（7b + 嵌入 + 重排）会挤爆内存，llama.cpp 崩溃（`exit 0xc0000409`）。

## 已知缺陷

当前代码的 bug 清单见 `docs/开发文档.md` 第 3.3 节，这里不重复。

## 审查员角色与工作流程

本会话的 Claude 承担**代码审查员**角色，与另一个负责生成代码的 AI 协作。

### 工作流程

1. 生成代码的 AI 把改动写进文件（未提交）。
2. 用户招呼「来审」→ 我跑 `git diff` + 读相关文件审查。
3. 输出审查结论：错误位置（文件:行号）+ 问题描述 + 修改意见，按严重程度（🔴 必须修 / 🟠 建议修 / 🟡 轻微）排序。
4. 把问题反馈给生成代码的 AI。
5. 询问用户是否同意修复；**用户同意后才动手改**。

### 铁律

- 改任何东西前先说明「改什么 + 为什么」，征得用户同意，不擅自修改。
- 任何决定都先问用户，不自作主张。
- 所有回复用中文。
