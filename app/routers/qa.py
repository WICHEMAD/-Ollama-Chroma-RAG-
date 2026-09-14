"""问答路由"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.rag.chain import get_rag_chain as _get_rag_chain
from app.agent.agent_executor import get_agent_executor as _get_agent_executor
router = APIRouter(prefix="/qa", tags=["qa"])

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = None  # ✅ 使用明确的类型
    model: Optional[str] = None  # 可选：指定对话模型（缺省走默认配置）
    thread_id: Optional[str] = None  # Agent 会话 id（多轮记忆用，普通问答忽略）

# 响应模型
class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict] = []
    context_length: int = 0
    rewritten_query: Optional[str] = None
    fallback: bool = False

# 延迟导入避免循环依赖
def get_rag_chain():
    return _get_rag_chain()

def get_agent_executor(model_name=None):
    return _get_agent_executor(model_name)

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    单轮问答接口

    Args:
        question: 用户问题
        history: 历史对话记录（可选）

    Returns:
        包含回答、来源和上下文数量的响应
    """
    try:
        # 1. 验证问题
        if not request.question or not request.question.strip():
            raise ValueError("问题不能为空")

        # 2. 处理历史对话（关键修复）
        history = []
        if request.history and isinstance(request.history, list):
            for msg in request.history:
                # 确保格式正确
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        # 3. 获取 RAG 链
        rag_chain = get_rag_chain()

        # 4. 调用问答（打印调试信息）
        print(f"DEBUG - 问题: {request.question}")
        print(f"DEBUG - 历史记录: {history}")

        result = rag_chain.answer(request.question, history, request.model)

        print(f"DEBUG - 结果: {result}")

        # 5. 返回响应
        return QuestionResponse(
            answer=result.get("answer", "根据已有文档无法回答"),
            sources=result.get("sources", []),
            context_length=result.get("context_length", 0),
            rewritten_query=result.get("rewritten_query"),
            fallback=result.get("fallback", False)
        )

    except Exception as e:
        print(f"ERROR - {str(e)}")  # 打印错误信息
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

@router.post("/stream")
async def stream_question(request: QuestionRequest):
    """
    流式问答接口
    """
    from fastapi.responses import StreamingResponse
    import json

    try:
        if not request.question or not request.question.strip():
            raise ValueError("问题不能为空")

        history = []
        if request.history and isinstance(request.history, list):
            for msg in request.history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

        rag_chain = get_rag_chain()

        async def generate_stream():
            try:
                for item in rag_chain.answer_stream(request.question, history, request.model):
                    if isinstance(item, str):
                        yield f"data: {json.dumps({'chunk': item})}\n\n"
                    else:
                        # 来源元数据 dict（含 sources / context_length / rewritten_query）
                        yield f"data: {json.dumps(item)}\n\n"
            except Exception as e:
                print(f"ERROR - 流式生成异常: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流式问答失败: {str(e)}")

# ==================== 新增 Agent 问答端点 ====================
@router.post("/agent", response_model=QuestionResponse)
async def ask_agent(request: QuestionRequest):
    """
    Agent 智能问答接口
    支持工具调用（文档检索、计算器等）的智能问答

    Args:
        question: 用户问题
        history: 历史对话记录（可选，当前版本暂不使用）

    Returns:
        包含回答的响应
    """
    try:
        # 1. 验证问题
        if not request.question or not request.question.strip():
            raise ValueError("问题不能为空")

        # 2. 获取 Agent 执行器（按所选模型）
        agent = get_agent_executor(request.model)

        # 3. 调用 Agent（只传递 question，history 可后续扩展）
        print(f"DEBUG - Agent 问题: {request.question}")
        answer = agent.run_agent(request.question, request.thread_id or "default")

        print(f"DEBUG - Agent 结果: {answer}")

        # 4. 构造响应（sources 暂时为空，context_length 填 0）
        return QuestionResponse(
            answer=answer,
            sources=[],  # 后续可从 Agent 输出中解析来源
            context_length=0,
            fallback=agent.last_fallback
        )

    except Exception as e:
        print(f"ERROR - Agent 执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {str(e)}")