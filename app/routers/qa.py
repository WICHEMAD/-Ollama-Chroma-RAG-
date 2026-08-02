"""问答路由"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.rag.chain import rag_chain
from app.agent.agent_executor import get_agent_executor
router = APIRouter(prefix="/qa", tags=["qa"])

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = None  # ✅ 使用明确的类型

# 响应模型
class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict] = []
    context_length: int = 0

# 延迟导入避免循环依赖
def get_rag_chain():
    return rag_chain

def get_agent_executor():
    return get_agent_executor()

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

        result = rag_chain.answer(request.question, history)

        print(f"DEBUG - 结果: {result}")

        # 5. 返回响应
        return QuestionResponse(
            answer=result.get("answer", "根据已有文档无法回答"),
            sources=result.get("sources", []),
            context_length=result.get("context_length", 0)
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
            for chunk in rag_chain.answer_stream(request.question, history):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
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

        # 2. 获取 Agent 执行器
        agent = get_agent_executor()

        # 3. 调用 Agent（只传递 question，history 可后续扩展）
        print(f"DEBUG - Agent 问题: {request.question}")
        answer = agent.run_agent(request.question)

        print(f"DEBUG - Agent 结果: {answer}")

        # 4. 构造响应（sources 暂时为空，context_length 填 0）
        return QuestionResponse(
            answer=answer,
            sources=[],  # 后续可从 Agent 输出中解析来源
            context_length=0
        )

    except Exception as e:
        print(f"ERROR - Agent 执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {str(e)}")