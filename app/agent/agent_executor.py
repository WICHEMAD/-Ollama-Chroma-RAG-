"""Agent 执行器服务（基于 LangChain 1.x create_agent API）"""
from typing import Optional, Any

from app.agent.tools import TOOLS
from app.models.ollama_client import build_chat_client, get_ollama_client
from langgraph.checkpoint.memory import InMemorySaver
import sys

# 提高 Python 递归深度限制，防止 Agent 执行时栈溢出
sys.setrecursionlimit(10000)
# ==================== 系统指令 ====================
SYSTEM_PROMPT = """你是一个智能助手，可以访问工具来帮助回答用户问题。

## 行为准则
1. 对于事实性问题，必须使用 document_search 工具在知识库中查找信息。
2. 对于数学计算，必须使用 calculator 工具。
3. 如果工具返回"未找到相关文档"或类似空结果，请如实告诉用户，不要编造答案。
4. 最终回答必须基于工具返回的信息，不要添加额外的编造内容。
5. 回答使用简体中文。
"""


class AgentExecutorService:
    """Agent 执行器，封装工具调用和多步推理"""

    def __init__(self, model_name=None):
        from app.models import registry

        self.model_name = model_name
        self.last_fallback = False
        # 云端优先：已注册云端模型 → 云端；否则本地兜底（config.CHAT_MODEL）
        if model_name and registry.get_model(model_name):
            self.llm_client = build_chat_client(model_name, temperature=0)
            self.is_cloud = True
        else:
            self.llm_client = get_ollama_client()
            self.is_cloud = False
        self.tools = TOOLS
        # 内存检查点：让 Agent 按 thread_id 记住多轮对话（后端重启后清空）
        self.checkpointer = InMemorySaver()
        self.agent_graph = self._create_agent_executor(self.llm_client)

    def _create_agent_executor(self, llm_client) -> "CompiledStateGraph":
        """使用 create_agent 构建 Agent（LangChain 1.x 新 API）"""
        from langchain.agents import create_agent
        agent_graph = create_agent(
            model=llm_client.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )
        return agent_graph

    def run_agent(self, query: str, thread_id: str = "default") -> str:
        """执行 Agent 推理（thread_id 区分不同会话，实现多轮记忆）"""
        self.last_fallback = False
        try:
            return self._invoke(self.agent_graph, query, thread_id)
        except Exception as e:
            error_msg = str(e)
            print(f"[Agent ERROR] {error_msg}")
            # 云端失败 → 回退本地兜底模型重试一次
            if self.is_cloud:
                print("[Agent] 云端模型调用失败，回退本地模型")
                self.last_fallback = True
                try:
                    local_graph = self._create_agent_executor(get_ollama_client())
                    return self._invoke(local_graph, query, thread_id)
                except Exception as e2:
                    return f"Agent 执行失败（含本地回退）: {e2}"
            if "iteration" in error_msg.lower() or "time limit" in error_msg.lower():
                return "思考过程超时，请尝试更简洁的问题。"
            return f"Agent 执行失败: {error_msg}"

    def _invoke(self, agent_graph, query: str, thread_id: str) -> str:
        """执行一次 agent_graph.invoke 并提取最后一条 AI 消息内容"""
        # 新 API 使用 messages 格式输入；thread_id 用于从检查点恢复历史
        result = agent_graph.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # 从返回的状态中提取最后一条消息的内容
        messages = result.get("messages", [])
        if not messages:
            return "Agent 未能生成有效回答，请换一种方式提问。"

        # 取最后一条 AIMessage 的内容作为回答
        last_msg = messages[-1]
        output = getattr(last_msg, "content", str(last_msg))

        if not output:
            return "Agent 未能生成有效回答，请换一种方式提问。"

        return str(output).strip()


# 按模型名缓存 Agent 执行器（不同模型可能 provider 不同）
_agent_executors: dict = {}


def get_agent_executor(model_name=None) -> AgentExecutorService:
    """按模型名获取（缓存的）Agent 执行器；model 为空走本地兜底。"""
    global _agent_executors
    key = model_name or ""  # 空模型名与本地兜底共用同一缓存键
    if key not in _agent_executors:
        _agent_executors[key] = AgentExecutorService(model_name=model_name)
    return _agent_executors[key]




if __name__ == "__main__":
    agent = get_agent_executor()
    print("=== 测试文档检索 ===")
    print(agent.run_agent("陈冠霖"))
    print("\n=== 测试计算器 ===")
    print(agent.run_agent("2+2等于多少？"))