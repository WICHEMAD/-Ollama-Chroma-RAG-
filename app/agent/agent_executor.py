"""Agent 执行器服务（基于 LangChain 1.x create_agent API）"""
from typing import Optional, Any

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from app.agent.tools import TOOLS
from app.models.ollama_client import OllamaClient
from app.config import config
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

    def __init__(self):
        # 1. 创建 Agent 专用 LLM 实例
        self.llm_client = OllamaClient(
            chat_model=config.AGENT_MODEL,
            temperature=0,
        )
        self.tools = TOOLS
        self.agent_graph = self._create_agent_executor()

    def _create_agent_executor(self) -> CompiledStateGraph:
        """使用 create_agent 构建 Agent（LangChain 1.x 新 API）"""
        agent_graph = create_agent(
            model=self.llm_client.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
        )
        return agent_graph

    def run_agent(self, query: str) -> str:
        """执行 Agent 推理"""
        try:
            # 新 API 使用 messages 格式输入
            result = self.agent_graph.invoke({
                "messages": [{"role": "user", "content": query}]}
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

        except Exception as e:
            error_msg = str(e)
            print(f"[Agent ERROR] {error_msg}")
            if "iteration" in error_msg.lower() or "time limit" in error_msg.lower():
                return "思考过程超时，请尝试更简洁的问题。"
            return f"Agent 执行失败: {error_msg}"


# 全局单例
_agent_executor_instance: Optional[AgentExecutorService] = None


def get_agent_executor() -> AgentExecutorService:
    """获取 Agent 执行器单例"""
    global _agent_executor_instance
    if _agent_executor_instance is None:
        _agent_executor_instance = AgentExecutorService()
    return _agent_executor_instance




if __name__ == "__main__":
    agent = get_agent_executor()
    print("=== 测试文档检索 ===")
    print(agent.run_agent("陈冠霖"))
    print("\n=== 测试计算器 ===")
    print(agent.run_agent("2+2等于多少？"))