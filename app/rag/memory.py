from typing import List, Dict, Optional
from app.models.ollama_client import OllamaClient


class ConversationMemory:
    """
    对话记忆管理类

    负责管理对话历史，在 Token 超限时自动将早期对话压缩成摘要
    """

    def __init__(self, max_token_limit: int = 2000, llm_client: Optional[OllamaClient] = None):
        """
        初始化对话记忆

        Args:
            max_token_limit: 触发摘要的 Token 阈值
            llm_client: LLM 客户端（用于生成摘要）
        """
        self.messages: List[Dict] = []  # 存储最近几轮原始对话
        self.summary: str = ""  # 早期对话的摘要
        self.max_token_limit = max_token_limit
        self.llm_client = llm_client or OllamaClient()

    def add_user_message(self, text: str) -> None:
        """
        添加用户消息

        Args:
            text: 用户输入文本
        """
        self.messages.append({"role": "user", "content": text})
        self._check_and_compress()

    def add_assistant_message(self, text: str) -> None:
        """
        添加助手消息

        Args:
            text: 助手回复文本
        """
        self.messages.append({"role": "assistant", "content": text})
        self._check_and_compress()

    def _count_tokens(self) -> int:
        """
        估算当前消息列表的总 Token 数

        Returns:
            int: 估算的 Token 数量
        """
        total_tokens = 0
        for msg in self.messages:
            total_tokens += self.llm_client.count_tokens(msg["content"])
        return total_tokens

    def _check_and_compress(self) -> None:
        """
        检查 Token 数量，超过阈值时进行压缩
        """
        total_tokens = self._count_tokens()

        if total_tokens > self.max_token_limit:
            self._compress_history()

    def _compress_history(self) -> None:
        """
        将早期对话压缩成摘要

        保留最近 2-3 轮对话，将其余对话总结成摘要
        """
        # 保留最近 3 轮对话（6 条消息：user/assistant 各 3 条）
        messages_to_keep = 6
        if len(self.messages) <= messages_to_keep:
            # 消息少但 Token 超限 → 至少要保留最近 2 条
            if len(self.messages) <= 2:
                return  # 只有 2 条，没法压缩

            # 保留最近 2 条，其余压缩
            messages_to_keep = 2

        # 提取需要压缩的早期消息
        messages_to_compress = self.messages[:-messages_to_keep]

        # 构建对话文本
        conversation_text = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in messages_to_compress
        )

        # 生成摘要
        summary_prompt = f"""
请将以下对话总结为一句话摘要，保持关键信息：

{conversation_text}

摘要：
""".strip()

        try:
            new_summary = self.llm_client.chat([
                {"role": "system", "content": "你是一个对话摘要助手，擅长将多轮对话浓缩成简洁的摘要。"},
                {"role": "user", "content": summary_prompt}
            ])

            # 合并新旧摘要
            if self.summary:
                self.summary = f"{self.summary}\n\n补充摘要：{new_summary}"
            else:
                self.summary = new_summary

            # 保留最近的消息
            self.messages = self.messages[-messages_to_keep:]

        except Exception as e:
            # 如果摘要失败，直接截断早期消息
            self.messages = self.messages[-messages_to_keep:]

    def get_context_for_prompt(self) -> str:
        """
        获取适合拼入 Prompt 的上下文字符串

        Returns:
            str: 摘要 + 最近几轮对话
        """
        parts = []

        # 如果有摘要，先添加摘要
        if self.summary:
            parts.append(f"【对话摘要】\n{self.summary}\n")

        # 添加最近的对话历史
        if self.messages:
            parts.append("【最近对话】")
            for msg in self.messages:
                role = "用户" if msg["role"] == "user" else "助手"
                parts.append(f"{role}：{msg['content']}")

        return "\n".join(parts)

    def clear(self) -> None:
        """清空所有对话记忆"""
        self.messages = []
        self.summary = ""

    def get_stats(self) -> Dict:
        """
        获取记忆状态统计

        Returns:
            Dict: 包含消息数量、Token 数、是否有摘要
        """
        return {
            "message_count": len(self.messages),
            "estimated_tokens": self._count_tokens(),
            "has_summary": bool(self.summary),
            "max_token_limit": self.max_token_limit
        }


# 创建全局单例
_memory_instance = None


def get_memory(max_token_limit: int = 2000, llm_client: Optional[OllamaClient] = None) -> ConversationMemory:
    """
    获取对话记忆单例

    Args:
        max_token_limit: Token 阈值
        llm_client: LLM 客户端

    Returns:
        ConversationMemory: 对话记忆实例
    """
    global _memory_instance

    if _memory_instance is None:
        _memory_instance = ConversationMemory(max_token_limit, llm_client)

    return _memory_instance

# 使用示例
if __name__ == "__main__":
    memory = get_memory()

    # 添加对话
    memory.add_user_message("你好，我想了解 RAG")
    memory.add_assistant_message("RAG 是检索增强生成技术...")
    memory.add_user_message("它有什么优势？")
    memory.add_assistant_message("RAG 可以缓解幻觉问题...")

    # 获取上下文
    context = memory.get_context_for_prompt()
    print(context)