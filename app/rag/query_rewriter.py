from typing import List, Optional
from app.config import config
from app.models.ollama_client import build_chat_client


class QueryRewriter:
    """
    查询改写器

    把用户口语化、日常化的问题改写成标准、规范的技术/法律术语问题，
    补充关键术语、去除口语与缩写、保留原意，使其更适合向量检索与关键词检索，
    从而提升后续检索准确率。

    - enabled=False 时原样透传，不改写。
    - 未提供聊天模型（model=None）或改写失败/返回空/超时（5s）时回退原始问题，保证主链路不被阻塞。
    - 改写用低温度（0.3）独立云端客户端，不复用生成客户端的缓存与温度。
    """

    # 改写专属参数：短超时（5s）+ 低温度（0.3），超时/失败即回退原问题，不拖慢主链路
    REWRITE_TIMEOUT = 5.0
    REWRITE_TEMPERATURE = 0.3

    SYSTEM_PROMPT = (
        "你是检索查询改写助手。把用户的口语化、日常化问题改写成标准、规范的技术或法律术语问题，"
        "补充关键术语、去除口语和缩写、保留原意，使其更适合向量检索和关键词检索。"
        "如果问题本身已经是规范表述，原样返回即可。只输出改写后的问题本身，不要任何解释。"
    )

    _REWRITE_EXAMPLES = (
        "示例1：\n"
        "【当前问题】车被别人撞了该找谁？\n"
        "改写为：发生道路交通事故后，责任如何认定？赔偿责任如何承担？\n\n"
        "示例2：\n"
        "【当前问题】买的食品吃坏肚子了怎么办？\n"
        "改写为：因食用不符合食品安全标准的食品受到损害，如何向生产者或经营者索赔？\n\n"
        "示例3：\n"
        "【对话历史】用户：什么是 RAG？ / 助手：RAG 是检索增强生成技术。\n"
        "【当前问题】它有什么优势？\n"
        "改写为：RAG 检索增强生成技术有什么优势？"
    )

    def __init__(self):
        self.enabled = config.QUERY_REWRITE_ENABLED

    @staticmethod
    def _format_history(history: Optional[List[dict]]) -> str:
        """把历史消息列表格式化为「用户：… / 助手：…」文本"""
        if not history:
            return "（无）"

        lines = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"助手：{content}")
        return "\n".join(lines) if lines else "（无）"

    @staticmethod
    def _clean(text: str) -> str:
        """清洗模型输出：去首尾空白与常见包裹引号"""
        if not text:
            return ""
        text = text.strip()
        # 去掉首尾成对的引号
        if len(text) >= 2 and text[0] in "\"'「『" and text[-1] in "\"'」』":
            text = text[1:-1].strip()
        return text

    def rewrite(self, question: str, history: Optional[List[dict]] = None,
                model: Optional[str] = None) -> str:
        """
        改写查询

        Args:
            question: 用户原始问题
            history: 历史对话记录（可选）
            model: 聊天用的云端模型名（可选，缺省不改写）

        Returns:
            str: 改写后的查询；未启用 / 无 model / 失败时返回原始 question
        """
        if not self.enabled:
            return question
        if not model:
            return question

        history_text = self._format_history(history)
        user_prompt = (
            f"{self._REWRITE_EXAMPLES}\n\n"
            f"现在改写：\n【对话历史】\n{history_text}\n【当前问题】\n{question}\n改写为："
        )

        try:
            client = build_chat_client(
                model,
                temperature=self.REWRITE_TEMPERATURE,
                request_timeout=self.REWRITE_TIMEOUT,
            )
            reply = client.chat([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])
            cleaned = self._clean(reply)
            return cleaned if cleaned else question
        except Exception as e:
            print(f"[QueryRewriter] 改写失败，回退原始问题: {e}")
            return question


# 全局单例
_rewriter_instance: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    """获取查询改写器单例"""
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter()
    return _rewriter_instance


if __name__ == "__main__":
    import sys
    rewriter = get_query_rewriter()
    model = sys.argv[1] if len(sys.argv) > 1 else None
    if not model:
        print("用法：python app/rag/query_rewriter.py <云端模型名>")
        sys.exit(0)
    for q in ["车被别人撞了该找谁？", "买的食品吃坏肚子了怎么办？"]:
        print(f"改写前: {q}")
        print(f"改写后: {rewriter.rewrite(q, model=model)}")
        print()
