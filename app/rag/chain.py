from typing import List, Dict, Generator, Optional
from langchain_core.documents import Document
from app.rag.retriever import Retriever, get_retriever
from app.models.ollama_client import OllamaClient, get_ollama_client, get_chat_client
from app.config import config
import re
from app.rag.memory import ConversationMemory, get_memory
from app.rag.query_rewriter import QueryRewriter, get_query_rewriter
class RAGChain:
    """
    RAG 问答链

    将检索、上下文拼接、LLM 调用整合为完整的问答流程
    """

    def __init__(self, retriever: Optional[Retriever] = None, llm_client: Optional[OllamaClient] = None, memory: Optional[ConversationMemory] = None):
        """
        初始化 RAG 链

        Args:
            retriever: 检索器实例（可选，默认创建）
            llm_client: LLM 客户端实例（可选，默认创建）
        """
        self.retriever = retriever or get_retriever()
        self.llm_client = llm_client or get_ollama_client()
        self.memory=memory or get_memory()
        self.query_rewriter = get_query_rewriter()
    def _build_context(self, docs: List[Document]) -> str:
        """
        将文档列表拼接成格式化的上下文字符串

        Args:
            docs: 检索到的文档列表

        Returns:
            str: 格式化的上下文
        """
        context_parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get('source', '未知来源')
            page = doc.metadata.get('page', '')
            score = doc.metadata.get('rerank_score', doc.metadata.get('similarity_score', ''))

            # 构建来源信息
            source_info = f"source: {source}"
            if page:
                source_info += f", page: {page}"
            if score:
                source_info += f", score: {score}"

            # 拼接单个文档块
            doc_str = f"[{i}] {source_info}\n{doc.page_content}\n"
            context_parts.append(doc_str)

        return "\n".join(context_parts)

    def _build_messages(self, question: str, context: str, history: Optional[List[dict]] = None) -> List[dict]:
        """
        构建消息列表

        Args:
            question: 用户问题
            context: 检索到的上下文
            history: 历史对话记录（可选）

        Returns:
            List[dict]: 消息列表
        """
        messages = []

        # 系统提示
        system_prompt = """
你是一个严谨的文档问答助手。请仅根据下面的参考文档内容回答问题。
如果无法从文档中找到答案，请回答"根据已有文档无法回答"，不要编造任何信息。
当引用文档内容时，请使用方括号标注来源编号（如 [1]）。
""".strip()
        messages.append({"role": "system", "content": system_prompt})

        # 如果有历史对话，加入摘要或最近几轮
        if history and isinstance(history, list):
            self.memory.clear()
            for msg in history:
                if msg.get("role") == "user":
                    self.memory.add_user_message(msg["content"])
                elif msg.get("role") == "assistant":
                    self.memory.add_assistant_message(msg["content"])

            history_context=self.memory.get_context_for_prompt()
            if history_context:
                messages.append({"role": "user", "content": f"【对话历史】\n{history_context}"})

        # 用户消息（包含上下文和问题）
        user_prompt = f"""
参考文档：
{context}

用户问题：{question}
请回答：
""".strip()
        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _extract_sources(self, answer: str, docs: List[Document]) -> List[Dict]:
        """
        从回答中提取引用的来源

        Args:
            answer: 模型生成的回答
            docs: 检索到的文档列表

        Returns:
            List[Dict]: 来源信息列表
        """


        sources = []
        # 匹配回答中的引用编号（如 [1], [2-3]）
        matches = re.findall(r'\[(\d+)\]', answer)

        for match in matches:
            try:
                idx = int(match) - 1  # 转换为 0 索引
                if 0 <= idx < len(docs):
                    doc = docs[idx]
                    source_info = {
                        "source": doc.metadata.get('source', '未知来源'),
                        "page": doc.metadata.get('page', None),
                        "chunk_id": doc.metadata.get('chunk_id', None),
                        "score": doc.metadata.get('rerank_score', doc.metadata.get('similarity_score', None))
                    }
                    # 去重
                    if source_info not in sources:
                        sources.append(source_info)
            except ValueError:
                continue

        return sources

    def answer(self, question: str, history: Optional[List[dict]] = None, model: Optional[str] = None) -> Dict:
        """
        获取完整回答
        """
        # 1. 改写查询（检索用改写后的 query，回答仍用原始 question）
        search_query = self.query_rewriter.rewrite(question, history, config.QUERY_REWRITE_MODEL)
        if search_query != question:
            print(f"DEBUG: 改写查询: {question!r} -> {search_query!r}")

        # 2. 检索相关文档
        docs = self.retriever.retrieve(search_query)
        print(f"DEBUG: 检索到 {len(docs)} 个文档")

        # ✅ 添加这部分调试代码
        print("\n=== 检索到的文档 ===")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get('source', '未知')
            page = doc.metadata.get('page', '未知')
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"[{i}] 来源: {source}, 页码: {page}")
            print(f"内容预览: {content_preview}\n")

        if not docs:
            return {
                "answer": "根据已有文档无法回答",
                "sources": []
            }

        # 2. 构建上下文
        context = self._build_context(docs)

        # ✅ 添加这行调试代码
        print(f"DEBUG: 上下文长度: {len(context)} 字符")

        # 3. 构建消息列表
        messages = self._build_messages(question, context, history)

        # 4. 调用 LLM（云端优先；云端失败回退本地兜底模型）
        llm_client = get_chat_client(model)
        fallback = False
        try:
            answer = llm_client.chat(messages)
        except Exception as e:
            if model:
                fallback = True
                print(f"[chain] 云端模型调用失败，回退本地模型: {e}")
                answer = get_ollama_client().chat(messages)
            else:
                raise

        # 5. 提取来源信息
        sources = self._extract_sources(answer, docs)

        return {
            "answer": answer,
            "sources": sources,
            "context_length": len(docs),
            "rewritten_query": search_query,
            "fallback": fallback
        }

    def answer_stream(self, question: str, history: Optional[List[dict]] = None, model: Optional[str] = None):
        """
        流式获取回答

        Args:
            question: 用户问题
            history: 历史对话记录（可选）

        Yields:
            str: 逐块的回答内容；最后 yield 一个 dict（来源元数据）
        """
        # 1. 改写查询（检索用改写后的 query，回答仍用原始 question）
        search_query = self.query_rewriter.rewrite(question, history, config.QUERY_REWRITE_MODEL)

        # 2. 检索相关文档
        docs = self.retriever.retrieve(search_query)

        if not docs:
            yield "根据已有文档无法回答"
            yield {"type": "sources", "sources": [], "context_length": 0, "rewritten_query": search_query, "fallback": False}
            return

        # 2. 构建上下文
        context = self._build_context(docs)

        # 3. 构建消息列表
        messages = self._build_messages(question, context, history)

        # 4. 流式调用 LLM（云端优先；云端在首个 chunk 前失败则回退本地）
        llm_client = get_chat_client(model)
        answer_parts = []
        fallback = False
        first = True
        try:
            for chunk in llm_client.chat_stream(messages):
                first = False
                answer_parts.append(chunk)
                yield chunk
        except Exception as e:
            if model and first:
                fallback = True
                print(f"[chain] 云端模型调用失败，回退本地模型: {e}")
                for chunk in get_ollama_client().chat_stream(messages):
                    answer_parts.append(chunk)
                    yield chunk
            else:
                raise

        full_answer = "".join(answer_parts)
        sources = self._extract_sources(full_answer, docs)
        yield {"type": "sources", "sources": sources, "context_length": len(docs), "rewritten_query": search_query, "fallback": fallback}

# 全局单例（懒加载，不在模块导入时实例化）
_rag_chain_instance = None


def get_rag_chain(retriever: Optional[Retriever] = None, llm_client: Optional[OllamaClient] = None) -> RAGChain:
    """
    获取 RAGChain 单例

    Args:
        retriever: 检索器实例
        llm_client: LLM 客户端实例

    Returns:
        RAGChain: RAG 链实例
    """
    global _rag_chain_instance

    if _rag_chain_instance is None:
        _rag_chain_instance = RAGChain(retriever, llm_client)

    return _rag_chain_instance

# 使用示例
if __name__ == "__main__":
    from app.vectordb.chroma_store import ChromaStore
    from app.rag.retriever import Retriever

    # 显式创建与入库时完全相同的存储实例
    store = ChromaStore(collection_name="my_knowledge_base")
    print("库统计：", store.get_collection_stats())

    retriever = Retriever(chroma_store=store, use_rerank=False)
    chain = RAGChain(retriever=retriever)

    result = chain.answer("陈冠霖的具体内容")
    print("答案:", result["answer"])
    print("来源:", result["sources"])