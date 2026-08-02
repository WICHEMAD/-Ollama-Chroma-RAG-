from typing import List, Optional, Tuple
from langchain_core.documents import Document
from app.vectordb.chroma_store import ChromaStore
from sentence_transformers import CrossEncoder
class Retriever:
    """
    检索器类

    支持基础检索和 Rerank 优化检索两种模式
    """

    def __init__(self, chroma_store: ChromaStore, use_rerank: bool = False):
        """
        初始化检索器

        Args:
            chroma_store: ChromaStore 实例
            use_rerank: 是否使用重排序优化
        """
        self.chroma_store = chroma_store
        self.use_rerank = use_rerank
        self.reranker = None

        # 如果启用 rerank，延迟加载重排序模型
        if use_rerank:
            self._init_reranker()

    def _init_reranker(self):
        try:
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                max_length=512  # 限制输入长度，加速计算
            )
        except Exception as e:
            raise RuntimeError(f"加载重排序模型失败: {str(e)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            List[Document]: 按相关性排序的文档列表
        """
        if self.use_rerank:
            return self._retrieve_with_rerank(query, top_k)
        else:
            return self._retrieve_basic(query, top_k)

    def _retrieve_basic(self, query: str, top_k: int) -> List[Document]:
        """
        基础检索：直接使用相似度搜索

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            List[Document]: 文档列表
        """
        results = self.chroma_store.similarity_search(query, k=top_k)
        return results

    def _retrieve_with_rerank(self, query: str, top_k: int) -> List[Document]:
        """
        Rerank 优化检索：先粗检索再精排序

        Args:
            query: 查询文本
            top_k: 最终返回数量

        Returns:
            List[Document]: 重排序后的文档列表
        """
        # 1. 粗检索：获取较多候选（如 20 条）
        rough_results = self.chroma_store.similarity_search_with_score(
            query,
            k=min(top_k * 4, 20)  # 取 top_k 的 4 倍或最多 20 条
        )

        if not rough_results:
            return []

        # 2. 准备重排序输入：(query, chunk) 对
        pairs = []
        doc_list = []

        for doc, score in rough_results:
            pairs.append([query, doc.page_content])
            doc_list.append(doc)

        # 3. 调用重排序模型
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        # 4. 合并结果并按 rerank 分数排序
        scored_docs = list(zip(doc_list, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 5. 更新文档元数据中的分数信息
        final_docs = []
        for i, (doc, score) in enumerate(scored_docs[:top_k], start=1):
            # 添加重排序后的排名和分数到元数据
            doc.metadata["rerank_rank"] = i
            doc.metadata["rerank_score"] = round(float(score), 4)
            final_docs.append(doc)

        return final_docs


# 创建全局单例检索器
_retriever_instance = None


def get_retriever(chroma_store: ChromaStore = None, use_rerank: bool = False) -> Retriever:
    """
    获取检索器单例

    Args:
        chroma_store: ChromaStore 实例（首次调用时需要）
        use_rerank: 是否使用重排序

    Returns:
        Retriever: 检索器实例
    """
    global _retriever_instance

    if _retriever_instance is None:
        if chroma_store is None:
            from app.vectordb.chroma_store import ChromaStore
            chroma_store = ChromaStore()
        _retriever_instance = Retriever(chroma_store, use_rerank)

    return _retriever_instance

# 使用示例
if __name__ == "__main__":
    # 测试启用 Rerank
    retriever = get_retriever(use_rerank=True)
    docs = retriever.retrieve("阿里", top_k=5)
    for doc in docs:
        print(f"来源: {doc.metadata.get('source')}")
        print(f"重排序分数: {doc.metadata.get('rerank_score')}")
        print(doc.page_content[:100] + "...\n")