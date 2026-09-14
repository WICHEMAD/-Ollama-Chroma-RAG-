from typing import List, Optional, Tuple
from langchain_core.documents import Document
from app.vectordb.chroma_store import ChromaStore
from app.config import config
from app.rag.sparse_retriever import get_bm25_retriever


def rrf_fusion(result_lists: List[List[str]], k: int = 60) -> List[str]:
    """
    RRF（倒数排名融合）：按文档 id 聚合多路排名，同一文档多路命中时分数叠加、只保留一条

    Args:
        result_lists: 每路检索返回的文档 id 列表（已按相关性降序）
        k: RRF 常数（默认 60）

    Returns:
        List[str]: 按融合分数降序的文档 id 列表
    """
    scores = {}
    for ranking in result_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]


class Retriever:
    """
    检索器类

    支持基础检索、Rerank 优化检索、混合检索（稠密 + BM25 稀疏）三种模式
    """

    def __init__(self, chroma_store: ChromaStore, use_rerank: bool = False,
                 rerank_model: Optional[str] = None,
                 rough_top_k: int = 20, final_top_k: int = 3):
        """
        初始化检索器

        Args:
            chroma_store: ChromaStore 实例
            use_rerank: 是否使用重排序优化
            rerank_model: 重排模型名（默认取 config.RERANK_MODEL）
            rough_top_k: 初检召回数量
            final_top_k: 最终喂给 LLM 的文档块数
        """
        self.chroma_store = chroma_store
        self.use_rerank = use_rerank
        self.rerank_model = rerank_model or config.RERANK_MODEL
        self.rough_top_k = rough_top_k
        self.final_top_k = final_top_k
        self.reranker = None

        # 如果启用 rerank，加载重排序模型
        if use_rerank:
            self._init_reranker()

    def _init_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                self.rerank_model,
                max_length=512  # 限制输入长度，加速计算
            )
        except Exception as e:
            # 重排模型加载失败（如首次未下载）→ 降级为基础检索，避免整条链路崩溃
            print(f"[WARN] 重排模型加载失败，已降级为基础检索: {e}")
            self.use_rerank = False
            self.reranker = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数量（默认取 final_top_k）

        Returns:
            List[Document]: 按相关性排序的文档列表
        """
        if config.ENABLE_HYBRID:
            return self._retrieve_hybrid(query, top_k)

        top_k = top_k or self.final_top_k
        if self.use_rerank and self.reranker is not None:
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
        # 1. 粗检索：获取较多候选（RETRIEVE_TOP_K 条）
        rough_results = self.chroma_store.similarity_search_with_score(
            query,
            k=self.rough_top_k
        )

        if not rough_results:
            return []

        doc_list = [doc for doc, _ in rough_results]

        # 2. 精排
        return self._rerank_docs(query, doc_list, top_k)

    def _rerank_docs(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """用 CrossEncoder 对候选文档精排，返回 top_k 条（带 rerank_rank / rerank_score 元数据）"""
        pairs = [[query, doc.page_content] for doc in docs]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        scored_docs = list(zip(docs, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        final_docs = []
        for i, (doc, score) in enumerate(scored_docs[:top_k], start=1):
            doc.metadata["rerank_rank"] = i
            doc.metadata["rerank_score"] = round(float(score), 4)
            final_docs.append(doc)

        return final_docs

    def _retrieve_hybrid(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        混合检索：稠密 + BM25 稀疏两路召回 → RRF 融合 → 可选 rerank 精排

        基础（无 rerank）：RRF 融合取 HYBRID_RRF_TOP_K 条返回
        rerank：RRF 融合取 HYBRID_RRF_TOP_K 候选 → 精排取 HYBRID_RERANK_TOP_K 条返回
        """
        # 1. 稠密召回（带 id）
        dense = self.chroma_store.similarity_search_with_id(query, k=config.HYBRID_DENSE_TOP_K)
        dense_ids = [doc_id for doc_id, _, _ in dense]

        # 2. 稀疏召回（BM25，懒加载建索引；失败降级为仅稠密）
        sparse = []
        sparse_ids = []
        try:
            bm25 = get_bm25_retriever()
            if not bm25.is_built:
                bm25.build_index(self.chroma_store.get_all_documents())
            sparse = bm25.search(query, k=config.HYBRID_SPARSE_TOP_K)
            sparse_ids = [doc_id for doc_id, _ in sparse]
        except Exception as e:
            print(f"[WARN] 稀疏检索失败，已降级为仅稠密召回: {e}")

        # 3. RRF 融合
        fused_ids = rrf_fusion([dense_ids, sparse_ids], k=config.RRF_K)

        # 4. id -> Document 映射，按融合顺序取 top-k 候选
        id_to_doc = {}
        for doc_id, doc, _ in dense:
            id_to_doc[doc_id] = doc
        for doc_id, doc in sparse:
            id_to_doc[doc_id] = doc

        fused_docs = []
        for doc_id in fused_ids:
            if doc_id in id_to_doc:
                fused_docs.append(id_to_doc[doc_id])
            if len(fused_docs) >= config.HYBRID_RRF_TOP_K:
                break

        if not fused_docs:
            return []

        # 5. rerank 精排 or 直接返回
        if self.use_rerank and self.reranker is not None:
            return self._rerank_docs(query, fused_docs, config.HYBRID_RERANK_TOP_K)
        return fused_docs


# 创建全局单例检索器
_retriever_instance = None


def get_retriever(chroma_store: ChromaStore = None, use_rerank: Optional[bool] = None,
                  rerank_model: Optional[str] = None,
                  rough_top_k: Optional[int] = None, final_top_k: Optional[int] = None) -> Retriever:
    """
    获取检索器单例

    Args:
        chroma_store: ChromaStore 实例（首次调用时需要）
        use_rerank: 是否使用重排序（None 时取 config.ENABLE_RERANK）

    Returns:
        Retriever: 检索器实例
    """
    global _retriever_instance

    if _retriever_instance is None:
        if chroma_store is None:
            from app.vectordb.chroma_store import ChromaStore
            chroma_store = ChromaStore()
        use_rerank = config.ENABLE_RERANK if use_rerank is None else use_rerank
        _retriever_instance = Retriever(
            chroma_store,
            use_rerank=use_rerank,
            rerank_model=rerank_model or config.RERANK_MODEL,
            rough_top_k=rough_top_k or config.RETRIEVE_TOP_K,
            final_top_k=final_top_k or config.RERANK_TOP_N,
        )

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