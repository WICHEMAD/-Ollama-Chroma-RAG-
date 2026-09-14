"""BM25 稀疏关键词检索器（混合检索的稀疏一路）

jieba 分词 + rank_bm25.BM25Okapi。索引为内存态，与 Chroma 实际存储的 chunk 对齐：
首次检索时由调用方（Retriever）懒加载建索引；入库 / 删除后调用 invalidate_bm25_index()
失效，下次检索时重建。采用全量重建而非增量，知识库 chunk 量不大（内部自用）。
"""
from typing import List, Tuple, Optional
from langchain_core.documents import Document


class BM25Retriever:
    """BM25 关键词检索器（稀疏召回）"""

    def __init__(self):
        self._index = None                              # BM25Okapi 实例
        self._docs: List[Tuple[str, Document]] = []     # (id, Document)，与索引行一一对应

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """建索引分词：jieba 标准模式"""
        import jieba
        return jieba.lcut(text)

    @staticmethod
    def _tokenize_query(text: str) -> List[str]:
        """查询分词：jieba 搜索引擎模式，切分更细，避免查询词与索引词粒度不一致而匹配不上"""
        import jieba
        return jieba.lcut_for_search(text)

    @property
    def is_built(self) -> bool:
        return self._index is not None

    def build_index(self, id_docs: List[Tuple[str, Document]]) -> None:
        """从 (id, Document) 列表重建 BM25 索引

        Args:
            id_docs: (chunk_id, Document) 列表，id 与 Chroma 的 chunk UUID 对齐
        """
        from rank_bm25 import BM25Okapi
        if not id_docs:
            self._index = None
            self._docs = []
            return
        tokenized = [self._tokenize(doc.page_content) for _, doc in id_docs]
        self._index = BM25Okapi(tokenized)
        self._docs = id_docs

    def search(self, query: str, k: int = 20) -> List[Tuple[str, Document]]:
        """检索，返回按 BM25 分数降序的 (id, Document) 列表

        Args:
            query: 查询文本
            k: 返回条数

        Returns:
            List[Tuple[str, Document]]: (chunk_id, Document)，已按相关性降序
        """
        if self._index is None or not self._docs:
            return []
        tokens = self._tokenize_query(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            zip(self._docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc_id, doc) for (doc_id, doc), _ in ranked[:k]]


# 全局单例（懒加载，不在模块导入时建索引，避免首帧 import jieba / rank_bm25 拖慢启动）
_bm25_retriever: Optional[BM25Retriever] = None


def get_bm25_retriever() -> BM25Retriever:
    """获取 BM25 检索器单例"""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever


def invalidate_bm25_index() -> None:
    """入库 / 删除后失效 BM25 索引，下次检索时懒重建"""
    global _bm25_retriever
    if _bm25_retriever is not None:
        _bm25_retriever._index = None
        _bm25_retriever._docs = []


if __name__ == "__main__":
    # 自测：构造几段模拟 chunk，验证分词 / 建索引 / 关键词搜索排序
    sample = [
        ("d1", Document(page_content="食品安全法规定，食品添加剂的使用应当符合食品安全标准。", metadata={"source": "食品安全法.txt"})),
        ("d2", Document(page_content="食品生产经营者应当建立食品安全自查制度。", metadata={"source": "食品安全法.txt"})),
        ("d3", Document(page_content="上市公司应当依法披露定期报告和临时报告。", metadata={"source": "证券法.txt"})),
        ("d4", Document(page_content="证券交易活动中，内幕信息知情人不得买卖证券。", metadata={"source": "证券法.txt"})),
    ]

    bm25 = BM25Retriever()
    bm25.build_index(sample)

    for q in ["食品添加剂", "信息披露", "食品安全"]:
        print(f"\n查询：{q}")
        print(f"  分词：{bm25._tokenize_query(q)}")
        for doc_id, doc in bm25.search(q, k=3):
            print(f"  [{doc_id}] ({doc.metadata.get('source')}) {doc.page_content[:30]}...")
