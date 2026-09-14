import os
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from app.config import config
from app.models.ollama_client import get_ollama_client
from pathlib import Path

# 当前文件路径 app/vectordb/chroma_store.py
current_file = Path(__file__)
# 向上三层到达项目根目录 ollama_rag_system
project_root = current_file.parent.parent.parent
# 向量库目录：根目录下 chroma_db
db_path = project_root / "chroma_db"
class ChromaStore:
    """
    Chroma 向量存储封装类

    负责文档的向量存储、检索和管理，复用 OllamaClient 的嵌入能力。
    """

    def __init__(self, collection_name: str = None, persist_dir: str = None):
        """
        初始化 ChromaStore

        Args:
            collection_name: 集合名称，用于隔离不同数据集
            persist_dir: 数据持久化目录
        """
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        # self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR

        project_root = Path(__file__).resolve().parent.parent.parent
        self.persist_dir = str(project_root / "chroma_db")

        # 确保目录存在（防止首次运行报错）
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # 复用全局单例 Ollama 客户端用于嵌入
        self.ollama_client = get_ollama_client()

        # 初始化 Chroma 实例
        self._init_chroma()

    def _init_chroma(self):
        """初始化 Chroma 向量数据库实例"""
        from langchain_chroma import Chroma
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.ollama_client.embeddings,  # 复用嵌入函数
            persist_directory=self.persist_dir
        )

    def add_documents(self, docs: List[Document]) -> List[str]:
        """
        添加文档到向量库

        Args:
            docs: Document 对象列表

        Returns:
            List[str]: 自动生成的文档 ID 列表

        Raises:
            Exception: 嵌入失败或存储失败时抛出
        """
        try:
            ids = self.chroma.add_documents(docs)
            return ids
        except Exception as e:
            raise Exception(f"添加文档失败: {str(e)}")

    def has_content_hash(self, content_hash: str) -> bool:
        """
        判断向量库中是否已存在指定内容哈希的文档

        Args:
            content_hash: 文档内容哈希（MD5）

        Returns:
            bool: 已存在返回 True，否则 False
        """
        try:
            result = self.chroma.get(where={"content_hash": content_hash}, limit=1)
            return bool(result.get("ids"))
        except Exception as e:
            # 查重失败不阻塞上传，按不存在处理（宁可重复也不误删）
            print(f"[WARN] 内容哈希查重失败: {e}")
            return False

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        按 ID 列表删除文档

        Args:
            ids: 要删除的文档 ID 列表
        """
        self.chroma.delete(ids=ids)
        # langchain_chroma 持久化 client 会自动落盘，无需手动 persist()

    def delete_by_filter(self, where: Dict) -> None:
        """
        按元数据条件删除文档

        Args:
            where: 过滤条件字典，如 {"source": "path/to/file.pdf"}
        """
        self.chroma.delete(where=where)
        # langchain_chroma 持久化 client 会自动落盘，无需手动 persist()

    def clear_collection(self) -> None:
        """清空整个集合（删除后重建）"""
        # 删除集合
        self.chroma.delete_collection()
        # 重新初始化
        self._init_chroma()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        相似度检索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            List[Document]: 匹配的文档列表（按相似度排序）
        """
        return self.chroma.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        带相似度分数的检索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            List[Tuple[Document, float]]: 匹配的文档及分数（分数越低越相似）
        """
        return self.chroma.similarity_search_with_score(query, k=k)

    def get_all_documents(self) -> List[Tuple[str, Document]]:
        """
        返回全量 (chunk_id, Document)，供 BM25 建索引

        Returns:
            List[Tuple[str, Document]]: (Chroma chunk UUID, Document) 列表
        """
        data = self.chroma._collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        result = []
        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else None
            result.append((doc_id, Document(page_content=content, metadata=meta or {})))
        return result

    def similarity_search_with_id(self, query: str, k: int = 20) -> List[Tuple[str, Document, float]]:
        """
        带 id 的稠密检索（供 RRF 融合对齐两路 chunk id）

        Args:
            query: 查询文本
            k: 返回条数

        Returns:
            List[Tuple[str, Document, float]]: (chunk_id, Document, distance)，距离越小越相似
        """
        embedding = self.ollama_client.embeddings.embed_query(query)
        res = self.chroma._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        id_list = (res.get("ids") or [[]])[0]
        doc_list = (res.get("documents") or [[]])[0]
        meta_list = (res.get("metadatas") or [[]])[0]
        dist_list = (res.get("distances") or [[]])[0]

        result = []
        for i, doc_id in enumerate(id_list):
            content = doc_list[i] if i < len(doc_list) else ""
            meta = meta_list[i] if i < len(meta_list) else None
            dist = dist_list[i] if i < len(dist_list) else 0.0
            result.append((doc_id, Document(page_content=content, metadata=meta or {}), dist))
        return result

    def get_collection_stats(self) -> Dict:
        """
        获取集合统计信息

        Returns:
            Dict: 包含文档数量等信息的字典
        """
        count = self.chroma._collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_dir": self.persist_dir
        }

    def list_documents(self) -> List[Dict]:
        """
        按文档（source）聚合知识库中的文档列表

        Returns:
            List[Dict]: 每项含 filename / size / chunks / uploaded_at
        """
        try:
            data = self.chroma._collection.get(include=["metadatas"])
        except Exception as e:
            raise Exception(f"获取文档元数据失败: {str(e)}")

        metadatas = data.get("metadatas") or []

        # 按 source 聚合：统计每个文件的 chunk 数，取最大上传时间
        grouped: Dict[str, Dict] = {}
        for meta in metadatas:
            if not meta:
                continue
            source = meta.get("source")
            if not source:
                continue
            entry = grouped.setdefault(source, {"chunks": 0, "uploaded_at": None})
            entry["chunks"] += 1
            up = meta.get("uploaded_at")
            if isinstance(up, (int, float)) and (
                entry["uploaded_at"] is None or up > entry["uploaded_at"]
            ):
                entry["uploaded_at"] = up

        docs = []
        for source, info in grouped.items():
            size = None
            if os.path.exists(source):
                size = os.path.getsize(source)
            docs.append({
                "filename": os.path.basename(source),
                "size": size,
                "chunks": info["chunks"],
                "uploaded_at": info["uploaded_at"],
            })

        # 按上传时间倒序（新的在前），无时间的排最后
        docs.sort(
            key=lambda d: (d["uploaded_at"] is not None, d["uploaded_at"] or 0),
            reverse=True,
        )
        return docs

if __name__ == "__main__":
    store = ChromaStore()
    print(store.get_collection_stats())