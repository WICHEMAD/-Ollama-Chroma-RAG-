from typing import List, Dict, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.config import config
from app.models.ollama_client import OllamaClient
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

        # 创建 Ollama 客户端用于嵌入
        self.ollama_client = OllamaClient()

        # 初始化 Chroma 实例
        self._init_chroma()

    def _init_chroma(self):
        """初始化 Chroma 向量数据库实例"""
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

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        按 ID 列表删除文档

        Args:
            ids: 要删除的文档 ID 列表
        """
        self.chroma.delete(ids=ids)
        self.chroma.persist()

    def delete_by_filter(self, where: Dict) -> None:
        """
        按元数据条件删除文档

        Args:
            where: 过滤条件字典，如 {"source": "path/to/file.pdf"}
        """
        self.chroma.delete(where=where)
        self.chroma.persist()

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

if __name__ == "__main__":
    store = ChromaStore()
    print(store.get_collection_stats())