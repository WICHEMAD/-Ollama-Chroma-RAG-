from pathlib import Path
from typing import List
from langchain_core.documents import Document
from app.document.loader import DocumentLoader
from app.document.splitter import sqlit_document
from app.vectordb.chroma_store import ChromaStore


class DocumentService:
    """
    文档服务类

    提供文档的完整处理流程：加载 → 切分 → 入库
    """

    def __init__(self):
        """初始化服务"""
        self.loader = DocumentLoader()
        self.store = ChromaStore()

    def process_and_store(self, file_path: str) -> dict:
        """
        完整流程：加载文档 → 切分 → 入库

        Args:
            file_path: 文件路径（支持 .pdf 和 .txt）

        Returns:
            dict: 处理结果包含文档ID列表和统计信息
        """
        # 1. 加载文档
        docs = self.loader.load(file_path)

        # 2. 切分文档
        chunks = sqlit_document(docs)

        # 3. 入库
        doc_ids = self.store.add_documents(chunks)

        # 4. 返回结果
        return {
            "file_path": file_path,
            "original_pages": len(docs),
            "chunks_created": len(chunks),
            "document_ids": doc_ids,
            "status": "success"
        }

    def batch_process(self, file_paths: List[str]) -> List[dict]:
        """
        批量处理多个文件

        Args:
            file_paths: 文件路径列表

        Returns:
            List[dict]: 每个文件的处理结果
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.process_and_store(file_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "file_path": file_path,
                    "status": "failed",
                    "error": str(e)
                })
        return results

    def delete_document(self, file_path: str) -> dict:
        """
        删除指定文件的所有文档块

        Args:
            file_path: 要删除的文件路径

        Returns:
            dict: 删除结果
        """
        try:
            self.store.delete_by_filter({"source": file_path})
            return {
                "file_path": file_path,
                "status": "success",
                "message": "文档已删除"
            }
        except Exception as e:
            return {
                "file_path": file_path,
                "status": "failed",
                "error": str(e)
            }

    def get_store_stats(self) -> dict:
        """
        获取向量库统计信息

        Returns:
            dict: 统计信息
        """
        return self.store.get_collection_stats()

# 使用示例
if __name__ == "__main__":
    service = DocumentService()

    # 处理单个文件
    service = DocumentService()
    current_file = Path(__file__)
    # 当前文件 app/document/service.py
    root_dir = current_file.parent.parent.parent
    pdf_path = root_dir / "data" / "陈冠霖优化版2.pdf"

    result = service.process_and_store(str(pdf_path))
    print(result)

    # 获取统计信息
    stats = service.get_store_stats()
    print(stats)