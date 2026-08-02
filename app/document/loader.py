import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from pathlib import Path
class DocumentLoader:
    from pathlib import Path
    from typing import List
    from langchain_core.documents import Document

class DocumentLoader:
    """文档加载器类，根据文件后缀自动选择解析方式"""

    def load(self, file_path: str) -> List[Document]:
        """
        加载文档，根据文件后缀选择对应的解析方法

        Args:
            file_path: 文件路径

        Returns:
            List[Document]: 文档对象列表
        """
        suffix = Path(file_path).suffix.lower()

        if suffix == '.pdf':
            return self._load_pdf(file_path)
        elif suffix == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}，仅支持 .pdf 和 .txt")

    # 将原有的 _load_pdf 方法替换为：
    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        解析 PDF 文件（使用 PyPDFLoader）

        Args:
            file_path: PDF 文件路径

        Returns:
            List[Document]: 每页一个 Document 对象
        """
        loader = PyPDFLoader(file_path)
        return loader.load()

    # 将原有的 _load_txt 方法替换为
    def _load_txt(self, file_path: str) -> List[Document]:
        """
        解析 TXT 文件（使用 TextLoader）

        Args:
            file_path: TXT 文件路径

        Returns:
            List[Document]: 包含单个 Document 对象的列表
        """
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

if __name__ == "__main__":
    loader = DocumentLoader()
    current_file=Path(__file__)
    root_dir=current_file.parent.parent.parent
    target_file=root_dir/"data"/"尺寸.txt"
    docs = loader.load(str(target_file))
    print(docs)
    target_file=root_dir/"data"/"陈冠霖优化版2.pdf"
    docs = loader.load(str(target_file))
    print(docs)

