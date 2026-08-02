from pathlib import Path
from typing import List
from app.document.loader import DocumentLoader
from charset_normalizer.utils import is_separator
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import config


def sqlit_document(
        document: list[Document],
        chunk_size: int = None,
        chunk_overlap: int = None,
) -> List[Document]:

    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    sqlitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[            "\n\n",  # 段落分隔
            "\n",    # 换行
            "。",    # 中文句号
            "！",    # 感叹号
            "？",    # 问号
            "；",    # 分号
            "，",    # 逗号
            ".",     # 英文句号
            "!",     # 英文感叹号
            "?",     # 英文问号
            ";",     # 英文分号
            ",",     # 英文逗号
            " ",     # 空格
            ""       # 无分隔符（最后兜底）]
        ],
        length_function= len
    )
    #执行切分
    split_documents=sqlitter.split_documents(document)

    #为每个切块添加序号
    for i, doc in enumerate(split_documents, start=1):
        doc.metadata["chunk_id"] = i
        doc.metadata["chunk_count"] = len(split_documents)

    return split_documents

if __name__ == "__main__":
    loader = DocumentLoader()
    current_file=Path(__file__)
    root_dir=current_file.parent.parent.parent
    target_file=root_dir/"data"/"陈冠霖优化版2.pdf"
    docs = loader.load(str(target_file))
    print(sqlit_document(docs))