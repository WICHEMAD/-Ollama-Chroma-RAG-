"""文档管理路由"""
import os
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from ..document.service import DocumentService
from ..vectordb.chroma_store import ChromaStore  # /stats、/list 接口用

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf", "text/plain"}
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    # 校验文件类型
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {file.content_type}。仅支持 PDF 和纯文本。"
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 保存文件
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 走统一入库流程：加载 → 去重 → 切分 → 入库
        result = DocumentService().process_and_store(file_path)

        if result["status"] == "duplicate":
            return {
                "status": "duplicate",
                "filename": file.filename,
                "skipped": True,
                "content_hash": result["content_hash"],
            }

        return {
            "status": "success",
            "filename": file.filename,
            "content_hash": result["content_hash"],
            "original_pages": result["original_pages"],
            "chunks_count": result["chunks_created"],
            "stored_count": result["chunks_created"],
        }

    except Exception as e:
        # 清理已保存的文件
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件保存或入库失败: {str(e)}"
        )


@router.get("/stats")
async def get_document_stats():
    """
    获取文档统计信息
    """
    try:
        # 修改后
        store = ChromaStore()

        # ✅ 添加兼容性检查
        if hasattr(store, 'get_collection_stats'):
            stats = store.get_collection_stats()
        else:
            # 如果没有该方法，返回默认信息
            stats = {
                "collection_name": "my_knowledge_base",
                "document_count": 0,
                "persist_dir": store.persist_dir if hasattr(store, 'persist_dir') else "unknown"
            }

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/list")
async def list_documents():
    """
    获取知识库文档列表（按文件聚合）
    """
    try:
        store = ChromaStore()
        documents = store.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")