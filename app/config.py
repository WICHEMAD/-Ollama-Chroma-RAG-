import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 修复 Windows 上 platform 模块因 subprocess 调用卡死的问题。
# torch / transformers 在 import 时会调用 platform.machine() -> platform.uname() -> platform.win32_ver()
# platform.win32_ver() 会先尝试 WMI，回退到 subprocess.check_output('cmd /c ver', shell=True)
# 在 uvicorn multiprocessing.spawn 子进程中，subprocess 调用可能永久挂起。
# 这里用 sys.getwindowsversion() 替换 platform.win32_ver()，彻底避免 subprocess 调用。
platform._wmi = None

_win32_ver_cached = None
_original_win32_ver = platform.win32_ver

def _patched_win32_ver(release='', version='', csd='', ptype=''):
    global _win32_ver_cached
    if _win32_ver_cached is not None:
        return _win32_ver_cached
    try:
        _win32_ver_cached = _original_win32_ver(release, version, csd, ptype)
    except Exception:
        wv = sys.getwindowsversion()
        _win32_ver_cached = (wv.major, wv.minor, wv.build, wv.platform)
    return _win32_ver_cached

platform.win32_ver = _patched_win32_ver

class Config:
    # 环境变量键与 .env 保持一致（LLM_MODEL 等），属性名仍用 CHAT_* 供代码引用
    CHAT_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")
    EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    CHAT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
    CHAT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 4096))
    CHAT_NUM_CTX = int(os.getenv("LLM_NUM_CTX", 4096))
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # 在 Config 类中添加以下两行
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    # 在 Config 类中添加以下配置
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_knowledge_base")
    AGENT_MODEL = os.getenv("AGENT_MODEL", "deepseek-r1:7b")
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 5))
    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    # 检索重排配置（二次重排，精排后取更少块以控制上下文 token）
    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"
    RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", 20))
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 3))
    RERANK_MODEL = os.getenv("RERANK_MODEL", "maidalun1020/bce-reranker-base_v1")
    # 混合检索配置（稠密 + BM25 稀疏两路召回 → RRF 融合 → 可选 rerank 精排）
    ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"
    HYBRID_DENSE_TOP_K = int(os.getenv("HYBRID_DENSE_TOP_K", 20))    # 稠密召回数
    HYBRID_SPARSE_TOP_K = int(os.getenv("HYBRID_SPARSE_TOP_K", 20))  # 稀疏召回数
    HYBRID_RRF_TOP_K = int(os.getenv("HYBRID_RRF_TOP_K", 10))        # RRF 融合后候选数（基础检索的返回数）
    HYBRID_RERANK_TOP_K = int(os.getenv("HYBRID_RERANK_TOP_K", 5))   # 精排后取数
    RRF_K = int(os.getenv("RRF_K", 60))                              # RRF 融合常数
    # 查询改写配置
    QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"
    QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "") or None  # 空则复用 CHAT_MODEL
    QUERY_REWRITE_TEMPERATURE = float(os.getenv("QUERY_REWRITE_TEMPERATURE", 0.0))
config = Config()   # 这个就是其他模块导入的名字