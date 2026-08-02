import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

class Config:
    CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-r1:1.5b")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.1))
    CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", 2048))
    CHAT_NUM_CTX = int(os.getenv("CHAT_NUM_CTX", 4096))
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
config = Config()   # 这个就是其他模块导入的名字
