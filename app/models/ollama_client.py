import sys
from pathlib import Path
# 将项目根目录（rag_system/）加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from typing import List, Generator, Union
from langchain_core.messages import HumanMessage, SystemMessage

# 导入 app 包内的 config 模块
from app.config import config

class OllamaClient:
    def __init__(self, chat_model=None, embed_model=None, temperature=None, max_tokens=None, num_ctx=None,
                 provider="ollama", base_url=None, api_key=None, request_timeout=None):
        self.chat_model = chat_model or config.CHAT_MODEL
        self.embed_model = embed_model or config.EMBED_MODEL
        self.temperature = temperature if temperature is not None else config.CHAT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.CHAT_MAX_TOKENS
        self.num_ctx = num_ctx or config.CHAT_NUM_CTX
        self.provider = provider
        self.request_timeout = request_timeout

        if provider == "openai":
            # OpenAI 兼容接口（百炼/DashScope、DeepSeek、Kimi、智谱、硅基流动等）
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.chat_model,
                base_url=base_url,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_timeout=self.request_timeout,
            )
        else:
            # 默认：本地 Ollama
            from langchain.chat_models import init_chat_model
            self.llm = init_chat_model(
                model=self.chat_model,
                model_provider="ollama",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                num_ctx=self.num_ctx,
                # base_url 可自定义，默认 http://localhost:11434
            )

        # 用 LangChain 初始化嵌入模型（延迟导入）。嵌入固定走本地 Ollama，
        # 因为换嵌入模型 = 向量维度变 = 需重建向量库。
        from langchain_ollama import OllamaEmbeddings
        self.embeddings = OllamaEmbeddings(
            model=self.embed_model,
            base_url=config.OLLAMA_BASE_URL,  # 如 "http://localhost:11434"
        )

    def chat(self, messages: List[dict]) -> str:
        # 将字典消息转为 LangChain 消息格式
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            # 其他角色可按需添加

        result = self.llm.invoke(lc_messages)
        return result.content

    def count_tokens(self, text: str) -> int:
        """简易估算token，规避ollama版本缺少count_tokens接口问题"""
        if not text:
            return 0
        return int(len(text) * 1.3)

    def chat_stream(self, messages: List[dict]) -> Generator[str, None, None]:
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))

        for chunk in self.llm.stream(lc_messages):
            if chunk.content:
                yield chunk.content

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        # embed_documents 接收字符串列表，返回向量列表
        return self.embeddings.embed_documents(texts)


_ollama_client_instance = None


def get_ollama_client(
    chat_model: str = None,
    embed_model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    num_ctx: int = None,
) -> OllamaClient:
    global _ollama_client_instance
    if _ollama_client_instance is None:
        _ollama_client_instance = OllamaClient(
            chat_model=chat_model,
            embed_model=embed_model,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
        )
    return _ollama_client_instance


_clients = {}  # 外部模型名 -> OllamaClient，按需缓存


def build_chat_client(model_name: str = None, temperature: float = None,
                      default_model: str = None, max_tokens: int = None,
                      request_timeout: float = None) -> OllamaClient:
    """按模型名构建云端聊天客户端。

    只接受已注册的外部（云端）模型名；未配置时抛错，不再 fallback 本地模型。
    延迟导入 registry 以避免与 app.models.__init__ 的循环依赖。
    """
    from app.models import registry

    spec = registry.get_model(model_name) if model_name else None
    if spec:
        return OllamaClient(
            chat_model=spec.get("model"),
            provider=spec.get("provider", "openai"),
            base_url=spec.get("base_url"),
            api_key=spec.get("api_key"),
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
    # 未配置云端模型：直接抛错，绝不用本地模型兜底
    raise ValueError("请先配置云端模型")


def get_chat_client(model_name: str = None) -> OllamaClient:
    """按模型名获取聊天客户端。

    - model_name 为空（前端未选/未配云端）→ 本地兜底模型（config.CHAT_MODEL）
    - model_name 是已注册的云端模型 → 云端客户端
    - model_name 未注册 → 抛错（无效模型名，不回退）
    """
    global _clients
    if not model_name:
        # 未配置/未选择云端模型：直接本地兜底
        return get_ollama_client()
    from app.models import registry
    if not registry.get_model(model_name):
        raise ValueError(f"未找到云端模型「{model_name}」，请先配置云端模型")
    if model_name not in _clients:
        _clients[model_name] = build_chat_client(model_name)
    return _clients[model_name]


if __name__ == "__main__":
    print("============普通对话测试===========")
    try:
        client = OllamaClient()
        reply = client.chat([
            {"role": "system", "content": "你是一个智能问答助手"},
            {"role": "user", "content": "介绍一下自己，说一下什么是RAG"},
        ])
        print(reply)
    except Exception as e:
        print(f"[chat]对话失败: {e}")
    print("============流式对话测试===========")
    try:
        client = OllamaClient()
        for chunk in client.chat_stream([
            {"role": "system", "content": "你是一个智能问答助手"},
            {"role": "user", "content": "介绍一下自己，说一下什么是RAG"},
        ]):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"[chat_stream]对话失败: {e}")
    # 3. 测试文本转向量
    print("\n=== 嵌入测试 ===")
    try:
        client = OllamaClient()
        vecs = client.embed("测试文本")
        print(f"向量维度: {len(vecs[0])}")  # qwen3-embedding:0.6b 通常是 768
    except Exception as e:
        print("失败:", e)