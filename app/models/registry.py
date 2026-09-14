"""外部模型注册表。

把用户添加的云端模型持久化到项目根目录的 models.json，
供 ollama_client 按模型名解析 provider / base_url / api_key。
"""
import json
import threading
from pathlib import Path
from typing import List, Optional, Dict

from app.config import config

# 存放位置：项目根目录 models.json（与 chroma_db、data 同级）
MODELS_FILE = Path(config.BASE_DIR) / "models.json"
_lock = threading.Lock()


def _load() -> List[Dict]:
    """读取全部外部模型；文件不存在或损坏时返回空列表。"""
    if not MODELS_FILE.exists():
        return []
    try:
        with open(MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return data
    return data.get("models", []) if isinstance(data, dict) else []


def _save(models: List[Dict]) -> None:
    MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)


def list_models() -> List[Dict]:
    with _lock:
        return _load()


def get_model(name: str) -> Optional[Dict]:
    """按名称查外部模型，找不到返回 None。"""
    for m in _load():
        if m.get("name") == name:
            return m
    return None


def add_model(spec: Dict) -> Dict:
    """添加或更新一个外部模型，返回写入后的 spec。"""
    with _lock:
        models = _load()
        name = spec.get("name")
        for i, m in enumerate(models):
            if m.get("name") == name:
                models[i] = spec
                _save(models)
                return spec
        models.append(spec)
        _save(models)
        return spec


def delete_model(name: str) -> bool:
    """按名称删除，返回是否真的删掉了。"""
    with _lock:
        models = _load()
        new_models = [m for m in models if m.get("name") != name]
        if len(new_models) == len(models):
            return False
        _save(new_models)
        return True
