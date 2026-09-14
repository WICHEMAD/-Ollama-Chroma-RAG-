"""模型列表路由：用户添加的外部（云端）模型。本地模型已固定，不再切换。"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models import registry

router = APIRouter(prefix="/models", tags=["models"])


class ExternalModelRequest(BaseModel):
    """添加外部模型的请求体。provider 目前只支持 openai（OpenAI 兼容）。"""
    name: str
    model: str
    provider: str = "openai"
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@router.get("")
async def list_models():
    """返回外部模型列表（本地模型已固定，不在切换列表内）。

    只返回前端需要的展示字段（name / model），
    绝不返回 api_key / base_url / provider 等敏感信息。
    """
    safe = []
    for m in registry.list_models():
        safe.append({
            "name": m.get("name"),
            "model": m.get("model"),
            "source": "external",
        })
    return {"models": safe}


@router.post("")
async def add_model(req: ExternalModelRequest):
    """添加（或按名称覆盖）一个外部模型。"""
    if not req.name.strip() or not req.model.strip():
        raise HTTPException(status_code=400, detail="名称和模型 ID 不能为空")
    spec = {
        "name": req.name.strip(),
        "model": req.model.strip(),
        "provider": req.provider,
        "base_url": req.base_url,
        "api_key": req.api_key,
    }
    registry.add_model(spec)
    return {"ok": True, "model": spec}


@router.delete("/{name}")
async def delete_model(name: str):
    """删除一个外部模型（只影响外部模型）。"""
    if not registry.delete_model(name):
        raise HTTPException(status_code=404, detail="未找到该外部模型")
    return {"ok": True}
