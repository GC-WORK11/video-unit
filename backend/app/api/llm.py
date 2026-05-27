"""LLM Settings API - OpenRouter Compatible Provider Setup."""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.llm import LLMClient, LLMConfig, PROVIDER_ENDPOINTS

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/llm", tags=["llm"])


class LLMConfigUpdate(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMStatus(BaseModel):
    provider: str
    model: str
    base_url: str
    configured: bool
    available_providers: list[str]
    models_for_provider: list[str]


# In-memory config (could persist to file/db)
_current_config = LLMConfig.from_env()


@router.get("/status", response_model=LLMStatus)
async def get_llm_status():
    """Get current LLM configuration status."""
    return LLMStatus(
        provider=_current_config.provider,
        model=_current_config.model,
        base_url=_current_config.base_url,
        configured=bool(_current_config.api_key),
        available_providers=list(PROVIDER_ENDPOINTS.keys()),
        models_for_provider=PROVIDER_ENDPOINTS.get(_current_config.provider, {}).get("models", []),
    )


@router.post("/config", response_model=LLMStatus)
async def update_llm_config(config: LLMConfigUpdate):
    """Update LLM configuration."""
    global _current_config
    
    if config.provider not in PROVIDER_ENDPOINTS:
        raise HTTPException(400, f"Unknown provider: {config.provider}")
    
    # Build new config
    base_url = config.base_url or PROVIDER_ENDPOINTS[config.provider]["base_url"]
    
    _current_config = LLMConfig(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key or _current_config.api_key,
        base_url=base_url,
    )
    
    log.info(f"LLM config updated: provider={config.provider}, model={config.model}")
    
    return LLMStatus(
        provider=_current_config.provider,
        model=_current_config.model,
        base_url=_current_config.base_url,
        configured=bool(_current_config.api_key),
        available_providers=list(PROVIDER_ENDPOINTS.keys()),
        models_for_provider=PROVIDER_ENDPOINTS.get(_current_config.provider, {}).get("models", []),
    )


@router.post("/test")
async def test_llm_connection():
    """Test LLM connection with current config."""
    global _current_config
    
    if not _current_config.api_key:
        raise HTTPException(400, "No API key configured")
    
    try:
        client = LLMClient(_current_config)
        response = await client.chat(
            messages=[{"role": "user", "content": "Hi, respond with just 'OK'"}],
            max_tokens=10,
        )
        await client.close()
        return {"success": True, "response": response}
    except Exception as e:
        log.error(f"LLM test failed: {e}")
        raise HTTPException(500, f"Connection failed: {str(e)}")


@router.get("/providers")
async def get_providers():
    """Get available providers and their endpoints."""
    return {
        "providers": {
            name: {
                "base_url": info["base_url"],
                "models": info["models"],
            }
            for name, info in PROVIDER_ENDPOINTS.items()
        }
    }
