"""Chat / Assistant API — hybrid MiniMax + Gemma 4 + Knowledge Base."""
import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Query

from app.assistant import hybrid_chat

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.post("/chat")
async def chat(
    message: str = Query(..., description="User message"),
    provider: str | None = Query(None, description="AI provider (minimax, openrouter, etc.)"),
    model: str | None = Query(None, description="Model name"),
    api_key: str | None = Query(None, description="API Key"),
    base_url: str | None = Query(None, description="Custom base URL"),
    session_id: str | None = Query(None, description="Optional session for context"),
):
    """Universal chat — Cloud-first reasoning grounded in local Knowledge Base.
    """
    target_provider = provider or "minimax"
    try:
        result = await asyncio.wait_for(
            hybrid_chat(
                user_message=message,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                session_context=f"session:{session_id}" if session_id else "",
            ),
            timeout=120.0,
        )
        return result
    except asyncio.TimeoutError:
        return {
            "response": "The reasoning engine timed out. Try Gemma 4 mode for local (faster) reasoning, or MiniMax mode for cloud speed.",
            "knowledge_used": False,
            "gemma_used": False,
            "kb_chunks": [],
            "gemma_reasoning": "",
            "mode": target_provider,
            "error": "timeout",
        }
    except Exception as e:
        log.error(f"Chat failed: {e}")
        return {
            "response": f"I encountered an error: {e}",
            "knowledge_used": False,
            "gemma_used": False,
            "kb_chunks": [],
            "gemma_reasoning": "",
            "mode": target_provider,
            "error": str(e),
        }


@router.get("/chat/status")
async def chat_status():
    """Check which chat backends are available."""
    from app.ollama import client as ollama_client
    from app.knowledge import service as knowledge_service

    ollama_alive = await ollama_client.is_ollama_alive()
    kb_status = knowledge_service.get_knowledge_status()

    import os
    minimax_key = bool(os.getenv("MINIMAX_API_KEY", ""))

    return {
        "minimax": {"available": minimax_key, "model": os.getenv("ANTHROPIC_MODEL", "MiniMax-M2.7-highspeed")},
        "gemma4": {"available": ollama_alive, "model": ollama_client.DEFAULT_MODEL},
        "knowledge_base": {"initialized": kb_status.get("knowledge_initialized", False), "chunks": kb_status.get("chunk_count", 0)},
        "mode_recommendation": "hybrid" if (ollama_alive and minimax_key) else "minimax" if minimax_key else "gemma4",
    }
