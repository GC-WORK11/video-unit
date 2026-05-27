"""AETHER Hybrid Assistant — MiniMax orchestrates + Gemma 4 reasons + Knowledge Base grounds.

Three-tier reasoning:
  1. Knowledge Base  → relevant physics/engineering context (fast, local)
  2. Gemma 4 (Ollama) → local physics reasoning on that context (private, GPU)
  3. MiniMax (cloud)  → user-facing explanation + tool calls (fast, smart)

The "small LLM brain" is Gemma 4 + ChromaDB.
The "fast reasoning engine" is MiniMax.
Together = best of both worlds.
"""
import asyncio
import logging
import os
from typing import Literal, Optional

import httpx

from app.knowledge import service as knowledge_service
from app.core.ai_client import UniversalAIClient, ProviderType

log = logging.getLogger(__name__)

# Default settings from env (fallback)
DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", "minimax")
DEFAULT_MODEL = os.getenv("AI_MODEL", "MiniMax-M2.7-highspeed")
DEFAULT_KEY = os.getenv("MINIMAX_API_KEY", "")

# Concurrent fetch of knowledge
KB_TIMEOUT = 8.0  # seconds

async def _fetch_knowledge_context(query: str, top_k: int = 4) -> str:
    """Fetch knowledge base context."""
    try:
        ctx = await asyncio.wait_for(
            asyncio.to_thread(knowledge_service.format_knowledge_context, query, top_k),
            timeout=KB_TIMEOUT,
        )
        return ctx
    except asyncio.TimeoutError:
        log.warning("Knowledge base query timed out")
        return ""
    except Exception as e:
        log.warning(f"Knowledge base query failed: {e}")
        return ""

async def hybrid_chat(
    user_message: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    session_context: str = "",
    conversation_history: list[dict] | None = None,
) -> dict:
    """Main entry point — Universal cloud-first chat with local knowledge-grounding.
    
    This version removes the mandatory local Ollama step for speed,
    instead grounding the selected cloud model in local physics data.
    """
    history = conversation_history or []
    
    # 1. Fetch Knowledge Base context (Fast & Local)
    kb_context = await _fetch_knowledge_context(user_message, top_k=5)
    kb_chunks = knowledge_service.query_knowledge(user_message, top_k=3)
    
    # 2. Setup Universal Client
    target_provider = (provider or DEFAULT_PROVIDER).lower()
    client = UniversalAIClient(
        provider=target_provider, # type: ignore
        api_key=api_key or DEFAULT_KEY,
        model=model or DEFAULT_MODEL,
        base_url=base_url
    )
    
    # 3. Build System Prompt
    system_msg = f"""You are AETHER Studio's assistant — a physics and mechanical engineering expert.
    
Grounded in local session data: {session_context}

CRITICAL RULES:
1. Every time you state a physical result, include the Confidence and Basis.
2. Distinguish between observed behavior (video) vs predicted (simulation) vs theoretical (knowledge base).
3. Use the following local knowledge base references to ensure accuracy.

---
LOCAL KNOWLEDGE BASE REFERENCE:
{kb_context or "No specific local knowledge found."}
---
"""

    messages = [
        {"role": "system", "content": system_msg},
        *history,
        {"role": "user", "content": user_message},
    ]
    
    # 4. Execute Cloud-First Reasoning
    try:
        response = await client.chat_completion(messages)
    except Exception as e:
        log.error(f"Chat completion failed: {e}")
        response = f"I encountered an error while connecting to {target_provider}: {str(e)}"
        if kb_context:
            response += f"\n\nHowever, I found this in the local knowledge base:\n{kb_context[:500]}..."

    return {
        "response": response,
        "knowledge_used": bool(kb_context),
        "kb_chunks": kb_chunks,
        "provider": target_provider,
        "model": client.model,
        "status": "success" if not "error" in response.lower() else "partial_failure"
    }
