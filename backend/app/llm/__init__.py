"""
AETHER LLM Client - OpenRouter Compatible
==========================================

Supports any OpenRouter-compatible API endpoint:
- MiniMax
- OpenAI
- Anthropic
- Local models via LM Studio, etc.
"""

import os
import logging
from typing import Generator, Optional
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM Provider Configuration."""
    provider: str = "minimax"  # minimax, openai, anthropic, openrouter
    model: str = "MiniMax-Embedding"
    api_key: Optional[str] = None
    base_url: str = "https://api.minimax.chat/v1"
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        # AETHER_LLM_API_KEY is the primary key, but fall back to MINIMAX_API_KEY for backwards compatibility
        api_key = os.getenv("AETHER_LLM_API_KEY") or os.getenv("MINIMAX_API_KEY", "")
        return cls(
            provider=os.getenv("AETHER_LLM_PROVIDER", "minimax"),
            model=os.getenv("AETHER_LLM_MODEL", "MiniMax-Embedding"),
            api_key=api_key if api_key else None,
            base_url=os.getenv("AETHER_LLM_BASE_URL", "https://api.minimax.chat/v1"),
        )


# Provider endpoints
PROVIDER_ENDPOINTS = {
    "minimax": {
        "base_url": "https://api.minimax.chat/v1",
        "chat": "/text/chatcompletion_v2",
        "models": [],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "chat": "/chat/completions",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "chat": "/chat/completions",
        "models": [],  # Fetched dynamically
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "chat": "/chat/completions",
        "models": [],
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "chat": "/chat/completions",
        "models": [],
    },
}


class LLMClient:
    """
    OpenRouter-compatible LLM client.
    
    Works with MiniMax, OpenAI, Anthropic, LM Studio, Ollama, etc.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(120.0, connect=30.0),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Send chat completion request.

        messages: [{"role": "user"|"assistant"|"system", "content": "..."}]
        """
        if not self.config.api_key:
            raise RuntimeError(
                f"No API key configured for LLM provider '{self.config.provider}'. "
                f"Set AETHER_LLM_API_KEY or MINIMAX_API_KEY environment variable. "
                f"Alternatively, use the /api/llm/config endpoint to set the API key."
            )

        model = model or self.config.model

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # MiniMax specific
        if "minimax" in self.config.base_url:
            payload["model"] = model
            endpoint = "/text/chatcompletion_v2"
        else:
            endpoint = "/chat/completions"
        
        try:
            if stream:
                return self._stream_chat(endpoint, payload)
            else:
                resp = await self.client.post(endpoint, json=payload)
                if resp.status_code != 200:
                    log.error(f"LLM error: {resp.status_code} - {resp.text}")
                    raise RuntimeError(f"LLM request failed: {resp.status_code}")
                
                data = resp.json()
                
                # Handle different response formats
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                elif "output" in data:  # MiniMax format
                    return data["output"]
                else:
                    return str(data)
                    
        except httpx.TimeoutException:
            raise TimeoutError(f"LLM request timed out after 120s")
        except Exception as e:
            log.error(f"LLM error: {e}")
            raise
    
    async def _stream_chat(self, endpoint: str, payload: dict) -> Generator[str, None, None]:
        """Stream chat completion."""
        async with self.client.stream("POST", endpoint, json=payload) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"LLM stream failed: {resp.status_code}")
            
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    import json
                    try:
                        data = json.loads(data_str)
                        if "choices" in data:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        elif "output" in data:
                            yield data["output"]
                    except json.JSONDecodeError:
                        continue


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def close_llm_client():
    global _llm_client
    if _llm_client:
        await _llm_client.close()
        _llm_client = None
