"""
AETHER Universal AI Client
==========================

Handles multiple AI providers (MiniMax, OpenRouter, Local Ollama, OpenAI)
using a unified interface.
"""

import logging
import os
from typing import List, Dict, Optional, Any, Literal
import httpx
import json

log = logging.getLogger(__name__)

# Provider Types
ProviderType = Literal["minimax", "openrouter", "openai", "ollama", "gemini"]

class UniversalAIClient:
    """
    Unified client for interacting with various AI model providers.
    """
    
    def __init__(
        self,
        provider: ProviderType = "minimax",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.base_url = base_url
        self.model = model
        
        # Default configurations
        self._setup_defaults()

    def _setup_defaults(self):
        """Set default URLs and models based on provider."""
        if self.provider == "minimax":
            self.base_url = self.base_url or "https://api.minimax.io/v1"
            self.model = self.model or "MiniMax-M2.7-highspeed"
        elif self.provider == "openrouter":
            self.base_url = self.base_url or "https://openrouter.ai/api/v1"
            self.model = self.model or "anthropic/claude-3.5-sonnet"
        elif self.provider == "openai":
            self.base_url = self.base_url or "https://api.openai.com/v1"
            self.model = self.model or "gpt-4o"
        elif self.provider == "ollama":
            self.base_url = self.base_url or "http://localhost:11434/api"
            self.model = self.model or "gemma2"
        elif self.provider == "gemini":
            self.base_url = self.base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
            self.model = self.model or "gemini-1.5-pro"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Execute a chat completion request across any provider."""

        # Local providers (Ollama, LM Studio) don't need API keys
        local_providers = {"ollama", "lmstudio"}
        if self.provider not in local_providers and not self.api_key:
            raise RuntimeError(
                f"No API key configured for provider '{self.provider}'. "
                f"Set MINIMAX_API_KEY (or AETHER_LLM_API_KEY) environment variable, "
                f"or use a local provider like 'ollama' or 'lmstudio'."
            )

        if self.provider == "ollama" and "/api" in self.base_url and "/v1" not in self.base_url:
            return await self._ollama_native_chat(messages, temperature, max_tokens)
            
        # Standard OpenAI-compatible flow
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Special headers for OpenRouter
            if self.provider == "openrouter":
                headers["HTTP-Referer"] = "https://aether-studio.io"
                headers["X-Title"] = "AETHER Studio"

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            try:
                url = f"{self.base_url.rstrip('/')}/chat/completions"
                resp = await client.post(url, headers=headers, json=payload)
                
                if resp.status_code != 200:
                    error_data = resp.json()
                    log.error(f"AI Provider {self.provider} failed: {error_data}")
                    raise RuntimeError(f"{self.provider.capitalize()} error: {json.dumps(error_data)}")
                
                data = resp.json()
                return data["choices"][0]["message"]["content"]
                
            except Exception as e:
                log.error(f"AI request failed for {self.provider}: {e}")
                raise

    async def _ollama_native_chat(self, messages, temperature, max_tokens):
        """Native Ollama API fallback if not using their OpenAI-compatible endpoint."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            url = f"{self.base_url.rstrip('/')}/chat"
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

async def get_ai_response(
    messages: List[Dict[str, str]],
    provider: ProviderType = "minimax",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Helper function for quick AI calls."""
    client = UniversalAIClient(provider=provider, api_key=api_key, model=model, base_url=base_url)
    return await client.chat_completion(messages)
