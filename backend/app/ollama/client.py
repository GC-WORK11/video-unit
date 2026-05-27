"""Ollama API client — local Gemma 4 for AETHER knowledge reasoning."""
import logging
from typing import Generator

import httpx

from app.core.config import DATA_DIR

log = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "nomic-embed-text"  # Fast embedding model for Ollama

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=OLLAMA_BASE,
            timeout=httpx.Timeout(180.0, connect=10.0),
            follow_redirects=True,
        )
    return _client


async def is_ollama_alive() -> bool:
    """Check if Ollama is running."""
    try:
        client = get_client()
        resp = await client.get("/api/tags")
        return resp.status_code == 200
    except Exception:
        return False


async def list_models() -> list[dict]:
    """List available Ollama models."""
    try:
        client = get_client()
        resp = await client.get("/api/tags")
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        log.warning(f"Ollama list_models failed: {e}")
        return []


async def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """Generate text from Ollama Gemma 4.

    Returns the full response text (or generator if streaming).
    """
    client = get_client()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["</answer>", "STOP"],
        },
    }

    if system:
        payload["system"] = system

    try:
        async with client.stream("POST", "/api/generate", json=payload) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama returned {resp.status_code}: {await resp.text()}")

            if stream:
                async def _stream():
                    async for line in resp.aiter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                yield token
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                return _stream()
            else:
                import json
                data = await resp.json()
                return data.get("response", "")

    except httpx.TimeoutException:
        raise TimeoutError(f"Ollama generation timed out after 180s for model {model}. Gemma 4 takes time to load on first request — retry.")
    except Exception as e:
        raise RuntimeError(f"Ollama generation failed: {e}")


async def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Chat completion via Ollama /api/chat endpoint.

    messages: [{"role": "user"|"assistant", "content": "..."}]
    """
    client = get_client()

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = await client.post("/api/chat", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama chat returned {resp.status_code}: {resp.text()}")
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except httpx.TimeoutException:
        raise TimeoutError(f"Ollama chat timed out after 180s")
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}")


async def get_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Get text embeddings from Ollama's embedding endpoint.

    Falls back to sentence-transformers if Ollama embedding fails.
    """
    try:
        client = get_client()
        embeddings = []
        for text in texts:
            resp = await client.post("/api/embeddings", json={"model": model, "prompt": text})
            if resp.status_code == 200:
                data = resp.json()
                embeddings.append(data.get("embedding", []))
            else:
                raise RuntimeError(f"Embedding failed: {resp.status_code}")
        return embeddings
    except Exception as e:
        log.warning(f"Ollama embeddings failed ({e}), falling back to sentence-transformers")
        from sentence_transformers import SentenceTransformer
        model_st = SentenceTransformer("all-MiniLM-L6-v2")
        return model_st.encode(texts, show_progress_bar=False).tolist()


async def pull_model(model: str) -> dict:
    """Trigger Ollama to pull/download a model."""
    client = get_client()
    async with client.stream("POST", "/api/pull", json={"name": model}) as resp:
        status = {}
        async for line in resp.aiter_lines():
            if line:
                import json
                try:
                    data = json.loads(line)
                    status = data
                    if data.get("status"):
                        log.info(f"Ollama pull: {data['status']}")
                except json.JSONDecodeError:
                    continue
        return status
