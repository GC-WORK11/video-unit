"""ChromaDB vector database — embedded local vector store for AETHER knowledge base."""
import asyncio
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
import httpx
import torch

from app.core.config import DATA_DIR

log = logging.getLogger(__name__)

# ChromaDB persistent storage
CHROMA_PATH = DATA_DIR / "knowledge_chroma"
EMBEDDING_COLLECTION = "aether_knowledge"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, CPU fast, 22MB

_client = None
_embedder = None


def get_chroma_client():
    """Get or create ChromaDB client (persistent to disk)."""
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
        )
        log.info(f"ChromaDB initialized at {CHROMA_PATH}")
    return _client


def get_embedder():
    """Lazy-load sentence-transformers embedder (CPU-only to avoid GPU OOM with Ollama)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        # Force CPU — Ollama holds the GPU for Gemma 4
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        log.info(f"Embedder loaded: {EMBEDDING_MODEL} (CPU)")
    return _embedder


def get_or_create_collection():
    """Get the AETHER knowledge collection, creating if needed."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=EMBEDDING_COLLECTION)
        log.info(f"Collection '{EMBEDDING_COLLECTION}' loaded: {collection.count()} chunks")
    except Exception:
        collection = client.create_collection(
            name=EMBEDDING_COLLECTION,
            metadata={"description": "AETHER physics & engineering knowledge base"},
        )
        log.info(f"Created collection '{EMBEDDING_COLLECTION}'")
    return collection


def add_documents(chunks: list[dict]):
    """Add document chunks to the knowledge base.

    Each chunk: {id, text, title, source, category, url}
    """
    collection = get_or_create_collection()
    embedder = get_embedder()

    texts = [c["text"] for c in chunks]
    try:
        embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
    except RuntimeError as e:
        if "CUDA" in str(e):
            log.warning(f"GPU OOM in embedder ({e}), falling back to CPU")
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            embeddings = embedder.encode(texts, show_progress_bar=False, device="cpu", convert_to_numpy=True).tolist()
        else:
            raise
    if not embeddings and texts:
        # Ollama embed fallback
        import asyncio
        embeddings = asyncio.get_event_loop().run_until_complete(
            _ollama_embed_batch(texts)
        )

    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "title": c.get("title", ""),
            "source": c.get("source", ""),
            "category": c.get("category", ""),
            "url": c.get("url", ""),
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    log.info(f"Added {len(chunks)} chunks to knowledge base")


async def _ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed via Ollama nomic-embed-text (runs on Ollama's GPU)."""
    try:
        client = httpx.AsyncClient(base_url="http://localhost:11434", timeout=60.0)
        embeddings = []
        for text in texts:
            resp = await client.post("/api/embeddings", json={"model": "nomic-embed-text", "prompt": text})
            resp.raise_for_status()
            embeddings.append(resp.json().get("embedding", []))
        await client.aclose()
        return embeddings
    except Exception as e:
        log.warning(f"Ollama embed fallback failed: {e}")
        return [[0.0] * 384 for _ in texts]


def query(query_text: str, top_k: int = 8, category: str | None = None) -> list[dict]:
    """Query the knowledge base.

    Returns top_k most relevant chunks.
    """
    collection = get_or_create_collection()
    embedder = get_embedder()

    query_embedding = embedder.encode([query_text], show_progress_bar=False).tolist()[0]

    where_filter = {"category": category} if category else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    chunks = []
    if results and results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "text": doc,
                "title": results["metadatas"][0][i].get("title", ""),
                "source": results["metadatas"][0][i].get("source", ""),
                "category": results["metadatas"][0][i].get("category", ""),
                "url": results["metadatas"][0][i].get("url", ""),
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
    return chunks


def get_stats() -> dict:
    """Get knowledge base statistics."""
    try:
        collection = get_or_create_collection()
        count = collection.count()
    except Exception:
        count = 0

    return {
        "chunk_count": count,
        "embedding_model": EMBEDDING_MODEL,
        "storage_path": str(CHROMA_PATH),
    }
