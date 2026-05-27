"""Knowledge Base API — query and manage AETHER's local knowledge brain."""
import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query

from app.knowledge import service as knowledge_service
from app.ollama import client as ollama_client

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/knowledge")


@router.get("/status")
async def knowledge_status():
    """Get knowledge base status and statistics."""
    return knowledge_service.get_knowledge_status()


@router.post("/initialize")
async def initialize_knowledge(background_tasks: BackgroundTasks):
    """Initialize the knowledge base with ArXiv papers.

    This runs in the background and fetches ~250 physics/engineering papers.
    """
    existing = knowledge_service.get_knowledge_status()
    if existing.get("knowledge_initialized"):
        return {"status": "already_initialized", **existing}

    async def _init():
        try:
            await knowledge_service.initialize_knowledge_base()
        except Exception as e:
            log.error(f"Background knowledge init failed: {e}")

    background_tasks.add_task(_init)
    return {"status": "initialization_started", "message": "Knowledge base is being populated in background"}


@router.post("/ingest")
async def ingest_arxiv_topic(
    query: str = Query(..., description="ArXiv search query"),
    category: str = Query("general", description="Knowledge category"),
    max_results: int = Query(10, ge=1, le=50),
):
    """Ingest ArXiv papers for a specific topic."""
    result = await knowledge_service.quick_ingest(query, category, max_results)
    return result


@router.get("/query")
async def query_knowledge(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    category: str | None = Query(None, description="Filter by category"),
):
    """Semantic search of the knowledge base."""
    chunks = knowledge_service.query_knowledge(q, top_k=top_k, category=category)
    return {
        "query": q,
        "results": chunks,
        "count": len(chunks),
    }


@router.post("/reason")
async def gemma4_reason(
    prompt: str,
    include_context: bool = True,
):
    """Use Gemma 4 (Ollama) for physics reasoning with knowledge base context.

    This is AETHER's "small LLM brain" — local reasoning grounded in the knowledge base.
    """
    context = ""
    if include_context:
        context = knowledge_service.format_knowledge_context(prompt, max_chunks=4)

    ollama_alive = await ollama_client.is_ollama_alive()
    models = await ollama_client.list_models() if ollama_alive else []

    if not ollama_alive:
        return {
            "status": "ollama_unavailable",
            "message": "Ollama is not running. Start it with: sudo systemctl start ollama",
            "ollama_alive": False,
        }

    try:
        response = await knowledge_service.gemma4_reason(prompt, context)
        return {
            "status": "success",
            "ollama_alive": True,
            "model": ollama_client.DEFAULT_MODEL,
            "response": response,
            "context_used": bool(context),
        }
    except Exception as e:
        log.error(f"Gemma4 reasoning failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "ollama_alive": ollama_alive,
        }


@router.get("/categories")
async def list_categories():
    """List available knowledge categories."""
    return {
        "categories": [
            {"id": "physics", "label": "Physics", "description": "Mechanics, dynamics, vibrations"},
            {"id": "mechanics", "label": "Mechanical Engineering", "description": "Belt drives, mechanisms, kinematics"},
            {"id": "control", "label": "Control Theory", "description": "MPC, PID, optimal control"},
            {"id": "vision", "label": "Computer Vision", "description": "SAM 2, tracking, depth estimation"},
            {"id": "robotics", "label": "Robotics", "description": "Kinematics, dynamics, planning"},
            {"id": "materials", "label": "Materials Science", "description": "Friction, elasticity, properties"},
            {"id": "engineering", "label": "Engineering", "description": "Digital twins, simulation, design"},
            {"id": "knowledge", "label": "Knowledge Systems", "description": "Scene graphs, knowledge representation"},
        ]
    }


@router.get("/ollama/status")
async def ollama_status():
    """Check Ollama service status."""
    alive = await ollama_client.is_ollama_alive()
    models = await ollama_client.list_models() if alive else []
    return {
        "alive": alive,
        "models": models,
        "default_model": ollama_client.DEFAULT_MODEL if alive else None,
        "endpoint": ollama_client.OLLAMA_BASE,
    }


@router.post("/ollama/pull")
async def pull_model(model: str = Query(...)):
    """Pull a model into Ollama (downloads weights)."""
    try:
        result = await ollama_client.pull_model(model)
        return {"status": "complete", "model": model, "result": result}
    except Exception as e:
        return {"status": "error", "model": model, "error": str(e)}
