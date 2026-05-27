"""AETHER Knowledge Service — orchestrates the local knowledge brain.

This is the "small LLM brain" inside AETHER.
It manages:
- ChromaDB vector store (knowledge base)
- ArXiv paper ingestion (physics, engineering, robotics)
- Ollama/Gemma 4 for local reasoning
- Hybrid search: semantic vector + keyword
"""
import asyncio
import hashlib
import logging
import time
from typing import Annotated

from fastapi import BackgroundTasks

from app.core.config import DATA_DIR
from app.knowledge import chromadb, arxiv_fetcher
from app.ollama import client as ollama_client

log = logging.getLogger(__name__)

# Persistent state
_ingestion_status: dict = {
    "status": "idle",  # idle | running | complete | error
    "papers_fetched": 0,
    "chunks_added": 0,
    "last_ingest": None,
    "error": None,
}
_knowledge_initialized = False


def seed_comprehensive_physics_kb(progress_callback=None) -> dict:
    """Seed the knowledge base with comprehensive physics formulas and data.
    
    Loads 500+ chunks covering:
    - Foundational formulas (Newton, energy, etc.)
    - Vibration dynamics (damping, resonance, etc.)
    - Material properties (steel, aluminum, titanium, etc.)
    - Mechanism templates (suspension, robots, drones, etc.)
    - Motors and actuators
    - Sensors and transducers
    - Control theory
    - Tolerances and standards
    - Failure modes
    - Case studies
    """
    from app.knowledge.physics_kb import ALL_CHUNKS
    
    try:
        if progress_callback:
            progress_callback(f"Seeding comprehensive physics KB ({len(ALL_CHUNKS)} chunks)...")
        
        # Format chunks for ChromaDB
        docs = []
        for chunk in ALL_CHUNKS:
            docs.append({
                "title": chunk["title"],
                "text": chunk["content"],
                "source": f"AETHER Physics KB - {chunk.get('category', 'general')}",
                "category": chunk.get("category", "general"),
                "tags": ", ".join(chunk.get("tags", [])),
                "url": "",
            })
        
        # Add to ChromaDB
        chromadb.add_documents(docs)
        
        return {
            "chunks_added": len(docs),
            "status": "complete",
        }
    except Exception as e:
        return {
            "chunks_added": 0,
            "status": "error",
            "error": str(e),
        }


def get_knowledge_status() -> dict:
    """Get current knowledge base status."""
    try:
        stats = chromadb.get_stats()
    except Exception:
        stats = {"chunk_count": 0, "embedding_model": "unknown", "storage_path": ""}

    return {
        **_ingestion_status,
        **stats,
        "knowledge_initialized": _knowledge_initialized,
    }


async def initialize_knowledge_base(progress_callback=None) -> dict:
    """Initialize AETHER's knowledge base with ArXiv papers.

    This runs on first launch. Fetches relevant physics/engineering papers
    and embeds them into ChromaDB.
    """
    global _knowledge_initialized, _ingestion_status

    if _knowledge_initialized:
        log.info("Knowledge base already initialized")
        return get_knowledge_status()

    _ingestion_status = {
        "status": "running",
        "papers_fetched": 0,
        "chunks_added": 0,
        "last_ingest": None,
        "error": None,
    }

    try:
        # Fetch ArXiv papers
        log.info("Starting ArXiv knowledge ingestion...")
        if progress_callback:
            progress_callback("Fetching ArXiv papers...")

        papers = arxiv_fetcher.fetch_arxiv_papers(arxiv_fetcher.ARXIV_QUERIES)
        _ingestion_status["papers_fetched"] = len(papers)

        if progress_callback:
            progress_callback(f"Processing {len(papers)} papers...")

        # Add foundational textbooks manually
        log.info("Adding textbook knowledge...")
        if progress_callback:
            progress_callback("Adding physics fundamentals...")

        textbook_chunks = _get_textbook_chunks()
        if textbook_chunks:
            chromadb.add_documents(textbook_chunks)
            _ingestion_status["chunks_added"] = len(textbook_chunks)

        # Seed comprehensive physics KB
        log.info("Seeding comprehensive physics knowledge base...")
        if progress_callback:
            progress_callback("Seeding comprehensive physics KB...")
        
        kb_result = seed_comprehensive_physics_kb(progress_callback)
        _ingestion_status["chunks_added"] += kb_result.get("chunks_added", 0)

        # Ingest ArXiv papers
        log.info(f"Ingesting {len(papers)} ArXiv papers...")
        if progress_callback:
            progress_callback(f"Ingesting {len(papers)} ArXiv papers...")

        chunks = arxiv_fetcher.ingest_papers(papers)
        _ingestion_status["chunks_added"] += len(chunks)

        _knowledge_initialized = True
        _ingestion_status["status"] = "complete"
        _ingestion_status["last_ingest"] = time.time()

        log.info(f"Knowledge base initialized: {len(chunks)} chunks from {len(papers)} papers")

        return get_knowledge_status()

    except Exception as e:
        log.error(f"Knowledge base initialization failed: {e}")
        _ingestion_status["status"] = "error"
        _ingestion_status["error"] = str(e)
        raise


def _get_textbook_chunks() -> list[dict]:
    """Hardcoded physics fundamentals every AETHER instance knows.

    These are high-value, compact knowledge chunks covering core physics.
    """
    fundamentals = [
        {
            "title": "Newton's Second Law — Force and Acceleration",
            "text": "Newton's Second Law states that F = ma, where F is the net force acting on a body, m is its mass, and a is the acceleration produced. For a belt drive system, the tension difference between the tight side and slack side of a belt creates a net force accelerating the carriage. This is the foundation for all mechanical dynamic analysis.",
            "source": "Physics Fundamentals",
            "category": "physics",
            "url": "",
        },
        {
            "title": "Belt Tension — Tight and Slack Side Analysis",
            "text": "In a belt drive system, the belt has a tight side (high tension Ft) and a slack side (low tension Fs). The difference Ft - Fs = F_net drives the load. The ratio Ft/Fs is determined by the capstan equation: Ft = Fs * exp(μθ), where μ is the coefficient of friction and θ is the wrap angle in radians. Increasing belt tension increases both Ft and Fs proportionally, reducing slip but increasing stress on bearings.",
            "source": "Mechanical Engineering Fundamentals",
            "category": "mechanics",
            "url": "",
        },
        {
            "title": "Friction — Coulomb Model",
            "text": "Coulomb friction states that the frictional force F_f = μN, where μ is the coefficient of friction and N is the normal force. In belt drives, friction between the belt and pulley surfaces determines how much torque can be transmitted without slip. Kinetic friction differs from static friction: static friction is higher, preventing initial slip, while kinetic friction governs continuous sliding.",
            "source": "Mechanical Engineering Fundamentals",
            "category": "materials",
            "url": "",
        },
        {
            "title": "Vibration Analysis — Natural Frequency",
            "text": "Every mechanical system has natural frequencies at which it vibrates most easily. For a mass-spring system, ω_n = sqrt(k/m), where k is the stiffness and m is the mass. Belt drive systems can exhibit axial vibration at frequencies determined by belt tension, carriage mass, and belt stiffness. Resonance occurs when excitation frequency matches natural frequency, causing large amplitude oscillations.",
            "source": "Vibration Theory",
            "category": "physics",
            "url": "",
        },
        {
            "title": "Lagrangian Mechanics — Generalized Coordinates",
            "text": "The Lagrangian L = T - V (kinetic energy minus potential energy) fully describes a mechanical system's dynamics. Using Euler-Lagrange equations: d/dt(∂L/∂q_dot) - ∂L/∂q = 0, where q is a generalized coordinate (like carriage position or belt angle). This approach automatically accounts for constraints and is preferred for complex mechanisms like gantry robots.",
            "source": "Analytical Mechanics",
            "category": "physics",
            "url": "",
        },
        {
            "title": "Damping — Viscous and Structural",
            "text": "Damping dissipates mechanical energy as heat. Viscous damping: F_d = -c*v, where c is the damping coefficient and v is velocity. Structural damping (hysteretic): energy loss per cycle is proportional to strain amplitude. In belt drives, damping comes from belt material hysteresis, air resistance, and bearing friction. Increasing damping reduces vibration amplitude but doesn't change natural frequency.",
            "source": "Vibration Theory",
            "category": "physics",
            "url": "",
        },
        {
            "title": "Model Predictive Control (MPC) — Overview",
            "text": "MPC solves a finite-horizon optimal control problem at each timestep. At time t, it: (1) measures current state x(t), (2) solves min_u Σ(k=0..N) [||x(k|t)-x_ref||² + ρ||u(k)||²] subject to constraints, (3) applies u(0|t), (4) repeats at t+1. For belt drives, MPC can anticipate tension changes and pre-compensate for disturbances, providing smoother tracking than PID.",
            "source": "Control Theory",
            "category": "control",
            "url": "",
        },
        {
            "title": "PID Control — Proportional-Integral-Derivative",
            "text": "PID control: u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de/dt, where e(t) is the error between setpoint and measured position. Kp provides immediate response, Ki eliminates steady-state error, and Kd damps oscillations. For a gantry/carriage system: Kp affects response speed, Ki removes position drift from friction, and Kd suppresses vibration from belt elasticity.",
            "source": "Control Theory",
            "category": "control",
            "url": "",
        },
        {
            "title": "Gaussian Splatting — 3D Reconstruction",
            "text": "Gaussian splatting represents a 3D scene as a set of colored ellipsoids (Gaussians). Each gaussian has position (mean), covariance (shape/size), and color/opacity. Rendering uses differentiable splatting: project 3D Gaussians to 2D, alpha-blend them. Training optimizes Gaussian parameters to match input images. Unlike NeRF which raysamples, splatting rasterizes — making it 10-100x faster for real-time applications.",
            "source": "Computer Vision",
            "category": "vision",
            "url": "",
        },
        {
            "title": "SAM 2 — Segment Anything Model 2",
            "text": "SAM 2 (Meta) extends SAM with video understanding. It uses a memory attention mechanism to propagate mask predictions across video frames, handling occlusions through its Predictor-Corrector loop. For object tracking, SAM 2 first segments the object in one frame (prompt), then propagates the mask forward and backward in time. Key innovation: lightweight Hiera backbone + hierarchical memory for long videos.",
            "source": "Computer Vision",
            "category": "vision",
            "url": "",
        },
        {
            "title": "Point Tracking — CoTracker3",
            "text": "CoTracker3 (Meta) tracks keypoints across video frames, handling long occlusions via its Transformer-based update mechanism. It maintains a belief state about track positions and updates it as new observations arrive. Tracks are initialized from a grid or user clicks. Key innovation: it can track through full occlusions by predicting where a point should be based on motion context, then correcting when the point reappears.",
            "source": "Computer Vision",
            "category": "vision",
            "url": "",
        },
        {
            "title": "Kinematics — Forward and Inverse",
            "text": "Forward kinematics: given joint angles θ, compute end-effector position x = f(θ). For a gantry with prismatic joints: x = x_base + L*cos(θ1) + dx*cos(θ1+θ2). Inverse kinematics: given desired position x, solve for joint angles θ = f^{-1}(x). This is needed for motion planning — we know where we want to go, but need to find what angles to command.",
            "source": "Robotics",
            "category": "robotics",
            "url": "",
        },
        {
            "title": "Scene Graphs — Object Relationship Representation",
            "text": "A scene graph represents a scene as a directed graph: nodes are objects, edges are relationships (touches, supports, connected-to, parent-of). Each node has properties (position, type, physics params). Each edge has a joint type (fixed, revolute, prismatic, belt). Scene graphs enable physics simulation by providing which objects interact and how. ROCG-PA is a specific scene graph schema for physical mechanisms.",
            "source": "Knowledge Representation",
            "category": "knowledge",
            "url": "",
        },
        {
            "title": "Digital Twin — Physics-Grounded Simulation",
            "text": "A digital twin is a live-updating virtual replica of a physical system. Unlike CAD models (which are static geometry) or 3D reconstructions (which are visual), a digital twin is physics-grounded: it has real physical parameters (mass, friction, stiffness) and simulates forward in time. The key property is fidelity: how closely does the simulation match reality? Measured by trajectory error between observed and simulated motion.",
            "source": "Engineering",
            "category": "engineering",
            "url": "",
        },
        {
            "title": "Error Metrics — Trajectory Error and RMS Error",
            "text": "Trajectory error measures how far the simulated path deviates from the observed real-world path. RMS Error = sqrt(mean((x_sim - x_obs)²)) across all timesteps. For a belt drive, trajectory error captures how well the physics model predicts carriage position over time. Tracking error is instantaneous position deviation, while accumulated error integrates over time and reflects drift from systematic bias.",
            "source": "Engineering",
            "category": "physics",
            "url": "",
        },
    ]

    chunks = []
    for item in fundamentals:
        chunk_id = hashlib.sha256(item["title"].encode()).hexdigest()[:16]
        chunks.append({**item, "id": chunk_id})
    return chunks


def query_knowledge(query_text: str, top_k: int = 5, category: str | None = None) -> list[dict]:
    """Query the knowledge base. Returns top_k relevant chunks."""
    try:
        return chromadb.query(query_text, top_k=top_k, category=category)
    except Exception as e:
        log.warning(f"Knowledge query failed: {e}")
        return []


def format_knowledge_context(query: str, max_chunks: int = 5) -> str:
    """Format knowledge base results as a context string for LLM prompts."""
    chunks = query_knowledge(query, top_k=max_chunks)
    if not chunks:
        return ""

    context_parts = ["[KNOWLEDGE BASE REFERENCE]"]
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"\n--- Source {i} ---")
        context_parts.append(f"Title: {chunk.get('title', 'Unknown')}")
        context_parts.append(f"Category: {chunk.get('category', 'general')}")
        if chunk.get("source"):
            context_parts.append(f"Source: {chunk['source']}")
        context_parts.append(f"Content: {chunk['text']}")

    context_parts.append("\n[END KNOWLEDGE BASE REFERENCE]")
    return "\n".join(context_parts)


async def gemma4_reason(
    prompt: str,
    context: str = "",
    system: str | None = None,
) -> str:
    """Use Gemma 4 (via Ollama) for local physics reasoning.

    If Ollama is slow/unavailable, falls back to returning the context.
    """
    if not context:
        context = "No relevant knowledge base entries found."

    system_prompt = system or (
        "You are AETHER's local physics reasoning engine. You are a world-class physicist and mechanical engineer. "
        "You reason step-by-step about mechanical systems, physics, control theory, and engineering. "
        "You are concise, precise, and cite specific formulas and parameters. "
        f"Here is relevant knowledge from the AETHER knowledge base:\n{context}\n\n"
        "Use this knowledge to answer the user's question. Be specific — include equations, values, and physical reasoning."
    )

    try:
        is_alive = await ollama_client.is_ollama_alive()
        if not is_alive:
            return f"[Gemma 4 unavailable — Ollama not running. Knowledge context: {context[:500]}]"

        response = await ollama_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=ollama_client.DEFAULT_MODEL,
            temperature=0.3,
            max_tokens=1024,
        )
        return response

    except TimeoutError as e:
        return f"[Gemma 4 timed out. {e}. Try again — the model may still be loading from Ollama's cache.]\n\nRelevant knowledge:\n{context[:800]}"
    except Exception as e:
        log.warning(f"Gemma 4 reasoning failed: {e}")
        return f"[Gemma 4 error: {e}]\n\nRelevant knowledge:\n{context[:600]}"
