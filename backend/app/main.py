"""AETHER Studio - FastAPI Backend Entry Point"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import health, sessions, videos, frames, reconstruction, simulation, perception, scene_graph, knowledge, assistant, orchestrator, llm, videounit
from app.core import config, logging as app_logging
from app import __version__

app_logging.configure()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"AETHER Studio backend v{__version__} starting...")
    log.info(f"Data dir: {config.DATA_DIR}")
    yield
    log.info("AETHER Studio shutting down...")


app = FastAPI(
    title="AETHER Studio API",
    description="Physics-grounded digital twin from video",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(config.DATA_DIR / "sessions")), name="static")

app.include_router(health.router, tags=["health"])
app.include_router(sessions.router, tags=["sessions"])
app.include_router(videos.router, tags=["videos"])
app.include_router(frames.router, tags=["frames"])
app.include_router(reconstruction.router, tags=["reconstruction"])
app.include_router(perception.router, tags=["perception"])
app.include_router(scene_graph.router, tags=["scene_graph"])
app.include_router(simulation.router, tags=["simulation"])
app.include_router(knowledge.router, tags=["knowledge"])
app.include_router(assistant.router, tags=["assistant"])
app.include_router(orchestrator.router, tags=["orchestrator"])
app.include_router(llm.router, tags=["llm"])
app.include_router(videounit.router, tags=["videounit"])


@app.get("/")
async def root():
    return {"name": "AETHER Studio", "version": "0.1.0", "status": "running"}


@app.get("/api")
async def api_root():
    return {"message": "AETHER Studio API v0.1.0", "docs": "/docs"}
