"""
AETHER Neural Core — Perception API Endpoint

POST /api/perception/{session_id}/run
Body: {"max_frames": 5, "run_3d": false}
Response: Complete perception results
"""
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import DATA_DIR

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/perception")

# ═══════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════

class PerceptionRequest(BaseModel):
    max_frames: int = 5
    run_3d: bool = False

class PerceptionResponse(BaseModel):
    pipeline_id: str
    session_id: str
    frame_count: int
    segmentation: dict
    depth: dict
    tracking: dict
    stages: dict
    total_time_s: float
    vram_peak_gb: float

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _get_session_frames(session_id: str, max_frames: int):
    """Load frames for a session."""
    frames_dir = Path(DATA_DIR) / "sessions" / session_id / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    frame_files = sorted(frames_dir.glob("*.png"))[:max_frames]
    if not frame_files:
        raise HTTPException(status_code=404, detail=f"No frames found for session: {session_id}")

    import cv2
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is not None:
            frames.append(img)

    return frames

# ═══════════════════════════════════════════════════════════════════
# GET /api/perception/mechanism_types
# ═══════════════════════════════════════════════════════════════════

@router.get("/perception/mechanism_types")
async def list_mechanism_types():
    """List available mechanism types for scene graph."""
    return {
        "types": [
            "belt_gantry",
            "drone",
            "human_motion",
            "vehicle",
            "robot_arm",
            "linkage",
            "pendulum",
            "rigid_body",
            "custom",
        ]
    }

# ═══════════════════════════════════════════════════════════════════
# GET /api/perception/{session_id}/status
# ═══════════════════════════════════════════════════════════════════

@router.get("/perception/{session_id}/status")
async def get_perception_status(session_id: str):
    """Check if perception has been run for a session."""
    frames_dir = Path(DATA_DIR) / "sessions" / session_id / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    frame_files = sorted(frames_dir.glob("*.png"))
    return {
        "session_id": session_id,
        "frame_count": len(frame_files),
        "perception_available": True,
    }

# ═══════════════════════════════════════════════════════════════════
# POST /api/perception/{session_id}/run
# ═══════════════════════════════════════════════════════════════════

@router.post("/perception/{session_id}/run", response_model=PerceptionResponse)
async def run_perception(session_id: str, req: PerceptionRequest = PerceptionRequest()):
    """Run the AETHER Neural Core perception pipeline.

    This endpoint:
    1. Loads frames from the session
    2. Runs SAM2 segmentation (lean config, ~1s)
    3. Runs MiDaS depth estimation (~0.1s per frame)
    4. Runs CoTracker3 point tracking (~0.04s per frame)
    5. Returns all results with timing

    Performance (RTX 3050, after model warmup):
      - First call: ~8s (includes model loading)
      - Subsequent calls: ~1.3s for 5 frames

    Args:
        session_id: The session to process
        req.max_frames: Max frames to process (default 5, max 16)
        req.run_3d: Enable 3D reconstruction (future feature)

    Returns:
        Complete perception results with masks, depth, tracks
    """
    import torch
    import cv2

    # Cap max_frames to CoTracker3 window size
    max_frames = min(req.max_frames, 16)
    pipeline_id = str(uuid.uuid4())[:8]

    log.info(f"[{pipeline_id}] Starting perception for session {session_id}")
    t0 = time.time()

    # Load frames
    try:
        frames = _get_session_frames(session_id, max_frames)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load frames: {e}")

    log.info(f"[{pipeline_id}] Loaded {len(frames)} frames")

    # Run AETHER Neural Core
    try:
        from app.perception.optimized.pipeline import get_neural_core
        core = get_neural_core()
        result = core.run(frames)
    except Exception as e:
        log.error(f"[{pipeline_id}] Neural core failed: {e}")
        raise HTTPException(status_code=500, detail=f"Perception pipeline failed: {e}")

    total_time = time.time() - t0

    return PerceptionResponse(
        pipeline_id=pipeline_id,
        session_id=session_id,
        frame_count=len(frames),
        segmentation=result["segmentation"],
        depth={
            "depth_map": result["depth"]["depth_map"].tolist() if hasattr(result["depth"]["depth_map"], 'tolist') else result["depth"]["depth_map"],
            "min_depth": result["depth"]["min_depth"],
            "max_depth": result["depth"]["max_depth"],
            "mean_depth": result["depth"]["mean_depth"],
            "method": result["depth"]["method"],
            "time_s": result["depth"]["time_s"],
        },
        tracking={
            "tracks": result["tracking"]["tracks"],
            "frame_count": result["tracking"]["frame_count"],
            "track_count": result["tracking"]["track_count"],
            "method": result["tracking"]["method"],
            "time_s": result["tracking"]["time_s"],
        },
        stages=result["stages"],
        total_time_s=total_time,
        vram_peak_gb=result.get("vram_peak_gb", 0.0),
    )


# ═══════════════════════════════════════════════════════════════════
# GET /api/perception/{session_id}/masks
# ═══════════════════════════════════════════════════════════════════

@router.get("/perception/{session_id}/masks")
async def get_masks(session_id: str):
    """Get segmentation masks for a session (run perception first)."""
    # This would normally fetch from a cache/db
    # For now, we just run perception
    req = PerceptionRequest(max_frames=1, run_3d=False)
    return await run_perception(session_id, req)
