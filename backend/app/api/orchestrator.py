"""
AETHER Complete Pipeline API
"""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
import cv2
import numpy as np

from app.orchestrator.complete_pipeline import AetherCompletePipeline
from app.core import config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/orchestrate", tags=["orchestrator"])


@router.get("/process")
async def process_video(
    session_id: str,
    question: str = "What is the physics of this mechanism?",
):
    """
    Process a session through the complete AETHER pipeline v2.
    """
    session_dir = config.DATA_DIR / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    # Load frames from the frames subdirectory
    frames_dir = session_dir / "frames"
    frames_files = sorted(frames_dir.glob("frame_*.png"))
    if not frames_files:
        raise HTTPException(404, "No frames found")
    
    frames = [cv2.imread(str(f)) for f in frames_files]
    frames = [f for f in frames if f is not None]
    
    if not frames:
        raise HTTPException(400, "Failed to load frames")
    
    # Process through pipeline v2
    pipeline = AetherCompletePipeline()
    try:
        result = await pipeline.process(
            frames=frames,
            session_id=session_id,
        )
        # Add question back to result for frontend
        result["question"] = question
        return result
    except Exception as e:
        import traceback
        log.error(f"Pipeline v2 failed: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(500, f"Pipeline v2 failed: {str(e)[:200]}")
    finally:
        pipeline.cleanup()


@router.get("/quick")
async def quick_analyze(
    session_id: str,
):
    """
    Quick analysis without full pipeline.
    
    Returns:
    - Mechanism type
    - Object count
    - Shape features
    - Quick simulation
    """
    session_dir = config.DATA_DIR / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    # Load frames from the frames subdirectory
    frames_dir = session_dir / "frames"
    frames_files = sorted(frames_dir.glob("frame_*.png"))
    if not frames_files:
        raise HTTPException(404, "No frames found")

    frame = cv2.imread(str(frames_files[0]))
    if frame is None:
        raise HTTPException(400, "Failed to load frame")
    
    # Dense segment (better detection)
    from app.perception.optimized.pipeline import DenseSegmenter
    segmenter = DenseSegmenter()
    masks = segmenter.generate(frame)
    segmenter.unload()
    
    # Quick identify
    from app.scene_graph.universal_builder import identify_mechanism, analyze_mask_shape
    
    shape_features = {}
    mask_dicts = []
    for i, m in enumerate(masks):
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
        bbox = [int(x) for x in m.get("bbox", [0, 0, 0, 0])]
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            x, y, w, h = bbox
            mask[y:y+h, x:x+w] = True
        
        shape_features[i] = analyze_mask_shape({"segmentation": mask})
        mask_dicts.append({"id": i})
    
    mechanism_type = identify_mechanism(mask_dicts, shape_features)
    
    # Quick simulate
    from app.physics.universal_simulator import UniversalPhysicsSimulator
    simulator = UniversalPhysicsSimulator()
    
    # Use mechanism-appropriate params
    param_overrides = {}
    if mechanism_type == "vehicle":
        param_overrides = {"chassis_mass": 50.0, "wheel_radius": 0.08}
    elif mechanism_type == "pendulum":
        param_overrides = {"rod_length": 0.5, "bob_mass": 1.0}
    elif mechanism_type == "drone":
        param_overrides = {"drone_mass": 1.0, "prop_radius": 0.08}
    
    sim_result = simulator.simulate(
        mechanism_type=mechanism_type,
        horizon_seconds=3.0,  # 3 seconds for proper trajectory
        param_overrides=param_overrides,
    )
    
    return {
        "session_id": session_id,
        "mechanism_type": mechanism_type,
        "n_objects": len(masks),
        "shape_features": {
            i: {"aspect": f.get("aspect_ratio"), "area": f.get("area")}
            for i, f in shape_features.items()
        },
        "simulation": {
            "success": sim_result.get("success", True),
            "duration": sim_result.get("duration", 3.0),
            "timesteps": sim_result.get("timesteps", 0),
        },
    }


@router.get("/status")
async def pipeline_status():
    """Get pipeline status and capabilities."""
    import torch
    from pathlib import Path

    gpu_available = torch.cuda.is_available()
    checkpoint_dir = Path("/home/govinda/aether/data/checkpoints")

    # Check actual model availability
    sam2_available = (checkpoint_dir / "sam2_hiera_small.pt").exists()
    cotracker_available = (checkpoint_dir / "scaled_online.pth").exists()
    encoder_available = (checkpoint_dir / "sam2_encoder_fp32.onnx").exists()

    return {
        "status": "ready" if gpu_available else "limited",
        "gpu_available": gpu_available,
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_available else 0,
        "models": {
            "sam2": sam2_available,
            "cotracker": cotracker_available,
            "sam2_encoder_onnx": encoder_available,
        },
        "stages": [
            {"name": "perception", "model": "SAM2", "ready": sam2_available},
            {"name": "scene_graph", "model": "AETHER Universal", "ready": True},
            {"name": "tracking", "model": "CoTracker3", "ready": cotracker_available},
            {"name": "inverse_dynamics", "model": "AETHER", "ready": True},
            {"name": "reconstruction", "model": "MiDaS", "ready": True},
            {"name": "simulation", "model": "MuJoCo/MJX", "ready": gpu_available},
            {"name": "knowledge", "model": "ChromaDB", "ready": True},
        ],
    }


@router.get("/analyzed/{session_id}")
async def get_analyzed_video(session_id: str):
    """
    Get the analyzed video with SAM2 masks overlaid.

    Returns the path to the analyzed video and whether it exists.
    The video is accessible at /static/{session_id}/analyzed.mp4
    """
    session_dir = config.DATA_DIR / "sessions" / session_id
    analyzed_video = session_dir / "analyzed.mp4"
    analyzed_frames_dir = session_dir / "analyzed_frames"

    exists = analyzed_video.exists()
    frames_exist = analyzed_frames_dir.exists()

    # Count frames if directory exists
    frame_count = 0
    if frames_exist:
        frame_count = len(list(analyzed_frames_dir.glob("frame_*.png")))

    return {
        "session_id": session_id,
        "analyzed_video_exists": exists,
        "analyzed_video_url": f"/static/{session_id}/analyzed.mp4" if exists else None,
        "analyzed_video_path": str(analyzed_video) if exists else None,
        "analyzed_frames_exists": frames_exist,
        "analyzed_frames_url": f"/static/{session_id}/analyzed_frames" if frames_exist else None,
        "analyzed_frames_count": frame_count,
    }
