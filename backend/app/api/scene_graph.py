"""
Universal Scene Graph API (BUILD 4)
"""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
import cv2
import numpy as np

from app.scene_graph.universal_builder import build_universal_scene_graph, identify_mechanism, analyze_mask_shape
from app.perception.optimized.pipeline import FastSegmenter
from app.core import config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scene-graph", tags=["scene_graph"])


@router.get("/build")
async def build_scene_graph(
    session_id: str,
    frame_index: int = 0,
):
    """Build universal scene graph from a session."""
    session_dir = config.DATA_DIR / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")
    
    frames = sorted(session_dir.glob("frame_*.png"))
    if not frames:
        raise HTTPException(404, "No frames found")
    
    frame = cv2.imread(str(frames[min(frame_index, len(frames)-1)]))
    if frame is None:
        raise HTTPException(400, "Failed to load frame")
    
    # Segment
    segmenter = FastSegmenter()
    masks = segmenter.generate(frame)
    segmenter.unload()
    
    # Convert to dict format
    mask_dicts = []
    for i, m in enumerate(masks):
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
        bbox = [int(x) for x in m.get("bbox", [0, 0, 0, 0])]
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            x, y, w, h = bbox
            mask[y:y+h, x:x+w] = True
        
        mask_dicts.append({
            "id": i,
            "segmentation": mask,
            "bbox": bbox,
            "area": m.get("area", 0),
        })
    
    # Build scene graph
    scene_graph = build_universal_scene_graph(
        masks=mask_dicts,
        frame_shape=frame.shape[:2],
    )
    
    proc_info = scene_graph.processing_info or {}
    
    return {
        "scene_id": scene_graph.scene_id,
        "mechanism_type": proc_info.get("mechanism_type", "unknown"),
        "mechanism_name": proc_info.get("mechanism_name", "Unknown"),
        "n_objects": len(scene_graph.objects),
        "n_edges": len(scene_graph.edges),
        "shape_features": proc_info.get("shape_features", {}),
        "objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "object_type": obj.object_type,
                "physics": obj.physics,
            }
            for obj in scene_graph.objects
        ],
    }


@router.get("/identify")
async def identify_mechanism_api(
    session_id: str,
    frame_index: int = 0,
):
    """Identify mechanism type without building full graph."""
    session_dir = config.DATA_DIR / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")
    
    frames = sorted(session_dir.glob("frame_*.png"))
    if not frames:
        raise HTTPException(404, "No frames found")
    
    frame = cv2.imread(str(frames[min(frame_index, len(frames)-1)]))
    if frame is None:
        raise HTTPException(400, "Failed to load frame")
    
    # Segment
    segmenter = FastSegmenter()
    masks = segmenter.generate(frame)
    segmenter.unload()
    
    # Analyze shapes
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
    
    # Identify
    mechanism_type = identify_mechanism(mask_dicts, shape_features)
    
    return {
        "mechanism_type": mechanism_type,
        "n_objects": len(masks),
        "shape_features": {
            i: {"aspect_ratio": f.get("aspect_ratio", 1.0), "compactness": f.get("compactness", 0.5), "area": f.get("area", 0)}
            for i, f in shape_features.items()
        },
    }
