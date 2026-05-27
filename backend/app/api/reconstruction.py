"""
3D Reconstruction API
"""

import logging, base64, io, time
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.reconstruction.mesh import (
    reconstruct_3d_from_frame,
    reconstruct_from_video,
    depth_to_point_cloud,
    save_mesh_obj,
    estimate_depth,
    segment_frame_dense,
)
from app.core import config

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reconstruction", tags=["reconstruction"])

class ReconstructionRequest(BaseModel):
    session_id: str
    frame_index: int = 0
    use_dense: bool = True


@router.get("/reconstruct")
async def reconstruct_3d(
    session_id: str = "",
    frame_index: int = 0,
    use_dense: bool = False,
):
    """Reconstruct 3D from a session frame."""
    import glob, cv2, os
    
    # Find frame
    session_dir = config.DATA_DIR / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    frames_dir = session_dir / "frames"
    frames = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    if not frames:
        raise HTTPException(404, "No frames found")
    
    frame_path = frames[min(frame_index, len(frames)-1)]
    frame = cv2.imread(str(frame_path))
    
    if frame is None:
        raise HTTPException(400, "Failed to load frame")
    
    # Reconstruct
    t0 = time.time()
    result = reconstruct_3d_from_frame(frame, use_dense=use_dense)
    
    # Convert numpy types to Python native
    import numpy as np
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    result = convert(result)
    result["time_seconds"] = time.time() - t0
    result["session_id"] = session_id
    result["frame_path"] = str(frame_path)
    
    return result


@router.get("/reconstruct/dense")
async def reconstruct_dense_get(session_id: str = ""):
    """Reconstruct 3D from session frame (dense segmentation)."""
    import cv2, numpy as np, torch
    from pathlib import Path
    
    if not session_id:
        raise HTTPException(400, "session_id required")
    
    session_dir = config.DATA_DIR / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    frames_dir = session_dir / "frames"
    frames = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    if not frames:
        raise HTTPException(404, "No frames found")

    frame = cv2.imread(str(frames[0]))
    if frame is None:
        raise HTTPException(400, "Failed to load frame")
    
    t0 = time.time()
    
    # Dense segmentation on full image
    masks = segment_frame_dense(frame, points_per_side=8)
    
    # Depth
    depth = estimate_depth(frame)
    
    # Point clouds
    point_clouds = []
    for i, m in enumerate(masks[:20]):  # Max 20 objects
        mask = m["segmentation"]
        if mask.sum() > 500:
            pc = depth_to_point_cloud(depth, mask)
            if len(pc) > 10:
                # Compute bounding box stats
                x_mean = pc[:, 0].mean()
                y_mean = pc[:, 1].mean()
                z_mean = pc[:, 2].mean()
                
                point_clouds.append({
                    "id": i,
                    "n_points": len(pc),
                    "area": int(mask.sum()),
                    "bbox": [float(x) for x in m["bbox"]],
                    "center_3d": [float(x_mean), float(y_mean), float(z_mean)],
                    "stability": float(m.get("stability_score", 0)),
                })
    
    # Save mesh for largest
    mesh_path = None
    if point_clouds:
        largest = max(point_clouds, key=lambda x: x["n_points"])
        # Get corresponding mask
        largest_mask = masks[largest["id"]]["segmentation"]
        pc = depth_to_point_cloud(depth, largest_mask)
        
        if len(pc) > 500:
            from app.reconstruction.mesh import point_cloud_to_mesh, save_mesh_obj
            
            # Downsample
            if len(pc) > 3000:
                idx = np.random.choice(len(pc), 3000, replace=False)
                pts = pc[idx]
            else:
                pts = pc
            
            mesh = point_cloud_to_mesh(pts)
            mesh_path = f"/tmp/recon_{int(time.time())}.obj"
            save_mesh_obj(mesh, mesh_path)
    
    return {
        "n_objects": len(masks),
        "n_point_clouds": len(point_clouds),
        "point_clouds": point_clouds[:10],  # Top 10
        "depth_stats": {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
        },
        "mesh_path": mesh_path,
        "time_seconds": time.time() - t0,
    }


@router.post("/reconstruct/from_frames")
async def reconstruct_from_frames(session_id: str, max_frames: int = 3):
    """Reconstruct 3D from multiple frames for better coverage."""
    import glob, cv2, os, torch
    
    session_dir = config.DATA_DIR / "sessions" / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")
    
    frames = sorted(session_dir.glob("frame_*.png"))[:max_frames]
    if not frames:
        raise HTTPException(404, "No frames found")
    
    all_frames = [cv2.imread(str(f)) for f in frames]
    all_frames = [f for f in all_frames if f is not None]
    
    t0 = time.time()
    result = reconstruct_from_video(all_frames, max_frames=max_frames)
    result["time_seconds"] = time.time() - t0
    
    return result


@router.get("/mesh/{session_id}")
async def get_mesh(session_id: str):
    """Get the latest mesh for a session."""
    mesh_path = Path(f"/tmp/recon_{session_id}.obj")
    if not mesh_path.exists():
        raise HTTPException(404, "Mesh not found")
    return FileResponse(str(mesh_path), media_type="model/obj")


@router.post("/export/urdf")
async def export_urdf(session_id: str, frame_index: int = 0):
    """Export as URDF for MuJoCo/Robot."""
    import glob, cv2, numpy as np
    
    session_dir = config.DATA_DIR / "sessions" / session_id
    frames_dir = session_dir / "frames"
    frames = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    frame = cv2.imread(str(frames[frame_index]))
    
    if frame is None:
        raise HTTPException(400, "Failed to load frame")
    
    # Simple URDF from bounding boxes
    from app.perception.optimized.pipeline import FastSegmenter
    
    segmenter = FastSegmenter()
    masks = segmenter.generate(frame)
    segmenter.unload()
    
    # Generate URDF with boxes
    urdf = """<?xml version="1.0"?>
<robot name="aether_reconstruction">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
"""
    
    for i, m in enumerate(masks[:10]):
        bbox = m.get("bbox", [0, 0, 100, 100])
        w, h = bbox[2], bbox[3]
        scale = 0.001  # meters per pixel
        urdf += f"""
  <link name="object_{i}">
    <visual>
      <origin xyz="{bbox[0]*scale} {bbox[1]*scale} 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="{w*scale} {h*scale} 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="{w*scale} {h*scale} 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  <joint name="joint_{i}" type="fixed">
    <parent link="base_link"/>
    <child link="object_{i}"/>
  </joint>
"""
    
    urdf += "</robot>"
    
    return {"urdf": urdf, "n_objects": len(masks)}
