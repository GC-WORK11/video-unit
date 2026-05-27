"""
AETHER 3D Reconstruction Pipeline
================================

SAM2 masks + MiDaS depth → Point cloud → Mesh → URDF/MuJoCo

PIPELINE:
1. Dense SAM2 segmentation (points_per_side=8 for better coverage)
2. MiDaS depth estimation
3. Mask + Depth → Point cloud per object
4. Optional: ICP alignment + mesh reconstruction
5. Export: OBJ, PLY, URDF
"""

import numpy as np
import cv2
import torch
import logging
from typing import Optional
from pathlib import Path

log = logging.getLogger(__name__)


def mask_to_polygon(mask: np.ndarray) -> list:
    """Extract polygon contour from binary mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    
    # Largest contour
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Flatten to list of (x, y) points
    return approx.squeeze().tolist()


def depth_to_point_cloud(
    depth_map: np.ndarray,
    mask: np.ndarray | None = None,
    K: np.ndarray | None = None,
    scale: float = 0.001,
) -> np.ndarray:
    """
    Convert depth map to 3D point cloud.
    
    Args:
        depth_map: HxW depth in meters
        mask: Optional HxW binary mask to filter points
        K: Camera intrinsic matrix (3x3). If None, assume pinhole with f=1
        scale: Depth scale factor (default: 0.001 for mm→m)
    
    Returns:
        Nx3 array of (x, y, z) points
    """
    h, w = depth_map.shape
    
    # Default intrinsics (normalized)
    if K is None:
        fx = fy = w * 0.8
        cx, cy = w / 2, h / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    # Generate pixel coordinates
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply mask if provided
    if mask is not None:
        valid = mask > 0
        u = u_coords[valid]
        v = v_coords[valid]
        d = depth_map[valid]
    else:
        u = u_coords.ravel()
        v = v_coords.ravel()
        d = depth_map.ravel()
    
    # Scale depth
    d = d * scale
    
    # Back-project to 3D
    # x = (u - cx) * d / fx
    # y = (v - cy) * d / fy
    # z = d
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d
    
    points = np.stack([x, y, z], axis=-1)
    
    # Filter invalid depth
    valid = (z > 0) & (z < 100)  # 0 to 100m
    return points[valid]


def segment_frame_dense(
    frame: np.ndarray,
    points_per_side: int = 8,
) -> list[dict]:
    """
    Dense SAM2 segmentation for better coverage.
    
    Args:
        frame: BGR image
        points_per_side: Grid density (8 = 64 points vs 16 with points_per_side=4)
    
    Returns:
        List of mask dicts with bbox, area, polygon
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    # Use cached model if available
    sam2 = build_sam2(
        "sam2_hiera_s.yaml",
        "/home/govinda/aether/data/checkpoints/sam2_hiera_small.pt",
        device="cuda",
    )
    sam2.eval()
    
    mask_gen = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=points_per_side,
        points_per_batch=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=100,
    )
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        masks = mask_gen.generate(frame_rgb)
    
    # Add polygon and mask
    result_masks = []
    for i, m in enumerate(masks):
        mask = m["segmentation"].astype(np.uint8)
        polygon = mask_to_polygon(mask)
        
        result_masks.append({
            "id": i,
            "segmentation": mask,
            "bbox": m["bbox"],
            "area": int(m["area"]),
            "polygon": polygon,
            "predicted_iou": float(m.get("predicted_iou", 0)),
            "stability_score": float(m.get("stability_score", 0)),
        })
    
    del sam2
    del mask_gen
    torch.cuda.empty_cache()
    
    return result_masks


def estimate_depth(frame: np.ndarray) -> np.ndarray:
    """Estimate depth using MiDaS."""
    from app.perception.optimized.pipeline import DepthEstimator
    
    estimator = DepthEstimator()
    result = estimator.estimate(frame)
    estimator.unload()
    del estimator
    torch.cuda.empty_cache()
    
    return result["depth_map"]


def point_cloud_to_mesh(
    points: np.ndarray,
    voxel_size: float = 0.01,
) -> dict:
    """
    Convert point cloud to mesh using simple voxelization.
    
    For production: use Open3D, PyMesh, or trimesh for Poisson surface reconstruction.
    This provides a simple alpha-shapes approach.
    
    Returns:
        dict with vertices, faces, normals
    """
    from scipy.spatial import Delaunay
    
    if len(points) < 4:
        return {"vertices": [], "faces": []}
    
    # Simple approach: use convex hull (alpha shapes for holes)
    try:
        # Downsample for speed
        if len(points) > 5000:
            idx = np.random.choice(len(points), 5000, replace=False)
            points = points[idx]
        
        # Convex hull
        points_2d = points[:, :2]  # Project to 2D for triangulation
        tri = Delaunay(points_2d)
        
        # Filter triangles by depth variance
        faces = []
        for simplex in tri.simplices:
            pts = points[simplex]
            z_vals = pts[:, 2]
            if z_vals.std() < 0.1:  # Filter warped triangles
                faces.append(simplex)
        
        return {
            "vertices": points.tolist(),
            "faces": faces if faces else [],
            "n_vertices": len(points),
            "n_faces": len(faces) if faces else 0,
        }
    except Exception as e:
        log.warning(f"Mesh reconstruction failed: {e}")
        return {"vertices": points.tolist(), "faces": [], "n_vertices": len(points), "n_faces": 0}


def save_mesh_obj(mesh: dict, filepath: str) -> bool:
    """Save mesh as OBJ file."""
    vertices = mesh.get("vertices", [])
    faces = mesh.get("faces", [])
    
    if not vertices:
        return False
    
    with open(filepath, "w") as f:
        f.write("# AETHER 3D Reconstruction\n")
        f.write(f"# vertices: {len(vertices)}\n")
        f.write(f"# faces: {len(faces)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    return True


def reconstruct_3d_from_frame(
    frame: np.ndarray,
    use_dense: bool = True,
) -> dict:
    """
    Full 3D reconstruction from a single frame.
    
    Returns:
        dict with:
        - masks: list of object masks
        - depth: depth map
        - point_clouds: list of Nx3 point clouds per object
        - mesh: combined mesh
    """
    # Get masks
    if use_dense:
        masks = segment_frame_dense(frame, points_per_side=8)
    else:
        from app.perception.optimized.pipeline import FastSegmenter
        segmenter = FastSegmenter()
        mask_dicts = segmenter.generate(frame)
        masks = []
        for i, m in enumerate(mask_dicts):
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            bbox = [int(x) for x in m.get("bbox", [0, 0, 0, 0])]
            if len(bbox) == 4:
                x, y, w, h = bbox
                mask[y:y+h, x:x+w] = 1
            masks.append({
                "id": i,
                "segmentation": mask,
                "bbox": bbox,
                "area": int(m.get("area", 0)),
            })
        segmenter.unload()
        del segmenter
    
    # Get depth
    depth = estimate_depth(frame)
    
    # Build point cloud per mask
    point_clouds = []
    for m in masks:
        mask = m["segmentation"]
        if mask.sum() < 100:  # Skip tiny masks
            continue
        
        pc = depth_to_point_cloud(depth, mask)
        if len(pc) > 10:
            point_clouds.append({
                "mask_id": m["id"],
                "area": m["area"],
                "points": pc,
                "n_points": len(pc),
            })
    
    # Combine all points for mesh
    all_points = np.vstack([pc["points"] for pc in point_clouds]) if point_clouds else np.array([])
    
    mesh = {}
    if len(all_points) > 100:
        mesh = point_cloud_to_mesh(all_points, voxel_size=0.01)
    
    return {
        "n_masks": len(masks),
        "n_point_clouds": len(point_clouds),
        "masks": masks,
        "depth_stats": {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
        },
        "point_clouds": [
            {"n_points": pc["n_points"], "mask_id": pc["mask_id"]} 
            for pc in point_clouds
        ],
        "mesh": {
            "n_vertices": mesh.get("n_vertices", 0),
            "n_faces": mesh.get("n_faces", 0),
        },
    }


def reconstruct_from_video(
    frames: list[np.ndarray],
    max_frames: int = 5,
) -> dict:
    """
    Reconstruct 3D from multiple frames.
    
    For each frame:
    1. Dense segmentation
    2. Depth estimation
    3. Point cloud per object
    
    Then: ICP alignment + mesh fusion (future)
    """
    results = []
    
    for i, frame in enumerate(frames[:max_frames]):
        result = reconstruct_3d_from_frame(frame)
        result["frame_index"] = i
        results.append(result)
        log.info(f"Frame {i}: {result['n_masks']} masks, {result['n_point_clouds']} point clouds")
    
    return {
        "n_frames": len(results),
        "frame_results": results,
    }
