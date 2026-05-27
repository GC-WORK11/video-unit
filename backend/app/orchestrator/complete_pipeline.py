"""
AETHER Complete Integrated Pipeline v2
================================

All phases working together with proper data flow.
"""

import time
import numpy as np
from pathlib import Path
import tempfile
import logging
import os
import cv2

from app.perception.tracking import get_pipeline as get_perception_pipeline
from app.scene_graph.kinematic_discovery import (
    discover_kinematic_structure,
    kinematic_tree_to_mjcf,
)
from app.physics.inertia_tensor import compute_exact_inertia
from app.physics.hamiltonian_physics import HamiltonianRegularizedSimulator
from app.physics.self_improving_physics import SelfImprovingPhysicsEngine
from app.physics.mjx.backprop_sim import BackpropMJX
from app.reconstruction.mesh import depth_to_point_cloud
from app.core import config

log = logging.getLogger(__name__)


class AetherCompletePipeline:
    """Complete AETHER pipeline with all phases integrated."""
    
    def __init__(self):
        self.perception = get_perception_pipeline()
        self.hamiltonian_sim = HamiltonianRegularizedSimulator(lambda_hamiltonian=0.1)
        self.self_improver = SelfImprovingPhysicsEngine(
            drift_threshold=0.05,
            ewc_lambda=1000.0,
        )
        
    async def process(self, frames, session_id="default"):
        """Complete pipeline."""
        start_time = time.time()
        results = {
            "session_id": session_id,
            "pipeline": "AETHER Complete v2",
            "n_frames": len(frames),
            "stages": {},
        }
        
        # STAGE 1: PERCEPTION
        stage_start = time.time()
        log.info("Stage 1: Perception")
        
        perception_result = self.perception.run_full_pipeline(frames)
        masks = perception_result["segmentation"]["masks"]
        tracking = perception_result["tracking"]
        depth_map = perception_result["depth"]["depth_map"]
        
        results["stages"]["perception"] = {
            "time_seconds": time.time() - stage_start,
            "n_masks": len(masks),
            "n_tracks": tracking["track_count"],
        }
        
        # STAGE 2: KINEMATIC DISCOVERY (Phase 1.1)
        stage_start = time.time()
        log.info("Stage 2: Kinematic Discovery (Phase 1.1)")
        
        trajectories = self._build_trajectories(tracking, masks, frames[0].shape)
        kin_tree = None
        
        if trajectories and len(trajectories) >= 2:
            traj_list = list(trajectories.values())
            max_len = max(len(t["xyz"]) for t in traj_list)
            n_points = len(traj_list)
            
            tracks_3d = np.zeros((max_len, n_points, 3))
            for i, traj in enumerate(traj_list):
                traj_xyz = np.array(traj["xyz"])
                for t in range(min(len(traj_xyz), max_len)):
                    tracks_3d[t, i] = traj_xyz[t]
            
            kin_tree = discover_kinematic_structure(tracks_3d, n_bodies=2)
        
        results["stages"]["kinematic_discovery"] = {
            "time_seconds": time.time() - stage_start,
            "n_bodies": kin_tree.n_bodies if kin_tree else 0,
            "n_joints": kin_tree.n_joints if kin_tree else 0,
            "joints": [
                {"type": j.joint_type.value, "confidence": j.confidence}
                for j in kin_tree.joints
            ] if kin_tree else [],
        }

        # STAGE 2.5: MJX SYSTEM IDENTIFICATION (using discovered kinematic structure)
        stage_start = time.time()
        log.info("Stage 2.5: MJX System ID (learning physical parameters)")

        mjx_learned_params = {}
        mjcf_path = None
        if kin_tree and trajectories and len(trajectories) >= 2:
            try:
                # Build MJCF from discovered kinematic tree
                mjcf_xml = kinematic_tree_to_mjcf(kin_tree)

                # Write MJCF to temp file for BackpropMJX
                with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", dir="/tmp") as f:
                    f.write(mjcf_xml)
                    mjcf_path = f.name

                # Initialize MJX backprop system ID
                backprop_mjx = BackpropMJX(mjcf_path)

                # Build qpos trajectory from xyz trajectories for each joint
                # Use trajectory data to compute joint angles
                traj_list = list(trajectories.values())
                if traj_list and len(traj_list) >= 2:
                    # For discovered kinematic tree, compute joint angles from body trajectories
                    # Get body centroids from kin_tree
                    if len(kin_tree.bodies) >= 2 and len(kin_tree.bodies[0].centroid_trajectory) > 0:
                        # Compute relative motion between first two bodies to get joint angles
                        body0_traj = kin_tree.bodies[0].centroid_trajectory
                        body1_traj = kin_tree.bodies[1].centroid_trajectory

                        # Ensure trajectories are same length
                        min_len = min(len(body0_traj), len(body1_traj), 100)
                        body0_traj = body0_traj[:min_len]
                        body1_traj = body1_traj[:min_len]

                        # Compute joint angles from relative position
                        rel_pos = body1_traj - body0_traj
                        qpos_obs = np.arctan2(rel_pos[:, 1], rel_pos[:, 0]).reshape(-1, 1)

                        # Run MJX system ID to learn physical parameters
                        mjx_result = backprop_mjx.learn_parameters(
                            qpos_obs=qpos_obs,
                            lr=0.05,
                            n_iterations=200,
                            target_params=["body_mass", "geom_friction"],
                        )

                        learned = mjx_result.get("learned_parameters", {})

                        # Map MJX learned params to simulator format
                        body_mass = learned.get("body_mass", np.array([1.0, 1.0]))
                        geom_friction = learned.get("geom_friction", np.array([0.5, 0.5]))

                        mjx_learned_params = {
                            "mass_kg": float(body_mass[1]) if len(body_mass) > 1 else float(body_mass[0]),
                            "friction": float(geom_friction[1]) if len(geom_friction) > 1 else float(geom_friction[0]),
                            "final_loss": mjx_result.get("final_loss", 0.0),
                            "n_iterations": 200,
                        }
                        log.info(f"  MJX learned: mass={mjx_learned_params['mass_kg']:.4f}, friction={mjx_learned_params['friction']:.4f}")

            except Exception as e:
                log.warning(f"MJX System ID failed: {e}")
                mjx_learned_params = {}
            finally:
                if mjcf_path and os.path.exists(mjcf_path):
                    os.unlink(mjcf_path)

        results["stages"]["mjx_system_id"] = {
            "time_seconds": time.time() - stage_start,
            "learned_params": mjx_learned_params,
        }

        # STAGE 3: EXACT INERTIA TENSOR (Phase 1.2)
        stage_start = time.time()
        log.info("Stage 3: Exact Inertia Tensor (Phase 1.2)")
        
        inertia_data = {}
        frame = frames[0]
        for i, m in enumerate(masks[:5]):
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
            bbox = [int(x) for x in m.get("bbox", [0, 0, 0, 0])]
            x, y, bw, bh = bbox
            if bw > 0 and bh > 0:
                mask[y:y+bh, x:x+bw] = True
                points_3d = depth_to_point_cloud(depth_map, mask)
                
                if len(points_3d) > 100:
                    inertia = compute_exact_inertia(points_3d)
                    inertia_data[f"body_{i}"] = {
                        "mass": inertia.mass,
                        "com": inertia.com.tolist(),
                        "I": inertia.tensor.tolist(),
                        "principal_moments": [inertia.I1, inertia.I2, inertia.I3],
                    }
        
        results["stages"]["inertia_tensor"] = {
            "time_seconds": time.time() - stage_start,
            "n_bodies": len(inertia_data),
            "bodies": inertia_data,
        }
        
        # STAGE 4: JAX + HAMILTONIAN PHYSICS (Phase 2.1 + 2.2)
        stage_start = time.time()
        log.info("Stage 4: JAX Physics + Hamiltonian (Phase 2.1 + 2.2)")
        
        learned_params = {}
        for obj_id, traj_data in trajectories.items():
            xyz = np.array(traj_data["xyz"])
            if len(xyz) > 10:
                # Use first 60 frames for learning
                x_obs = xyz[:60]
                if len(x_obs) > 10:
                    x0 = x_obs[0]
                    v0 = np.gradient(x_obs, axis=0)[0] if len(x_obs) > 1 else np.zeros(3)
                    
                    params, loss, metrics = self.hamiltonian_sim.learn_params(
                        x0, v0, x_obs,
                        lr=0.01,
                        n_iterations=100,
                        lambda_hamiltonian=0.1,
                    )
                    
                    learned_params[obj_id] = {
                        "mass_kg": float(params[0]),
                        "stiffness_Nm": float(params[1]),
                        "damping_Nsm": float(params[2]),
                        "hamiltonian_loss": metrics["hamiltonian_loss"],
                        "energy_drift": metrics["energy_drift"],
                    }
        
        results["stages"]["jax_physics"] = {
            "time_seconds": time.time() - stage_start,
            "n_learned": len(learned_params),
            "learned_params": learned_params,
        }

        # STAGE 5: V-NEXT DIGITAL TWIN (Phase 4)
        stage_start = time.time()
        log.info("Stage 5: V-NEXT Digital Twin (Phase 4)")

        from app.physics.vnext_complete import get_vnext_engine

        # Build parts list from masks
        parts = [
            {"name": f"obj_{i}", "bbox": m.get("bbox", [0, 0, 0, 0])}
            for i, m in enumerate(masks[:5])
        ]

        # Build point_tracks from tracking data: (n_points, n_timesteps, 3)
        point_tracks = None
        if tracking["tracks"] and len(tracking["tracks"]) > 0:
            n_frames = len(tracking["tracks"])
            n_tracks = tracking["track_count"]
            h, w = frames[0].shape[:2]
            track_array = np.zeros((n_tracks, n_frames, 3))
            for f_idx, frame_tracks in enumerate(tracking["tracks"]):
                for track in frame_tracks:
                    track_id = int(track["id"])
                    if 0 <= track_id < n_tracks:
                        tx = track["x"] * 0.001
                        ty = track["y"] * 0.001
                        tz = 1.0 + (track["y"] / h) * 0.5
                        track_array[track_id, f_idx] = [tx, ty, tz]
            point_tracks = track_array

        twin = None
        if parts and point_tracks is not None:
            try:
                engine = get_vnext_engine()
                twin = await engine.process_video_data(
                    name=session_id,
                    parts=parts,
                    trajectories=trajectories,
                    point_tracks=point_tracks,
                )
            except Exception as e:
                log.warning(f"V-NEXT twin creation failed: {e}")
                twin = None

        results["stages"]["vnext_digital_twin"] = {
            "time_seconds": time.time() - stage_start,
            "created": twin is not None,
        }
        results["digital_twin"] = twin

        # STAGE 6: SELF-IMPROVING ENGINE (Phase 3)
        stage_start = time.time()
        log.info("Stage 6: Self-Improving Engine (Phase 3)")
        
        for obj_id, params in learned_params.items():
            trajectory = trajectories.get(obj_id, {}).get("xyz", [])
            if len(trajectory) > 5:
                self.self_improver.process_observation(
                    obj_id,
                    np.array(trajectory),
                    params,
                )
        
        all_mechanisms = self.self_improver.get_all_mechanisms()
        
        results["stages"]["self_improvement"] = {
            "time_seconds": time.time() - stage_start,
            "n_mechanisms": len(all_mechanisms),
            "mechanisms": all_mechanisms,
        }
        
        # STAGE 7: MuJoCo
        stage_start = time.time()
        log.info("Stage 7: MuJoCo Generation")
        
        if kin_tree:
            mjcf = kinematic_tree_to_mjcf(kin_tree)
        else:
            mjcf = self._build_mjcf_from_inertia(learned_params, inertia_data)
        
        results["stages"]["mujoco"] = {
            "time_seconds": time.time() - stage_start,
            "xml_generated": True,
        }
        
        results["total_time_seconds"] = time.time() - start_time
        results["mujoco_xml"] = mjcf
        results["learned_physical_params"] = mjx_learned_params

        # Generate annotated video with masks overlaid
        session_dir = config.DATA_DIR / "sessions" / session_id
        analyzed_video_path = session_dir / "analyzed.mp4"
        analyzed_frames_dir = session_dir / "analyzed_frames"

        self.generate_marked_video(
            frames=frames,
            masks=masks,
            tracking=tracking,
            output_path=analyzed_video_path,
            fps=5.0,
        )

        # Also export individual marked frames
        self.export_marked_frames(
            frames=frames,
            masks=masks,
            tracking=tracking,
            output_dir=analyzed_frames_dir,
            prefix="frame",
        )

        results["analyzed_video_path"] = str(analyzed_video_path)
        results["analyzed_frames_dir"] = str(analyzed_frames_dir)

        log.info(f"✅ Complete pipeline: {results['total_time_seconds']:.1f}s")

        return results
    
    def _build_trajectories(self, tracking, masks, frame_shape):
        """Build trajectories from tracking data."""
        trajectories = {}
        tracks_raw = tracking["tracks"]
        
        if len(tracks_raw) > 0:
            h, w = frame_shape[:2]
            
            for mask_idx, m in enumerate(masks[:10]):
                obj_id = f"obj_{mask_idx}"
                bbox = [int(x) for x in m.get("bbox", [0, 0, 0, 0])]
                x, y, bw, bh = bbox
                
                obj_tracks = []
                for track_id in range(tracking["track_count"]):
                    t0 = tracks_raw[0][track_id]
                    if x <= t0["x"] <= x + bw and y <= t0["y"] <= y + bh:
                        full_track = [tracks_raw[f_idx][track_id] for f_idx in range(len(tracks_raw))]
                        obj_tracks.append(full_track)
                
                if obj_tracks:
                    avg_xyz = []
                    for f_idx in range(len(tracks_raw)):
                        fx = np.mean([t[f_idx]["x"] for t in obj_tracks])
                        fy = np.mean([t[f_idx]["y"] for t in obj_tracks])
                        fz = 1.0 + (fy / h) * 0.5
                        avg_xyz.append([fx * 0.001, fy * 0.001, fz])
                    
                    trajectories[obj_id] = {"xyz": np.array(avg_xyz)}
        
        return trajectories
    
    def _build_mjcf_from_inertia(self, learned_params, inertia_data):
        """Build MuJoCo from inertia data."""
        bodies = []
        for body_id, inertia in inertia_data.items():
            com = inertia["com"]
            I = inertia["I"]
            mass = inertia["mass"]

            body = f"""<body name="{body_id}" pos="{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass:.4f}" fullinertia="{I[0][0]:.6f} {I[1][1]:.6f} {I[2][2]:.6f} {I[0][1]:.6f} {I[0][2]:.6f} {I[1][2]:.6f}"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.2 0.8 0.8 1"/>
    </body>"""
            bodies.append(body)

        return f"""<mujoco model="aether_complete">
  <option integrator="implicitfast"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01" friction="0.8 0.01 0.01"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""

    def generate_marked_video(self, frames, masks, tracking, output_path, fps=5.0):
        """
        Generate an annotated video with SAM2 masks overlaid on frames.

        Args:
            frames: List of numpy arrays (from cv2.imread)
            masks: List of mask dictionaries from perception stage
            tracking: Tracking data dictionary
            output_path: Path to write the MP4 video
            fps: Frames per second for the output video

        Returns:
            Path to the generated video file
        """
        if not frames:
            log.warning("No frames provided for video generation")
            return None

        # Color palette for different objects (BGR format for cv2)
        color_palette = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 87, 255),    # Blue
            (255, 255, 87),   # Yellow
            (255, 87, 255),   # Magenta
            (87, 255, 255),   # Cyan
            (255, 165, 87),   # Orange
            (165, 87, 255),   # Purple
            (87, 165, 255),   # Light Blue
            (255, 87, 165),   # Pink
        ]

        h, w = frames[0].shape[:2]

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            log.error(f"Failed to open VideoWriter for {output_path}")
            return None

        # Build track_id to color mapping from tracking data
        track_colors = {}
        if tracking and tracking.get("tracks"):
            for frame_idx, frame_tracks in enumerate(tracking["tracks"]):
                for track in frame_tracks:
                    track_id = track["id"]
                    if track_id not in track_colors:
                        track_colors[track_id] = color_palette[len(track_colors) % len(color_palette)]

        try:
            for frame_idx, frame in enumerate(frames):
                annotated = frame.copy()

                # Draw masks with colors based on tracking
                for mask_idx, mask_data in enumerate(masks):
                    bbox = mask_data.get("bbox", [0, 0, 0, 0])
                    x, y, bw, bh = [int(v) for v in bbox]

                    if bw <= 0 or bh <= 0:
                        continue

                    # Get color for this object
                    color = color_palette[mask_idx % len(color_palette)]

                    # Create mask visualization (filled polygon at 30% opacity)
                    mask_visual = np.zeros_like(annotated)

                    # Try to get segmentation mask
                    seg = mask_data.get("segmentation")
                    if seg is not None and isinstance(seg, np.ndarray) and seg.size > 0:
                        # Use actual segmentation if available
                        if seg.dtype != np.uint8:
                            seg = (seg * 255).astype(np.uint8)
                        # Resize if needed
                        if seg.shape[:2] != (h, w):
                            seg = cv2.resize(seg, (w, h))
                        # Find contours and fill with solid color
                        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(mask_visual, contours, -1, color, -1)
                    else:
                        # Fall back to bbox
                        cv2.rectangle(mask_visual, (x, y), (x + bw, y + bh), color, -1)

                    # Overlay mask on frame with 30% opacity
                    cv2.addWeighted(annotated, 1.0, mask_visual, 0.3, 0, dst=annotated)

                    # Draw contour outline
                    if seg is not None and isinstance(seg, np.ndarray) and seg.size > 0:
                        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(annotated, contours, -1, color, 2)
                    else:
                        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

                    # Draw label with track ID if available
                    label = f"Obj {mask_idx}"
                    if tracking and tracking.get("tracks") and frame_idx < len(tracking["tracks"]):
                        for track in tracking["tracks"][frame_idx]:
                            track_bbox = track.get("bbox", [0, 0, 0, 0])
                            if len(track_bbox) == 4:
                                tx, ty, tbw, tbh = [int(v) for v in track_bbox]
                                # Check if this track matches this mask by bbox overlap
                                if (x <= tx <= x + bw and y <= ty <= y + bh):
                                    label = f"Track {track['id']}"
                                    break

                    # Draw label
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = max(y - 10, label_size[1] + 10)
                    cv2.rectangle(annotated, (x, label_y - label_size[1] - 5),
                                  (x + label_size[0], label_y + 5), color, -1)
                    cv2.putText(annotated, label, (x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Draw frame number
                cv2.putText(annotated, f"Frame {frame_idx + 1}/{len(frames)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                writer.write(annotated)

        finally:
            writer.release()

        log.info(f"Generated marked video: {output_path}")
        return str(output_path)

    def export_marked_frames(self, frames, masks, tracking, output_dir, prefix="marked"):
        """
        Export annotated frames as individual PNG images.

        Args:
            frames: List of numpy arrays (from cv2.imread)
            masks: List of mask dictionaries from perception stage
            tracking: Tracking data dictionary
            output_dir: Directory to write the images
            prefix: Filename prefix for saved images

        Returns:
            List of paths to the exported images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate a single marked frame to get the annotation
        # We reuse the video generation logic but just save frames instead
        color_palette = [
            (255, 87, 87), (87, 255, 87), (87, 87, 255), (255, 255, 87),
            (255, 87, 255), (87, 255, 255), (255, 165, 87), (165, 87, 255),
            (87, 165, 255), (255, 87, 165),
        ]

        h, w = frames[0].shape[:2]
        output_paths = []

        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            for mask_idx, mask_data in enumerate(masks):
                bbox = mask_data.get("bbox", [0, 0, 0, 0])
                x, y, bw, bh = [int(v) for v in bbox]

                if bw <= 0 or bh <= 0:
                    continue

                color = color_palette[mask_idx % len(color_palette)]

                # Get segmentation if available
                seg = mask_data.get("segmentation")
                mask_visual = np.zeros_like(annotated)

                if seg is not None and isinstance(seg, np.ndarray) and seg.size > 0:
                    if seg.dtype != np.uint8:
                        seg = (seg * 255).astype(np.uint8)
                    if seg.shape[:2] != (h, w):
                        seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask_visual, contours, -1, color, -1)
                else:
                    cv2.rectangle(mask_visual, (x, y), (x + bw, y + bh), color, -1)

                cv2.addWeighted(annotated, 1.0, mask_visual, 0.3, 0, dst=annotated)

                if seg is not None and isinstance(seg, np.ndarray) and seg.size > 0:
                    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, color, 2)
                else:
                    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

                label = f"Obj {mask_idx}"
                cv2.putText(annotated, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            output_path = output_dir / f"{prefix}_{frame_idx:05d}.png"
            cv2.imwrite(str(output_path), annotated)
            output_paths.append(str(output_path))

        log.info(f"Exported {len(output_paths)} marked frames to {output_dir}")
        return output_paths

    def cleanup(self):
        self.perception.unload_all()


async def run_complete_pipeline(frames, session_id="default"):
    pipeline = AetherCompletePipeline()
    try:
        return await pipeline.process(frames, session_id)
    finally:
        pipeline.cleanup()
