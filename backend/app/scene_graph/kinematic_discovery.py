"""
AETHER Unsupervised Kinematic Discovery
=======================================
Discovers kinematic structure from visual trajectories WITHOUT templates.
"""

import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import logging

log = logging.getLogger(__name__)


class JointType(Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"     # 1 rotational DOF (hinge)
    PRISMATIC = "prismatic"   # 1 translational DOF (slider)
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    FREE = "free"


@dataclass
class RigidBody:
    id: int
    point_indices: np.ndarray
    centroid_trajectory: np.ndarray
    rotation_trajectory: np.ndarray
    
    
@dataclass  
class DiscoveredJoint:
    parent_id: int
    child_id: int
    joint_type: JointType
    axis: Optional[np.ndarray] = None
    plane_normal: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class KinematicTree:
    bodies: List[RigidBody]
    joints: List[DiscoveredJoint]
    root_id: int = 0
    n_points_total: int = 0
    n_bodies: int = 0
    n_joints: int = 0


def compute_motion_coherence_matrix(tracks: np.ndarray) -> np.ndarray:
    """Compute pairwise motion coherence - points with similar displacement vectors cluster."""
    n_frames, n_points = tracks.shape[:2]
    n_dim = tracks.shape[2] if tracks.ndim == 3 else 2
    
    # displacements: (n_frames-1, n_points, n_dim)
    displacements = np.diff(tracks, axis=0)
    
    # For each pair of points, compute correlation of their displacement patterns
    affinity = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                affinity[i, j] = 1.0
            else:
                # Flatten displacements for each point
                d1 = displacements[:, i, :].flatten()
                d2 = displacements[:, j, :].flatten()
                
                norm1 = np.linalg.norm(d1) + 1e-8
                norm2 = np.linalg.norm(d2) + 1e-8
                corr = np.dot(d1, d2) / (norm1 * norm2)
                affinity[i, j] = max(0, corr)
    
    return affinity


def cluster_rigid_bodies(tracks: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
    """Cluster trajectory points into rigid bodies using spectral clustering."""
    n_frames, n_points = tracks.shape[:2]
    
    affinity = compute_motion_coherence_matrix(tracks)
    
    if n_clusters is None:
        n_clusters = min(8, max(2, n_points // 10))
    
    try:
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=10
        )
        return spectral.fit_predict(affinity)
    except Exception as e:
        log.warning(f"Spectral clustering failed: {e}, using simple assignment")
        return np.arange(n_points) % n_clusters


def compute_rigid_body_transformations(
    tracks: np.ndarray,
    labels: np.ndarray,
    body_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rigid body motion using SVD Procrustes analysis."""
    n_frames = tracks.shape[0]
    
    mask = labels == body_id
    body_points = tracks[:, mask]
    
    if body_points.shape[1] < 3:
        return np.zeros((n_frames, 3)), np.tile(np.eye(3), (n_frames, 1, 1))
    
    ref_centroid = body_points[0].mean(axis=0)
    ref_points = body_points[0] - ref_centroid
    
    centroids = []
    rotations = []
    
    for t in range(n_frames):
        curr_centroid = body_points[t].mean(axis=0)
        curr_points = body_points[t] - curr_centroid
        
        H = ref_points.T @ curr_points
        U, _, Vt = np.linalg.svd(H)
        
        S = np.eye(min(H.shape))
        S[-1, -1] = np.linalg.det(Vt.T @ U)
        
        R_opt = Vt.T @ S @ U.T
        
        # Pad to 3x3 if needed
        if R_opt.shape[0] == 2:
            R_opt = np.block([[R_opt, np.zeros((2, 1))], [np.zeros(3)]])
        
        centroids.append(curr_centroid)
        rotations.append(R_opt)
    
    return np.array(centroids), np.array(rotations)


def analyze_joint_dof(
    parent_traj: np.ndarray,
    child_traj: np.ndarray,
    parent_rot: Optional[np.ndarray],
    child_rot: Optional[np.ndarray],
) -> Tuple[JointType, Optional[np.ndarray], float]:
    """Analyze relative motion between two bodies to classify joint type."""
    n_frames = len(parent_traj)
    
    rel_translation = child_traj - parent_traj
    translation_variance = np.var(rel_translation, axis=0)
    
    rotation_variance = 0.0
    mean_axis = None
    
    if parent_rot is not None and child_rot is not None:
        try:
            rotvecs = []
            for t in range(n_frames):
                try:
                    rel_rot = parent_rot[t].T @ child_rot[t]
                    rv = R.from_matrix(rel_rot)
                    rotvecs.append(rv.as_rotvec())
                except:
                    pass
            
            if rotvecs:
                rotvecs = np.array(rotvecs)
                rotation_variance = float(np.var(np.linalg.norm(rotvecs, axis=1)))
                mean_axis = np.mean(rotvecs, axis=0)
                norm = np.linalg.norm(mean_axis)
                if norm > 1e-6:
                    mean_axis = mean_axis / norm
                else:
                    mean_axis = None
        except Exception as e:
            log.warning(f"Rotation analysis failed: {e}")
    
    total_trans = np.sum(translation_variance)
    
    if total_trans < 1e-6 and rotation_variance < 1e-6:
        return JointType.FIXED, None, 1.0
    
    if total_trans > 100 and rotation_variance > 0.1:
        return JointType.FREE, None, 0.5
    
    trans_dof = translation_variance > np.mean(translation_variance) * 0.5
    
    # Revolute: rotation present, minimal translation
    if rotation_variance > 0.01 and np.sum(trans_dof) <= 1:
        if mean_axis is not None:
            return JointType.REVOLUTE, mean_axis, min(1.0, rotation_variance * 10)
        return JointType.REVOLUTE, np.array([0, 0, 1]), min(1.0, rotation_variance * 10)
    
    # Prismatic: translation along 1 axis, minimal rotation
    if np.sum(trans_dof) >= 1 and rotation_variance < 0.01:
        dominant_axis = np.argmax(translation_variance)
        axis = np.eye(3)[dominant_axis].astype(float)
        if len(rel_translation) > 0 and rel_translation[0, dominant_axis] < 0:
            axis = -axis
        confidence = float(translation_variance[dominant_axis] / (total_trans + 1e-8))
        return JointType.PRISMATIC, axis, min(1.0, confidence)
    
    if rotation_variance > 0.01 and np.sum(trans_dof) == 1:
        return JointType.CYLINDRICAL, mean_axis, 0.6
    
    return JointType.FIXED, None, 0.3


def discover_kinematic_structure(
    tracks: np.ndarray,
    n_bodies: Optional[int] = None,
) -> KinematicTree:
    """Main entry point: Discover complete kinematic structure from trajectories."""
    log.info(f"Kinematic discovery: {tracks.shape[0]} frames, {tracks.shape[1]} points")
    
    labels = cluster_rigid_bodies(tracks, n_bodies)
    n_clusters = len(np.unique(labels))
    log.info(f"Clustered into {n_clusters} rigid bodies")
    
    bodies = []
    for body_id in range(n_clusters):
        centroid_traj, rot_traj = compute_rigid_body_transformations(tracks, labels, body_id)
        bodies.append(RigidBody(
            id=body_id,
            point_indices=np.where(labels == body_id)[0],
            centroid_trajectory=centroid_traj,
            rotation_trajectory=rot_traj
        ))
    
    joints = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            joint_type, axis, confidence = analyze_joint_dof(
                bodies[i].centroid_trajectory,
                bodies[j].centroid_trajectory,
                bodies[i].rotation_trajectory,
                bodies[j].rotation_trajectory
            )
            
            if joint_type != JointType.FIXED and confidence > 0.3:
                rel_pos = bodies[j].centroid_trajectory - bodies[i].centroid_trajectory
                closest_idx = np.argmin(np.linalg.norm(rel_pos, axis=1))
                
                joints.append(DiscoveredJoint(
                    parent_id=i,
                    child_id=j,
                    joint_type=joint_type,
                    axis=axis if joint_type == JointType.REVOLUTE else None,
                    plane_normal=axis if joint_type == JointType.PRISMATIC else None,
                    position=(bodies[i].centroid_trajectory[closest_idx] + 
                             bodies[j].centroid_trajectory[closest_idx]) / 2,
                    confidence=confidence
                ))
                log.info(f"  Joint {i} -> {j}: {joint_type.value} (conf={confidence:.2f})")
    
    return KinematicTree(
        bodies=bodies,
        joints=joints,
        root_id=0,
        n_points_total=tracks.shape[1],
        n_bodies=n_clusters,
        n_joints=len(joints)
    )


def kinematic_tree_to_mjcf(tree: KinematicTree) -> str:
    """Convert discovered kinematic tree to MuJoCo XML."""
    bodies_xml = []
    
    for body in tree.bodies:
        pos = body.centroid_trajectory[0] if len(body.centroid_trajectory) > 0 else np.zeros(3)
        pos_str = " ".join(map(str, pos))
        body_xml = f'    <body name="body_{body.id}" pos="{pos_str}">'
        
        connected_joints = [j for j in tree.joints if j.parent_id == body.id]
        
        for joint in connected_joints:
            if joint.joint_type == JointType.REVOLUTE:
                axis_str = " ".join(map(str, joint.axis)) if joint.axis is not None else "0 0 1"
                pos_j = " ".join(map(str, joint.position)) if joint.position is not None else "0 0 0"
                body_xml += f'\n      <joint name="j_{body.id}_{joint.child_id}" type="hinge" pos="{pos_j}" axis="{axis_str}"/>'
            elif joint.joint_type == JointType.PRISMATIC:
                axis_str = " ".join(map(str, joint.plane_normal)) if joint.plane_normal is not None else "1 0 0"
                pos_j = " ".join(map(str, joint.position)) if joint.position is not None else "0 0 0"
                body_xml += f'\n      <joint name="j_{body.id}_{joint.child_id}" type="slide" pos="{pos_j}" axis="{axis_str}"/>'
            elif joint.joint_type == JointType.FREE:
                body_xml += '\n      <freejoint/>'
        
        body_xml += f'\n      <geom type="box" size="0.05 0.05 0.05" mass="1.0"/>'
        body_xml += '\n    </body>'
        bodies_xml.append(body_xml)
    
    bodies_str = "\n".join(bodies_xml)
    
    return f"""<mujoco model="aether_discovered">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option integrator="implicitfast"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.01" rgba=".3 .3 .3 1" friction="0.8 0.01 0.01"/>
{bodies_str}
  </worldbody>
</mujoco>"""
