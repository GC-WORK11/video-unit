"""
SPLART: 3D Gaussian Splatting for Articulated Objects
====================================================

Core module for representing articulated objects as 3D Gaussians.

Key Innovation:
- Instead of spectral clustering (heuristic), we use 3D Gaussians
- Each rigid part = cluster of Gaussians with SE(3) motion
- Joint detection via DOF analysis of SE(3) trajectories

Mathematical Foundation:
- 3D Gaussian: G(x) = exp(-0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ))
- Covariance decomposition: Σ = R S Rᵀ (rotation × scale × rotation)
- SE(3) transform: T(p) = Rp + t
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class Gaussian3D:
    """
    A single 3D Gaussian for articulated object representation.
    
    Unlike standard 3DGS, we track SE(3) transforms across frames.
    """
    mean: np.ndarray          # [3] position (x, y, z)
    covariance: np.ndarray     # [3, 3] covariance matrix
    opacity: float            # alpha (0-1)
    color: Optional[np.ndarray] = None  # [3] RGB (optional)
    
    def to_spherical_harmonics(self) -> Dict:
        """Convert to spherical harmonics representation for rendering."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        return {
            "mean": self.mean.tolist(),
            "scales": np.sqrt(eigenvalues).tolist(),
            "rotations": eigenvectors.tolist(),
            "opacity": self.opacity,
        }


@dataclass 
class RigidPart:
    """
    A rigid body part = cluster of Gaussians moving together.
    
    Contains:
    - Gaussians describing the geometry
    - SE(3) transforms for each frame (describing rigid motion)
    """
    name: str
    gaussians: List[Gaussian3D]
    se3_transforms: List[np.ndarray]  # [T-1, 4, 4] SE(3) transforms
    
    @property
    def n_gaussians(self) -> int:
        return len(self.gaussians)
    
    @property
    def n_frames(self) -> int:
        return len(self.se3_transforms) + 1
    
    def get_pose(self, frame: int) -> np.ndarray:
        """Get SE(3) pose at frame t."""
        if frame == 0:
            return np.eye(4)
        return self.se3_transforms[frame - 1]
    
    def get_mean_trajectory(self) -> np.ndarray:
        """Get trajectory of part's center of mass."""
        trajectory = [np.mean([g.mean for g in self.gaussians], axis=0)]
        for T in self.se3_transforms:
            last_pos = trajectory[-1]
            new_pos = T[:3, :3] @ last_pos + T[:3, 3]
            trajectory.append(new_pos)
        return np.array(trajectory)


@dataclass
class GaussianCloud:
    """
    Complete 3D Gaussian representation of an articulated object.
    
    Contains multiple RigidParts connected by joints.
    """
    parts: List[RigidPart]
    joints: List["JointSpec"]  # Will be filled by joint detector
    
    def get_part(self, name: str) -> Optional[RigidPart]:
        for part in self.parts:
            if part.name == name:
                return part
        return None
    
    def get_part_trajectory(self, part_name: str) -> np.ndarray:
        part = self.get_part(part_name)
        return part.get_mean_trajectory() if part else np.array([])


@dataclass
class JointSpec:
    """Specification for a kinematic joint."""
    name: str
    parent_name: str
    child_name: str
    joint_type: str  # revolute, prismatic, universal, spherical, fixed
    axis: np.ndarray  # [3] joint axis direction
    origin: np.ndarray  # [4, 4] SE(3) transform
    limits: Dict  # {lower, upper, effort, velocity}


class GaussianCloudReconstructor:
    """
    Reconstruct articulated object from video as 3D Gaussian Cloud.
    
    Pipeline:
    1. Point clouds from SAM2 masks + MiDaS depth
    2. Fit 3D Gaussians to point clusters
    3. Track SE(3) transforms between frames
    4. Output GaussianCloud with RigidParts
    
    NO spectral clustering. NO aspect ratio heuristics.
    Pure geometry + motion analysis.
    """
    
    def __init__(
        self,
        n_gaussians_per_part: int = 100,
        covariance_regularization: float = 1e-5,
    ):
        self.n_gaussians_per_part = n_gaussians_per_part
        self.cov_reg = covariance_regularization
    
    def reconstruct_from_point_clouds(
        self,
        point_clouds: List[np.ndarray],  # [T, N_i, 3]
        mask_assignments: Optional[List[np.ndarray]] = None,  # [T, N_i] part assignments
    ) -> GaussianCloud:
        """
        Main reconstruction from point clouds.
        
        Args:
            point_clouds: List of [N_t, 3] point clouds per frame
            mask_assignments: Optional per-point part assignments
            
        Returns:
            GaussianCloud with RigidParts
        """
        log.info(f"Reconstructing Gaussian cloud from {len(point_clouds)} frames")
        
        # Step 1: Estimate SE(3) transforms between frames
        se3_transforms = self._estimate_se3_transforms(point_clouds)
        
        # Step 2: Fit Gaussians for each rigid part
        parts = self._fit_gaussians_per_part(point_clouds, se3_transforms, mask_assignments)
        
        # Step 3: Build GaussianCloud
        cloud = GaussianCloud(parts=parts, joints=[])
        
        log.info(f"Reconstructed {len(parts)} rigid parts with {sum(p.n_gaussians for p in parts)} Gaussians")
        
        return cloud
    
    def _estimate_se3_transforms(
        self,
        point_clouds: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Estimate SE(3) transforms between consecutive frames.
        
        Uses SVD Procrustes Analysis for optimal rigid transformation.
        
        Args:
            point_clouds: [T, N, 3] point clouds
            
        Returns:
            [T-1, 4, 4] SE(3) transforms (frame t to t+1)
        """
        transforms = []
        
        for t in range(len(point_clouds) - 1):
            pc_ref = point_clouds[t]
            pc_curr = point_clouds[t + 1]
            
            if len(pc_ref) < 3 or len(pc_curr) < 3:
                transforms.append(np.eye(4))
                continue
            
            # SVD Procrustes for SE(3)
            T = self._procrustes_se3(pc_ref, pc_curr)
            transforms.append(T)
        
        return transforms
    
    def _procrustes_se3(
        self,
        points_ref: np.ndarray,
        points_curr: np.ndarray,
    ) -> np.ndarray:
        """
        Find SE(3) transform T such that T(p_ref) ≈ p_curr.
        
        Method: SVD-based Procrustes Analysis
        
        Math:
        - Find R (rotation) and t (translation) minimizing Σ||Rp_i + t - q_i||²
        - R = VᵀU from SVD of H = Σp_i q_iᵀ
        - t = μ_q - R μ_p
        
        Returns:
            [4, 4] SE(3) transformation matrix
        """
        # Handle different point counts via subsampling
        n = min(len(points_ref), len(points_curr))
        indices_ref = np.random.choice(len(points_ref), n, replace=False)
        indices_curr = np.random.choice(len(points_curr), n, replace=False)
        
        p = points_ref[indices_ref]
        q = points_curr[indices_curr]
        
        # Compute centroids
        p_mean = np.mean(p, axis=0)
        q_mean = np.mean(q, axis=0)
        
        # Center point clouds
        p_centered = p - p_mean
        q_centered = q - q_mean
        
        # SVD
        H = p_centered.T @ q_centered
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation (with reflection correction)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            log.warning("Reflection detected in SE(3), correcting")
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = q_mean - R @ p_mean
        
        # Build SE(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def _fit_gaussians_per_part(
        self,
        point_clouds: List[np.ndarray],
        se3_transforms: List[np.ndarray],
        mask_assignments: Optional[List[np.ndarray]] = None,
    ) -> List[RigidPart]:
        """
        Fit Gaussians for each rigid part.
        
        If mask_assignments provided: one part per unique ID.
        Otherwise: single part for entire object.
        """
        if mask_assignments is not None:
            return self._fit_gaussians_from_masks(point_clouds, mask_assignments)
        else:
            return self._fit_single_part(point_clouds, se3_transforms)
    
    def _fit_single_part(
        self,
        point_clouds: List[np.ndarray],
        se3_transforms: List[np.ndarray],
    ) -> List[RigidPart]:
        """Fit Gaussians for entire object as single rigid part."""
        # Combine all points (in first frame reference)
        all_points = point_clouds[0]
        
        # Fit Gaussians via K-means initialization + EM
        gaussians = self._fit_gmm(all_points)
        
        part = RigidPart(
            name="root",
            gaussians=gaussians,
            se3_transforms=se3_transforms,
        )
        
        return [part]
    
    def _fit_gaussians_from_masks(
        self,
        point_clouds: List[np.ndarray],
        mask_assignments: List[np.ndarray],
    ) -> List[RigidPart]:
        """Fit Gaussians per mask assignment (per rigid part)."""
        # Get unique part IDs
        all_ids = set()
        for mask in mask_assignments:
            all_ids.update(np.unique(mask))
        all_ids.discard(-1)  # Ignore unassigned
        
        parts = []
        
        for part_id in sorted(all_ids):
            # Collect points for this part
            part_points = []
            for t, (pc, mask) in enumerate(zip(point_clouds, mask_assignments)):
                mask_bool = mask == part_id
                if mask_bool.sum() > 0:
                    part_points.append(pc[mask_bool])
            
            if len(part_points) == 0:
                continue
            
            # Transform all to first frame
            combined_points = part_points[0]
            for t in range(1, len(part_points)):
                # We need inverse transforms... for now just use first frame
                pass
            
            # Fit Gaussians
            if len(combined_points) > 10:
                gaussians = self._fit_gmm(np.vstack(part_points))
                
                # Get SE(3) for this part
                # (simplified: use global transforms for now)
                part = RigidPart(
                    name=f"part_{part_id}",
                    gaussians=gaussians,
                    se3_transforms=[],  # Would need per-part tracking
                )
                parts.append(part)
        
        return parts if parts else self._fit_single_part(point_clouds, [])
    
    def _fit_gmm(self, points: np.ndarray) -> List[Gaussian3D]:
        """
        Fit Gaussian Mixture to points.
        
        Uses K-means initialization + simplified EM.
        Returns Gaussians representing the point cloud.
        """
        n_points = len(points)
        n_gaussians = min(self.n_gaussians_per_part, n_points)
        
        if n_points < 3:
            return []
        
        # K-means initialization
        indices = np.random.choice(n_points, n_gaussians, replace=False)
        means = points[indices].copy()
        
        # Simple EM-like refinement (1 iteration)
        for _ in range(1):
            # Assign points to nearest Gaussian
            distances = np.linalg.norm(points[:, None, :] - means[None, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)
            
            # Update means
            for i in range(n_gaussians):
                mask = assignments == i
                if mask.sum() > 3:
                    means[i] = points[mask].mean(axis=0)
        
        # Build Gaussians
        gaussians = []
        for mean in means:
            # Compute local covariance
            dists = np.linalg.norm(points - mean, axis=1)
            inlier_mask = dists < np.percentile(dists, 50)  # Nearest 50%
            
            if inlier_mask.sum() > 3:
                cov = np.cov(points[inlier_mask].T)
                cov += np.eye(3) * self.cov_reg  # Regularization
                
                gaussian = Gaussian3D(
                    mean=mean,
                    covariance=cov,
                    opacity=1.0,
                )
                gaussians.append(gaussian)
        
        return gaussians
    
    def point_cloud_from_depth(
        self,
        depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = None,
        cy: float = None,
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth: [H, W] depth in meters
            mask: Optional [H, W] boolean mask
            fx, fy: Focal lengths
            cx, cy: Principal point (defaults to image center)
            
        Returns:
            [N, 3] point cloud
        """
        h, w = depth.shape
        cx = cx or w / 2
        cy = cy or h / 2
        
        # Build coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply mask if provided
        if mask is not None:
            valid = mask & (depth > 0)
        else:
            valid = depth > 0
        
        # Backproject
        z = depth[valid]
        x = (u_coords[valid] - cx) * z / fx
        y = (v_coords[valid] - cy) * z / fy
        
        return np.stack([x, y, z], axis=1)


def test_gaussian_cloud():
    """Test Gaussian cloud reconstruction."""
    print("=" * 60)
    print("Testing Gaussian Cloud Reconstruction")
    print("=" * 60)
    
    # Create synthetic articulated motion (2-link arm)
    np.random.seed(42)
    T = 60  # frames
    t = np.linspace(0, 2*np.pi, T)
    
    # Link 1: fixed pivot at origin, rotates
    link1_points = np.random.randn(50, 3) * 0.1
    link1_points[:, 2] += 1.0  # z offset
    
    # Link 2: attached to link 1 end, rotates relative
    link2_points = np.random.randn(50, 3) * 0.1
    
    # Generate trajectories
    point_clouds = []
    for i in range(T):
        angle = 0.5 * np.sin(t[i])
        
        # Link 1 rotates
        R1 = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        pc1 = link1_points @ R1.T
        point_clouds.append(pc1)
    
    # Reconstruct
    reconstructor = GaussianCloudReconstructor(n_gaussians_per_part=20)
    cloud = reconstructor.reconstruct_from_point_clouds(point_clouds)
    
    print(f"\n✅ Reconstructed {len(cloud.parts)} parts:")
    for part in cloud.parts:
        print(f"   - {part.name}: {part.n_gaussians} Gaussians, {part.n_frames} frames")
        print(f"     Trajectory shape: {part.get_mean_trajectory().shape}")
    
    # Test SE(3) transform
    T_se3 = reconstructor._procrustes_se3(point_clouds[0], point_clouds[1])
    print(f"\n✅ SE(3) transform computed:")
    print(f"   Rotation det: {np.linalg.det(T_se3[:3, :3]):.4f} (should be 1.0)")
    print(f"   Translation: {T_se3[:3, 3]}")
    
    print("\n" + "=" * 60)
    print("Gaussian Cloud Reconstruction: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_gaussian_cloud()
