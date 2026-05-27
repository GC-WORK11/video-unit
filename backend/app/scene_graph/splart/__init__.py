"""
SPLART: 3D Gaussian Splatting for Articulated Objects
====================================================

V-NEXT replacement for spectral clustering-based kinematic discovery.

Uses:
- 3D Gaussian Splatting for point cloud representation
- SE(3) transformation analysis for rigid body motion
- Pure mathematical DOF analysis (no aspect ratio heuristics)
"""

from .gaussian_cloud import GaussianCloud, GaussianCloudReconstructor
from .reconstruct import SPLARTReconstructor, discover_kinematic_structure

__all__ = [
    "GaussianCloud",
    "GaussianCloudReconstructor", 
    "SPLARTReconstructor",
    "discover_kinematic_structure",
]
